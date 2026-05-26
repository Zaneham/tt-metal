// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/global_circular_buffer.hpp"

#include <algorithm>
#include <memory>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::global_circular_buffer {

GlobalCircularBuffer create_global_circular_buffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) {
    return tt::tt_metal::experimental::CreateGlobalCircularBuffer(
        device, sender_receiver_core_mapping, size, buffer_type);
}

GlobalCircularBuffer create_global_circular_buffer(
    MeshDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) {
    return tt::tt_metal::experimental::CreateGlobalCircularBuffer(
        device, sender_receiver_core_mapping, size, buffer_type);
}

GlobalCircularBuffer create_global_circular_buffer_with_dram_senders(
    MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type) {
    return tt::tt_metal::experimental::CreateGlobalCircularBufferWithDramSenders(
        mesh_device, bank_to_receivers, size, buffer_type);
}

GlobalCircularBuffer create_global_circular_buffer_for_matmul_1d(
    MeshDevice* mesh_device,
    const std::vector<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>& program_configs,
    const std::vector<tt::tt_metal::Tensor>& weights,
    uint32_t size,
    BufferType buffer_type) {
    TT_FATAL(!program_configs.empty(), "Must provide at least one program config");
    TT_FATAL(
        program_configs.size() == weights.size(),
        "Expected one weight tensor per program config; got {} configs and {} weights",
        program_configs.size(),
        weights.size());
    TT_FATAL(size > 0, "size must be > 0");

    // All matmuls share the same GCB receiver rectangle, so they must all agree on the
    // ring shape and per-bank receiver count.
    const auto& first = program_configs.front();
    TT_FATAL(
        first.gather_in0,
        "create_global_circular_buffer_for_matmul_1d requires gather_in0=true on every program "
        "config; config[0] has gather_in0=false");
    TT_FATAL(first.num_global_cb_receivers > 0, "config[0].num_global_cb_receivers must be > 0");

    const auto& grid = first.compute_with_storage_grid_size;
    const uint32_t ring_cols = grid.x;
    const uint32_t ring_rows = grid.y;
    const uint32_t ring_size = ring_cols * ring_rows;
    const uint32_t num_dram_banks = mesh_device->dram_grid_size().x;
    const uint32_t num_recv_per_bank = static_cast<uint32_t>(first.num_global_cb_receivers);

    TT_FATAL(
        ring_cols == num_dram_banks,
        "program_config[0].compute_with_storage_grid_size.x ({}) must equal num_dram_banks ({})",
        ring_cols,
        num_dram_banks);
    TT_FATAL(
        ring_rows == num_recv_per_bank,
        "program_config[0].compute_with_storage_grid_size.y ({}) must equal num_global_cb_receivers ({})",
        ring_rows,
        num_recv_per_bank);

    // Validate every (config, weight) pair against the matmul invariants, and collect
    // the largest in1_block_size to size the buffer.
    uint32_t max_in1_block_size = 0;
    for (size_t i = 0; i < program_configs.size(); ++i) {
        const auto& cfg = program_configs[i];
        const auto& w = weights[i];

        TT_FATAL(cfg.gather_in0, "config[{}].gather_in0 must be true", i);
        TT_FATAL(cfg.num_global_cb_receivers > 0, "config[{}].num_global_cb_receivers must be > 0", i);
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x == ring_cols && cfg.compute_with_storage_grid_size.y == ring_rows,
            "config[{}] has compute_with_storage_grid_size {{{}, {}}}; must match config[0] {{{}, {}}} "
            "(all matmuls sharing a GCB must use the same receiver rectangle)",
            i,
            cfg.compute_with_storage_grid_size.x,
            cfg.compute_with_storage_grid_size.y,
            ring_cols,
            ring_rows);
        TT_FATAL(
            static_cast<uint32_t>(cfg.num_global_cb_receivers) == num_recv_per_bank,
            "config[{}].num_global_cb_receivers ({}) must match config[0] ({}); the GCB has a single "
            "receiver-per-bank count shared across all matmuls",
            i,
            cfg.num_global_cb_receivers,
            num_recv_per_bank);

        // ---- Weight shape & dtype ----
        const auto& wp = w.padded_shape();
        TT_FATAL(wp.rank() >= 2, "weights[{}] must be at least 2D; got rank {}", i, wp.rank());
        const auto& tile = w.tensor_spec().tile();
        const uint32_t tile_h = tile.get_height();
        const uint32_t tile_w = tile.get_width();
        const uint32_t weight_K = wp[-2];
        const uint32_t weight_N = wp[-1];
        TT_FATAL(weight_K % tile_h == 0, "weights[{}] K ({}) must be tile-aligned (tile_h={})", i, weight_K, tile_h);
        TT_FATAL(weight_N % tile_w == 0, "weights[{}] N ({}) must be tile-aligned (tile_w={})", i, weight_N, tile_w);
        const uint32_t weight_K_tiles = weight_K / tile_h;
        const uint32_t weight_N_tiles = weight_N / tile_w;

        // ---- The silent-hang check ----
        TT_FATAL(
            weight_K_tiles % ring_size == 0,
            "weights[{}] K must be divisible by ring_size in tiles. Got weight_K_tiles={}, ring_size={} "
            "(remainder={}). The matmul activation grid would pad K beyond what the prefetcher pushes "
            "and the receivers would wait forever for in1 pages.",
            i,
            weight_K_tiles,
            ring_size,
            weight_K_tiles % ring_size);
        TT_FATAL(
            weight_K_tiles % cfg.in0_block_w == 0,
            "weights[{}] K ({} tiles) must be divisible by config[{}].in0_block_w ({})",
            i,
            weight_K_tiles,
            i,
            cfg.in0_block_w);
        TT_FATAL(
            weight_N_tiles % num_dram_banks == 0,
            "weights[{}] N ({} tiles) must be divisible by num_dram_banks ({})",
            i,
            weight_N_tiles,
            num_dram_banks);

        // ---- Weight DRAM shard layout ----
        TT_FATAL(w.buffer() != nullptr && w.buffer()->is_dram(), "weights[{}] must live in DRAM", i);
        const auto& shard_shape = w.buffer()->shard_spec().shape();
        const uint32_t shard_K = shard_shape[0];
        const uint32_t shard_N = shard_shape[1];
        TT_FATAL(
            shard_K == weight_K,
            "weights[{}] DRAM shard K ({}) must equal full K ({}); weight must be width-sharded across "
            "banks with each bank holding the full K dimension",
            i,
            shard_K,
            weight_K);
        TT_FATAL(
            shard_N * num_dram_banks == weight_N,
            "weights[{}] DRAM shard N ({}) * num_dram_banks ({}) must equal full N ({})",
            i,
            shard_N,
            num_dram_banks,
            weight_N);
        const uint32_t shard_N_tiles = shard_N / tile_w;
        TT_FATAL(
            shard_N_tiles % num_recv_per_bank == 0,
            "weights[{}] per-bank N ({} tiles) must be divisible by num_global_cb_receivers ({})",
            i,
            shard_N_tiles,
            num_recv_per_bank);

        const uint32_t per_recv_N_tiles = shard_N_tiles / num_recv_per_bank;
        TT_FATAL(
            per_recv_N_tiles == cfg.per_core_N,
            "config[{}].per_core_N ({}) must equal weights[{}] per-receiver N ({} = shard_N_tiles {} "
            "/ num_global_cb_receivers {})",
            i,
            cfg.per_core_N,
            i,
            per_recv_N_tiles,
            shard_N_tiles,
            num_recv_per_bank);

        // ---- This matmul's in1 block size ----
        // gather_in0 matmul derives its effective in0_block_w from weight_K_tiles / ring_size,
        // not from cfg.in0_block_w (which is typically left at 1 in the program config). Use the
        // same derivation here so the GCB page matches what the matmul will actually consume.
        const uint32_t actual_in0_block_w = weight_K_tiles / ring_size;
        const uint32_t bytes_per_tile = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(w.dtype()));
        const uint32_t in1_block_size = static_cast<uint32_t>(actual_in0_block_w * cfg.per_core_N) * bytes_per_tile;
        TT_FATAL(in1_block_size > 0, "config[{}] in1_block_size computed as 0", i);
        max_in1_block_size = std::max(max_in1_block_size, in1_block_size);
    }

    // ---- Validate the caller-supplied size fits the remote-CB page-count cap and at
    // least one full layer's worth of pages (the matmul does wait_front(num_blocks)).
    //
    // No L1 budget check here — receivers may have very different L1 usage on top of the
    // GCB (matmul in0/in1/out/interm CBs etc.) and we don't have enough context at the
    // factory to compute a real cap. Callers must size the GCB to fit their own L1.
    //
    // kMaxCbPagesBytes is a conservative cap on fifo_aligned_num_pages = fifo_size /
    // REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE. The remote-CB receiver tracks pages with
    // 32-bit counters wrapped at 2^31 (noc_fast_atomic_increment wrap=31) and computes
    // free_pages = fifo_aligned_num_pages - (pages_sent - pages_acked) in unsigned 32-bit
    // arithmetic. Keeping fifo_aligned_num_pages well under 2^16 leaves orders of magnitude
    // between the counter range and any plausible in-flight count, so signed/unsigned
    // interpretation of the difference can never misfire.
    constexpr uint32_t kMaxCbPagesBytes = 65000u * 16u;
    const uint32_t num_blocks = ring_size;
    const uint32_t min_size = max_in1_block_size * num_blocks;
    TT_FATAL(
        size >= min_size,
        "GCB size ({} B) must be at least num_blocks * largest in1_block ({} * {} = {} B); the "
        "matmul does wait_front(num_blocks) so it needs that many pages buffered before it consumes.",
        size,
        num_blocks,
        max_in1_block_size,
        min_size);
    TT_FATAL(
        size <= kMaxCbPagesBytes,
        "GCB size ({} B) exceeds the remote-CB page-count cap ({} B). Reduce size.",
        size,
        kMaxCbPagesBytes);

    // ---- Build bank_to_receivers (row-major through the ring rectangle) ----
    std::vector<std::pair<uint32_t, CoreRangeSet>> bank_to_receivers;
    bank_to_receivers.reserve(num_dram_banks);
    for (uint32_t b = 0; b < num_dram_banks; ++b) {
        std::vector<CoreRange> ranges;
        ranges.reserve(num_recv_per_bank);
        for (uint32_t k = 0; k < num_recv_per_bank; ++k) {
            const uint32_t ring_pos = b * num_recv_per_bank + k;
            const uint32_t col = ring_pos % ring_cols;
            const uint32_t row = ring_pos / ring_cols;
            ranges.emplace_back(CoreCoord{col, row}, CoreCoord{col, row});
        }
        bank_to_receivers.emplace_back(b, CoreRangeSet(ranges));
    }

    return tt::tt_metal::experimental::CreateGlobalCircularBufferWithDramSenders(
        mesh_device, bank_to_receivers, size, buffer_type);
}

}  // namespace ttnn::global_circular_buffer
