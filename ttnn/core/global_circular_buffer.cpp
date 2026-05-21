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
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const tt::tt_metal::Tensor& weight,
    uint32_t num_buffered_blocks,
    BufferType buffer_type) {
    TT_FATAL(
        program_config.gather_in0,
        "create_global_circular_buffer_for_matmul_1d requires gather_in0=true; the DRAM-sender "
        "GCB path only supports the gather_in0 matmul factory");
    TT_FATAL(program_config.num_global_cb_receivers > 0, "num_global_cb_receivers must be > 0");
    TT_FATAL(num_buffered_blocks > 0, "num_buffered_blocks must be > 0");

    const uint32_t num_dram_banks = mesh_device->dram_grid_size().x;
    const uint32_t num_recv_per_bank = static_cast<uint32_t>(program_config.num_global_cb_receivers);
    const auto& grid = program_config.compute_with_storage_grid_size;
    const uint32_t ring_cols = grid.x;
    const uint32_t ring_rows = grid.y;
    const uint32_t ring_size = ring_cols * ring_rows;

    TT_FATAL(
        ring_cols == num_dram_banks,
        "program_config.compute_with_storage_grid_size.x ({}) must equal num_dram_banks ({}); the "
        "receiver rectangle has one column per DRAM bank",
        ring_cols,
        num_dram_banks);
    TT_FATAL(
        ring_rows == num_recv_per_bank,
        "program_config.compute_with_storage_grid_size.y ({}) must equal num_global_cb_receivers ({}); "
        "the receiver rectangle has num_global_cb_receivers rows",
        ring_rows,
        num_recv_per_bank);

    // ---- Weight shape & dtype ----
    const auto& weight_padded = weight.padded_shape();
    TT_FATAL(weight_padded.rank() >= 2, "Weight must be at least 2D; got rank {}", weight_padded.rank());
    const auto& tile = weight.tensor_spec().tile();
    const uint32_t tile_h = tile.get_height();
    const uint32_t tile_w = tile.get_width();
    const uint32_t weight_K = weight_padded[-2];
    const uint32_t weight_N = weight_padded[-1];
    TT_FATAL(weight_K % tile_h == 0, "Weight K ({}) must be tile-aligned (tile_h={})", weight_K, tile_h);
    TT_FATAL(weight_N % tile_w == 0, "Weight N ({}) must be tile-aligned (tile_w={})", weight_N, tile_w);
    const uint32_t weight_K_tiles = weight_K / tile_h;
    const uint32_t weight_N_tiles = weight_N / tile_w;

    // ---- The silent-hang check: weight K must be divisible by ring_size ----
    //
    // The gather_in0 matmul derives in0 block geometry from the activation's sharded K
    // (K_per_shard, rounded up to a tile). If weight_K % ring_size != 0, K_per_shard
    // padded across ring_size workers exceeds weight_K, so the matmul reads more in1
    // blocks than the prefetcher pushes -> remote_cb_wait_front hangs.
    TT_FATAL(
        weight_K_tiles % ring_size == 0,
        "Weight K must be divisible by ring_size in tiles. Got weight_K_tiles={}, ring_size={} "
        "(remainder={}). If we proceeded, the matmul activation grid would pad K beyond what the "
        "prefetcher pushes and the receivers would wait forever for in1 pages.",
        weight_K_tiles,
        ring_size,
        weight_K_tiles % ring_size);
    TT_FATAL(
        weight_K_tiles % program_config.in0_block_w == 0,
        "Weight K ({} tiles) must be divisible by in0_block_w ({})",
        weight_K_tiles,
        program_config.in0_block_w);
    TT_FATAL(
        weight_N_tiles % num_dram_banks == 0,
        "Weight N ({} tiles) must be divisible by num_dram_banks ({}) so it shards evenly across banks",
        weight_N_tiles,
        num_dram_banks);

    // ---- Weight DRAM shard layout ----
    TT_FATAL(
        weight.buffer() != nullptr && weight.buffer()->is_dram(),
        "Weight must live in DRAM (got buffer_type={})",
        static_cast<int>(weight.buffer() ? weight.buffer()->buffer_type() : tt::tt_metal::BufferType::L1));
    const auto& shard_shape = weight.buffer()->shard_spec().shape();
    const uint32_t shard_K = shard_shape[0];
    const uint32_t shard_N = shard_shape[1];
    TT_FATAL(
        shard_K == weight_K,
        "Weight DRAM shard K ({}) must equal full K ({}); weight must be width-sharded across banks "
        "with each bank holding the full K dimension",
        shard_K,
        weight_K);
    TT_FATAL(
        shard_N * num_dram_banks == weight_N,
        "Weight DRAM shard N ({}) * num_dram_banks ({}) must equal full N ({})",
        shard_N,
        num_dram_banks,
        weight_N);
    const uint32_t shard_N_tiles = shard_N / tile_w;
    TT_FATAL(
        shard_N_tiles % num_recv_per_bank == 0,
        "Weight per-bank N ({} tiles) must be divisible by num_global_cb_receivers ({})",
        shard_N_tiles,
        num_recv_per_bank);

    const uint32_t per_recv_N_tiles = shard_N_tiles / num_recv_per_bank;
    TT_FATAL(
        per_recv_N_tiles == program_config.per_core_N,
        "Matmul per_core_N ({}) must equal per-receiver N ({} = shard_N_tiles {} / num_global_cb_receivers {})",
        program_config.per_core_N,
        per_recv_N_tiles,
        shard_N_tiles,
        num_recv_per_bank);

    // ---- GCB size: must be an exact multiple of in1_block_size ----
    //
    // remote_cb_wait_front does wrap-adjustment using (fifo_size % in1_block_size). If
    // that remainder is nonzero, the receiver miscounts at the layer boundary and waits
    // forever for pages the sender will never push.
    const uint32_t bytes_per_tile = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(weight.dtype()));
    const uint32_t in1_block_size =
        static_cast<uint32_t>(program_config.in0_block_w * program_config.per_core_N) * bytes_per_tile;
    TT_FATAL(in1_block_size > 0, "in1_block_size computed as 0");

    // Leave room in L1 for matmul in0/out/interm CBs. ~1.4MB is a conservative bound
    // that's worked in practice; callers needing more can lower num_buffered_blocks.
    constexpr uint32_t kL1Cap = 1'400'000;
    constexpr uint32_t kMaxCbPagesBytes = 65000u * 16u;  // <65535 pages * 16B page = ~1MB
    const uint32_t upper_l1 = std::max(in1_block_size, kL1Cap > in1_block_size ? kL1Cap - in1_block_size : 0);
    const uint32_t upper = std::min(upper_l1, kMaxCbPagesBytes);
    const uint32_t desired = in1_block_size * num_buffered_blocks;
    uint32_t gcb_size = std::min(desired, upper);
    gcb_size = std::max(in1_block_size, (gcb_size / in1_block_size) * in1_block_size);

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
        mesh_device, bank_to_receivers, gcb_size, buffer_type);
}

}  // namespace ttnn::global_circular_buffer
