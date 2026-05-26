// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <tt-metalium/global_circular_buffer.hpp>
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::global_circular_buffer {

// Single Device APIs
GlobalCircularBuffer create_global_circular_buffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

// Multi Device APIs
GlobalCircularBuffer create_global_circular_buffer(
    MeshDevice* mesh_device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

// DRAM-sender variant: senders are programmable DRAM cores identified by DRAM bank id.
// The returned GlobalCircularBuffer is the same type as the worker variant; the sender
// domain is queryable via tt::tt_metal::experimental::sender_core_type(gcb).
GlobalCircularBuffer create_global_circular_buffer_with_dram_senders(
    MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

// Build a DRAM-sender GCB shaped to feed one or more 1D ring matmuls (gather_in0=true)
// from the given weight tensors. The receiver rectangle is laid out as `num_dram_banks`
// columns x `num_global_cb_receivers` rows starting at (0, 0); bank `b` owns the b-th
// contiguous stride of `num_global_cb_receivers` cores in row-major order.
//
// One (program_config, weight) pair per matmul. Each pair is validated independently:
//   * weight K is tile-aligned AND divisible by ring_size (so activation K_per_shard is
//     integer-tile, the silent-hang case where matmul pads K beyond what the prefetcher
//     pushes),
//   * weight N shards evenly across DRAM banks and per-bank N splits evenly across
//     receivers,
//   * matmul's per_core_N matches the weight per-receiver N.
// All configs must agree on compute_with_storage_grid_size and num_global_cb_receivers
// (the GCB has one receiver rectangle shared across all consumer matmuls).
//
// `gcb_size` is picked as a multiple of LCM(in1_block_size for each config), large enough
// to buffer the biggest matmul's `num_buffered_blocks` worth of in1 (matches the production
// llama-70B pattern where a single GCB feeds XQKV/WO/FF1/FF2 with different in1_block sizes
// and the GCB size is a common multiple of all of them).
//
// Picking num_buffered_blocks:
//   * 1 = no overlap; DRISC pushes a block, matmul consumes it, repeat. Throughput is the
//     sum of both stages.
//   * 2 = double-buffer ping-pong (minimum useful value). DRISC writes block N+1 while
//     matmul reads block N. Throughput becomes max(prefetch_time, matmul_time).
//   * 4 (default) = comfortable slack against jitter; fits in L1 for typical shapes.
//   * num_blocks of largest matmul (= max(weight_K_tiles / in0_block_w)) = full-layer
//     decoupling. The DRISC can start pushing layer N+1 while the matmul finishes layer
//     N. Above this, more buffering does not increase throughput (DRISC just stalls on
//     remote_cb_reserve_back). This is what production llama 70B picks.
// Larger values are clamped by an L1 budget (currently ~1.4 MB) so the GCB leaves room
// for the matmul's in0/out/interm CBs. If you hit L1 OOM, lower this.
GlobalCircularBuffer create_global_circular_buffer_for_matmul_1d(
    MeshDevice* mesh_device,
    const std::vector<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>& program_configs,
    const std::vector<tt::tt_metal::Tensor>& weights,
    uint32_t num_buffered_blocks = 4,
    BufferType buffer_type = BufferType::L1);

}  // namespace ttnn::global_circular_buffer
