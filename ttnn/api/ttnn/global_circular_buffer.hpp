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
// MeshDevice-only because the per-mesh DRISC L1 arena lives on MeshDeviceImpl. The
// returned GlobalCircularBuffer is the same type as the worker variant; the sender
// domain is queryable via tt::tt_metal::experimental::sender_core_type(gcb).
GlobalCircularBuffer create_global_circular_buffer_with_dram_senders(
    MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

// Build a DRAM-sender GCB shaped to feed a 1D ring matmul (gather_in0=true) with the
// given weight tensor, with size/page-stride/receiver-layout derived from the matmul's
// program config. The receiver rectangle is laid out as `num_dram_banks` columns x
// `num_global_cb_receivers` rows starting at (0, 0); bank `b` owns the b-th contiguous
// stride of `num_global_cb_receivers` cores in row-major order.
//
// Validates: weight K is tile-aligned AND divisible by ring_size (so activation
// K_per_shard is integer-tile, the silent-hang case where matmul pads K beyond what
// the prefetcher pushes); weight N shards evenly across DRAM banks and per-bank N
// splits evenly across receivers; matmul's per_core_N matches the weight per-receiver N.
//
// `num_buffered_blocks` controls how many in1 blocks fit in the GCB ring per receiver.
GlobalCircularBuffer create_global_circular_buffer_for_matmul_1d(
    MeshDevice* mesh_device,
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const tt::tt_metal::Tensor& weight,
    uint32_t num_buffered_blocks = 4,
    BufferType buffer_type = BufferType::L1);

}  // namespace ttnn::global_circular_buffer
