// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::test {

// Byte-for-byte validator companion to ttnn.dram_prefetcher /
// ttnn.experimental.start_dram_core_prefetcher. Enqueues a Program loading a
// validator kernel on each receiver core of the supplied GCB. For each pushed
// page the kernel reads the receiver's expected tile range from `source_tensor`
// via TensorAccessor (no hand-rolled bank arithmetic) and memcmps against the
// received page; on any mismatch it DPRINTs the failing (layer, block, word)
// triple plus the diverging bytes and hangs the core.
//
// `global_cb` may be either a worker-sender or DRAM-sender GCB; the kernel is
// identical for both. `source_tensor` must be the same width-sharded DRAM tensor
// the prefetcher is being driven with. The per-(bank, receiver, block) -> tile
// mapping the kernel uses is documented in
// tt_metal/impl/buffers/prefetcher_matmul_design.md §3 ("Per-block source tiles").
void test_dram_prefetcher_validator(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const ttnn::Tensor& source_tensor,
    uint32_t num_layers,
    uint32_t print_stride,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb);

}  // namespace ttnn::operations::experimental::test
