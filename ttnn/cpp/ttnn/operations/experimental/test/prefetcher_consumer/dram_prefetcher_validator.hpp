// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace ttnn::operations::experimental::test {

// Diagnostic companion to ttnn.dram_prefetcher / ttnn.start_dram_core_prefetcher. Enqueues
// a Program that loads a validator kernel on each receiver core of the supplied GCB. The
// kernel does `wait_front(1); peek-and-DPRINT-first-16-bytes; pop_front(1)` for num_iters
// iterations, then polls briefly for any extra pages (indicating sender overshoot) and
// either DPRINTs success or DPRINTs + hangs on overflow.
//
// `global_cb` may be either a worker-sender or DRAM-sender GCB; the receiver-side kernel
// is identical (it just consumes from remote CB index 31).
void test_dram_prefetcher_validator(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint32_t num_iters,
    uint32_t page_size_bytes,
    uint32_t print_stride,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb);

}  // namespace ttnn::operations::experimental::test
