// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_consumer_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "dram_prefetcher_consumer.hpp"
#include "dram_prefetcher_validator.hpp"

namespace ttnn::operations::dram_prefetcher_consumer::detail {

void bind_dram_prefetcher_consumer(nb::module_& mod) {
    ttnn::bind_function<"dram_prefetcher_consumer">(
        mod,
        R"doc(
            Bench-only consumer companion to ttnn.dram_prefetcher. Builds and enqueues a
            program that loads a discard receiver kernel on each receiver core of the supplied
            GCB, looping num_iters times of wait_front(1)+pop_front(1). Used to measure
            prefetcher push bandwidth without matmul receiver-side effects.

            Args:
                mesh_device: the MeshDevice to enqueue on.
                num_iters (int): total pages each receiver should consume (= num_layers * num_blocks).
                page_size_bytes (int): receiver-side page size; must match what the sender pushes
                    (in0_block_w_tiles * n_tiles_per_receiver * tile_bytes).
                global_cb (GlobalCircularBuffer): either a worker-sender or DRAM-sender GCB.
        )doc",
        &ttnn::dram_prefetcher_consumer,
        nb::arg("mesh_device"),
        nb::arg("num_iters"),
        nb::arg("page_size_bytes"),
        nb::kw_only(),
        nb::arg("global_cb"));

    ttnn::bind_function<"dram_prefetcher_validator">(
        mod,
        R"doc(
            Diagnostic receiver: wait_front(1) + DPRINT(first 16 bytes) + pop_front(1) for
            num_iters iterations, then polls briefly for any extra pages (sender overshoot)
            and DPRINTs success or DPRINTs + hangs on overflow. Used to debug prefetcher
            push behavior without involving the matmul kernels.

            Args:
                mesh_device: the MeshDevice to enqueue on.
                num_iters (int): expected total pages each receiver should see.
                page_size_bytes (int): receiver-side page size (must match the sender push size).
                print_stride (int): DPRINT every Nth iter; first/last always logged. 0 = first/last only.
                global_cb (GlobalCircularBuffer): worker-sender or DRAM-sender GCB.
        )doc",
        &ttnn::dram_prefetcher_validator,
        nb::arg("mesh_device"),
        nb::arg("num_iters"),
        nb::arg("page_size_bytes"),
        nb::arg("print_stride"),
        nb::kw_only(),
        nb::arg("global_cb"));
}

}  // namespace ttnn::operations::dram_prefetcher_consumer::detail
