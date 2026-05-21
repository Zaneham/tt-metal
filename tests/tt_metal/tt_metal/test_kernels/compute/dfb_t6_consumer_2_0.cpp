// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) Tensix consumer for run_single_dfb_program_2_0.
// Just drains credits; host reads L1 directly to verify the data landed.

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);

    DataflowBuffer dfb(dfb::in);

    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; ++tile_id) {
        dfb.wait_front(1);
        dfb.pop_front(1);
    }
    dfb.finish();
}
