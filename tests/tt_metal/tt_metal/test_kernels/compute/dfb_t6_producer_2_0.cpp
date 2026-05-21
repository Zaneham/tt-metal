// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) Tensix producer for run_single_dfb_program_2_0.
// Host pre-fills the DFB L1 ring; this kernel only posts credits.

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);

    DataflowBuffer dfb(dfb::out);

    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; ++tile_id) {
        dfb.reserve_back(1);
        dfb.push_back(1);
    }
    dfb.finish();
}
