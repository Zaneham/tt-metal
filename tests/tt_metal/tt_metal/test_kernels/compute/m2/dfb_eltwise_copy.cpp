// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) eltwise copy compute kernel.
// Identity: copy each tile from dfb::in to dfb::out via copy_tile + pack_tile.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_arg(args::per_core_tile_cnt);

    unary_op_init_common(0, 1);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        acquire_dst();
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);
        copy_tile(dfb_in.get_id(), 0, 0);
        pack_tile(0, dfb_out.get_id());
        dfb_in.pop_front(1);
        dfb_out.push_back(1);
        release_dst();
    }
}
