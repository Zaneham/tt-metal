// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"
#include "experimental/dataflow_buffer.h"
#ifndef ARCH_QUASAR
#include "experimental/circular_buffer.h"
#endif

// Mirrors eltwise_copy.cpp but inserts an SFPU relu_tile() between unpack and pack.
// Output = max(0, input) per element.
//
// relu_tile is the one SFPU op currently ported on Quasar (the rest of the
// Quasar SFPU LLK is empty), so this kernel works on Quasar/WH/BH alike --
// useful when you want to exercise the SFPU path on Quasar before more ops
// land.
//
// Compile-time args:
//   0: per_core_tile_cnt   - number of tiles to process
//   1: use_dfbs (bool)     - 1 to use DFBs (Quasar + WH/BH DFB path), 0 to use CBs (WH/BH legacy)
void kernel_main() {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    constexpr bool use_dfbs = get_compile_time_arg_val(1) == 1;

#ifdef ARCH_QUASAR
    static_assert(use_dfbs, "DFBs need to be used for Quasar!");
#endif

    if constexpr (use_dfbs) {
        unary_op_init_common(0, 1);
    } else {
        unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
    }
    relu_tile_init();

    if constexpr (use_dfbs) {
        experimental::DataflowBuffer dfb_in(0);
        experimental::DataflowBuffer dfb_out(1);
        for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
            acquire_dst();

            dfb_in.wait_front(1);
            dfb_out.reserve_back(1);
            copy_tile(dfb_in.get_id(), 0, 0);
            relu_tile(0);
            pack_tile(0, dfb_out.get_id());
            dfb_in.pop_front(1);
            dfb_out.push_back(1);

            release_dst();
        }
    } else {
#ifndef ARCH_QUASAR
        experimental::CircularBuffer cb0(tt::CBIndex::c_0);
        experimental::CircularBuffer cb16(tt::CBIndex::c_16);
        for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
            acquire_dst();

            cb0.wait_front(1);
            cb16.reserve_back(1);
            copy_tile(tt::CBIndex::c_0, 0, 0);
            relu_tile(0);
            pack_tile(0, tt::CBIndex::c_16);
            cb0.pop_front(1);
            cb16.push_back(1);

            release_dst();
        }
#endif
    }
}
