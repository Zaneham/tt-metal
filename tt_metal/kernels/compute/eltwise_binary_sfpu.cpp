// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_binary.h"

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// WH/BH binary SFPU kernel: copy_tile both operands into DST[0]/DST[1], run SFPU op, pack DST[0].
// Matches the copy_tile + fp32_dest_acc path used by eltwise_binary_quasar_SFPU.cpp on Quasar.

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    binary_op_init_common(cb_in0, cb_in1, cb_out);
#ifdef SFPU_OP_INIT_0
    SFPU_OP_INIT_0
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_wait_front(cb_in0, per_core_block_size);
        cb_wait_front(cb_in1, per_core_block_size);
        cb_reserve_back(cb_out, per_core_block_size);

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            acquire_dst();
            copy_tile_to_dst_init_short(cb_in0);
            copy_tile(cb_in0, i, 0);
            copy_tile_to_dst_init_short(cb_in1);
            copy_tile(cb_in1, i, 1);
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            pack_tile(0, cb_out);
            release_dst();
        }

        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);
        cb_push_back(cb_out, per_core_block_size);
    }
}
