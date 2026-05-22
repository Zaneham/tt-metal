// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_binary.h"

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#ifdef SFPU_BINARY_OP
#include "api/compute/eltwise_unary/eltwise_unary.h"
#endif

#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);
    uint32_t acc_to_dst = get_arg_val<uint32_t>(2);

    constexpr uint32_t dfb_in0_id = get_compile_time_arg_val(0);
    constexpr uint32_t dfb_in1_id = get_compile_time_arg_val(1);
    constexpr uint32_t dfb_out_id = get_compile_time_arg_val(2);
    experimental::DataflowBuffer dfb_in0(dfb_in0_id);
    experimental::DataflowBuffer dfb_in1(dfb_in1_id);
    experimental::DataflowBuffer dfb_out(dfb_out_id);
    sfpu_unpack_to_dest_hw_init(dfb_in0.get_id(), dfb_in1.get_id(), dfb_out.get_id());
#ifdef SFPU_OP_INIT_0
    SFPU_OP_INIT_0
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        dfb_in0.wait_front(per_core_block_size);
        dfb_in1.wait_front(per_core_block_size);
        dfb_out.reserve_back(per_core_block_size);

        // SFPU binary path (e.g. div_binary). Each pair is its own
        // acquire/release: bring LHS into DST[0], RHS into DST[1], run the op
        // (SFPU_OP_CHAIN_0 expands to e.g. div_binary_tile(0, 1, 0)) so the
        // result lands at DST[0], then pack DST[0]. Two DST slots suffice
        // regardless of per_core_block_size.
        // Re-init before each unpack so the unpacker is pointed at the right
        // input DFB; init configures unpacker state for a single source.
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile_to_dst_init_short(dfb_in0.get_id());
            unpack_tile_to_dest(dfb_in0.get_id(), i, 0);
            copy_tile_to_dst_init_short(dfb_in1.get_id());
            unpack_tile_to_dest(dfb_in1.get_id(), i, 1);
            unpack_tile_to_dest_section_done();
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            sfpu_op_dest_section_done();
            pack_tile<true>(0, dfb_out.get_id(), i);
            pack_tile_dest_dvalid_section_done();
        }

        dfb_in0.pop_front(per_core_block_size);
        dfb_in1.pop_front(per_core_block_size);
        dfb_out.push_back(per_core_block_size);
    }
}
