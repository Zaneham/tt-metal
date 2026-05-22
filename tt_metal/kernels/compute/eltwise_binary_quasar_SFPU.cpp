// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_binary.h"

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#ifdef SFPU_BINARY_OP
#include "api/compute/eltwise_unary/eltwise_unary.h"
#endif

void kernel_main() {
    const uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    const uint32_t per_core_block_size = get_arg(args::per_core_block_size);
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);

    // One-time HW init: dest dvalid chain (unpack -> SFPU -> pack) and unpack/pack config.
    sfpu_unpack_to_dest_hw_init(dfb_in0.get_id(), dfb_in1.get_id(), dfb_out.get_id());
#ifdef SFPU_OP_INIT_0
    SFPU_OP_INIT_0
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        // Wait for reader to produce both operands; reserve output slots for compute.
        dfb_in0.wait_front(per_core_block_size);
        dfb_in1.wait_front(per_core_block_size);
        dfb_out.reserve_back(per_core_block_size);

        // Per-tile: unpack LHS/RHS to DST[0]/DST[1], run SFPU op, pack DST[0].
        // Re-init before each unpack so the unpacker points at the right input DFB.
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

        // Release consumed inputs and publish completed output tiles.
        dfb_in0.pop_front(per_core_block_size);
        dfb_in1.pop_front(per_core_block_size);
        dfb_out.push_back(per_core_block_size);
    }
}
