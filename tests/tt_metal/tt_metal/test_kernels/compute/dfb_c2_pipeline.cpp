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

// C2 compute kernel: three-DFB pipeline that loops a tile through the same
// TRISC twice using the SFPU (relu) between stages.
//
// DFB ids (caller registers in this exact order):
//   0  inter, DM → TRISC      (DM producer fills dfb_in)
//   1  intra, TRISC → TRISC   (self-loop; this kernel is both producer and consumer)
//   2  inter, TRISC → DM      (DM consumer drains dfb_out)
//
// Per tile:
//   stage A: unpack dfb_in  → SFPU relu → pack dfb_self
//   stage B: unpack dfb_self → SFPU relu → pack dfb_out
//
// For positive bf16 inputs, double-relu is identity, so the host verifies
// output ≈ input (bf16 tolerance, same approach as DMTensixDMTest2xDFB1Sx1S_Relu).
//
// Compile-time args:
//   0: per_core_tile_cnt  - number of tiles to process
void kernel_main() {
    const uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    // Init for unary pipeline; both stages reuse the same SFPU op (relu).
    unary_op_init_common(0, 2);
    relu_tile_init();

    experimental::DataflowBuffer dfb_in(0);
    experimental::DataflowBuffer dfb_self(1);
    experimental::DataflowBuffer dfb_out(2);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        // Stage A: dfb_in → relu → dfb_self  (inter-tensix in, intra-tensix self-loop out)
        acquire_dst();
        dfb_in.wait_front(1);
        dfb_self.reserve_back(1);
        copy_tile(dfb_in.get_id(), 0, 0);
        relu_tile(0);
        pack_tile(0, dfb_self.get_id());
        dfb_in.pop_front(1);
        dfb_self.push_back(1);
        release_dst();

        // Stage B: dfb_self → relu → dfb_out (intra-tensix self-loop in, inter-tensix out)
        acquire_dst();
        dfb_self.wait_front(1);
        dfb_out.reserve_back(1);
        copy_tile(dfb_self.get_id(), 0, 0);
        relu_tile(0);
        pack_tile(0, dfb_out.get_id());
        dfb_self.pop_front(1);
        dfb_out.push_back(1);
        release_dst();
    }
}
