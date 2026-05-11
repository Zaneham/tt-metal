// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif

// Binary SFPU compute kernel — Quasar-compatible single-source layout.
//
// Adapts the working Quasar LLK div pattern (tests/sources/quasar/sfpu_binary_
// div_quasar_test.cpp + test_sfpu_binary_div_quasar.py): both operands live in
// one contiguous L1 buffer so the unpacker's buffer descriptor never changes
// across operand boundaries (which produced zero output on Quasar when copy_
// tile was issued against a different CB).
//
// Host responsibilities:
//   * Pack inputs INTERLEAVED in one DRAM buffer as
//     [LHS_0, RHS_0, LHS_1, RHS_1, ...] — i.e. pair `i` sits at CB indices
//     2*i and 2*i + 1.
//   * Reader pushes 2 tiles per pair into the single input CB/DFB.
//
// Per acquire/release the kernel only ever touches DST[0] and DST[1], so it
// fits comfortably inside one half-sync section regardless of how many pairs
// we process.
//
// Macros (set by host via defines):
//   SFPU_OP_INIT_0  - once before the main loop (e.g. div_binary_tile_init())
//   SFPU_OP_CHAIN_0 - per pair inside the loop. LHS is at DST[0], RHS at
//                     DST[1], result written back to DST[0].
void kernel_main() {
    // Compile-time args set by the host:
    //   0: per_core_block_cnt — number of outer blocks. The current test uses
    //      1 block and lets the inner loop handle every tile pair.
    //   1: per_core_block_dim — number of (LHS, RHS) PAIRS per block. The
    //      reader is expected to push 2 * per_core_block_dim tiles.
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    // Open the input/output staging surface. Pre-Quasar uses CircularBuffers
    // (CBs c_0/c_16); Quasar uses DataflowBuffers whose IDs come in as extra
    // compile-time args. From here on the loop body is identical.
#ifdef ARCH_QUASAR
    constexpr uint32_t dfb_in_id = get_compile_time_arg_val(2);
    constexpr uint32_t dfb_out_id = get_compile_time_arg_val(3);
    experimental::DataflowBuffer buff_in(dfb_in_id);
    experimental::DataflowBuffer buff_out(dfb_out_id);
    const uint32_t in_id = buff_in.get_id();
    const uint32_t out_id = buff_out.get_id();
#else
    experimental::CircularBuffer buff_in(tt::CBIndex::c_0);
    experimental::CircularBuffer buff_out(tt::CBIndex::c_16);
    const uint32_t in_id = tt::CBIndex::c_0;
    const uint32_t out_id = tt::CBIndex::c_16;
#endif

    // One-time SFPU setup. SFPU_OP_INIT_0 is the op-specific init (e.g.
    // div_binary_tile_init) injected by the host via the kernel `defines` map.
    init_sfpu(in_id, out_id);
#ifdef SFPU_OP_INIT_0
    SFPU_OP_INIT_0
#endif
    copy_tile_to_dst_init_short(in_id);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve enough output slots for this block up-front, so the packer
        // can push results as soon as each pair is done.
        buff_out.reserve_back(per_core_block_dim);

        for (uint32_t i = 0; i < per_core_block_dim; ++i) {
            // Pair-level pipeline:
            //   acquire DST -> wait for 2 input tiles -> copy into DST[0]/DST[1]
            //   -> run SFPU op (result lands in DST[0]) -> commit -> pack
            //   -> pop the consumed pair -> release DST.
            tile_regs_acquire();
            // Wait for one (LHS, RHS) pair to land in the input buffer.
            buff_in.wait_front(2);

            // Bring the two operands into the math DST register file.
            // LHS at DST[0], RHS at DST[1] — the convention SFPU_OP_CHAIN_0
            // (e.g. div_binary_tile(0, 1, 0)) reads from.
            copy_tile(in_id, 0, 0);
            copy_tile(in_id, 1, 1);
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            tile_regs_commit();

            // Hand DST off to the packer thread, then emit DST[0] (the result)
            // into the output buffer.
            tile_regs_wait();
            pack_tile(0, out_id);
            buff_in.pop_front(2);
            tile_regs_release();
        }

        // Publish the whole block of results downstream.
        buff_out.push_back(per_core_block_dim);
    }
}
