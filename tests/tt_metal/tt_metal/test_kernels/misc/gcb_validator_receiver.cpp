// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Validator receiver for the prefetcher-vs-matmul contract.
// See docs/prefetcher_matmul_design.md for the contract this validates.
//
// Per page:
//   1. wait_front(1)
//   2. read the first 16 bytes of the received page from the receiver's local CB
//   3. DPRINT (iter, fifo_rd_ptr, first 4 u32s)
//   4. pop_front(1)
// After num_iters iterations, attempt one more peek with bounded polling to
// detect the "sender pushed too many pages" case. If detected, DPRINT and hang.
//
// Compile args:
//   0: remote_cb_id           (= 31)
//   1: num_iters              (expected total pages this receiver should see)
//   2: print_stride           (DPRINT every Nth iter; first/last always logged)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"
#include "api/debug/dprint.h"

namespace {

constexpr uint32_t kExtraPollCycles = 1u << 18;  // ~262k spin iterations; should be plenty

}  // namespace

void kernel_main() {
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_iters = get_compile_time_arg_val(1);
    constexpr uint32_t print_stride = get_compile_time_arg_val(2);

    DPRINT << "VALIDATOR_START num_iters=" << num_iters << ENDL();

    for (uint32_t i = 0; i < num_iters; ++i) {
        experimental::remote_cb_wait_front(remote_cb_id, 1);

        RemoteReceiverCBInterface& remote_cb = get_remote_receiver_cb_interface(remote_cb_id);
        volatile tt_l1_ptr uint32_t* page_data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.fifo_rd_ptr);
        const uint32_t u0 = page_data[0];
        const uint32_t u1 = page_data[1];
        const uint32_t u2 = page_data[2];
        const uint32_t u3 = page_data[3];

        const bool log = (i < 4) || (i + 1 == num_iters) || (print_stride > 0 && (i % print_stride == 0));
        if (log) {
            DPRINT << "VALIDATOR iter=" << i << " rd_ptr=0x" << HEX() << remote_cb.fifo_rd_ptr << " u0=0x" << u0
                   << " u1=0x" << u1 << " u2=0x" << u2 << " u3=0x" << u3 << ENDL();
        }

        experimental::remote_cb_pop_front(remote_cb_id, 1);
    }

    DPRINT << "VALIDATOR_LOOP_DONE num_iters=" << num_iters << ENDL();

    // Bounded-poll for an extra page: if it arrives, sender overshot.
    volatile tt_l1_ptr uint32_t* pages_acked_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_remote_receiver_cb_interface(remote_cb_id).aligned_pages_acked_ptr);
    volatile tt_l1_ptr uint32_t* pages_sent_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_remote_receiver_cb_interface(remote_cb_id).aligned_pages_acked_ptr - L1_ALIGNMENT);
    for (uint32_t spin = 0; spin < kExtraPollCycles; ++spin) {
        invalidate_l1_cache();
        const uint32_t sent = *pages_sent_ptr;
        const uint32_t acked = *pages_acked_ptr;
        if (sent != acked) {
            DPRINT << "VALIDATOR_OVERFLOW: sender pushed an extra page; pages_sent=" << sent << " pages_acked=" << acked
                   << ENDL();
            // Intentional hang so the dispatch timeout surfaces this core.
            while (true) {
                ;
            }
        }
    }

    DPRINT << "VALIDATOR_DONE ok num_iters=" << num_iters << ENDL();
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc_async_atomic_barrier();
}
