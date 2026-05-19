// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/semaphore.h"
#include "api/debug/dprint.h"
#include "experimental/dataflow_buffer.h"

// C1 compute kernel: INTRA self-loop on dfb_self (id 0), bookended by semaphore
// synchronization with the DM producer and DM consumer kernels.
//
// Flow:
//   1. Wait on sem_input_ready (signal from DM producer that dfb_self's L1 has data).
//   2. Run dfb_t6_intra pattern on dfb_self for num_entries iterations:
//        PACK adds +1 to each L1 entry in place;
//        UNPACK adds +1 to each L1 entry in place;
//        net L1 transform = preload + 2 per uint32 word.
//   3. dfb_self.finish() ensures the credit flow has fully drained.
//   4. PACK side signals sem_output_ready so the DM consumer can read the result.
//
// Compile-time args:
//   0: num_entries          - tile count
//   1: words_per_entry      - words per dfb_self entry (entry_size / sizeof(uint32_t))
//   2: sem_input_ready_id   - DM producer → compute signal
//   3: sem_output_ready_id  - compute → DM consumer signal
void kernel_main() {
    constexpr uint32_t num_entries = get_compile_time_arg_val(0);
    constexpr uint32_t words_per_entry = get_compile_time_arg_val(1);
    constexpr uint32_t sem_input_ready_id = get_compile_time_arg_val(2);
    constexpr uint32_t sem_output_ready_id = get_compile_time_arg_val(3);

    experimental::DataflowBuffer dfb_self(0);
    ckernel::Semaphore sem_input_ready(sem_input_ready_id);
    ckernel::Semaphore sem_output_ready(sem_output_ready_id);

    sem_input_ready.wait(1);

    for (uint32_t i = 0; i < num_entries; ++i) {
        dfb_self.reserve_back(1);
#ifdef UCK_CHLKC_PACK
        {
            const uint32_t addr = dfb_self.get_write_ptr() << 4;
            volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(addr);
            const uint32_t before = entry[0];
            for (uint32_t w = 0; w < words_per_entry; ++w) {
                entry[w] += 1;
            }
            DPRINT << "C1 PACK i=" << i << " addr=0x" << HEX() << addr << " before=0x" << before << " after=0x"
                   << entry[0] << DEC() << ENDL();
        }
#endif
        dfb_self.push_back(1);

        dfb_self.wait_front(1);
#ifdef UCK_CHLKC_UNPACK
        {
            uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
            if (trisc_id == 0) {
                const uint32_t addr = dfb_self.get_read_ptr() << 4;
                volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(addr);
                const uint32_t before = entry[0];
                for (uint32_t w = 0; w < words_per_entry; ++w) {
                    entry[w] += 1;
                }
                DPRINT << "C1 UNPACK i=" << i << " addr=0x" << HEX() << addr << " before=0x" << before << " after=0x"
                       << entry[0] << DEC() << ENDL();
            }
        }
#endif
        dfb_self.pop_front(1);
    }

    dfb_self.finish();

    // PACK signals after finish() — by that point all UNPACK credits have been
    // acked, so the L1 mutation is complete (last UNPACK +1 has landed).
#ifdef UCK_CHLKC_PACK
    sem_output_ready.up(1);
#endif
}
