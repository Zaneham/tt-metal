// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Variant of dfb_t6_intra_2_0.cpp that declares ta::dfb_ring as a no-op binding.
// Used by the M2 INTRA tests so the borrowed-memory L1 ring tensor (which lets
// the host pre-fill / read back the DFB's L1 region via WriteShard/ReadShard)
// can be bound to this kernel. The kernel never reads the tensor.

#include "api/compute/common.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t entries_per_neo = get_arg(args::entries_per_neo);
    constexpr uint32_t words_per_entry = get_arg(args::words_per_entry);

    DataflowBuffer dfb(dfb::self);
    // No-op binding: forces ta::dfb_ring to be visible so the host can bind
    // the borrowed-memory L1 ring tensor to this kernel. The kernel never reads it.
    (void)TensorAccessor(ta::dfb_ring);

#ifdef UCK_CHLKC_UNPACK
    uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
#endif

    for (uint32_t i = 0; i < entries_per_neo; ++i) {
        dfb.reserve_back(1);
#ifdef UCK_CHLKC_PACK
        {
            volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(dfb.get_write_ptr() << 4);
            for (uint32_t w = 0; w < words_per_entry; ++w) {
                entry[w] += 1;
            }
        }
#endif
        dfb.push_back(1);

        dfb.wait_front(1);
#ifdef UCK_CHLKC_UNPACK
        if (trisc_id == 0) {
            volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(dfb.get_read_ptr() << 4);
            for (uint32_t w = 0; w < words_per_entry; ++w) {
                entry[w] += 1;
            }
        }
#endif
        dfb.pop_front(1);
    }
    dfb.finish();
}
