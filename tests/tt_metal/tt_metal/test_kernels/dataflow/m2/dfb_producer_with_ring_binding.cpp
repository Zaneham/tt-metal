// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Variant of m2/dfb_producer.cpp that declares ta::dfb_ring as a no-op binding
// in addition to the normal ta::src_tensor. Used by borrowed-memory DFB tests:
// every TensorParameter declared on a ProgramSpec must be bound to >=1 kernel,
// so the ring tensor (which the kernel does not actually read) gets attached
// here as a placeholder. The DFB's L1 base address is patched in by Metal at
// SetProgramRunParameters time via the host-side borrowed_from mechanism;
// the kernel just produces into dfb::out as usual.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);

    const uint32_t chunk_offset = get_arg(args::chunk_offset);
    const uint32_t entries_per_core = get_arg(args::entries_per_core);

    DataflowBuffer dfb(dfb::out);
    Noc noc;
    const auto tensor_accessor = TensorAccessor(ta::src_tensor);
    // No-op binding: forces ta::dfb_ring to be visible so the host can bind
    // the borrowed-memory ring tensor to this kernel. The kernel never reads it.
    (void)TensorAccessor(ta::dfb_ring);

    const uint32_t producer_idx = get_my_thread_id();
    const uint32_t num_producers = get_num_threads();
    const uint32_t entry_size = dfb.get_entry_size();

    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; ++tile_id) {
        const uint32_t page_id = chunk_offset + tile_id * num_producers + producer_idx;
        if (page_id >= chunk_offset + entries_per_core) {
            break;
        }
        if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
            noc.async_read<Noc::TxnIdMode::ENABLED>(tensor_accessor, dfb, {.page_id = page_id}, {});
#endif
        } else {
            dfb.reserve_back(1);
            noc.async_read(tensor_accessor, dfb, entry_size, {.page_id = page_id}, {});
            noc.async_read_barrier();
            dfb.push_back(1);
        }
    }
    dfb.finish();
}
