// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Variant of dfb_producer.cpp that pre-increments the TC `posted` counter to a
// near-wrap value before the main producer loop. Used by D1 to exercise the
// uint16 `posted` wrap point with only a handful of real tile pushes instead
// of 65k+ — runtime drops from hours to seconds on the Quasar emulator while
// still catching the same bug class (firmware code that uses a wider shadow
// accumulator than the 16-bit HW register).
//
// CTA 3 (kPreloadPostedValue):
//   0       → no preload, identical behavior to dfb_producer.cpp
//   N > 0   → on entry, advance every TC slot's `posted` field by N via direct
//             HW register write (bypasses the ring-credit check). Consumer must
//             apply the same N to `acked` so the ring still looks empty after
//             preload (posted - acked == 0).
//
// Pairs with: dfb_consumer_with_tc_preload.cpp

#include "experimental/dataflow_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"

void kernel_main() {
    const uint32_t src_addr_base = get_compile_time_arg_val(0);
    const uint32_t num_entries_per_producer = get_compile_time_arg_val(1);
    constexpr uint32_t implicit_sync = get_compile_time_arg_val(2);
    constexpr uint32_t kPreloadPostedValue = get_compile_time_arg_val(3);
    constexpr auto src_args = TensorAccessorArgs<4>();

    uint32_t producer_mask = get_arg_val<uint32_t>(0);
    const uint32_t chunk_offset = get_arg_val<uint32_t>(1);
    const uint32_t entries_per_core = get_arg_val<uint32_t>(2);
    const uint32_t num_producers = static_cast<uint32_t>(__builtin_popcount(producer_mask));

    experimental::DataflowBuffer dfb(0);
    experimental::Noc noc;

#ifdef ARCH_QUASAR
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    uint32_t producer_idx = static_cast<uint32_t>(__builtin_popcount(producer_mask & ((1u << hartid) - 1u)));
#else
    uint32_t producer_idx = 0;
#endif

    // TC posted-counter preload: only producer 0 issues the increment so the
    // counter is bumped exactly once per TC slot regardless of num_producers.
    if constexpr (kPreloadPostedValue > 0) {
#ifdef ARCH_QUASAR
        if (producer_idx == 0) {
            dfb.preload_posted_counter(kPreloadPostedValue);
        }
#endif
    }

    uint32_t entry_size = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(src_args, src_addr_base);

    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id++) {
        const uint32_t page_id = chunk_offset + tile_id * num_producers + producer_idx;
        if (page_id >= chunk_offset + entries_per_core) {
            break;
        }
        if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
            noc.async_read<experimental::Noc::TxnIdMode::ENABLED>(tensor_accessor, dfb, {.page_id = page_id}, {});
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
