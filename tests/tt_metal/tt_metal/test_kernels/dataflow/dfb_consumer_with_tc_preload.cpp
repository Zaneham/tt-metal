// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Variant of dfb_consumer.cpp that pre-increments the TC `acked` counter to a
// near-wrap value before the main consumer loop. Pairs with
// dfb_producer_with_tc_preload.cpp. See that file for rationale.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t dst_addr_base = get_compile_time_arg_val(0);
    const uint32_t num_entries_per_consumer = get_compile_time_arg_val(1);
    const uint32_t blocked_consumer = get_compile_time_arg_val(2);
    constexpr uint32_t implicit_sync = get_compile_time_arg_val(3);
    constexpr uint32_t kPreloadAckedValue = get_compile_time_arg_val(4);
    constexpr auto dst_args = TensorAccessorArgs<5>();

    uint32_t consumer_mask = get_arg_val<uint32_t>(0);
    uint32_t logical_dfb_id = get_arg_val<uint32_t>(1);
    const uint32_t chunk_offset = get_arg_val<uint32_t>(2);
    const uint32_t entries_per_core = get_arg_val<uint32_t>(3);
    const uint32_t num_consumers = static_cast<uint32_t>(__builtin_popcount(consumer_mask));

    DataflowBuffer dfb(logical_dfb_id);
    Noc noc;

#ifdef ARCH_QUASAR
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    uint32_t consumer_idx = static_cast<uint32_t>(__builtin_popcount(consumer_mask & ((1u << hartid) - 1u)));
#else
    uint32_t consumer_idx = 0;
#endif

    if constexpr (kPreloadAckedValue > 0) {
#ifdef ARCH_QUASAR
        if (consumer_idx == 0) {
            dfb.preload_acked_counter(kPreloadAckedValue);
        }
#endif
    }

    uint32_t entry_size = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(dst_args, dst_addr_base);

    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; tile_id++) {
        uint32_t page_id = 0;
        if constexpr (blocked_consumer) {
            page_id = chunk_offset + tile_id;
        } else {
            page_id = chunk_offset + tile_id * num_consumers + consumer_idx;
        }
        if (page_id >= chunk_offset + entries_per_core) {
            break;
        }
        if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
            noc.async_write<Noc::TxnIdMode::ENABLED>(dfb, tensor_accessor, {}, {.page_id = page_id});
#endif
        } else {
            dfb.wait_front(1);
            noc.async_write(dfb, tensor_accessor, entry_size, {}, {.page_id = page_id});
            noc.async_write_barrier();
            dfb.pop_front(1);
        }
    }
    dfb.finish();
    dfb.write_barrier(noc);
}
