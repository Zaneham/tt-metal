// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "experimental/noc.h"
#include "experimental/tensor.h"
#include "experimental/noc_semaphore.h"

// C1 DM consumer: wait for the compute kernel's "output ready" semaphore, then
// NOC-write each tile of dfb_self's L1 ring out to DRAM (one ring slot per page).
//
// We bypass the DFB API for the L1 source because dfb_self is INTRA-scope —
// DM kernels can't be its consumer through the DFB API.
//
// Compile-time args:
//   0: dst_addr_base    - DRAM base address of out_buffer
//   1: num_entries      - tile count
//   2: entry_size       - bytes per tile
//   3: sem_output_ready - semaphore id for "output ready" signal
//   4+: TensorAccessor compile-time args for dst DRAM tensor
//
// Runtime arg:
//   0: dfb_self_l1      - L1 byte address of dfb_self's ring base
void kernel_main() {
    constexpr uint32_t dst_addr_base = get_compile_time_arg_val(0);
    constexpr uint32_t num_entries = get_compile_time_arg_val(1);
    constexpr uint32_t entry_size = get_compile_time_arg_val(2);
    constexpr uint32_t sem_output_ready_id = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();

    const uint32_t dfb_self_l1 = get_arg_val<uint32_t>(0);

    experimental::Semaphore sem_output_ready(sem_output_ready_id);
    const auto tensor_accessor = TensorAccessor(dst_args, dst_addr_base);

    sem_output_ready.wait(1);

    for (uint32_t i = 0; i < num_entries; ++i) {
        const uint32_t l1_src = dfb_self_l1 + i * entry_size;
        const uint64_t noc_dst = tensor_accessor.get_noc_addr(i);
        noc_async_write(l1_src, noc_dst, entry_size);
    }
    noc_async_write_barrier();
}
