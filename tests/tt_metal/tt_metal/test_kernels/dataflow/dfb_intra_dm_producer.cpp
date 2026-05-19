// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "experimental/noc.h"
#include "experimental/tensor.h"
#include "experimental/noc_semaphore.h"

// C1 DM producer: NOC-read each tile from a DRAM tensor into dfb_self's L1 ring
// at a host-passed L1 address (one ring slot per page). After all transfers,
// signal the compute kernel via a semaphore so it knows the data is ready.
//
// We bypass the DFB API for the L1 destination because dfb_self is INTRA-scope
// (Tensix-only) — DM kernels can't be its producer through the DFB API.
//
// Compile-time args:
//   0: src_addr_base    - DRAM base address of in_buffer
//   1: num_entries      - tile count
//   2: entry_size       - bytes per tile
//   3: sem_input_ready  - semaphore id for "input ready" signal
//   4+: TensorAccessor compile-time args for src DRAM tensor
//
// Runtime arg:
//   0: dfb_self_l1      - L1 byte address of dfb_self's ring base
void kernel_main() {
    constexpr uint32_t src_addr_base = get_compile_time_arg_val(0);
    constexpr uint32_t num_entries = get_compile_time_arg_val(1);
    constexpr uint32_t entry_size = get_compile_time_arg_val(2);
    constexpr uint32_t sem_input_ready_id = get_compile_time_arg_val(3);
    constexpr auto src_args = TensorAccessorArgs<4>();

    const uint32_t dfb_self_l1 = get_arg_val<uint32_t>(0);

    experimental::Semaphore sem_input_ready(sem_input_ready_id);
    const auto tensor_accessor = TensorAccessor(src_args, src_addr_base);

    for (uint32_t i = 0; i < num_entries; ++i) {
        const uint64_t noc_src = tensor_accessor.get_noc_addr(i);
        const uint32_t l1_dst = dfb_self_l1 + i * entry_size;
        noc_async_read(noc_src, l1_dst, entry_size);
    }
    noc_async_read_barrier();

    sem_input_ready.up(1);
}
