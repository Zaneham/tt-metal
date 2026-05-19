// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

// L1 to L1 send (loopback — sends to self, Metal 2.0)
void kernel_main() {
    constexpr uint32_t src_addr = get_arg(args::src_addr);
    constexpr uint32_t dst_addr = get_arg(args::dst_addr);
    constexpr uint32_t page_size_bytes = get_arg(args::page_size);
    constexpr uint32_t test_id = get_arg(args::test_id);

    // varargs: [0]=num_of_transactions, [1]=transaction_num_pages, [2]=dest_x, [3]=dest_y.
    const uint32_t num_of_transactions = get_vararg(0);
    const uint32_t transaction_num_pages = get_vararg(1);
    uint32_t dest_x = get_vararg(2);
    uint32_t dest_y = get_vararg(3);

    const uint32_t transaction_size_bytes = transaction_num_pages * page_size_bytes;

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    Noc noc(noc_index);
    UnicastEndpoint unicast_endpoint;

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            noc.async_write(
                unicast_endpoint,
                unicast_endpoint,
                transaction_size_bytes,
                {.addr = src_addr},
                {.noc_x = dest_x, .noc_y = dest_y, .addr = dst_addr});
        }
        noc.async_write_barrier();
    }

    // Signal completion via semaphore increment on self
    Semaphore sem(sem::sem_name);
    sem.up(noc, dest_x, dest_y, 1);
}
