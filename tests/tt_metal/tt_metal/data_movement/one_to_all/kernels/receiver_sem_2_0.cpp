// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc_semaphore.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

// Receiver semaphore kernel (Metal 2.0)
// Uses semaphore bindings:
//   sender_sem - receiver remotely increments this on the sender to ack readiness
//   recv_sem   - bound to the same SemaphoreSpec as sender's "sync_sem" (init=1, target=mst+sub)
//                receiver resets this to 0 before each round, then waits for sender to set it to 1
void kernel_main() {
    constexpr uint32_t num_of_transactions = get_arg(args::num_transactions);
    constexpr uint32_t pages_per_transaction = get_arg(args::pages_per_tx);
    constexpr uint32_t bytes_per_page = get_arg(args::bytes_per_page);
    constexpr uint32_t test_id = get_arg(args::test_id);
    constexpr uint32_t sender_coords = get_arg(args::sender_coords);

    constexpr uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;

    // Unpack sender physical coordinates
    const uint32_t sender_x = sender_coords >> 16;
    const uint32_t sender_y = sender_coords & 0xFFFF;

    experimental::Noc noc(noc_index);

    // sender_sem: we remotely increment this on the sender core to signal readiness
    experimental::Semaphore sender_sem_handle(sem::sender_sem);
    // recv_sem: we wait on this each round; sender sets it to 1 via multicast
    experimental::Semaphore recv_sem(sem::recv_sem);

    {
        DeviceZoneScopedN("RISCV1");

        for (uint32_t i = 0; i < num_of_transactions; i++) {
            // Reset recv_sem to 0 to signal readiness for next data
            recv_sem.set(0);

            // Increment sender's semaphore to ack readiness
            sender_sem_handle.up(noc, sender_x, sender_y, 1);

            // Wait for sender to multicast sync_sem = 1 (i.e., data is ready)
            recv_sem.wait(1);
        }
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
}
