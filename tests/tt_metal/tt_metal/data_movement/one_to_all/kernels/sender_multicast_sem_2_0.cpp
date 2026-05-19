// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/endpoints.h"
#include "experimental/noc_semaphore.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

// Sender semaphore kernel (Metal 2.0)
// Uses semaphore bindings:
//   sender_sem  - sender waits for all receivers to ack (init=0, target=mst)
//   sync_sem    - sender signals receivers each transaction (init=1, target=mst+sub)
//                 receivers bind this as "recv_sem" to wait on
void kernel_main() {
    constexpr uint32_t mst_base_addr = get_arg(args::mst_base_addr);
    constexpr uint32_t sub_base_addr = get_arg(args::sub_base_addr);
    constexpr uint32_t num_of_transactions = get_arg(args::num_transactions);
    constexpr uint32_t pages_per_transaction = get_arg(args::pages_per_tx);
    constexpr uint32_t bytes_per_page = get_arg(args::bytes_per_page);
    constexpr uint32_t test_id = get_arg(args::test_id);
    constexpr uint32_t num_subordinates = get_arg(args::num_subordinates);
    constexpr bool is_linked = get_arg(args::is_linked);
    constexpr bool loopback = get_arg(args::loopback);
    uint32_t start_x = get_arg(args::start_x);
    uint32_t start_y = get_arg(args::start_y);
    uint32_t end_x = get_arg(args::end_x);
    uint32_t end_y = get_arg(args::end_y);

    constexpr uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;

    if (noc_index == 1) {
        std::swap(start_x, end_x);
        std::swap(start_y, end_y);
    }

    experimental::Noc noc(noc_index);
    experimental::UnicastEndpoint unicast_endpoint;
    experimental::MulticastEndpoint multicast_endpoint;

    // sender_sem: receivers increment this to ack readiness (init=0, only on sender)
    experimental::Semaphore sender_sem(sem::sender_sem);
    // sync_sem: sender sets this on receivers each round to signal data is ready (init=1 on sender)
    // Receivers bind this same semaphore as "recv_sem"
    experimental::Semaphore sync_sem(sem::sync_sem);

    constexpr experimental::Noc::McastMode include_src =
        loopback ? experimental::Noc::McastMode::INCLUDE_SRC : experimental::Noc::McastMode::EXCLUDE_SRC;

    for (uint32_t i = 0; i < num_of_transactions - 1; i++) {
        // Wait for all receivers to signal readiness
        sender_sem.wait(num_subordinates);
        sender_sem.set(0);

        // Multicast data to all subordinates
        noc.async_write_multicast<include_src>(
            unicast_endpoint,
            multicast_endpoint,
            bytes_per_transaction,
            num_subordinates,
            {.addr = mst_base_addr},
            {.noc_x_start = start_x,
             .noc_y_start = start_y,
             .noc_x_end = end_x,
             .noc_y_end = end_y,
             .addr = sub_base_addr},
            is_linked);

        // Signal all receivers that data is ready (multicast sync_sem value = 1 to receivers' sync_sem slot)
        sync_sem.set_multicast(noc, start_x, start_y, end_x, end_y, num_subordinates, is_linked);
    }

    // Last transaction: sent without linking so the VC is freed
    sender_sem.wait(num_subordinates);
    sender_sem.set(0);

    noc.async_write_multicast<include_src>(
        unicast_endpoint,
        multicast_endpoint,
        bytes_per_transaction,
        num_subordinates,
        {.addr = mst_base_addr},
        {.noc_x_start = start_x,
         .noc_y_start = start_y,
         .noc_x_end = end_x,
         .noc_y_end = end_y,
         .addr = sub_base_addr});

    sync_sem.set_multicast(noc, start_x, start_y, end_x, end_y, num_subordinates);

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
    DeviceTimestampedData("Number of subordinates", num_subordinates);
    DeviceTimestampedData("NoC Index", noc_index);
}
