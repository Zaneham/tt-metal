// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "api/debug/dprint_pages.h"
#include "api/debug/dprint.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    constexpr uint32_t test_id = get_arg(args::test_id);

    // Sweep params + per-call runtime values as varargs (avoids JIT cache reuse).
    //   [0] num_packets, [1] packet_size_bytes, [2] master_l1_addr,
    //   [3] subordinate_l1_addr, [4] responder_x, [5] responder_y
    uint32_t num_packets = get_vararg(0);
    uint32_t packet_size_bytes = get_vararg(1);
    uint32_t master_l1_addr = get_vararg(2);
    uint32_t subordinate_l1_addr = get_vararg(3);
    uint32_t responder_x_coord = get_vararg(4);
    uint32_t responder_y_coord = get_vararg(5);

    Noc noc(noc_index);
    UnicastEndpoint unicast_endpoint;

    DeviceTimestampedData("Number of transactions", num_packets);
    DeviceTimestampedData("Transaction size in bytes", packet_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    {
        DeviceZoneScopedN("RISCV0");
        noc.set_async_write_state(
            unicast_endpoint, packet_size_bytes, {.noc_x = responder_x_coord, .noc_y = responder_y_coord, .addr = subordinate_l1_addr});

        for (uint32_t i = 0; i < num_packets; i++) {
            noc.async_write_with_state(
                unicast_endpoint,
                unicast_endpoint,
                packet_size_bytes,
                {.addr = master_l1_addr},
                {.noc_x = responder_x_coord, .noc_y = responder_y_coord, .addr = subordinate_l1_addr});
        }
        noc.async_write_barrier();
    }
}
