// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) sequential Tensix producer for 4 concurrent DFBs.
// Parallel to ../dfb_t6_seq_producer.cpp; M2 uses 4 named DFB bindings instead
// of a runtime-determined DFB count.
//
// A single Neo thread loops dfb::buf_0..buf_3, calling reserve_back/push_back
// for each entry. The host pre-fills each DFB's L1 ring (via borrowed_from
// tensors) before LaunchProgram; this kernel just emits credits so the DM
// consumers can drain.
//
// The ta::ring_0..ring_3 no-op bindings let the host attach borrowed-memory L1
// tensors to this kernel (every TensorParameter must be bound to >=1 kernel).

#include "api/compute/common.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <typename Dfb>
static inline void signal_one_dfb(Dfb& dfb, uint32_t num_entries_per_producer) {
    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; ++tile_id) {
        dfb.reserve_back(1);
        dfb.push_back(1);
    }
    dfb.finish();
}

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);

    (void)TensorAccessor(ta::ring_0);
    (void)TensorAccessor(ta::ring_1);
    (void)TensorAccessor(ta::ring_2);
    (void)TensorAccessor(ta::ring_3);

    {
        DataflowBuffer dfb(dfb::buf_0);
        signal_one_dfb(dfb, num_entries_per_producer);
    }
    {
        DataflowBuffer dfb(dfb::buf_1);
        signal_one_dfb(dfb, num_entries_per_producer);
    }
    {
        DataflowBuffer dfb(dfb::buf_2);
        signal_one_dfb(dfb, num_entries_per_producer);
    }
    {
        DataflowBuffer dfb(dfb::buf_3);
        signal_one_dfb(dfb, num_entries_per_producer);
    }
}
