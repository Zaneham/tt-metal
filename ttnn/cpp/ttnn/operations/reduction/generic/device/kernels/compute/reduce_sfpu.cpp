// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Int32 reduce via SFPU (see reduce_sfpu_helpers_compute.hpp). Host sets REDUCE_OP,
// REDUCE_DIM, REDUCE_FORMAT, REDUCE_NEGATE; CT args Ht, Wt, NC like reduce.cpp.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_sfpu_helpers_compute.hpp"

#ifndef REDUCE_NEGATE
#define REDUCE_NEGATE 0
#endif

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);

    compute_kernel_lib::reduce_sfpu<REDUCE_OP, REDUCE_DIM, REDUCE_FORMAT, /*negate=*/(REDUCE_NEGATE != 0)>(
        tt::CBIndex::c_0,
        tt::CBIndex::c_2,
        tt::CBIndex::c_3,
        compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
}
