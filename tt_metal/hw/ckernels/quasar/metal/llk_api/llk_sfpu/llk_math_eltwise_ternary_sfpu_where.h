// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_where.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_eltwise_ternary_sfpu.h"
#include "llk_math_eltwise_ternary_sfpu_params.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_where_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::where>();
    sfpu::_init_where_();
}

// Forwards to the new Blackhole-style ternary SFPU dispatch. The underlying
// `_calculate_where_<SFPU_ITERATIONS>` takes DEST tile indices and computes
// per-tile SFPU offsets internally; the params wrapper sets section base to
// DEST tile 0 and runs the kernel face-by-face.
template <bool APPROXIMATE, [[maybe_unused]] DataFormat data_format>
inline void llk_math_eltwise_ternary_sfpu_where(
    uint dst_index0, uint dst_index1, uint dst_index2, uint odst, int vector_mode = (int)VectorMode::RC) {
    LLK_ASSERT(vector_mode == (int)VectorMode::RC, "Quasar currently only supports vector mode RC");
    _llk_math_eltwise_ternary_sfpu_params_(
        sfpu::_calculate_where_<APPROXIMATE, SFPU_ITERATIONS>, dst_index0, dst_index1, dst_index2, odst);
}

}  // namespace ckernel
