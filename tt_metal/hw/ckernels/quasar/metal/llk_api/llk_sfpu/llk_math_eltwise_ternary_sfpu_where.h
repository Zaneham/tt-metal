// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "ckernel_sfpu_where.h"
#include "llk_assert.h"
#include "llk_defs.h"

namespace ckernel {

template <[[maybe_unused]] bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_where_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::where>(sfpu::_init_where_);
}

// Tile-stride in SFPU dest_reg_addr units for a 32x32 tile (32 rows × 2 units/row = 64).
// Offsets passed to _calculate_where_ are multiples of this stride, selecting which
// tile (relative to dst_index0's section base) each operand is read from or written to.
template <[[maybe_unused]] bool APPROXIMATE, [[maybe_unused]] DataFormat data_format>
inline void llk_math_eltwise_ternary_sfpu_where(
    uint dst_index0, uint dst_index1, uint dst_index2, uint odst, int vector_mode = (int)VectorMode::RC) {
    LLK_ASSERT(vector_mode == (int)VectorMode::RC, "Quasar currently only supports vector mode RC");
    constexpr int TILE_STRIDE = 64;
    _llk_math_eltwise_sfpu_params_(
        sfpu::_calculate_where_,
        dst_index0,
        SFPU_ITERATIONS,
        0,
        static_cast<int>(dst_index1 - dst_index0) * TILE_STRIDE,
        static_cast<int>(dst_index2 - dst_index0) * TILE_STRIDE,
        static_cast<int>(odst - dst_index0) * TILE_STRIDE);
}

}  // namespace ckernel
