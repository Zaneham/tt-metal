// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "llk_assert.h"
#include "sfpu/ckernel_sfpu_add.h"

namespace ckernel {

inline void llk_math_eltwise_binary_sfpu_add_int_init() { _llk_math_eltwise_sfpu_init_(); }

template <bool APPROXIMATE, int ITERATIONS, DataFormat DATA_FORMAT, bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_add_int(
    uint32_t idst0, uint32_t idst1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    LLK_ASSERT(vector_mode == (int)VectorMode::RC, "Quasar currently only supports vector mode RC");
    static_assert(DATA_FORMAT == DataFormat::Int32, "Quasar SFPU add_int currently supports Int32 only");
    constexpr int tile_stride = NUM_FACES * FACE_R_DIM;
    const int in0_offset = static_cast<int>(idst0) * tile_stride;
    const int in1_offset = static_cast<int>(idst1) * tile_stride;
    const int out_offset = static_cast<int>(odst) * tile_stride;

    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_add_int_<APPROXIMATE, ITERATIONS, 0, SIGN_MAGNITUDE_FORMAT>,
        0,
        DATA_FORMAT,
        ITERATIONS,
        in0_offset,
        in1_offset,
        out_offset);
}

}  // namespace ckernel
