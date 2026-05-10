// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

/**
 * @file reduce_sfpu_helpers_compute.hpp
 * @brief SFPU reduce for tile MAX/MIN when the FPU reduce path cannot be used
 *
 * Provides one entry point, `reduce_sfpu`, symmetric to `compute_kernel_lib::reduce` in
 * reduce_helpers_compute.hpp for the streaming WaitAndPopPerTile-style flow:
 * - Row reduction (REDUCE_ROW): reduces W, outputs Ht tiles per batch
 * - Column reduction (REDUCE_COL): reduces H, outputs Wt tiles per batch
 *
 * Difference from `reduce()`:
 * - `reduce()` uses the FPU GMPOOL path (`reduce_tile`). That path does not produce correct
 *   integer MAX/MIN for Int32 on device.
 * - `reduce_sfpu()` uses SFPU `sfpu_reduce` for within-tile reduction and
 *   `binary_max_int32_tile` / `binary_min_int32_tile` for cross-tile folds along the reduce axis.
 *
 * Supported today:
 * - `pool_type`: MAX or MIN (MIN over W uses host lowering to MAX with `negate=true`)
 * - `format`: Int32 only
 * - `reduce_dim`: REDUCE_ROW or REDUCE_COL (no REDUCE_SCALAR here; HW uses two passes)
 *
 * Library responsibilities (same spirit as reduce_helpers_compute.hpp):
 * - DST tile_regs acquire/commit/wait/release
 * - CB wait/pop/reserve/push for input and output
 * - pack_tile to write reduced tiles
 * - Packer reduce mask (sfpu_reduce does not configure it; this helper does)
 *
 * IMPORTANT: Do not call `compute_kernel_hw_startup()` before `reduce_sfpu`; this helper runs
 * on the SFPU path and calls `init_sfpu` / `copy_tile_to_dst_init_short` itself.
 *
 * IMPORTANT: The scaler CB must contain one tile before entry (same contract as `reduce()`).
 * `sfpu_reduce` does not use it; the helper waits and pops it so dataflow matches the FPU kernel.
 *
 * Basic usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/reduce_sfpu_helpers_compute.hpp"
 *
 *   compute_kernel_lib::reduce_sfpu<ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW, DataFormat::Int32, false>(
 *       cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
 */

namespace compute_kernel_lib {

/**
 * @brief SFPU reduce for Int32 MAX/MIN along one tile axis
 *
 * @tparam pool_type   PoolType::MAX or PoolType::MIN (MIN + REDUCE_ROW is rejected at compile time;
 *                     host uses MAX + negate for W-axis MIN).
 * @tparam reduce_dim  ReduceDim::REDUCE_ROW (W) or ReduceDim::REDUCE_COL (H).
 * @tparam format      DataFormat::Int32 (only supported format).
 * @tparam negate      If true, negate tiles in DST before fold/reduce and negate result before pack
 *                     (host lowers MIN to -MAX(-x), same idea as FPU reduce_*_neg kernels).
 *
 * @param input_cb_id    Tiles to reduce (streaming order matches `reduce()` for same dim).
 * @param scaler_cb_id   Scaler tile CB (waited/popped; not passed into sfpu_reduce).
 * @param output_cb_id   Reduced output tiles.
 * @param input_block_shape  Tile grid (Ht, Wt, NC); uses ReduceInputBlockShape from reduce_helpers_compute.hpp.
 */
template <ckernel::PoolType pool_type, ckernel::ReduceDim reduce_dim, DataFormat format, bool negate = false>
ALWI void reduce_sfpu(
    uint32_t input_cb_id, uint32_t scaler_cb_id, uint32_t output_cb_id, ReduceInputBlockShape input_block_shape);

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_sfpu_helpers_compute.inl"
