// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_sync.h"
#include "llk_unpack_common_api.h"
#include "llk_unpack_tilize.h"
#include "api/dataflow/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK TILIZE
 *************************************************************************/

/**
 * @brief Initializes the unpacker for tilize operations on Quasar.
 *
 * Configures UNP_A stride registers and programs the MOP for tilizing
 * block_ct_dim tiles from row-major L1 data into face format in SrcA.
 *
 * For 32-bit dest formats (Float32/Int32) routes through the
 * UNP_DEST path so the unpacker writes tilized data directly to DEST,
 * and sets up the unpack-thread dest section base for SyncHalf. For all
 * other formats the existing UNP_A path is used.
 *
 * @param operand       The input dataflow buffer identifier.
 * @param full_ct_dim   Number of tiles in a full row of the input tensor.
 * @param block_ct_dim  Number of tiles per MOP invocation (defaults to 1).
 */
inline void llk_unpack_tilize_init(
    const std::uint32_t operand, const std::uint32_t full_ct_dim, const std::uint32_t block_ct_dim = 1) {
    const std::uint32_t operand_id = get_operand_id(operand);

    // TODO: Once narrow-tile is supported c_dim_faces will be variable.
    constexpr std::uint32_t c_dim_faces = 2;

    const std::uint32_t dst_format = unpack_dst_format[operand_id];
    if (dst_format == (std::uint32_t)DataFormat::Float32 ||
        dst_format == (std::uint32_t)DataFormat::Int32) {
        _llk_unpack_tilize_init_<p_unpacr::UNP_DEST, DST_ACCUM_MODE, c_dim_faces>(
            operand_id, full_ct_dim, block_ct_dim);

        if constexpr (DST_SYNC_MODE == DstSync::SyncHalf) {
            _reset_dest_register_offset_();
            _set_dest_section_base_<ckernel::unpack::TRISC_ID>(_get_dest_buffer_base_());
        }
    } else {
        _llk_unpack_tilize_init_<p_unpacr::UNP_A, DST_ACCUM_MODE, c_dim_faces>(
            operand_id, full_ct_dim, block_ct_dim);
    }
}

/**
 * @brief Tilizes a block of tiles from L1 row-major layout into SrcA or DEST.
 *
 * Computes the L1 face index from the DFB read position and the input
 * tile index, then runs the MOP configured by llk_unpack_tilize_init.
 *
 * For 32-bit destination formats (Float32/Int32), each tile goes through
 * the unpack-to-dest semaphore handshake:
 *  - wait MATH_PACK < max AND UNPACK_MATH < max (drained)
 *  - unpack-to-dest via UNP_DEST
 *  - post UNPACK_MATH so math thread knows the bank is filled
 *  - advance dest section in SyncHalf
 *
 * For other formats the existing UNP_A loop is used.
 *
 * @param operand          The input dataflow buffer identifier.
 * @param block_c_tiles    Number of tiles in one block row (must match BLOCK_CT_DIM from init).
 * @param input_tile_index Starting tile index (encodes row offset via block_c_tiles stride).
 */
inline void llk_unpack_tilize_block(
    const std::uint32_t operand, const std::uint32_t block_c_tiles, const std::uint32_t input_tile_index = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);

    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);
    const std::uint32_t R_DIM_FACES = (num_faces == 2 && !narrow_tile) ? 1 : 2;
    const std::uint32_t faces_per_entry = R_DIM_FACES * face_r_dim;

    const LocalDFBInterface& local_dfb = g_dfb_interface[operand_id];
    const std::uint32_t rd_entry_idx = local_dfb.tc_slots[local_dfb.tc_idx].rd_entry_idx;

    // TODO (SK) #42757: Remove ct_dim loop when block_ct_dim unpacking optimization implemented.
    // BLOCK_CT_DIM is currently hardcoded to 1 in tilize_init (see compute/tilize.h), so the MOP
    // emits one SrcA dvalid per invocation. Loop to match the per-tile math consumption same
    // structural pattern as BH/WH llk_unpack_tilize_block
    const std::uint32_t l1_base_idx = (rd_entry_idx + input_tile_index) * faces_per_entry;

    const std::uint32_t dst_format = unpack_dst_format[operand_id];
    if (dst_format == (std::uint32_t)DataFormat::Float32 ||
        dst_format == (std::uint32_t)DataFormat::Int32) {
        for (std::uint32_t t = 0; t < block_c_tiles; t++) {
            _llk_sync_wait_<p_stall::STALL_UNPACK>(semaphore::MATH_PACK,   p_stall::STALL_ON_MAX);
            _llk_sync_wait_<p_stall::STALL_UNPACK>(semaphore::UNPACK_MATH, p_stall::STALL_ON_MAX);

            _llk_unpack_tilize_<p_unpacr::UNP_DEST>(l1_base_idx + t);

            _llk_sync_post_<p_stall::UNPACK0>(semaphore::UNPACK_MATH);

            if constexpr (DST_SYNC_MODE == DstSync::SyncHalf) {
                _llk_sync_advance_dest_section_<ckernel::unpack::TRISC_ID, true /*EN_32BIT_DEST*/, p_stall::UNPACK0>();
            }
        }
    } else {
        for (std::uint32_t t = 0; t < block_c_tiles; t++) {
            _llk_unpack_tilize_<p_unpacr::UNP_A>(l1_base_idx + t);
        }
    }
}
