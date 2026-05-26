// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_operands.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY
 *************************************************************************/

/**
 *
 * @brief Initialize eltwise unary datacopy operations
 *
 * BH-style template signature so the compute API can pipe UnpackToDestEn through
 * as a template parameter. When `unpack_to_dest=true` and the operand format is
 * 32-bit, this init is a no-op — unpack writes dest directly via UNP_DEST and
 * math is just a sync forwarder.
 *
 * @tparam type sets which src register to copy from, values = <A2D, B2D>
 * @tparam IS_32b_DEST_EN set if math destination register is set to Float32/Int32 mode
 * @tparam src_b_bcast_type Broadcast type for SrcB (kept for BH API compatibility)
 * @tparam unpack_to_dest Set true to enable the UNP_DEST routing decision
 * @param operand: The input operand circular buffer
 */
template <
    DataCopyType type,
    bool IS_32b_DEST_EN,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);

    if constexpr (unpack_to_dest) {
        const std::uint32_t dst_format = unpack_dst_format[operand_id];
        if (dst_format == (std::uint32_t)DataFormat::Float32 || dst_format == (std::uint32_t)DataFormat::Int32) {
            return;
        }
    }

    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    _llk_math_eltwise_unary_datacopy_init_<type, IS_32b_DEST_EN>(
        num_faces * face_r_dim /*num_rows_per_matrix*/, 1 /*num_matrices*/);
}

/**
 * @brief Performs an eltwise unary datacopy for a single tile.
 *
 * BH-style template signature. When `unpack_to_dest=true` and the operand
 * format is 32-bit, this is a no-op — unpack writes dest directly.
 *
 * @param dst_index Tile index into the destination register.
 * @param operand The input operand logical dataflow buffer id.
 */
template <
    DataCopyType type,
    bool IS_32b_DEST_EN,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy(const std::uint32_t dst_index, const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);

    if constexpr (unpack_to_dest) {
        const std::uint32_t dst_format = unpack_dst_format[operand_id];
        if (dst_format == (std::uint32_t)DataFormat::Float32 || dst_format == (std::uint32_t)DataFormat::Int32) {
            return;
        }
    }

    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    _llk_math_eltwise_unary_datacopy_(num_faces * face_r_dim, dst_index);
}

/**
 * @brief Performs an eltwise unary datacopy for a block of tiles.
 *
 * @param start_dst_index Starting tile index in the destination register.
 * @param ntiles Number of tiles to copy to the destination register.
 * @param operand The input operand logical dataflow buffer id.
 */
template <
    DataCopyType type,
    bool IS_32b_DEST_EN,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_block(
    const std::uint32_t start_dst_index, const std::uint32_t ntiles, const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);

    if constexpr (unpack_to_dest) {
        const std::uint32_t dst_format = unpack_dst_format[operand_id];
        if (dst_format == (std::uint32_t)DataFormat::Float32 || dst_format == (std::uint32_t)DataFormat::Int32) {
            return;
        }
    }

    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);

    for (std::uint32_t dst_index = start_dst_index; dst_index < start_dst_index + ntiles; dst_index++) {
        _llk_math_eltwise_unary_datacopy_(num_faces * face_r_dim, dst_index);
    }
}

/**
 * @brief Legacy non-template overloads kept for Quasar-only call sites in compute API.
 * Delegate to the BH-style template variants with unpack_to_dest=UnpackToDestEn.
 */
inline void llk_math_eltwise_unary_datacopy(const std::uint32_t dst_index, const std::uint32_t operand) {
    llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
        dst_index, operand);
}

inline void llk_math_eltwise_unary_datacopy_block(
    const std::uint32_t start_dst_index, const std::uint32_t ntiles, const std::uint32_t operand) {
    llk_math_eltwise_unary_datacopy_block<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
        start_dst_index, ntiles, operand);
}
