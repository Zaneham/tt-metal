// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_sync.h"
#include "llk_unpack_unary_operand.h"
#include "llk_unpack_common_api.h"
#include "api/dataflow/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK A
 *************************************************************************/

/**
 *
 * @brief Initialize unpacker0 with dest reuse support
 *
 * Mirrors the Blackhole/Wormhole BH-style signature so the compute API can pipe
 * UnpackToDestEn through as a template parameter. The runtime format check
 * inside still selects UNP_DEST for 32-bit formats vs UNP_A otherwise — the
 * template gate just controls whether the UNP_DEST routing is considered at all.
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_init(
    [[maybe_unused]] const std::uint32_t transpose_of_faces = 0,
    [[maybe_unused]] const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);

    static_assert(acc_to_dest == false, "acc_to_dest is not yet supported on Quasar");
    static_assert(BType == BroadcastType::NONE, "Only BroadcastType::NONE is supported on Quasar right now");

    // TODO (tt-metal #42916): Once runtime asserts are added, add asserts for unsupported features above and for valid
    // transpose_of_faces and within_face_16x16_transpose values

    if constexpr (unpack_to_dest) {
        const std::uint32_t dst_format = unpack_dst_format[operand_id];
        if (dst_format == (std::uint32_t)DataFormat::Float32 || dst_format == (std::uint32_t)DataFormat::Int32) {
            _llk_unpack_unary_operand_init_<p_unpacr::UNP_DEST, false /*TRANSPOSE_EN*/, DST_ACCUM_MODE>(operand_id);
            return;
        }
    }

    // For Quasar, the unp_sel field is ignored if binary_reuse_dest != EltwiseBinaryReuseDestType::NONE
    _llk_unpack_unary_operand_init_<
        p_unpacr::UNP_A,
        false /* TRANSPOSE_EN */,
        DST_ACCUM_MODE /* IS_32b_DEST_EN */,
        binary_reuse_dest>(operand_id);
}

/**
 * @brief Legacy 2-template overload kept for Quasar-only call sites that thread TRANSPOSE_EN
 * (e.g. transpose_wh, bcast). Mirrors the BH-style routing: when `UnpackToDestEn` is on and
 * the operand format is 32-bit, route through UNP_DEST; otherwise UNP_A with TRANSPOSE_EN.
 */
template <bool TRANSPOSE_EN, bool IS_32b_DEST_EN>
inline void llk_unpack_A_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);

    if constexpr (UnpackToDestEn) {
        const std::uint32_t dst_format = unpack_dst_format[operand_id];
        if (dst_format == (std::uint32_t)DataFormat::Float32 || dst_format == (std::uint32_t)DataFormat::Int32) {
            _llk_unpack_unary_operand_init_<p_unpacr::UNP_DEST, false /*TRANSPOSE_EN*/, IS_32b_DEST_EN>(operand_id);
            return;
        }
    }

    _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, TRANSPOSE_EN, IS_32b_DEST_EN>(operand_id);
}

/**
 *
 * @brief Unpacks a single operand with dest reuse support
 *
 * Mirrors the Blackhole/Wormhole BH-style signature. The `unpack_to_dest` template
 * gate decides whether the UNP_DEST/semaphore path is considered; the runtime
 * format check then selects between UNP_DEST (32-bit) and UNP_A (other).
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(operand_id);
    const std::uint32_t l1_tile_index =
        local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_entry_idx + tile_index;

    static_assert(acc_to_dest == false, "acc_to_dest is not yet supported on Quasar");
    static_assert(BType == BroadcastType::NONE, "Only BroadcastType::NONE is supported on Quasar right now");

    if constexpr (unpack_to_dest) {
        const std::uint32_t dst_format = unpack_dst_format[operand_id];
        if (dst_format == (std::uint32_t)DataFormat::Float32 || dst_format == (std::uint32_t)DataFormat::Int32) {
            // Producer of UNPACK_MATH. The math thread as middleman chain has two single counting
            // sems with max=N each without an extra wait on MATH_PACK, unpack could race
            // 2N iterations ahead of pack and overwrite a bank that pack has not read yet.
            // Wait on both: math has drained (UNPACK_MATH < max) AND pack has drained
            // (MATH_PACK < max). Combined this keeps unpack within N iterations of pack.
            _llk_sync_wait_<p_stall::STALL_UNPACK>(semaphore::MATH_PACK, p_stall::STALL_ON_MAX);
            _llk_sync_wait_<p_stall::STALL_UNPACK>(semaphore::UNPACK_MATH, p_stall::STALL_ON_MAX);

            // WH/BH-style address coupling: snoop math's SEC1 (math owns the bank pointer).
            // Unpack no longer maintains its own dest_register_offset — eliminates the
            // 3-state-machine drift bug seen in multi-tile SyncHalf transpose.
            ckernel::trisc::cfg[DEST_TARGET_REG_CFG_MATH_SEC0_Offset_ADDR32] =
                ckernel::trisc::cfg[DEST_TARGET_REG_CFG_MATH_SEC1_Offset_ADDR32];

            // Drain UNPACK0 before posting "filled" so the post does not race the writes math reads.
            _llk_unpack_unary_operand_<p_unpacr::UNP_DEST>(l1_tile_index);
            _llk_sync_post_<p_stall::UNPACK0>(semaphore::UNPACK_MATH);
            return;
        }
    }

    WAYPOINT("UPAW");
    // For Quasar, the unp_sel field is ignored if binary_reuse_dest != EltwiseBinaryReuseDestType::NONE
    _llk_unpack_unary_operand_<p_unpacr::UNP_A, binary_reuse_dest>(l1_tile_index);
    WAYPOINT("UPAD");
}

/**
 * @brief Legacy 2-arg overload kept for Quasar-only call sites. Delegates to the
 * BH-style overload with unpack_to_dest=UnpackToDestEn.
 */
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(operand, tile_index);
}

/**
 * @brief Unpacks a contiguous block of tiles with unpacker0.
 *
 * @param operand The logical dataflow buffer id.
 * @param start_tile_index The starting tile index within the input buffer.
 * @param ntiles The number of consecutive tiles to unpack.
 *
 * The tiles are read from the operand buffer starting at start_tile_index
 * and unpacked into srcA one tile at a time.
 */
// TODO: AM; Optimize block calls by using ntiles per unpack, issue #40798
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_block(
    const std::uint32_t operand, const std::uint32_t start_tile_index, const std::uint32_t ntiles) {
    static_assert(acc_to_dest == false, "acc_to_dest is not yet supported on Quasar");
    static_assert(BType == BroadcastType::NONE, "Only BroadcastType::NONE is supported on Quasar right now");

    for (uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        llk_unpack_A<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(operand, tile_index);
    }
}

/**
 * @brief Legacy 3-arg overload. Delegates to the BH-style overload with unpack_to_dest=UnpackToDestEn.
 */
inline void llk_unpack_A_block(
    const std::uint32_t operand, const std::uint32_t start_tile_index, const std::uint32_t ntiles) {
    llk_unpack_A_block<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        operand, start_tile_index, ntiles);
}
