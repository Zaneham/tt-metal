// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "cfg_defines.h"
#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_proj_params.h"
#include "ckernel_template.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "tensix_types.h"

namespace ckernel::trisc
{
// Num of words in buffer descriptor struct
constexpr static std::uint32_t BD_NUM_WORDS = 3;

using ckernel::FACE_C_DIM;
using ckernel::FACE_R_DIM;
using ckernel::TILE_C_DIM;
using ckernel::TILE_R_DIM;

// Default number of faces
constexpr static std::uint32_t NUM_FACES = 4;

// Default number of tiles processed per one _llk_* execute call
static constexpr std::uint32_t NUM_TILES = 1;

// Number of rows that can fit into Dest
static constexpr std::uint32_t DEST_REGISTER_FULL_SIZE = 64 * FACE_R_DIM;
static constexpr std::uint32_t DEST_REGISTER_HALF_SIZE = DEST_REGISTER_FULL_SIZE >> 1;

// Number of bits used to represent data format in unpacker/packer config.
// The reason we keep only the bottom 4 bits is because the HW only has 4 bits to represent the dataformat.
// Essentially, only using the bottom 4 bits when programming the HW dataformats is not a bug, because
// the higher bits should be used to program other registers.
// Also, uint32 does not require special handling because both int32 and uint32 are stored the same way in DEST
// (just the highest bit is interpreted as a sign bit vs. a magnitude bit by the user). So when the packer
// needs to read DEST in order to pack the data out it reads the same data either way and just moves 32bits in both cases.
// Uint8 requires special handling because when int8 is put into DEST, the sign bit actually gets put
// to the MSB of the 32bit container, rather than to bit 8. So for int8 the packer will read the 7 LSBs + 1 MSB,
// but for uint8 the packer will read the 8 LSBs.
constexpr std::uint32_t DATA_FORMAT_BIT_COUNT = 4;
// Mask to extract data format bits
constexpr std::uint32_t DATA_FORMAT_CONFIG_MASK = (1 << DATA_FORMAT_BIT_COUNT) - 1;

// Points to the config space
std::uint32_t volatile* const cfg = (std::uint32_t volatile*)TENSIX_CFG_BASE;
// Points to the buffer table
buffer_descriptor_u volatile* const bd_table = (buffer_descriptor_u volatile* const)(cfg + BUFFER_DESCRIPTOR_TABLE_REG0_L1_BASE_ADDR_ADDR32);

constexpr std::uint32_t NUM_WORDS_TILE_CNT = 8;

typedef struct
{
    std::uint32_t reserved0                    : 32;
    std::uint32_t reset                        : 32;
    std::uint32_t posted                       : 32;
    std::uint32_t acked                        : 32;
    std::uint32_t buf_capacity                 : 32;
    std::uint32_t reserved1                    : 32;
    std::uint32_t tiles_avail_interrupt_thresh : 32;
    std::uint32_t space_avail_interrupt_thresh : 32;
} tile_counter_t;

static_assert(sizeof(tile_counter_t) == NUM_WORDS_TILE_CNT * sizeof(std::uint32_t), "tile_counter_t must be 96b!");

typedef union
{
    std::uint32_t words[NUM_WORDS_TILE_CNT];
    tile_counter_t f;
} tile_counter_u;

// Points to the tile counters
tile_counter_u volatile* const tile_counters = (tile_counter_u volatile* const)TILE_COUNTERS_BASE;

// Destination register offset, offset = 0 -> targets dest bank 0, offset = 512 for 16bit dest, 256 for 32bit dest -> targets dest bank 1
static std::uint32_t dest_register_offset = 0;

enum DataFormatConfigSet : std::uint8_t
{
    DEFAULT               = 0,
    MOV_SRC2DST_OPS_INT32 = 1
};

// global variable for all 4 threads
inline volatile DataFormatConfigSet data_format_config_set = DataFormatConfigSet::DEFAULT;

/**
* @brief Check divisibility by power of 2
* @param value: input value to check divisibility
* @param power_of_two_divisor: divisor that must be a power of 2, will check if
    value % power_of_two_divisor == 0
*/
inline bool _divisible_by_pow_two_(const std::uint32_t value, const std::uint32_t power_of_two_divisor)
{
    return ((value & (power_of_two_divisor - 1)) == 0);
}

/**
 * @brief Populates buffer table entry for TDMA engines
 * @param buf_desc_id: Buffer descriptor id into the buffer descriptor table
 * @param buf_desc: Contains L1 buffer descriptor information
 */
inline void _configure_buf_desc_table_(const std::uint32_t buf_desc_id, const buffer_descriptor_u& buf_desc)
{
    for (std::uint32_t i = 0; i < BD_NUM_WORDS; i++)
    {
        bd_table[buf_desc_id].words[i] = buf_desc.words[i];
    }
}

enum class DstTileShape : std::uint8_t
{
    Tile32x1  = 1,
    Tile32x2  = 2,
    Tile32x4  = 3,
    Tile32x8  = 4,
    Tile32x16 = 5,
    Tile32x32 = 6
};

/**
 * @brief Sets the destination register base address, each Trisc0/1/2/3 has separate
 * registers for setting dest base address.
 * @tparam TRISC_ID: Trisc core which is executing this function, values = [0, 1, 2, 3]
 * @param base_addr: Base address for destination register
 * Bank 0 -> base_addr = 0 - 511
 * Bank 1 -> base_addr = 512 - 1023
 */
template <std::uint8_t TRISC_ID>
inline void _set_dest_section_base_(const std::uint32_t base_addr)
{
    if constexpr (TRISC_ID == 0)
    {
        cfg[DEST_TARGET_REG_CFG_MATH_SEC0_Offset_ADDR32] = base_addr;
    }
    else if constexpr (TRISC_ID == 1)
    {
        cfg[DEST_TARGET_REG_CFG_MATH_SEC1_Offset_ADDR32] = base_addr;
    }
    else if constexpr (TRISC_ID == 2)
    {
        cfg[DEST_TARGET_REG_CFG_MATH_SEC2_Offset_ADDR32] = base_addr;
    }
    else
    {
        cfg[DEST_TARGET_REG_CFG_MATH_SEC3_Offset_ADDR32] = base_addr;
    }
}

/**
 * @brief Returns dest buffer base addr
 * If dest register is set to 16bit mode:
 *     Bank 0 -> addr = 0
 *     Bank 1 -> addr = 512
 * If dest register is set to 32bit mode:
 *     Bank 0 -> addr = 0
 *     Bank 1 -> addr = 256
 */
inline std::uint32_t _get_dest_buffer_base_()
{
    return dest_register_offset;
}

inline constexpr static std::uint32_t masked_data_format(std::uint32_t data_format)
{
    return data_format & DATA_FORMAT_CONFIG_MASK;
}

constexpr static std::uint32_t SCALE_DATUM_SIZE(std::uint32_t format, std::uint32_t datum_count)
{
    switch (masked_data_format(format))
    {
        case (to_underlying(DataFormat::Int32)):
        case (to_underlying(DataFormat::Float32)):
            return (datum_count << 2);
        case (to_underlying(DataFormat::Float16)):
        case (to_underlying(DataFormat::Float16_b)):
            return (datum_count << 1);
        default:
            return datum_count;
    };
}

/**
 * All the following functions are added to enable Math <-> Pack synchronization
 *
 * Another Issue:
 * Some of the following enums/functions are needed to increment to arbitrary addresses in dest, the functions can be removed
 * if a Tensix instruction is added to address the full dest
 *
 * The following functions should be removed once the dvalid scheme is completely used for all LLK operations
 */

/**
 * @brief Set destination register offset variable to 0
 */
inline void _reset_dest_register_offset_()
{
    dest_register_offset = 0;
}

/**
 * @brief Update destination register offset, offset can only toggle between 0 & 512 for 16bit dest, 0 & 256 for 32bit dest
 */
template <bool EN_32BIT_DEST>
inline void _update_dest_register_offset_()
{
    constexpr std::uint32_t dest_bank1_offset = EN_32BIT_DEST ? DEST_REGISTER_HALF_SIZE >> 1 : DEST_REGISTER_HALF_SIZE;
    dest_register_offset                      = (dest_register_offset == 0) ? dest_bank1_offset : 0;
}

struct mutex
{
    constexpr static std::uint32_t REG_RMW = 0; // used for atomic register read-modify-write from different threads
};

struct semaphore
{
    constexpr static std::uint32_t MATH_PACK = 1; // math <-> pack sync on dest register

    constexpr static std::uint16_t t6_sem(const std::uint8_t sem_index)
    {
        return (1 << sem_index);
    }
};

// Tensix thread semaphore post optionally stalled
// Can stall on up to 3 resources at a time
template <std::uint32_t WaitRes0 = p_stall::NOTHING, std::uint32_t WaitRes1 = p_stall::NOTHING, std::uint32_t WaitRes2 = p_stall::NOTHING>
inline void t6_semaphore_post(const std::uint8_t index)
{
    if constexpr (WaitRes0 != p_stall::NOTHING)
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, WaitRes2, WaitRes1, WaitRes0);
    }

    TTI_SEMPOST(0, semaphore::t6_sem(index));
}

// Tensix thread semaphore get optionally stalled
// Can stall on up to 3 resources at a time
template <std::uint32_t WaitRes0 = p_stall::NOTHING, std::uint32_t WaitRes1 = p_stall::NOTHING, std::uint32_t WaitRes2 = p_stall::NOTHING>
inline void t6_semaphore_get(const std::uint8_t index)
{
    if constexpr (WaitRes0 != p_stall::NOTHING)
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, WaitRes2, WaitRes1, WaitRes0);
    }

    TTI_SEMGET(0, semaphore::t6_sem(index));
}

/**
 * @brief Set packer's dest register offset to the current dest bank base.
 *
 * In SyncHalf mode, alternates between bank 0 (offset 0) and bank 1
 * (DEST_REGISTER_HALF_SIZE for 16-bit dest, DEST_REGISTER_HALF_SIZE/2 for 32-bit dest).
 * In SyncFull mode, always reads from offset 0.
 */
template <std::uint32_t PACK_SEL, ckernel::DstSync DST>
inline void _set_packer_dest_registers_()
{
    static_assert(DST == ckernel::DstSync::SyncHalf || DST == ckernel::DstSync::SyncFull);
    std::uint32_t dest_buffer_base_offset = (DST == ckernel::DstSync::SyncFull) ? 0 : _get_dest_buffer_base_();

    if constexpr (PACK_SEL == p_pacr::PACK0)
    {
        cfg[THCON_PACKER0_REG0_SRC_ADDR_OFFSET_ADDR32] = dest_buffer_base_offset;
    }
    else
    {
        cfg[THCON_PACKER1_REG0_SRC_ADDR_OFFSET_ADDR32] = dest_buffer_base_offset;
    }
}

/**
 * @brief Determines whether the source register format and Float32 destination register format are a supported combination
 *
 * @param src_reg_fmt: The source register format
 */
inline bool _is_src_fmt_fp32_dest_compatible_(const DataFormat src_reg_fmt)
{
    return src_reg_fmt == DataFormat::Float16_b || src_reg_fmt == DataFormat::Float16 || src_reg_fmt == DataFormat::Tf32 ||
           src_reg_fmt == DataFormat::MxFp4_2x_A || src_reg_fmt == DataFormat::MxFp4_2x_B;
}

/**
 * @brief Determines whether the source register format and Int32 destination register format are a supported combination
 *
 * @param src_reg_fmt: The source register format
 */
inline bool _is_src_fmt_int32_dest_compatible_(const DataFormat src_reg_fmt)
{
    return src_reg_fmt == DataFormat::Int8 || src_reg_fmt == DataFormat::UInt8 || src_reg_fmt == DataFormat::Int8_2x || src_reg_fmt == DataFormat::UInt8_2x;
}

template <bool EN_IMPLIED_MATH_FORMAT, bool EN_32BIT_DEST>
inline void _configure_default_data_format_state_(DataFormat srcA_format, DataFormat srcB_format)
{
    TTI_ATGETM(mutex::REG_RMW);
    if (data_format_config_set != DataFormatConfigSet::DEFAULT)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::SFPU1, p_stall::MATH);
        cfg_rmw(DISABLE_IMPLIED_SRCA_FMT_SEC0_Base_RMW, !EN_IMPLIED_MATH_FORMAT);
        cfg_rmw(DISABLE_IMPLIED_SRCA_FMT_SEC1_Base_RMW, !EN_IMPLIED_MATH_FORMAT);
        cfg_rmw(DISABLE_IMPLIED_SRCA_FMT_SEC2_Base_RMW, !EN_IMPLIED_MATH_FORMAT);
        cfg_rmw(DISABLE_IMPLIED_SRCA_FMT_SEC3_Base_RMW, !EN_IMPLIED_MATH_FORMAT);
        cfg_rmw(DISABLE_IMPLIED_SRCB_FMT_SEC0_Base_RMW, !EN_IMPLIED_MATH_FORMAT);
        cfg_rmw(DISABLE_IMPLIED_SRCB_FMT_SEC1_Base_RMW, !EN_IMPLIED_MATH_FORMAT);
        cfg_rmw(DISABLE_IMPLIED_SRCB_FMT_SEC2_Base_RMW, !EN_IMPLIED_MATH_FORMAT);
        cfg_rmw(DISABLE_IMPLIED_SRCB_FMT_SEC3_Base_RMW, !EN_IMPLIED_MATH_FORMAT);

        const bool EN_FP32_DEST_FORMAT  = _is_src_fmt_fp32_dest_compatible_(srcA_format) && _is_src_fmt_fp32_dest_compatible_(srcB_format);
        const bool EN_INT32_DEST_FORMAT = _is_src_fmt_int32_dest_compatible_(srcA_format) && _is_src_fmt_int32_dest_compatible_(srcB_format);
        LLK_ASSERT(!(EN_FP32_DEST_FORMAT && EN_INT32_DEST_FORMAT), "Cannot have Int32 dest & Float32 dest at the same time");

        std::uint8_t SRCA_FORMAT_MASKED = static_cast<std::uint8_t>(srcA_format) & 0xFF;
        std::uint8_t SRCB_FORMAT_MASKED = static_cast<std::uint8_t>(srcB_format) & 0xFF;

        cfg_rmw(ALU_FORMAT_SPEC_REG_SrcA_val_RMW, SRCA_FORMAT_MASKED);
        cfg_rmw(ALU_FORMAT_SPEC_REG_SrcA_override_RMW, 0x1);
        cfg_rmw(ALU_FORMAT_SPEC_REG_SrcB_val_RMW, SRCB_FORMAT_MASKED);
        cfg_rmw(ALU_FORMAT_SPEC_REG_SrcB_override_RMW, 0x1);

        cfg_rmw(ALU_FORMAT_SPEC_REG0_SrcA_RMW, SRCA_FORMAT_MASKED);
        cfg_rmw(ALU_FORMAT_SPEC_REG1_SrcB_RMW, SRCB_FORMAT_MASKED);

        cfg_rmw(ALU_ACC_CTRL_Fp32_enabled_RMW, EN_32BIT_DEST && EN_FP32_DEST_FORMAT);
        cfg_rmw(ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW, EN_32BIT_DEST && EN_FP32_DEST_FORMAT);
        cfg_rmw(ALU_ACC_CTRL_INT8_math_enabled_RMW, EN_32BIT_DEST && EN_INT32_DEST_FORMAT);

        data_format_config_set = DataFormatConfigSet::DEFAULT;
    }
    TTI_ATRELM(mutex::REG_RMW);
}

inline void _configure_mov_src2dst_ops_data_format_state_(DataFormat srcA_format, DataFormat srcB_format)
{
    TTI_ATGETM(mutex::REG_RMW);
    if (data_format_config_set != DataFormatConfigSet::MOV_SRC2DST_OPS_INT32)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::SFPU1, p_stall::MATH);
        cfg_rmw(DISABLE_IMPLIED_SRCA_FMT_SEC0_Base_RMW, 1);
        cfg_rmw(DISABLE_IMPLIED_SRCA_FMT_SEC1_Base_RMW, 1);
        cfg_rmw(DISABLE_IMPLIED_SRCA_FMT_SEC2_Base_RMW, 1);
        cfg_rmw(DISABLE_IMPLIED_SRCA_FMT_SEC3_Base_RMW, 1);
        cfg_rmw(DISABLE_IMPLIED_SRCB_FMT_SEC0_Base_RMW, 1);
        cfg_rmw(DISABLE_IMPLIED_SRCB_FMT_SEC1_Base_RMW, 1);
        cfg_rmw(DISABLE_IMPLIED_SRCB_FMT_SEC2_Base_RMW, 1);
        cfg_rmw(DISABLE_IMPLIED_SRCB_FMT_SEC3_Base_RMW, 1);

        std::uint8_t SRCA_FORMAT_MASKED = static_cast<std::uint8_t>(srcA_format) & 0xFF;
        std::uint8_t SRCB_FORMAT_MASKED = static_cast<std::uint8_t>(srcB_format) & 0xFF;

        cfg_rmw(ALU_FORMAT_SPEC_REG_SrcA_val_RMW, SRCA_FORMAT_MASKED);
        cfg_rmw(ALU_FORMAT_SPEC_REG_SrcA_override_RMW, 0x1);
        cfg_rmw(ALU_FORMAT_SPEC_REG_SrcB_val_RMW, SRCB_FORMAT_MASKED);
        cfg_rmw(ALU_FORMAT_SPEC_REG_SrcB_override_RMW, 0x1);

        cfg_rmw(ALU_FORMAT_SPEC_REG0_SrcA_RMW, SRCA_FORMAT_MASKED);
        cfg_rmw(ALU_FORMAT_SPEC_REG1_SrcB_RMW, SRCB_FORMAT_MASKED);

        cfg_rmw(ALU_ACC_CTRL_Fp32_enabled_RMW, 1);
        cfg_rmw(ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW, 0);
        cfg_rmw(ALU_ACC_CTRL_INT8_math_enabled_RMW, 0);

        data_format_config_set = DataFormatConfigSet::MOV_SRC2DST_OPS_INT32;
    }
    TTI_ATRELM(mutex::REG_RMW);
}

} // namespace ckernel::trisc
