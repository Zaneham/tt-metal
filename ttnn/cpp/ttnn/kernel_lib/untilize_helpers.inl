// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file untilize_helpers.inl
 * @brief Implementation of untilize helper functions
 *
 * This file contains the implementation details for the untilize() function.
 * It should only be included by untilize_helpers.hpp.
 */

#include "ttnn/cpp/ttnn/kernel_lib/cb_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tile_format_helpers.hpp"
#include "api/dataflow/circular_buffer.h"

namespace compute_kernel_lib {

// =============================================================================
// Fast Untilize Gate Helpers
// =============================================================================

template <uint32_t input_cb, uint32_t output_cb>
constexpr bool has_supported_fast_untilize_format() {
    constexpr auto input_format = cb_l1_format<input_cb>();
    constexpr auto output_format = cb_l1_format<output_cb>();

    // Production untilize is bit-exact for Float32 today. The fast fp32 LLK path
    // is native fp32 DEST but still narrows input through the current SrcA route,
    // so keep fp32 out of automatic selection until that path is lossless.
    constexpr bool supported_input = input_format == static_cast<std::uint32_t>(DataFormat::Float16_b) ||
                                     input_format == static_cast<std::uint32_t>(DataFormat::Bfp8_b) ||
                                     input_format == static_cast<std::uint32_t>(DataFormat::Bfp4_b);
    constexpr bool supported_output = output_format == static_cast<std::uint32_t>(DataFormat::Float16_b);

    return supported_input && supported_output;
}

template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb>
constexpr bool can_use_fast_untilize() {
#ifdef ARCH_BLACKHOLE
    return block_width_tiles >= 2 && cb_has_32x32_tiles<input_cb>() && cb_has_32x32_tiles<output_cb>() &&
           has_supported_fast_untilize_format<input_cb, output_cb>();
#else
    return false;
#endif
}

// =============================================================================
// Block Splitting Helper for Wide Untilize
// =============================================================================

/**
 * @brief Compute number of blocks needed to split a wide row into DEST-sized chunks
 *
 * Finds the largest divisor of total_width that is <= max_block_width
 * This ensures optimal block size while respecting DEST register limits.
 *
 * @param total_width Total width in tiles to be split
 * @param max_block_width Maximum block width (DEST register limit)
 * @return Number of blocks needed
 */
constexpr uint32_t compute_num_blocks(uint32_t total_width, uint32_t max_block_width) {
    for (uint32_t block_width = max_block_width; block_width > 0; --block_width) {
        if (total_width % block_width == 0) {
            return total_width / block_width;
        }
    }
    return total_width;  // fallback: 1 tile per block
}

template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb>
struct UntilizeDispatchConfig {
    static constexpr bool use_fast = can_use_fast_untilize<block_width_tiles, input_cb, output_cb>();
    static constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    static constexpr bool use_block_based_pack_path = !use_fast && (block_width_tiles > dest_limit);
    static constexpr uint32_t num_sub_blocks =
        use_block_based_pack_path ? compute_num_blocks(block_width_tiles, dest_limit) : 1;
    static constexpr uint32_t sub_block_width =
        use_block_based_pack_path ? (block_width_tiles / num_sub_blocks) : block_width_tiles;
};

template <untilize_config::WaitMode wait_mode>
ALWI void untilize_wait_for_block(CircularBuffer& in_cb, const uint32_t tile_count) {
    if constexpr (wait_mode == untilize_config::WaitMode::WaitBlock) {
        in_cb.wait_front(tile_count);
    }
}

// =============================================================================
// Standalone Init/Uninit Wrapper Functions Implementations
// =============================================================================

template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb, untilize_config::RemapMode remap_mode>
ALWI void untilize_init() {
    using dispatch = UntilizeDispatchConfig<block_width_tiles, input_cb, output_cb>;
    constexpr bool configure_remap = (remap_mode == untilize_config::RemapMode::Configure);

    if constexpr (dispatch::use_fast) {
        if constexpr (configure_remap) {
            fast_untilize_init<block_width_tiles>(input_cb, output_cb);
        } else {
            fast_untilize_init_skip_remap<block_width_tiles>(input_cb, output_cb);
        }
    } else if constexpr (dispatch::use_block_based_pack_path) {
        if constexpr (configure_remap) {
            pack_untilize_init<dispatch::sub_block_width, block_width_tiles>(input_cb, output_cb);
        } else {
            pack_untilize_init_skip_remap<dispatch::sub_block_width, block_width_tiles>(input_cb, output_cb);
        }
    } else {
        if constexpr (configure_remap) {
            pack_untilize_init<block_width_tiles, block_width_tiles>(input_cb, output_cb);
        } else {
            pack_untilize_init_skip_remap<block_width_tiles, block_width_tiles>(input_cb, output_cb);
        }
    }
}

template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb>
ALWI void untilize_uninit() {
    using dispatch = UntilizeDispatchConfig<block_width_tiles, input_cb, output_cb>;

    if constexpr (dispatch::use_fast) {
        fast_untilize_uninit<block_width_tiles>(output_cb);
    } else {
        pack_untilize_uninit(output_cb);
    }
}

// =============================================================================
// Main Untilize Function Implementation
// =============================================================================

template <
    uint32_t block_width_tiles,
    uint32_t input_cb,
    uint32_t output_cb,
    untilize_config::InitUninitMode init_uninit_mode,
    untilize_config::WaitMode wait_mode,
    untilize_config::ReconfigureRegisterDatatypeMode reconfig_mode,
    untilize_config::RemapMode remap_mode>
ALWI void untilize(uint32_t num_blocks) {
    // Compile-time validation
    static_assert(input_cb != output_cb, "Untilize cannot be done in-place: input_cb and output_cb must be different");
    static_assert(block_width_tiles > 0, "block_width_tiles must be greater than 0");
    static_assert(input_cb < 32, "Invalid input_cb: must be less than 32");
    static_assert(output_cb < 32, "Invalid output_cb: must be less than 32");

    // Runtime parameter validation
    ASSERT(num_blocks > 0);

    // Untilize output must not be a block float format (Bfp8/4/2 and _b variants).
    // Block floats have shared exponents that break tile-to-row-major reinterpretation.
    PACK(ASSERT(!is_block_float_format(pack_dst_format[output_cb])));

    using dispatch = UntilizeDispatchConfig<block_width_tiles, input_cb, output_cb>;

    // Determine if we're doing data type reconfiguration
    constexpr bool use_unpack_reconfig =
        (reconfig_mode == untilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure) ||
        (reconfig_mode == untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    constexpr bool use_pack_reconfig =
        (reconfig_mode == untilize_config::ReconfigureRegisterDatatypeMode::PackReconfigure) ||
        (reconfig_mode == untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    // Reconfigure register datatypes if requested
    if constexpr (use_unpack_reconfig) {
        reconfig_data_format_srca(input_cb);
    }

    if constexpr (use_pack_reconfig) {
        pack_reconfig_data_format(output_cb);
    }

    // Validate CB capacity.
    // Guarded because get_local_cb_interface() references cb_interface, which is
    // not defined for the MATH TRISC (trisc.cc excludes it via #if !defined(UCK_CHLKC_MATH)).
    PACK(ASSERT(get_cb_num_pages(output_cb) >= block_width_tiles));
    if constexpr (dispatch::use_fast) {
        UNPACK(ASSERT(get_cb_num_pages(input_cb) >= block_width_tiles));
    } else if constexpr (dispatch::use_block_based_pack_path) {
        UNPACK(ASSERT(get_cb_num_pages(input_cb) >= dispatch::sub_block_width));
    } else {
        UNPACK(ASSERT(get_cb_num_pages(input_cb) >= block_width_tiles));
    }

    // =================================================================
    // INITIALIZATION
    // =================================================================

    if constexpr (
        init_uninit_mode == untilize_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == untilize_config::InitUninitMode::InitOnly) {
        untilize_init<block_width_tiles, input_cb, output_cb, remap_mode>();
    }

    // =================================================================
    // UPFRONT WAITING (if requested)
    // =================================================================

    // Construct CircularBuffer objects for sync operations
    CircularBuffer in_cb(input_cb);
    CircularBuffer out_cb(output_cb);

    if constexpr (wait_mode == untilize_config::WaitMode::WaitUpfront) {
        uint32_t total_tiles = block_width_tiles * num_blocks;
        in_cb.wait_front(total_tiles);
    }

    // =================================================================
    // MAIN PROCESSING LOOP
    // =================================================================

    if constexpr (dispatch::use_fast) {
        // =============================================================
        // BH FAST UNTILIZE PATH
        // =============================================================

        for (uint32_t r = 0; r < num_blocks; ++r) {
            untilize_wait_for_block<wait_mode>(in_cb, block_width_tiles);
            out_cb.reserve_back(block_width_tiles);
            fast_untilize_block<block_width_tiles>(input_cb, output_cb);
            in_cb.pop_front(block_width_tiles);
            out_cb.push_back(block_width_tiles);
        }

    } else if constexpr (dispatch::use_block_based_pack_path) {
        // =============================================================
        // BLOCK-BASED PACK UNTILIZE PATH
        // Used when width exceeds DEST limit
        // Splits wide rows into multiple sub-blocks that each fit in DEST
        // =============================================================

        for (uint32_t r = 0; r < num_blocks; ++r) {
            out_cb.reserve_back(block_width_tiles);
            for (uint32_t b = 0; b < dispatch::num_sub_blocks; ++b) {
                untilize_wait_for_block<wait_mode>(in_cb, dispatch::sub_block_width);
                pack_untilize_block<dispatch::sub_block_width, block_width_tiles>(input_cb, 1, output_cb, b);
                in_cb.pop_front(dispatch::sub_block_width);
            }
            out_cb.push_back(block_width_tiles);
        }

    } else {
        // =============================================================
        // PACK UNTILIZE PATH (SINGLE-PASS)
        // Used when width fits in DEST (optimal path)
        // =============================================================

        for (uint32_t r = 0; r < num_blocks; ++r) {
            untilize_wait_for_block<wait_mode>(in_cb, block_width_tiles);
            out_cb.reserve_back(block_width_tiles);
            pack_untilize_block<block_width_tiles, block_width_tiles>(input_cb, 1, output_cb, 0);
            in_cb.pop_front(block_width_tiles);
            out_cb.push_back(block_width_tiles);
        }
    }

    // =================================================================
    // CLEANUP
    // =================================================================

    if constexpr (
        init_uninit_mode == untilize_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == untilize_config::InitUninitMode::UninitOnly) {
        untilize_uninit<block_width_tiles, input_cb, output_cb>();
    }
}

}  // namespace compute_kernel_lib
