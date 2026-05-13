// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sort_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>

#include <cstdint>

namespace ttnn::prim {

// Single row - single core
struct SortProgramFactorySingleRowSingleCore {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SortParams&, const SortInputs&, std::vector<Tensor>&);
};

// SortProgramFactoryCrossCoreDataExchange - single row, multi core with processing multiple tiles on one core with
// cross core data exchange
struct SortProgramFactoryCrossCoreDataExchange {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SortParams&, const SortInputs&, std::vector<Tensor>&);

    /**
     * @brief Strategies for slicing work across cores in cross-core data exchange sort.
     */
    enum class CrossCoreDataExchangeSortSlicingStrategy : uint8_t {
        USE_AS_MANY_CORES,  ///< Use all available cores to process the same line, optimizing for latency.
        FILL_CORES_FIRST,   ///< Fill cores sequentially before assigning additional work.
    };

    static uint32_t get_number_of_tiles_per_core(
        uint32_t total_number_of_cores,
        uint32_t Wt,
        const DataType& input_dtype,
        const DataType& index_dtype,
        CrossCoreDataExchangeSortSlicingStrategy slicing_strategy =
            CrossCoreDataExchangeSortSlicingStrategy::USE_AS_MANY_CORES);

    static uint32_t rounddown_pow2(uint32_t n);
};

// Single row - multi core
struct SortProgramFactorySingleRowMultiCore {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SortParams&, const SortInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::prim
