// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <cmath>
#include <cstdint>

namespace ttnn::prim {

using namespace tt::tt_metal;

// Single row - single core
ProgramDescriptor SortProgramFactorySingleRowSingleCore::create_descriptor(
    const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors) {
    const bool is_row_major = (tensor_args.input_tensor.layout() == Layout::ROW_MAJOR);

    const tt::DataFormat input_tensor_cb_data_format =
        datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat value_tensor_cb_data_format = datatype_to_dataformat_converter(output_tensors.at(0).dtype());
    const tt::DataFormat index_tensor_cb_data_format = datatype_to_dataformat_converter(output_tensors.at(1).dtype());

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_cb_data_format);
    const uint32_t value_tensor_tile_size = tile_size(value_tensor_cb_data_format);
    const uint32_t index_tensor_tile_size = tile_size(index_tensor_cb_data_format);

    auto* input_buffer = tensor_args.input_tensor.buffer();
    auto* value_buffer = output_tensors.at(0).buffer();
    auto* index_buffer = output_tensors.at(1).buffer();

    const auto input_shape =
        is_row_major ? tensor_args.input_tensor.logical_shape() : tensor_args.input_tensor.padded_shape();
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tt::constants::TILE_HEIGHT;
    const uint32_t Wt = input_shape[3] / tt::constants::TILE_WIDTH;

    const uint32_t element_size_bytes = tt::datum_size(input_tensor_cb_data_format);
    const uint32_t index_element_size_bytes = tt::datum_size(index_tensor_cb_data_format);
    const uint32_t W_value_bytes = input_shape[3] * element_size_bytes;
    const uint32_t W_index_bytes = input_shape[3] * index_element_size_bytes;

    constexpr uint32_t num_cb_unit = 2;
    constexpr uint32_t cb_in_units = 2 * num_cb_unit;

    auto* device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t total_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    const uint32_t all_core_utilization_loop_count = Ht / total_number_of_cores;
    const uint32_t all_core_utilization_loop_residuum = Ht % total_number_of_cores;

    const bool is_32_bit_index = index_tensor_cb_data_format == tt::DataFormat::UInt32;
    const bool is_32_bit_data = is_32_bit_index || input_tensor_cb_data_format == tt::DataFormat::Float32;

    CoreRangeSet core_range;
    if (Ht >= total_number_of_cores) {
        core_range = CoreRangeSet(
            CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1}));
    } else {
        const uint32_t core_grid_calculated_rows_number = Ht / compute_with_storage_grid_size.x;
        const uint32_t core_grid_calculated_columns_number = Ht % compute_with_storage_grid_size.x;

        if (core_grid_calculated_rows_number == 0 && core_grid_calculated_columns_number == 0) {
            core_range = CoreRangeSet(CoreCoord({0, 0}));
        } else if (core_grid_calculated_rows_number == 0) {
            core_range = CoreRangeSet(CoreRange({0, 0}, {core_grid_calculated_columns_number - 1, 0}));
        } else {
            core_range = CoreRangeSet(
                CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, core_grid_calculated_rows_number - 1}));
            if (core_grid_calculated_columns_number != 0) {
                const CoreRange additional_range(
                    {0, core_grid_calculated_rows_number},
                    {core_grid_calculated_columns_number - 1, core_grid_calculated_rows_number});
                core_range = core_range.merge(CoreRangeSet(additional_range));
            }
        }
    }

    ProgramDescriptor desc;

    // -----------------------------------------------------------------------
    // Circular buffers
    // -----------------------------------------------------------------------
    constexpr uint32_t input_tensor_cb_index = tt::CBIndex::c_0;
    {
        const uint32_t cb0_tiles = is_row_major ? Wt : cb_in_units;
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb0_tiles * input_tensor_tile_size,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(input_tensor_cb_index),
                .data_format = input_tensor_cb_data_format,
                .page_size = input_tensor_tile_size,
            }}},
        });
    }

    constexpr uint32_t index_tensor_cb_index = tt::CBIndex::c_1;
    {
        const uint32_t cb1_tiles = is_row_major ? Wt : cb_in_units;
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb1_tiles * index_tensor_tile_size,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index_tensor_cb_index),
                .data_format = index_tensor_cb_data_format,
                .page_size = index_tensor_tile_size,
            }}},
        });
    }

    constexpr uint32_t input_tensor_transposed_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Wt * input_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_tensor_transposed_cb_index),
            .data_format = input_tensor_cb_data_format,
            .page_size = input_tensor_tile_size,
        }}},
    });

    constexpr uint32_t index_tensor_transposed_cb_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Wt * index_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_tensor_transposed_cb_index),
            .data_format = index_tensor_cb_data_format,
            .page_size = index_tensor_tile_size,
        }}},
    });

    constexpr uint32_t value_tensor_cb_index = tt::CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cb_unit * value_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(value_tensor_cb_index),
            .data_format = value_tensor_cb_data_format,
            .page_size = value_tensor_tile_size,
        }}},
    });

    constexpr uint32_t index_tensor_output_cb_index = tt::CBIndex::c_5;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cb_unit * index_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_tensor_output_cb_index),
            .data_format = index_tensor_cb_data_format,
            .page_size = index_tensor_tile_size,
        }}},
    });

    constexpr uint32_t synchronization_cb_index = tt::CBIndex::c_6;
    constexpr uint32_t synchronization_cb_size = tt::constants::TILE_HW * sizeof(uint8_t);
    desc.cbs.push_back(CBDescriptor{
        .total_size = synchronization_cb_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(synchronization_cb_index),
            .data_format = tt::DataFormat::UInt8,
            .page_size = synchronization_cb_size,
        }}},
    });

    constexpr uint32_t rm_input_cb_index = tt::CBIndex::c_7;
    constexpr uint32_t rm_value_output_cb_index = tt::CBIndex::c_8;
    constexpr uint32_t rm_index_output_cb_index = tt::CBIndex::c_9;
    constexpr uint32_t rm_post_sort_index_cb_index = tt::CBIndex::c_10;
    if (is_row_major) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = tt::constants::TILE_HEIGHT * W_value_bytes,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rm_input_cb_index),
                .data_format = input_tensor_cb_data_format,
                .page_size = W_value_bytes,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = tt::constants::TILE_HEIGHT * W_value_bytes,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rm_value_output_cb_index),
                .data_format = value_tensor_cb_data_format,
                .page_size = W_value_bytes,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = tt::constants::TILE_HEIGHT * W_index_bytes,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rm_index_output_cb_index),
                .data_format = index_tensor_cb_data_format,
                .page_size = W_index_bytes,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = Wt * index_tensor_tile_size,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rm_post_sort_index_cb_index),
                .data_format = index_tensor_cb_data_format,
                .page_size = index_tensor_tile_size,
            }}},
        });
    }

    // -----------------------------------------------------------------------
    // Kernels
    // -----------------------------------------------------------------------
    const uint32_t loop_count = all_core_utilization_loop_count ? all_core_utilization_loop_count : 1;
    const bool needs_residuum_bump =
        (all_core_utilization_loop_count != 0) && (all_core_utilization_loop_residuum != 0);

    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_index,
        index_tensor_output_cb_index,
        Wt,
        Ht,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        static_cast<uint32_t>(is_row_major),
        rm_input_cb_index,
        rm_index_output_cb_index,
        W_value_bytes,
        W_index_bytes,
    };
    TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*index_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        value_tensor_cb_index,
        index_tensor_cb_index,
        Wt,
        Ht,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        static_cast<uint32_t>(is_32_bit_data),
        static_cast<uint32_t>(is_row_major),
        rm_value_output_cb_index,
        W_value_bytes,
    };
    TensorAccessorArgs(*value_buffer).append_to(writer_compile_time_args);

    const std::vector<uint32_t> compute_compile_time_args = {
        input_tensor_cb_index,
        index_tensor_cb_index,
        input_tensor_transposed_cb_index,
        index_tensor_transposed_cb_index,
        value_tensor_cb_index,
        index_tensor_output_cb_index,
        Wt,
        static_cast<uint32_t>(attributes.descending),
        static_cast<uint32_t>(attributes.stable),
        synchronization_cb_index,
        static_cast<uint32_t>(is_row_major),
        rm_input_cb_index,
        rm_value_output_cb_index,
        rm_index_output_cb_index,
        rm_post_sort_index_cb_index,
    };

    std::vector<UnpackToDestMode> unpack_to_dest_mode_vector(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (input_tensor_cb_data_format == tt::DataFormat::Float32) {
        unpack_to_dest_mode_vector[input_tensor_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode_vector[input_tensor_transposed_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode_vector[value_tensor_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        if (is_row_major) {
            unpack_to_dest_mode_vector[rm_input_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        }
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/reader_single_row_single_core.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_range;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/writer_single_row_single_core.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_range;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/compute/sort_single_row_single_core.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_range;
    compute_desc.compile_time_args = compute_compile_time_args;
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = is_32_bit_data,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode_vector),
    };

    // Per-core runtime args: first all_core_utilization_loop_residuum cores (in grid order) get
    // loop_count+1, the rest get loop_count — matching the old SetRuntimeArgs residuum override.
    uint32_t bumped_count = 0;
    for (uint32_t core_y = 0; core_y < compute_with_storage_grid_size.y; core_y++) {
        for (uint32_t core_x = 0; core_x < compute_with_storage_grid_size.x; core_x++) {
            const CoreCoord core{core_x, core_y};
            if (!core_range.contains(core)) {
                continue;
            }
            const bool bump = needs_residuum_bump && (bumped_count < all_core_utilization_loop_residuum);
            const uint32_t this_loop = bump ? (loop_count + 1) : loop_count;
            if (bump) {
                bumped_count++;
            }

            reader_desc.runtime_args.emplace_back(
                core, KernelDescriptor::CoreRuntimeArgs{input_buffer->address(), index_buffer->address(), this_loop});
            writer_desc.runtime_args.emplace_back(
                core, KernelDescriptor::CoreRuntimeArgs{value_buffer->address(), this_loop});
            compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{this_loop});
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

// SortProgramFactoryCrossCoreDataExchange - single row, multi core with processing multiple tiles on one core with
// cross core data exchange
ProgramDescriptor SortProgramFactoryCrossCoreDataExchange::create_descriptor(
    const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors) {
    const bool is_row_major = (tensor_args.input_tensor.layout() == Layout::ROW_MAJOR);

    const tt::DataFormat input_tensor_cb_data_format =
        datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat value_tensor_cb_data_format = datatype_to_dataformat_converter(output_tensors.at(0).dtype());
    const tt::DataFormat index_tensor_cb_data_format = datatype_to_dataformat_converter(output_tensors.at(1).dtype());
    const tt::DataFormat packer_unpacker_sync_cb_data_format = tt::DataFormat::Float16_b;

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_cb_data_format);
    const uint32_t value_tensor_tile_size = tile_size(value_tensor_cb_data_format);
    const uint32_t index_tensor_tile_size = tile_size(index_tensor_cb_data_format);
    const uint32_t packer_unpacker_sync_tile_size = tile_size(packer_unpacker_sync_cb_data_format);

    auto* input_buffer = tensor_args.input_tensor.buffer();
    auto* value_buffer = output_tensors.at(0).buffer();
    auto* index_buffer = output_tensors.at(1).buffer();

    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();

    const auto input_shape =
        is_row_major ? tensor_args.input_tensor.logical_shape() : tensor_args.input_tensor.padded_shape();
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tile_height;
    const uint32_t Wt = input_shape[3] / tile_width;

    const uint32_t value_element_bytes = tt::datum_size(value_tensor_cb_data_format);
    const uint32_t index_element_bytes = tt::datum_size(index_tensor_cb_data_format);

    auto* const device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t total_number_of_cores_physical = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;
    const uint32_t total_number_of_cores_virtual = rounddown_pow2(total_number_of_cores_physical);
    uint32_t number_of_tiles_per_core = get_number_of_tiles_per_core(
        total_number_of_cores_virtual,
        Wt,
        tensor_args.input_tensor.dtype(),
        output_tensors.at(1).dtype(),
        CrossCoreDataExchangeSortSlicingStrategy::USE_AS_MANY_CORES);
    number_of_tiles_per_core = std::min(number_of_tiles_per_core, Wt);

    const uint32_t all_core_utilization_count = (Wt + number_of_tiles_per_core - 1) / number_of_tiles_per_core;

    const uint32_t W_value_slice_bytes = number_of_tiles_per_core * tile_width * value_element_bytes;
    const uint32_t W_index_slice_bytes = number_of_tiles_per_core * tile_width * index_element_bytes;

    TT_FATAL(
        all_core_utilization_count <= total_number_of_cores_virtual,
        "All core utilization count exceeds total number of cores. Utilized cores: {}, Total cores: {}",
        all_core_utilization_count,
        total_number_of_cores_virtual);

    const bool is_32_bit_index = index_tensor_cb_data_format == tt::DataFormat::UInt32;
    const bool is_32_bit_data = is_32_bit_index || input_tensor_cb_data_format == tt::DataFormat::Float32;

    CoreRangeSet core_range;
    if (all_core_utilization_count == total_number_of_cores_physical) {
        core_range = CoreRangeSet(
            CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1}));
    } else if (all_core_utilization_count == total_number_of_cores_virtual) {
        const uint32_t core_grid_calculated_rows_number =
            (all_core_utilization_count / compute_with_storage_grid_size.x) - 1;
        const uint32_t core_grid_calculated_columns_number =
            all_core_utilization_count % compute_with_storage_grid_size.x;
        core_range =
            CoreRangeSet(CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, core_grid_calculated_rows_number}));
        if (core_grid_calculated_columns_number != 0) {
            const CoreRange additional_range(
                {0, core_grid_calculated_rows_number + 1},
                {core_grid_calculated_columns_number - 1, core_grid_calculated_rows_number + 1});
            core_range = core_range.merge(CoreRangeSet(additional_range));
        }
    } else {
        const uint32_t core_grid_calculated_rows_number = all_core_utilization_count / compute_with_storage_grid_size.x;
        const uint32_t core_grid_calculated_columns_number =
            all_core_utilization_count % compute_with_storage_grid_size.x;

        if (core_grid_calculated_rows_number == 0 && core_grid_calculated_columns_number == 0) {
            core_range = CoreRangeSet(CoreCoord({0, 0}));
        } else if (core_grid_calculated_rows_number == 0) {
            core_range = CoreRangeSet(CoreRange({0, 0}, {core_grid_calculated_columns_number - 1, 0}));
        } else {
            core_range = CoreRangeSet(
                CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, core_grid_calculated_rows_number - 1}));
            if (core_grid_calculated_columns_number != 0) {
                const CoreRange additional_range(
                    {0, core_grid_calculated_rows_number},
                    {core_grid_calculated_columns_number - 1, core_grid_calculated_rows_number});
                core_range = core_range.merge(CoreRangeSet(additional_range));
            }
        }
    }

    // Physical core lookup table (fixed per device/core-range; recreated each call)
    std::vector<uint32_t> physical_core_lookup_table_data;
    for (const auto& range : core_range.ranges()) {
        for (const auto& core_coord : range) {
            const auto physical_core = device->worker_core_from_logical_core(core_coord);
            physical_core_lookup_table_data.emplace_back(physical_core.x);
            physical_core_lookup_table_data.emplace_back(physical_core.y);
        }
    }
    const TensorSpec physical_core_lookup_table_spec(
        ttnn::Shape{1, physical_core_lookup_table_data.size()},
        TensorLayout{DataType::UINT32, PageConfig{Layout::ROW_MAJOR}, MemoryConfig()});
    Tensor physical_core_lookup_table_tensor =
        Tensor::from_vector(std::move(physical_core_lookup_table_data), physical_core_lookup_table_spec);
    physical_core_lookup_table_tensor = physical_core_lookup_table_tensor.to_device(device);
    auto* const physical_core_lookup_table_tensor_buffer = physical_core_lookup_table_tensor.buffer();
    const tt::DataFormat physical_core_lookup_table_cb_data_format =
        datatype_to_dataformat_converter(physical_core_lookup_table_tensor.dtype());
    const uint32_t physical_core_lookup_table_tile_size = tile_size(physical_core_lookup_table_cb_data_format);

    ProgramDescriptor desc;

    // -----------------------------------------------------------------------
    // Circular buffers
    // -----------------------------------------------------------------------
    constexpr uint32_t cb_scale_factor = 2;

    constexpr uint32_t input_tensor_cb_index = tt::CBIndex::c_0;
    {
        const uint32_t cb0_tiles = is_row_major ? number_of_tiles_per_core : cb_scale_factor;
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb0_tiles * input_tensor_tile_size,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(input_tensor_cb_index),
                .data_format = input_tensor_cb_data_format,
                .page_size = input_tensor_tile_size,
            }}},
        });
    }

    constexpr uint32_t index_tensor_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_scale_factor * index_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_tensor_cb_index),
            .data_format = index_tensor_cb_data_format,
            .page_size = index_tensor_tile_size,
        }}},
    });

    constexpr uint32_t input_tensor_transposed_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = number_of_tiles_per_core * input_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_tensor_transposed_cb_index),
            .data_format = input_tensor_cb_data_format,
            .page_size = input_tensor_tile_size,
        }}},
    });

    constexpr uint32_t index_tensor_transposed_cb_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = number_of_tiles_per_core * index_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_tensor_transposed_cb_index),
            .data_format = index_tensor_cb_data_format,
            .page_size = index_tensor_tile_size,
        }}},
    });

    constexpr uint32_t value_tensor_cb_index = tt::CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_scale_factor * value_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(value_tensor_cb_index),
            .data_format = value_tensor_cb_data_format,
            .page_size = value_tensor_tile_size,
        }}},
    });

    constexpr uint32_t index_tensor_output_cb_index = tt::CBIndex::c_5;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_scale_factor * index_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_tensor_output_cb_index),
            .data_format = index_tensor_cb_data_format,
            .page_size = index_tensor_tile_size,
        }}},
    });

    constexpr uint32_t value_tensor_intermediate_cb_index = tt::CBIndex::c_6;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_scale_factor * value_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(value_tensor_intermediate_cb_index),
            .data_format = value_tensor_cb_data_format,
            .page_size = value_tensor_tile_size,
        }}},
    });

    constexpr uint32_t index_tensor_intermediate_cb_index = tt::CBIndex::c_7;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_scale_factor * index_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_tensor_intermediate_cb_index),
            .data_format = index_tensor_cb_data_format,
            .page_size = index_tensor_tile_size,
        }}},
    });

    constexpr uint32_t value_tensor_peer_cb_index = tt::CBIndex::c_8;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_scale_factor * value_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(value_tensor_peer_cb_index),
            .data_format = value_tensor_cb_data_format,
            .page_size = value_tensor_tile_size,
        }}},
    });

    constexpr uint32_t index_tensor_peer_cb_index = tt::CBIndex::c_9;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_scale_factor * index_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_tensor_peer_cb_index),
            .data_format = index_tensor_cb_data_format,
            .page_size = index_tensor_tile_size,
        }}},
    });

    constexpr uint32_t physical_core_lookup_table_cb_index = tt::CBIndex::c_10;
    desc.cbs.push_back(CBDescriptor{
        .total_size = physical_core_lookup_table_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(physical_core_lookup_table_cb_index),
            .data_format = physical_core_lookup_table_cb_data_format,
            .page_size = physical_core_lookup_table_tile_size,
        }}},
    });

    constexpr uint32_t packer_unpacker_sync_cb_index = tt::CBIndex::c_11;
    desc.cbs.push_back(CBDescriptor{
        .total_size = packer_unpacker_sync_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(packer_unpacker_sync_cb_index),
            .data_format = packer_unpacker_sync_cb_data_format,
            .page_size = packer_unpacker_sync_tile_size,
        }}},
    });

    constexpr uint32_t rm_input_cb_index = tt::CBIndex::c_12;
    constexpr uint32_t rm_value_output_cb_index = tt::CBIndex::c_13;
    constexpr uint32_t rm_index_output_cb_index = tt::CBIndex::c_14;
    constexpr uint32_t rm_post_sort_index_cb_index = tt::CBIndex::c_15;
    if (is_row_major) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = tt::constants::TILE_HEIGHT * W_value_slice_bytes,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rm_input_cb_index),
                .data_format = input_tensor_cb_data_format,
                .page_size = W_value_slice_bytes,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = tt::constants::TILE_HEIGHT * W_value_slice_bytes,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rm_value_output_cb_index),
                .data_format = value_tensor_cb_data_format,
                .page_size = W_value_slice_bytes,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = tt::constants::TILE_HEIGHT * W_index_slice_bytes,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rm_index_output_cb_index),
                .data_format = index_tensor_cb_data_format,
                .page_size = W_index_slice_bytes,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = number_of_tiles_per_core * index_tensor_tile_size,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rm_post_sort_index_cb_index),
                .data_format = index_tensor_cb_data_format,
                .page_size = index_tensor_tile_size,
            }}},
        });
    }

    // -----------------------------------------------------------------------
    // Semaphores — IDs 0, 1, 2 (sequential, no other semaphores on these cores)
    // -----------------------------------------------------------------------
    constexpr uint32_t semaphore_exchange_readers = 0;
    // semaphore ID 1 is unnamed (used internally by the cross-core exchange kernels)
    constexpr uint32_t semaphore_barrier = 2;
    desc.semaphores.push_back(SemaphoreDescriptor{.id = 0, .core_ranges = core_range, .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{.id = 1, .core_ranges = core_range, .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{.id = 2, .core_ranges = core_range, .initial_value = 0});

    // -----------------------------------------------------------------------
    // Kernels
    // -----------------------------------------------------------------------
    std::vector<uint32_t> reader_compile_time_args = {
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        input_tensor_cb_index,
        index_tensor_output_cb_index,
        value_tensor_intermediate_cb_index,
        index_tensor_intermediate_cb_index,
        value_tensor_peer_cb_index,
        index_tensor_peer_cb_index,
        physical_core_lookup_table_cb_index,
        Ht,
        Wt,
        number_of_tiles_per_core,
        all_core_utilization_count,
        !attributes.descending,
        semaphore_exchange_readers,
        semaphore_barrier,
        static_cast<uint32_t>(is_row_major),
        rm_input_cb_index,
        rm_index_output_cb_index,
        W_value_slice_bytes,
        W_index_slice_bytes,
    };
    TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*index_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*physical_core_lookup_table_tensor_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        index_tensor_cb_index,
        value_tensor_cb_index,
        value_tensor_peer_cb_index,
        physical_core_lookup_table_cb_index,
        Wt,
        Ht,
        number_of_tiles_per_core,
        total_number_of_cores_virtual,
        semaphore_exchange_readers,
        static_cast<uint32_t>(is_32_bit_data),
        static_cast<uint32_t>(is_row_major),
        rm_value_output_cb_index,
        W_value_slice_bytes,
    };
    TensorAccessorArgs(*value_buffer).append_to(writer_compile_time_args);

    const std::vector<uint32_t> compute_compile_time_args = {
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        Ht,
        Wt,
        number_of_tiles_per_core,
        all_core_utilization_count,
        !attributes.descending,
        input_tensor_cb_index,
        index_tensor_cb_index,
        input_tensor_transposed_cb_index,
        index_tensor_transposed_cb_index,
        value_tensor_cb_index,
        index_tensor_output_cb_index,
        value_tensor_intermediate_cb_index,
        index_tensor_intermediate_cb_index,
        value_tensor_peer_cb_index,
        index_tensor_peer_cb_index,
        packer_unpacker_sync_cb_index,
        static_cast<uint32_t>(is_row_major),
        rm_input_cb_index,
        rm_value_output_cb_index,
        rm_index_output_cb_index,
        rm_post_sort_index_cb_index,
    };

    std::vector<UnpackToDestMode> unpack_to_dest_mode_vector(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (input_tensor_cb_data_format == tt::DataFormat::Float32) {
        unpack_to_dest_mode_vector[input_tensor_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode_vector[input_tensor_transposed_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode_vector[value_tensor_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode_vector[value_tensor_intermediate_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode_vector[value_tensor_peer_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        if (is_row_major) {
            unpack_to_dest_mode_vector[rm_input_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        }
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/reader_cross_core_data_exchange.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_range;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/writer_cross_core_data_exchange.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_range;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/compute/sort_cross_core_data_exchange.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_range;
    compute_desc.compile_time_args = compute_compile_time_args;
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = is_32_bit_data,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode_vector),
    };

    // Per-core runtime args (same for all cores in this factory)
    for (const auto& range : core_range.ranges()) {
        for (const auto& core : range) {
            reader_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    input_buffer->address(),
                    index_buffer->address(),
                    physical_core_lookup_table_tensor_buffer->address()});
            writer_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{value_buffer->address()});
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

uint32_t SortProgramFactoryCrossCoreDataExchange::get_number_of_tiles_per_core(
    uint32_t total_number_of_cores,
    uint32_t Wt,
    const DataType& input_dtype,
    const DataType& index_dtype,
    CrossCoreDataExchangeSortSlicingStrategy slicing_strategy) {
    TT_FATAL(total_number_of_cores != 0, "number of cores cannot be 0");
    switch (slicing_strategy) {
        case CrossCoreDataExchangeSortSlicingStrategy::USE_AS_MANY_CORES: {
            constexpr uint32_t MIN_TILES_PER_CORE = 2;
            constexpr uint32_t MAX_TILES_PER_CORE = 128;
            const auto max_val = std::max(Wt / total_number_of_cores, MIN_TILES_PER_CORE);
            return std::min(MAX_TILES_PER_CORE, max_val);
        }
        case CrossCoreDataExchangeSortSlicingStrategy::FILL_CORES_FIRST:
        default: {
            if (input_dtype == DataType::FLOAT32 || input_dtype == DataType::UINT32 || input_dtype == DataType::INT32 ||
                index_dtype == DataType::INT32 || index_dtype == DataType::UINT32) {
                return 64;
            }
            break;
        }
    }

    return 128;
}

uint32_t SortProgramFactoryCrossCoreDataExchange::rounddown_pow2(uint32_t n) {
    if (n == 0) {
        return 0;
    }
    return 1 << (31 - std::countl_zero(n));
}

// Single row - multi core
ProgramDescriptor SortProgramFactorySingleRowMultiCore::create_descriptor(
    const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors) {
    const tt::DataFormat input_tensor_cb_data_format =
        datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat value_tensor_cb_data_format = datatype_to_dataformat_converter(output_tensors.at(0).dtype());
    const tt::DataFormat index_tensor_cb_data_format = datatype_to_dataformat_converter(output_tensors.at(1).dtype());

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_cb_data_format);
    const uint32_t value_tensor_tile_size = tile_size(value_tensor_cb_data_format);
    const uint32_t index_tensor_tile_size = tile_size(index_tensor_cb_data_format);

    auto* const input_buffer = tensor_args.input_tensor.buffer();
    auto* const value_buffer = output_tensors.at(0).buffer();
    auto* const index_buffer = output_tensors.at(1).buffer();

    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();

    const bool is_row_major = (tensor_args.input_tensor.layout() == Layout::ROW_MAJOR);

    const auto input_shape =
        is_row_major ? tensor_args.input_tensor.logical_shape() : tensor_args.input_tensor.padded_shape();
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tile_height;
    const uint32_t Wt = input_shape[3] / tile_width;

    const uint32_t value_element_size = tt::datum_size(input_tensor_cb_data_format);
    const uint32_t W_tile_bytes = tile_width * value_element_size;

    auto* device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t total_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    const uint32_t total_work_units = Wt / 2;
    const uint32_t number_of_available_cores = total_number_of_cores - 1;

    const uint32_t all_core_utilization_loop_count = total_work_units / number_of_available_cores;

    const bool is_32_bit_index = index_tensor_cb_data_format == tt::DataFormat::UInt32;
    const bool is_32_bit_data = is_32_bit_index || input_tensor_cb_data_format == tt::DataFormat::Float32;

    const uint32_t log2Wt = std::log2(Wt);

    CoreCoord coordinator_core = {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1};
    CoreRangeSet core_range;
    if (all_core_utilization_loop_count > 0) {
        core_range = CoreRangeSet(
            CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 2}));
        core_range = core_range.merge<CoreRangeSet>(CoreRangeSet(CoreRange(
            {0, compute_with_storage_grid_size.y - 1},
            {compute_with_storage_grid_size.x - 2, compute_with_storage_grid_size.y - 1})));
    } else {
        const uint32_t core_grid_calculated_rows_number = total_work_units / compute_with_storage_grid_size.x;
        const uint32_t core_grid_calculated_columns_number = total_work_units % compute_with_storage_grid_size.x;

        if (core_grid_calculated_rows_number == 0 && core_grid_calculated_columns_number == 0) {
            core_range = CoreRangeSet(CoreCoord({0, 0}));
        } else if (core_grid_calculated_rows_number == 0) {
            core_range = CoreRangeSet(CoreRange({0, 0}, {core_grid_calculated_columns_number - 1, 0}));
        } else {
            core_range = CoreRangeSet(
                CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, core_grid_calculated_rows_number - 1}));
            if (core_grid_calculated_columns_number != 0) {
                const CoreRange additional_range(
                    {0, core_grid_calculated_rows_number},
                    {core_grid_calculated_columns_number - 1, core_grid_calculated_rows_number});
                core_range = core_range.merge(CoreRangeSet(additional_range));
            }
        }
    }
    CoreRangeSet all_core_set({CoreRange(coordinator_core)});
    all_core_set = all_core_set.merge<CoreRangeSet>(core_range);

    constexpr uint32_t buffer_scale_factor = 2;

    ProgramDescriptor desc;

    // -----------------------------------------------------------------------
    // Circular buffers (c_0–c_5 on all cores, c_6–c_9 RM-only)
    // -----------------------------------------------------------------------
    constexpr uint32_t input_tensor_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = buffer_scale_factor * input_tensor_tile_size,
        .core_ranges = all_core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_tensor_cb_index),
            .data_format = input_tensor_cb_data_format,
            .page_size = input_tensor_tile_size,
        }}},
    });

    constexpr uint32_t index_tensor_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = buffer_scale_factor * index_tensor_tile_size,
        .core_ranges = all_core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_tensor_cb_index),
            .data_format = index_tensor_cb_data_format,
            .page_size = index_tensor_tile_size,
        }}},
    });

    constexpr uint32_t input_tensor_transposed_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = buffer_scale_factor * input_tensor_tile_size,
        .core_ranges = all_core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_tensor_transposed_cb_index),
            .data_format = input_tensor_cb_data_format,
            .page_size = input_tensor_tile_size,
        }}},
    });

    constexpr uint32_t index_tensor_transposed_cb_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = buffer_scale_factor * index_tensor_tile_size,
        .core_ranges = all_core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_tensor_transposed_cb_index),
            .data_format = index_tensor_cb_data_format,
            .page_size = index_tensor_tile_size,
        }}},
    });

    constexpr uint32_t input_tensor_output_cb_index = tt::CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = buffer_scale_factor * value_tensor_tile_size,
        .core_ranges = all_core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_tensor_output_cb_index),
            .data_format = value_tensor_cb_data_format,
            .page_size = value_tensor_tile_size,
        }}},
    });

    constexpr uint32_t index_tensor_output_cb_index = tt::CBIndex::c_5;
    desc.cbs.push_back(CBDescriptor{
        .total_size = buffer_scale_factor * index_tensor_tile_size,
        .core_ranges = all_core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_tensor_output_cb_index),
            .data_format = index_tensor_cb_data_format,
            .page_size = index_tensor_tile_size,
        }}},
    });

    const uint32_t index_element_size = tt::datum_size(index_tensor_cb_data_format);
    const uint32_t W_index_bytes = tile_width * index_element_size;

    constexpr uint32_t rm_coord_value_row_cb_index = tt::CBIndex::c_6;
    constexpr uint32_t rm_coord_index_row_cb_index = tt::CBIndex::c_7;
    constexpr uint32_t rm_worker_input_value_cb_index = tt::CBIndex::c_6;
    constexpr uint32_t rm_worker_input_index_cb_index = tt::CBIndex::c_7;
    constexpr uint32_t rm_worker_output_value_cb_index = tt::CBIndex::c_8;
    constexpr uint32_t rm_worker_output_index_cb_index = tt::CBIndex::c_9;
    if (is_row_major) {
        const CoreRangeSet coordinator_core_set{CoreRange(coordinator_core)};
        constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;

        // Coordinator bounce buffers (c_6, c_7) — single-page width
        desc.cbs.push_back(CBDescriptor{
            .total_size = W_tile_bytes,
            .core_ranges = coordinator_core_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rm_coord_value_row_cb_index),
                .data_format = input_tensor_cb_data_format,
                .page_size = W_tile_bytes,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = W_index_bytes,
            .core_ranges = coordinator_core_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rm_coord_index_row_cb_index),
                .data_format = index_tensor_cb_data_format,
                .page_size = W_index_bytes,
            }}},
        });

        // Worker RM CBs (c_6, c_7, c_8, c_9) — 2 tile-block rows each
        desc.cbs.push_back(CBDescriptor{
            .total_size = 2 * TILE_H * W_tile_bytes,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rm_worker_input_value_cb_index),
                .data_format = input_tensor_cb_data_format,
                .page_size = W_tile_bytes,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = 2 * TILE_H * W_index_bytes,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rm_worker_input_index_cb_index),
                .data_format = index_tensor_cb_data_format,
                .page_size = W_index_bytes,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = 2 * TILE_H * W_tile_bytes,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rm_worker_output_value_cb_index),
                .data_format = input_tensor_cb_data_format,
                .page_size = W_tile_bytes,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = 2 * TILE_H * W_index_bytes,
            .core_ranges = core_range,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rm_worker_output_index_cb_index),
                .data_format = index_tensor_cb_data_format,
                .page_size = W_index_bytes,
            }}},
        });
    }

    // -----------------------------------------------------------------------
    // Semaphores — IDs 0 and 1 on all cores
    // -----------------------------------------------------------------------
    constexpr uint32_t coordinator_to_cores_semaphore_id = 0;
    constexpr uint32_t cores_to_coordinator_semaphore_id = 1;
    desc.semaphores.push_back(SemaphoreDescriptor{.id = 0, .core_ranges = all_core_set, .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{.id = 1, .core_ranges = all_core_set, .initial_value = 0});

    // -----------------------------------------------------------------------
    // Kernels
    // -----------------------------------------------------------------------
    const auto coordinator_core_physical_coord = device->worker_core_from_logical_core(coordinator_core);
    const auto start_core_logical = core_range.ranges()[0].start_coord;
    const auto start_core_physical_coord = device->worker_core_from_logical_core(start_core_logical);
    const auto end_core_physical_coord = device->worker_core_from_logical_core(coordinator_core);

    std::vector<uint32_t> coordinator_compile_time_args = {
        total_work_units,
        Wt,
        Ht,
        total_number_of_cores,
        number_of_available_cores,
        input_tensor_cb_index,
        index_tensor_cb_index,
        static_cast<uint32_t>(is_32_bit_data)};
    TensorAccessorArgs(*input_buffer).append_to(coordinator_compile_time_args);
    TensorAccessorArgs(*value_buffer).append_to(coordinator_compile_time_args);
    TensorAccessorArgs(*index_buffer).append_to(coordinator_compile_time_args);
    coordinator_compile_time_args.push_back(static_cast<uint32_t>(is_row_major));
    coordinator_compile_time_args.push_back(static_cast<uint32_t>(rm_coord_value_row_cb_index));
    coordinator_compile_time_args.push_back(static_cast<uint32_t>(rm_coord_index_row_cb_index));
    coordinator_compile_time_args.push_back(W_tile_bytes);
    coordinator_compile_time_args.push_back(W_index_bytes);
    coordinator_compile_time_args.push_back(tile_width);

    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_index,
        index_tensor_cb_index,
        Wt,
        Ht,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        number_of_available_cores};
    TensorAccessorArgs(*value_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*index_buffer).append_to(reader_compile_time_args);
    reader_compile_time_args.push_back(static_cast<uint32_t>(is_row_major));
    reader_compile_time_args.push_back(static_cast<uint32_t>(rm_worker_input_value_cb_index));
    reader_compile_time_args.push_back(static_cast<uint32_t>(rm_worker_input_index_cb_index));
    reader_compile_time_args.push_back(W_tile_bytes);
    reader_compile_time_args.push_back(W_index_bytes);

    std::vector<uint32_t> writer_compile_time_args = {
        input_tensor_output_cb_index,
        index_tensor_output_cb_index,
        Wt,
        Ht,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        number_of_available_cores};
    TensorAccessorArgs(*value_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(*index_buffer).append_to(writer_compile_time_args);
    writer_compile_time_args.push_back(static_cast<uint32_t>(is_row_major));
    writer_compile_time_args.push_back(static_cast<uint32_t>(rm_worker_output_value_cb_index));
    writer_compile_time_args.push_back(static_cast<uint32_t>(rm_worker_output_index_cb_index));
    writer_compile_time_args.push_back(W_tile_bytes);
    writer_compile_time_args.push_back(W_index_bytes);

    std::vector<uint32_t> compute_compile_time_args = {
        input_tensor_cb_index,
        index_tensor_cb_index,
        input_tensor_transposed_cb_index,
        index_tensor_transposed_cb_index,
        input_tensor_output_cb_index,
        index_tensor_output_cb_index,
        Wt,
        Ht,
        number_of_available_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y,
        static_cast<uint32_t>(attributes.descending),
        static_cast<uint32_t>(attributes.stable),
        log2Wt};
    compute_compile_time_args.push_back(static_cast<uint32_t>(is_row_major));
    compute_compile_time_args.push_back(static_cast<uint32_t>(rm_worker_input_value_cb_index));
    compute_compile_time_args.push_back(static_cast<uint32_t>(rm_worker_input_index_cb_index));
    compute_compile_time_args.push_back(static_cast<uint32_t>(rm_worker_output_value_cb_index));
    compute_compile_time_args.push_back(static_cast<uint32_t>(rm_worker_output_index_cb_index));

    std::vector<UnpackToDestMode> unpack_to_dest_mode_vector(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (input_tensor_cb_data_format == tt::DataFormat::Float32) {
        unpack_to_dest_mode_vector[input_tensor_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode_vector[input_tensor_transposed_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode_vector[input_tensor_output_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode_vector[rm_worker_input_value_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    // Coordinator kernel (single core)
    KernelDescriptor coordinator_desc;
    coordinator_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/coordinator_single_row_multi_core.cpp";
    coordinator_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    coordinator_desc.core_ranges = CoreRangeSet{CoreRange(coordinator_core)};
    coordinator_desc.compile_time_args = std::move(coordinator_compile_time_args);
    coordinator_desc.config = ReaderConfigDescriptor{};
    coordinator_desc.runtime_args.emplace_back(
        coordinator_core,
        KernelDescriptor::CoreRuntimeArgs{
            start_core_physical_coord.x,
            start_core_physical_coord.y,
            end_core_physical_coord.x,
            end_core_physical_coord.y,
            coordinator_to_cores_semaphore_id,
            cores_to_coordinator_semaphore_id,
            core_range.num_cores(),
            input_buffer->address(),
            value_buffer->address(),
            index_buffer->address()});

    // Reader kernel (worker cores)
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/reader_single_row_multi_core.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_range;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer kernel (worker cores)
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/writer_single_row_multi_core.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_range;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Compute kernel (worker cores)
    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/compute/sort_single_row_multi_core.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_range;
    compute_desc.compile_time_args = std::move(compute_compile_time_args);
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = is_32_bit_data,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode_vector),
    };

    // Per-core runtime args for reader, writer (same args for all worker cores)
    for (const auto& range : core_range.ranges()) {
        for (const auto& core : range) {
            reader_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    value_buffer->address(),
                    index_buffer->address(),
                    coordinator_core_physical_coord.x,
                    coordinator_core_physical_coord.y,
                    coordinator_to_cores_semaphore_id,
                    cores_to_coordinator_semaphore_id});
            writer_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    value_buffer->address(),
                    index_buffer->address(),
                    coordinator_core_physical_coord.x,
                    coordinator_core_physical_coord.y,
                    coordinator_to_cores_semaphore_id,
                    cores_to_coordinator_semaphore_id});
        }
    }

    desc.kernels.push_back(std::move(coordinator_desc));
    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
