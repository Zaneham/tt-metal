// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_validator.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace ttnn {

namespace {

constexpr uint32_t kValidatorRemoteCBId = 31;

}  // namespace

void dram_prefetcher_validator(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint32_t num_iters,
    uint32_t page_size_bytes,
    uint32_t print_stride,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb) {
    using namespace tt::tt_metal;

    TT_FATAL(mesh_device != nullptr, "mesh_device required");
    TT_FATAL(num_iters > 0, "num_iters must be > 0");
    TT_FATAL(page_size_bytes > 0, "page_size_bytes must be > 0");

    const CoreRangeSet receiver_cores = global_cb.receiver_cores();
    TT_FATAL(receiver_cores.num_cores() > 0, "GCB has no receiver cores");

    Program program = CreateProgram();

    // Receiver-side CB sized to the page the sender pushes. wait_front/pop_front operate in
    // units of this page size. Matches the matmul receiver's in1_block_size_bytes when used
    // as a validator stand-in for the matmul consumer.
    CircularBufferConfig cb_config(page_size_bytes);
    cb_config.remote_index(kValidatorRemoteCBId)
        .set_page_size(page_size_bytes)
        .set_data_format(tt::DataFormat::Float16_b);
    experimental::CreateCircularBuffer(program, receiver_cores, cb_config, global_cb);

    const std::vector<uint32_t> compile_args = {kValidatorRemoteCBId, num_iters, print_stride};
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/gcb_validator_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = compile_args});

    tt::tt_metal::distributed::MeshCoordinateRange device_range(tt::tt_metal::distributed::MeshCoordinate(0, 0));
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(device_range, std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/false);
}

}  // namespace ttnn
