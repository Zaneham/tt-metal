// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_consumer.hpp"

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_cache.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace ttnn::operations::experimental::test {

namespace {

constexpr uint32_t kRemoteCBId = 31;

// Empty shared-variables struct: this op has no runtime args to override on cache hit.
struct ConsumerSharedVars {};
using ConsumerCachedWorkload = tt::tt_metal::program_cache::detail::CachedMeshWorkload<ConsumerSharedVars>;

}  // namespace

void test_dram_prefetcher_consumer(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint32_t num_iters,
    uint32_t page_size_bytes,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb) {
    using namespace tt::tt_metal;

    TT_FATAL(mesh_device != nullptr, "mesh_device required");
    TT_FATAL(num_iters > 0, "num_iters must be > 0");
    TT_FATAL(page_size_bytes > 0, "page_size_bytes must be > 0");

    const CoreRangeSet receiver_cores = global_cb.receiver_cores();
    TT_FATAL(receiver_cores.num_cores() > 0, "GCB has no receiver cores");

    // Hash key: same inputs that determine the Program shape (num_iters + page_size_bytes)
    // plus the GCB identity (config_address is unique per GCB instance on this device).
    const uint64_t program_hash = ttsl::hash::hash_objects_with_default_seed(
        num_iters, page_size_bytes, static_cast<uint64_t>(global_cb.config_address()));

    auto& program_cache = mesh_device->get_program_cache();
    const bool cache_enabled = program_cache.is_enabled();
    if (cache_enabled && program_cache.contains(program_hash)) {
        auto& cached = program_cache.get(program_hash);
        auto& cached_workload = cached.cached_program.template get<ConsumerCachedWorkload>();
        tt::tt_metal::distributed::EnqueueMeshWorkload(
            mesh_device->mesh_command_queue(), cached_workload.workload, /*blocking=*/false);
        return;
    }

    Program program = CreateProgram();

    // Configure the receiver-side CB. set_page_size matches what the sender resizes the CB to
    // (in_block_w_tiles * n_tiles_per_recv * tile_bytes); receiver wait_front/pop_front operate
    // in units of this page size.
    CircularBufferConfig cb_config(page_size_bytes);
    cb_config.remote_index(kRemoteCBId).set_page_size(page_size_bytes).set_data_format(tt::DataFormat::Float16_b);
    tt::tt_metal::experimental::CreateCircularBuffer(program, receiver_cores, cb_config, global_cb);

    const std::vector<uint32_t> compile_args = {kRemoteCBId, num_iters};
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/gcb_bench_discard_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = compile_args});

    tt::tt_metal::distributed::MeshCoordinateRange device_range(tt::tt_metal::distributed::MeshCoordinate(0, 0));
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(device_range, std::move(program));

    if (cache_enabled) {
        ConsumerCachedWorkload cached_workload{std::move(workload), ConsumerSharedVars{}};
        program_cache.insert(
            program_hash,
            tt::tt_metal::program_cache::detail::CachedProgramFactory{std::move(cached_workload), 0});
        auto& stored = program_cache.get(program_hash);
        auto& cached_ref = stored.cached_program.template get<ConsumerCachedWorkload>();
        tt::tt_metal::distributed::EnqueueMeshWorkload(
            mesh_device->mesh_command_queue(), cached_ref.workload, /*blocking=*/false);
    } else {
        tt::tt_metal::distributed::EnqueueMeshWorkload(
            mesh_device->mesh_command_queue(), workload, /*blocking=*/false);
    }
}

}  // namespace ttnn::operations::experimental::test
