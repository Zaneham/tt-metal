// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include "impl/data_format/bfloat16_utils.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"

namespace tt::tt_metal {

enum class DFBPorCType : uint8_t { DM, TENSIX };

class DFBImplicitSyncParamFixture : public MeshDeviceFixture, public ::testing::WithParamInterface<bool> {};

static std::string ImplicitSyncParamName(const ::testing::TestParamInfo<bool>& info) {
    return info.param ? "ImplicitSyncTrue" : "ImplicitSyncFalse";
}

// expected_output, when set, is compared against the device output instead of input.
// This is used for Tensix→DM ring-pressure tests where the device cycles through fewer
// unique ring slots than entries_per_core, so output != input by design.
void execute_program_and_verify(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    Program& program,
    const std::shared_ptr<distributed::MeshBuffer>& in_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& out_buffer,
    distributed::MeshCoordinate& zero_coord,
    std::vector<uint32_t>& input,
    bool verify_output = true,
    std::optional<std::vector<uint32_t>> expected_output = std::nullopt) {
    distributed::WriteShard(mesh_device->mesh_command_queue(), in_buffer, input, zero_coord, true);

    if (mesh_device->get_devices()[0]->arch() == ARCH::QUASAR) {
        // TODO #38042: Need to wait for data to be written, the barrier needs to be uplifted for Quasar
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        std::vector<uint32_t> rdback_dram;
        distributed::ReadShard(mesh_device->mesh_command_queue(), rdback_dram, in_buffer, zero_coord, true);

        tt_driver_atomics::mfence();

        EXPECT_EQ(rdback_dram, input);
    }

    // Execute using slow dispatch (DFBs not yet supported in MeshWorkload path)
    IDevice* device = mesh_device->get_devices()[0];
    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    std::vector<uint32_t> output;
    distributed::ReadShard(mesh_device->mesh_command_queue(), output, out_buffer, zero_coord, true);

    if (verify_output) {
        const std::vector<uint32_t>& expected = expected_output ? *expected_output : input;
        if (expected != output) {
            log_info(tt::LogTest, "Printing expected");
            for (auto i : expected) {
                std::cout << i << " ";
            }
            std::cout << std::endl;
            log_info(tt::LogTest, "Printing output");
            for (auto i : output) {
                std::cout << i << " ";
            }
        }
        EXPECT_EQ(expected, output);
    }
}

// Runs a single DFB program on one or more cores and verifies output == input.
//
// When core_range_set contains N > 1 cores the global DRAM buffers are sized
// N x entries_per_core x entry_size and each core receives a unique
// chunk_offset (= core_idx * entries_per_core) so it accesses a disjoint
// slice of the buffer.  Multi-core use requires DM producer and consumer.
void run_single_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::dfb::DataflowBufferConfig& dfb_config,
    DFBPorCType producer_type,
    DFBPorCType consumer_type,
    const CoreRangeSet& core_range_set = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))),
    std::optional<uint32_t> num_entries_in_buffer = std::nullopt,
    // Quasar only: when set, bypasses the auto-allocator and places the DM consumer kernel on
    // exactly these DM processors. Used by B1b to force DM0 to be idle while a subordinate DM
    // runs the consumer — the original PR's fix scenario, which the auto-allocator can't produce
    // (it always fills from DM0 up).
    std::optional<std::set<DataMovementProcessor>> consumer_dm_processors_override = std::nullopt) {
    TT_FATAL(
        !(producer_type == DFBPorCType::TENSIX && consumer_type == DFBPorCType::TENSIX),
        "Both producer and consumer cannot be Tensix. At least one must be a DM kernel for NOC transfers.");
    TT_FATAL(
        core_range_set.num_cores() == 1 ||
            (producer_type == DFBPorCType::DM && consumer_type == DFBPorCType::DM),
        "Multi-core DFB programs only support DM producer and consumer.");

    const auto arch = mesh_device->get_devices()[0]->arch();
    const bool is_quasar = (arch == ARCH::QUASAR);

    if (!is_quasar) {
        // WH/BH DM: one BRISC (RISCV_0) as producer and one NCRISC (RISCV_1) as consumer.
        // Configs with num_producers > 1 or num_consumers > 1 require multi-threaded DM
        // which is not available on WH/BH.
        if (dfb_config.num_producers > 1 || dfb_config.num_consumers > 1) {
            GTEST_SKIP() << "WH/BH DFB supports only 1 DM producer (BRISC) and 1 DM consumer (NCRISC)";
        }
        // read_in / write_out are Quasar-only; the device-side kernel would fail to compile
        // if enable_implicit_sync=true is propagated as a compile-time arg.
        dfb_config.enable_implicit_sync = false;
    }

    Program program = CreateProgram();
    auto zero_coord = distributed::MeshCoordinate(0, 0);

    const uint32_t num_cores = core_range_set.num_cores();
    const uint32_t entries_per_core = num_entries_in_buffer.has_value() ? num_entries_in_buffer.value() : dfb_config.num_entries;
    const uint32_t entry_size = dfb_config.entry_size;
    // page_size = entry_size makes every entry independently addressable by page_id.
    const uint32_t total_buffer_size = num_cores * entries_per_core * entry_size;
    distributed::DeviceLocalBufferConfig local_buffer_config{.page_size = entry_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = total_buffer_size};
    auto in_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    auto out_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());

    log_info(tt::LogTest, "In Buffer:  [address: {} B, size: {} B]", in_buffer->address(), in_buffer->size());
    log_info(tt::LogTest, "Out Buffer: [address: {} B, size: {} B]", out_buffer->address(), out_buffer->size());

    // Ceiling division so every producer gets a loop bound that covers the largest slice.
    // Producers whose page_id would exceed entries_per_core use the runtime bounds
    // check in the kernel to skip the out-of-range iteration.
    uint32_t num_entries_per_producer = (entries_per_core + dfb_config.num_producers - 1) / dfb_config.num_producers;
    const bool is_all = (dfb_config.cap == dfb::AccessPattern::ALL);
    std::vector<uint32_t> producer_cta = {
        (uint32_t)in_buffer->address(), num_entries_per_producer, (uint32_t)dfb_config.enable_implicit_sync};
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(producer_cta);

    KernelHandle producer_kernel;
    if (producer_type == DFBPorCType::DM) {
        const std::string dm_producer_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp";
        if (is_quasar) {
            producer_kernel = experimental::quasar::CreateKernel(
                program,
                dm_producer_kernel_path,
                core_range_set,
                experimental::quasar::QuasarDataMovementConfig{
                    .num_threads_per_cluster = dfb_config.num_producers, .compile_args = producer_cta});
        } else {
            producer_kernel = CreateKernel(
                program,
                dm_producer_kernel_path,
                core_range_set,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .compile_args = producer_cta});
        }
    } else {
        const std::string t6_producer_kernel_path = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_producer.cpp";
        if (is_quasar) {
            producer_kernel = CreateKernel(
                program,
                t6_producer_kernel_path,
                core_range_set,
                experimental::quasar::QuasarComputeConfig{
                    .num_threads_per_cluster = dfb_config.num_producers, .compile_args = producer_cta});
        } else {
            producer_kernel = CreateKernel(
                program, t6_producer_kernel_path, core_range_set, ComputeConfig{.compile_args = producer_cta});
        }
    }

    uint32_t num_entries_per_consumer =
        is_all ? entries_per_core : (entries_per_core + dfb_config.num_consumers - 1) / dfb_config.num_consumers;
    std::vector<uint32_t> consumer_cta = {
        (uint32_t)out_buffer->address(),
        num_entries_per_consumer,
        (uint32_t)is_all,
        (uint32_t)dfb_config.enable_implicit_sync};
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(consumer_cta);

    KernelHandle consumer_kernel;
    if (consumer_type == DFBPorCType::DM) {
        const std::string dm_consumer_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp";
        if (is_quasar) {
            if (consumer_dm_processors_override.has_value()) {
                TT_FATAL(
                    consumer_dm_processors_override->size() == dfb_config.num_consumers,
                    "consumer_dm_processors_override size must match num_consumers");
                experimental::quasar::QuasarDataMovementConfig dm_cfg{
                    .num_threads_per_cluster = dfb_config.num_consumers, .compile_args = consumer_cta};
                auto kernel = std::make_shared<experimental::quasar::QuasarDataMovementKernel>(
                    KernelSource(dm_consumer_kernel_path, KernelSource::FILE_PATH),
                    core_range_set,
                    dm_cfg,
                    *consumer_dm_processors_override);
                consumer_kernel = program.impl().add_kernel(kernel, HalProgrammableCoreType::TENSIX);
            } else {
                consumer_kernel = experimental::quasar::CreateKernel(
                    program,
                    dm_consumer_kernel_path,
                    core_range_set,
                    experimental::quasar::QuasarDataMovementConfig{
                        .num_threads_per_cluster = dfb_config.num_consumers, .compile_args = consumer_cta});
            }
        } else {
            consumer_kernel = CreateKernel(
                program,
                dm_consumer_kernel_path,
                core_range_set,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .compile_args = consumer_cta});
        }
    } else {
        const std::string t6_consumer_kernel_path = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_consumer.cpp";
        if (is_quasar) {
            consumer_kernel = CreateKernel(
                program,
                t6_consumer_kernel_path,
                core_range_set,
                experimental::quasar::QuasarComputeConfig{
                    .num_threads_per_cluster = dfb_config.num_consumers, .compile_args = consumer_cta});
        } else {
            consumer_kernel = CreateKernel(
                program, t6_consumer_kernel_path, core_range_set, ComputeConfig{.compile_args = consumer_cta});
        }
    }

    auto logical_dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_range_set, dfb_config);
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, logical_dfb_id, producer_kernel, consumer_kernel);

    auto dfb = program.impl().get_dataflow_buffer(logical_dfb_id);
    const uint32_t producer_mask = dfb->config.producer_risc_mask;
    const uint32_t consumer_mask = dfb->config.consumer_risc_mask;

    // Build a per-core chunk-offset map (used for both runtime args and L1 pre-fill/verify).
    std::map<CoreCoord, uint32_t> core_to_chunk_offset;
    uint32_t core_idx = 0;
    for (const CoreRange& cr : core_range_set.ranges()) {
        for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
            for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
                core_to_chunk_offset[CoreCoord(x, y)] = core_idx++ * entries_per_core;
            }
        }
    }

    for (const CoreRange& cr : core_range_set.ranges()) {
        for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
            for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
                const CoreCoord core(x, y);
                const uint32_t chunk_offset = core_to_chunk_offset.at(core);
                SetRuntimeArgs(program, producer_kernel, core, {producer_mask, chunk_offset, entries_per_core});
                SetRuntimeArgs(
                    program,
                    consumer_kernel,
                    core,
                    {consumer_mask, (uint32_t)logical_dfb_id, chunk_offset, entries_per_core});
            }
        }
    }

    // Generate input once; shared by in_buffer write, L1 pre-fill, and verification.
    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, total_buffer_size / sizeof(uint32_t));

    IDevice* device = mesh_device->get_devices()[0];
    // const uint32_t words_per_core = entries_per_core * entry_size / sizeof(uint32_t);

    // For Tensix → DM: pre-fill each core's DFB L1 with its input chunk so the
    // Tensix producer kernel can read from L1 while DM consumer drains to DRAM.
    //
    // l1_by_core addresses are not populated until allocate_dataflow_buffers() runs
    // during program compilation. Since this is a single-DFB test it is always placed at the L1 base allocator address.
    //
    // IMPORTANT: the slice written to L1 must be exactly the physical ring size
    // (dfb->total_size() bytes = num_entries * entry_size). Writing more than the ring
    // size would corrupt L1 beyond the ring (kernel stack, config structures, etc.).
    // For ring-pressure tests (entries_per_core > num_entries) only the first
    // num_entries slots are filled; the producer kernel cycles through those same
    // slots repeatedly, which is the expected behaviour.
    if (producer_type == DFBPorCType::TENSIX) {
        const uint32_t dfb_l1_addr =
            static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
        const uint32_t ring_words = dfb->total_size() / sizeof(uint32_t);
        for (const CoreRange& cr : core_range_set.ranges()) {
            for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
                for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
                    const CoreCoord core(x, y);
                    const uint32_t co = core_to_chunk_offset.at(core);
                    const uint32_t wpe = entry_size / sizeof(uint32_t);
                    std::vector<uint32_t> slice(ring_words, 0);
                    for (uint32_t p = 0; p < dfb_config.num_producers; p++) {
                        for (uint32_t e = 0; e < num_entries_per_producer; e++) {
                            const uint32_t page_id = co + e * dfb_config.num_producers + p;
                            if (page_id >= co + entries_per_core) {
                                break;
                            }
                            // Ring layout depends on stride_in_entries, which is set by the
                            // consumer access pattern:
                            //   STRIDED: stride = num_producers → interleaved (slot = e*P + p)
                            //   ALL: stride = 1 → TC-first   (slot = p*E + e)
                            const uint32_t dst_slot = (dfb_config.cap == dfb::AccessPattern::ALL)
                                                          ? (p * num_entries_per_producer + e)
                                                          : (e * dfb_config.num_producers + p);

                            // Stop once all physical ring slots are filled; for ring-pressure
                            // tests the remaining iterations would alias back to already-filled
                            // slots, so there is nothing new to write.
                            if (dst_slot >= dfb_config.num_entries) {
                                break;
                            }

                            std::copy(
                                input.begin() + page_id * wpe,
                                input.begin() + page_id * wpe + wpe,
                                slice.begin() + dst_slot * wpe);
                        }
                    }
                    detail::WriteToDeviceL1(device, core, dfb_l1_addr, slice);
                }
            }
        }
    }

    // For Tensix → DM ring-pressure tests (entries_per_core > num_entries), the
    // Tensix producer cycles through the same num_entries ring slots indefinitely.
    // Each STRIDED consumer c always reads ring slot (c % num_entries), which was
    // pre-filled with input page c.  The expected out_buffer page p therefore
    // contains the data from ring slot (p % num_consumers) % num_entries, not
    // input[p].  Build the corrected expected vector so the verification is sound.
    std::optional<std::vector<uint32_t>> tensix_dm_expected;
    if (producer_type == DFBPorCType::TENSIX && consumer_type == DFBPorCType::DM &&
        entries_per_core > dfb_config.num_entries && dfb_config.cap == dfb::AccessPattern::STRIDED) {
        const uint32_t wpe = entry_size / sizeof(uint32_t);
        tensix_dm_expected.emplace(num_cores * entries_per_core * wpe, 0u);
        for (const CoreRange& cr : core_range_set.ranges()) {
            for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
                for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
                    const CoreCoord core(x, y);
                    const uint32_t co = core_to_chunk_offset.at(core);
                    for (uint32_t p = 0; p < entries_per_core; p++) {
                        // Consumer c = p % num_consumers always reads the ring slot it
                        // was assigned (slot = c % num_entries), which holds input[co + c].
                        const uint32_t ring_slot = (p % dfb_config.num_consumers) % dfb_config.num_entries;
                        std::copy(
                            input.begin() + (co + ring_slot) * wpe,
                            input.begin() + (co + ring_slot + 1) * wpe,
                            tensix_dm_expected->begin() + (co + p) * wpe);
                    }
                }
            }
        }
    }

    // Launch program; verify out_buffer only for DM → DM paths (Tensix consumer
    // does not write to DRAM, so out_buffer verification is skipped there).
    execute_program_and_verify(
        mesh_device,
        program,
        in_buffer,
        out_buffer,
        zero_coord,
        input,
        /*verify_output=*/(consumer_type == DFBPorCType::DM),
        tensix_dm_expected);

    // For DM → Tensix: verify each core's DFB L1 against the expected input chunk.
    if (consumer_type == DFBPorCType::TENSIX) {
        for (const auto& group : dfb->groups) {
            for (const auto& [core, alloc_addr] : group.l1_by_core) {
                const uint32_t co = core_to_chunk_offset.at(core);
                std::vector<uint32_t> l1_data;
                detail::ReadFromDeviceL1(device, core, alloc_addr, dfb->total_size(), l1_data);
                const uint32_t wpe_v = entry_size / sizeof(uint32_t);
                // Physical ring holds dfb_config.num_entries entries; for ring-pressure
                // tests (entries_per_core > dfb_config.num_entries) the ring wraps and
                // only the last ring_capacity writes per producer survive in L1.
                // Size expected to l1_data so the comparison is against what is actually there.
                const uint32_t total_ring_words = dfb->total_size() / sizeof(uint32_t);
                std::vector<uint32_t> expected(total_ring_words, 0);
                if (dfb_config.cap == dfb::AccessPattern::ALL) {
                    // ALL consumer: ring is TC-first (stride_in_entries=1).
                    // Each producer p has ring_capacity consecutive ring slots.
                    // After wrapping, only the last ring_capacity entries from each
                    // producer survive: e in [num_entries_per_producer - ring_capacity, ...).
                    const uint32_t ring_capacity = dfb_config.num_entries / dfb_config.num_producers;
                    const uint32_t last_e_base = num_entries_per_producer - ring_capacity;
                    for (uint32_t p = 0; p < dfb_config.num_producers; p++) {
                        for (uint32_t c = 0; c < ring_capacity; c++) {
                            const uint32_t ring_slot = p * ring_capacity + c;
                            const uint32_t e = last_e_base + c;
                            const uint32_t page_id = co + e * dfb_config.num_producers + p;
                            if (page_id >= co + entries_per_core) {
                                break;
                            }
                            std::copy(
                                input.begin() + page_id * wpe_v,
                                input.begin() + page_id * wpe_v + wpe_v,
                                expected.begin() + ring_slot * wpe_v);
                        }
                    }
                } else {
                    // STRIDED consumer: ring is interleaved, matching sequential input order.
                    // For ring-pressure tests (entries_per_core > dfb_config.num_entries) only
                    // the last dfb_config.num_entries entries survive in L1; copy that suffix.
                    const uint32_t ring_start_page = co + entries_per_core - dfb_config.num_entries;
                    std::copy(
                        input.begin() + ring_start_page * wpe_v,
                        input.begin() + ring_start_page * wpe_v + total_ring_words,
                        expected.begin());
                }
                if (expected != l1_data) {
                    std::cout << "expected: ";
                    for (const auto& e : expected) {
                        std::cout << e << " ";
                    }
                    std::cout << std::endl;
                    std::cout << "l1_data: ";
                    for (const auto& l : l1_data) {
                        std::cout << l << " ";
                    }
                    std::cout << std::endl;
                }
                EXPECT_EQ(expected, l1_data) << "DFB L1 mismatch on core (" << core.x << "," << core.y << ")";
            }
        }
    }
}

// =====================================================================================
// Gap 7 harness – concurrent DFBs on the same core (TC allocator stress)
//
// Runs `num_dfbs` independent 1Sx1S DM→DM DFBs simultaneously on core (0,0).
//
// Thread assignment (Quasar has 8 DM threads total):
//   Producer threads : DM[0 .. num_dfbs-1]  (combined_producer_mask = low  num_dfbs bits)
//   Consumer threads : DM[num_dfbs .. 2*num_dfbs-1] (combined_consumer_mask = next num_dfbs bits)
//
// All num_dfbs DFBs are created in a single Program so their TCs are allocated
// simultaneously, stressing the TC allocator.  Each DFB is bound to the same
// multi-producer and multi-consumer kernel; each DM thread derives its DFB ID from
// its position in the combined mask (via mhartid) and owns a contiguous slice of
// the shared DRAM buffers.
//
// num_dfbs must satisfy: 2 * num_dfbs <= 8 (Quasar DM thread limit).
// dfb_config must use num_producers=1, num_consumers=1 (1Sx1S).
// =====================================================================================
void run_concurrent_dfbs_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t num_dfbs,
    experimental::dfb::DataflowBufferConfig& dfb_config) {
    TT_FATAL(
        mesh_device->get_devices()[0]->arch() == ARCH::QUASAR,
        "Concurrent DFB tests require Quasar (multi-threaded DM)");
    TT_FATAL(
        dfb_config.num_producers == 1 && dfb_config.num_consumers == 1,
        "run_concurrent_dfbs_program requires 1Sx1S per DFB");
    TT_FATAL(2 * num_dfbs <= 6, "2*num_dfbs must fit in the 6 available Quasar DM threads (DM0 reserved for dispatch)");

    const CoreCoord core(0, 0);
    const CoreRangeSet core_range_set(CoreRange(core, core));

    const uint32_t entry_size      = dfb_config.entry_size;
    const uint32_t entries_per_dfb = dfb_config.num_entries;
    const uint32_t total_buf_size  = num_dfbs * entries_per_dfb * entry_size;

    distributed::DeviceLocalBufferConfig local_cfg{.page_size = entry_size, .buffer_type = BufferType::DRAM};
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto in_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = total_buf_size}, local_cfg, mesh_device.get());
    auto out_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = total_buf_size}, local_cfg, mesh_device.get());

    log_info(
        tt::LogTest,
        "Gap7: {} concurrent 1Sx1S DFBs, {} entries/DFB, total {}B in/out buffers",
        num_dfbs,
        entries_per_dfb,
        total_buf_size);

    Program program = CreateProgram();

    // Create one producer kernel instance and one consumer kernel instance per DFB,
    // each running on a single DM thread.  Encoding dfb_id and chunk_offset in the CTA
    // makes each instance unique so the runtime allocates a distinct DM thread to it.
    // This ensures BindDataflowBufferToProducerConsumerKernels registers exactly 1 DM
    // thread as producer and 1 as consumer for each DFB – matching num_producers=1 /
    // num_consumers=1 in the DFB config.  A shared N-thread kernel bound to N DFBs
    // would erroneously register all N threads as producers for every DFB, causing the
    // consumer to wait for N acks that never arrive (hang).
    std::vector<KernelHandle> producer_kernels, consumer_kernels;
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        const uint32_t chunk_offset = i * entries_per_dfb;

        std::vector<uint32_t> prod_cta = {
            entries_per_dfb,
            (uint32_t)dfb_config.enable_implicit_sync,
            (uint32_t)in_buffer->address(),
            i,              // dfb_id
            chunk_offset};  // chunk_offset
        tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(prod_cta);

        producer_kernels.push_back(experimental::quasar::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_multi_producer.cpp",
            core_range_set,
            experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .compile_args = prod_cta}));

        std::vector<uint32_t> cons_cta = {
            entries_per_dfb,
            (uint32_t)dfb_config.enable_implicit_sync,
            (uint32_t)out_buffer->address(),
            i,              // dfb_id
            chunk_offset};  // chunk_offset
        tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(cons_cta);

        consumer_kernels.push_back(experimental::quasar::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_multi_consumer.cpp",
            core_range_set,
            experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .compile_args = cons_cta}));
    }

    // All DFBs created in the same Program so TCs are allocated simultaneously.
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        auto dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_range_set, dfb_config);
        experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program, dfb_id, producer_kernels[i], consumer_kernels[i]);
    }

    // No runtime args needed: dfb_id and chunk_offset are already in each kernel's CTA.

    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 100, total_buf_size / sizeof(uint32_t));
    execute_program_and_verify(mesh_device, program, in_buffer, out_buffer, zero_coord, input);
}

// =====================================================================================
// Tensix→DM concurrent DFBs
//
// Runs num_dfbs independent 1Sx1S Tensix→DM DFBs on core (0,0) with a single
// Neo thread looping through all DFBs sequentially and num_dfbs DM consumer threads
// running concurrently (each draining its own DFB the moment entries arrive).
//
// Using a sequential Tensix kernel (dfb_t6_seq_producer.cpp) avoids any dependency
// on Neo hartid values: the single Neo thread signals DFB_0 fully, waits for acks,
// then moves to DFB_1.  DM consumer threads start simultaneously but block on
// wait_front until their DFB has entries.
// =====================================================================================
void run_concurrent_tensix_dm_dfbs_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t num_dfbs,
    experimental::dfb::DataflowBufferConfig& dfb_config) {
    TT_FATAL(
        mesh_device->get_devices()[0]->arch() == ARCH::QUASAR,
        "Concurrent Tensix→DM DFB tests require Quasar");
    TT_FATAL(
        dfb_config.num_producers == 1 && dfb_config.num_consumers == 1,
        "run_concurrent_tensix_dm_dfbs_program requires 1Sx1S per DFB");
    TT_FATAL(num_dfbs <= 6, "num_dfbs must fit in the 6 available Quasar DM threads (DM0 reserved for dispatch)");

    const CoreCoord core(0, 0);
    const CoreRangeSet core_range_set(CoreRange(core, core));
    IDevice* device = mesh_device->get_devices()[0];
    auto zero_coord = distributed::MeshCoordinate(0, 0);

    const uint32_t entry_size     = dfb_config.entry_size;
    const uint32_t entries_per_dfb = dfb_config.num_entries;
    const uint32_t wpe = entry_size / sizeof(uint32_t);
    const uint32_t buf_size = entries_per_dfb * entry_size;

    // Separate DRAM out_buffer per DFB (DM consumers write here).
    std::vector<std::shared_ptr<distributed::MeshBuffer>> out_buffers;
    distributed::DeviceLocalBufferConfig local_cfg{.page_size = entry_size, .buffer_type = BufferType::DRAM};
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        distributed::ReplicatedBufferConfig buf_cfg{.size = buf_size};
        out_buffers.push_back(distributed::MeshBuffer::create(buf_cfg, local_cfg, mesh_device.get()));
    }

    // Generate input – each DFB gets its own slice (entries_per_dfb entries).
    const uint32_t total_words = num_dfbs * entries_per_dfb * wpe;
    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, total_words);

    Program program = CreateProgram();

    // Tensix sequential producer: 1 Neo thread, loops through num_dfbs DFBs.
    std::vector<uint32_t> producer_cta = {num_dfbs, entries_per_dfb};
    auto producer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_seq_producer.cpp",
        core_range_set,
        experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = 1, .compile_args = producer_cta});

    // DM concurrent consumers: one 1-thread kernel instance per DFB.
    // Encoding dfb_id and the per-DFB dst_addr in each instance's CTA makes each
    // instance unique, so the runtime allocates a separate DM thread to it and
    // BindDataflowBufferToProducerConsumerKernels registers exactly 1 consumer per
    // DFB – matching num_consumers=1 in the DFB config.
    const uint32_t num_entries_per_consumer = entries_per_dfb;  // 1Sx1S: sole consumer gets all

    std::vector<KernelHandle> consumer_kernels;
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        std::vector<uint32_t> cons_cta = {
            num_entries_per_consumer,
            (uint32_t)dfb_config.enable_implicit_sync,
            (uint32_t)out_buffers[i]->address(),
            i};  // dfb_id
        tt::tt_metal::TensorAccessorArgs(out_buffers[i]).append_to(cons_cta);

        consumer_kernels.push_back(experimental::quasar::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_multi_consumer_sep.cpp",
            core_range_set,
            experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .compile_args = cons_cta}));
    }

    // Create all DFBs in the same Program (simultaneous TC allocation).
    std::vector<uint32_t> dfb_ids;
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        auto dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_range_set, dfb_config);
        experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program, dfb_id, producer_kernel, consumer_kernels[i]);
        dfb_ids.push_back(dfb_id);
    }

    // Producer runtime args: nothing extra needed (loop bounds are in CTA).
    SetRuntimeArgs(program, producer_kernel, core, {});
    // No consumer runtime args: dfb_id and dst_addr are in each instance's CTA.

    // Pre-fill each DFB's L1 ring with its input slice.
    // DFBs are allocated consecutively starting at the base L1 allocator address.
    auto dfb0 = program.impl().get_dataflow_buffer(dfb_ids[0]);
    const uint32_t l1_base = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    const uint32_t ring_stride = dfb0->total_size();  // bytes; same for all identical configs
    const uint32_t ring_words  = ring_stride / sizeof(uint32_t);

    for (uint32_t i = 0; i < num_dfbs; ++i) {
        const uint32_t dfb_l1_addr = l1_base + i * ring_stride;
        std::vector<uint32_t> slice(ring_words, 0u);
        for (uint32_t e = 0; e < entries_per_dfb; ++e) {
            const uint32_t src = (i * entries_per_dfb + e) * wpe;
            std::copy(input.begin() + src, input.begin() + src + wpe, slice.begin() + e * wpe);
        }
        detail::WriteToDeviceL1(device, core, dfb_l1_addr, slice);
    }

    // Launch program (slow dispatch; same pattern as execute_program_and_verify).
    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    // Verify: each out_buffer must match its DFB's L1 pre-fill slice.
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        std::vector<uint32_t> output;
        distributed::ReadShard(mesh_device->mesh_command_queue(), output, out_buffers[i], zero_coord, true);

        std::vector<uint32_t> expected(entries_per_dfb * wpe);
        for (uint32_t e = 0; e < entries_per_dfb; ++e) {
            const uint32_t src = (i * entries_per_dfb + e) * wpe;
            std::copy(input.begin() + src, input.begin() + src + wpe, expected.begin() + e * wpe);
        }
        EXPECT_EQ(expected, output) << "TensixDM concurrent DFB " << i << " output mismatch";
    }
}

// =====================================================================================
// Gap 7 harness – sequential DM→DM DFBs
//
// N DM producer threads and N DM consumer threads cooperate sequentially through
// num_dfbs DFBs
//
// Producer threads: DM[0..num_producers-1]
// Consumer threads: DM[num_producers..num_producers+num_consumers-1]
// =====================================================================================
void run_sequential_dfbs_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::vector<experimental::dfb::DataflowBufferConfig>& configs) {
    TT_FATAL(!configs.empty(), "configs must not be empty");
    TT_FATAL(
        mesh_device->get_devices()[0]->arch() == ARCH::QUASAR,
        "Sequential multi-DFB TC-exhaustion tests require Quasar");

    const uint32_t num_dfbs      = static_cast<uint32_t>(configs.size());
    const uint32_t num_producers = configs[0].num_producers;
    const uint32_t num_consumers = configs[0].num_consumers;
    const uint32_t num_entries   = configs[0].num_entries;
    const uint32_t entry_size    = configs[0].entry_size;

    for (const auto& c : configs) {
        TT_FATAL(c.num_producers == num_producers && c.num_consumers == num_consumers &&
                     c.num_entries == num_entries && c.entry_size == entry_size,
            "All DFB configs must share num_producers/num_consumers/num_entries/entry_size");
    }

    // Producer DM threads: low num_producers bits; consumer threads: next num_consumers bits.
    TT_FATAL(
        num_producers + num_consumers <= 6,
        "num_producers + num_consumers must fit in 6 available Quasar DM threads (DM0 reserved for dispatch)");

    const uint32_t producer_mask = (1u << num_producers) - 1u;
    const uint32_t consumer_mask = ((1u << num_consumers) - 1u) << num_producers;

    const CoreCoord core(0, 0);
    const CoreRangeSet core_range_set(CoreRange(core, core));
    IDevice* device = mesh_device->get_devices()[0];
    auto zero_coord = distributed::MeshCoordinate(0, 0);

    const uint32_t buf_size = num_entries * entry_size;

    // Separate in_buffer and out_buffer per DFB.
    std::vector<std::shared_ptr<distributed::MeshBuffer>> in_buffers, out_buffers;
    distributed::DeviceLocalBufferConfig local_cfg{.page_size = entry_size, .buffer_type = BufferType::DRAM};
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        distributed::ReplicatedBufferConfig buf_cfg{.size = buf_size};
        in_buffers.push_back(distributed::MeshBuffer::create(buf_cfg, local_cfg, mesh_device.get()));
        out_buffers.push_back(distributed::MeshBuffer::create(buf_cfg, local_cfg, mesh_device.get()));
    }

    // Generate and write input data for each DFB's in_buffer.
    std::vector<std::vector<uint32_t>> inputs(num_dfbs);
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        inputs[i] = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, buf_size / sizeof(uint32_t));
        distributed::WriteShard(mesh_device->mesh_command_queue(), in_buffers[i], inputs[i], zero_coord, true);
    }

    if (mesh_device->get_devices()[0]->arch() == ARCH::QUASAR) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        for (uint32_t i = 0; i < num_dfbs; ++i) {
            std::vector<uint32_t> rdback;
            distributed::ReadShard(mesh_device->mesh_command_queue(), rdback, in_buffers[i], zero_coord, true);
            tt_driver_atomics::mfence();
            EXPECT_EQ(rdback, inputs[i]) << "in_buffer[" << i << "] DRAM write verify failed";
        }
    }

    Program program = CreateProgram();

    const uint32_t num_entries_per_producer =
        (num_entries + num_producers - 1) / num_producers;  // ceiling; all configs share this

    // Sequential DM producer kernel.
    std::vector<uint32_t> prod_cta = {num_entries_per_producer, (uint32_t)configs[0].enable_implicit_sync};
    tt::tt_metal::TensorAccessorArgs(in_buffers[0]).append_to(prod_cta);

    auto producer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_seq_producer.cpp",
        core_range_set,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = num_producers, .compile_args = prod_cta});

    // Sequential DM consumer kernel.
    std::vector<uint32_t> cons_cta = {(uint32_t)configs[0].enable_implicit_sync};
    tt::tt_metal::TensorAccessorArgs(out_buffers[0]).append_to(cons_cta);

    auto consumer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_seq_consumer.cpp",
        core_range_set,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = num_consumers, .compile_args = cons_cta});

    // Create all DFBs in the same Program (simultaneous TC allocation stress).
    std::vector<uint32_t> dfb_ids;
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        auto cfg_copy = configs[i];  // CreateDataflowBuffer takes by const ref; copy is fine
        auto dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_range_set, cfg_copy);
        experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program, dfb_id, producer_kernel, consumer_kernel);
        dfb_ids.push_back(dfb_id);
    }

    // Producer runtime args: mask | num_dfbs | src_addr[0..N-1]
    std::vector<uint32_t> prod_rta = {producer_mask, num_dfbs};
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        prod_rta.push_back(static_cast<uint32_t>(in_buffers[i]->address()));
    }
    SetRuntimeArgs(program, producer_kernel, core, prod_rta);

    // Consumer runtime args: mask | num_dfbs | dst_addr[0..N-1] | epc[0..N-1] | is_all[0..N-1]
    std::vector<uint32_t> cons_rta = {consumer_mask, num_dfbs};
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        cons_rta.push_back(static_cast<uint32_t>(out_buffers[i]->address()));
    }
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        const bool is_all = (configs[i].cap == dfb::AccessPattern::ALL);
        const uint32_t epc = is_all ? num_entries : (num_entries + num_consumers - 1) / num_consumers;
        cons_rta.push_back(epc);
    }
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        cons_rta.push_back(configs[i].cap == dfb::AccessPattern::ALL ? 1u : 0u);
    }
    SetRuntimeArgs(program, consumer_kernel, core, cons_rta);

    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    // Verify: out_buffer[i] should equal in_buffer[i] (strided and all alike).
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        std::vector<uint32_t> output;
        distributed::ReadShard(mesh_device->mesh_command_queue(), output, out_buffers[i], zero_coord, true);
        EXPECT_EQ(inputs[i], output) << "Sequential DFB " << i << " output mismatch";
    }
}

// Compute transformation applied by the TRISC between DFB_in and DFB_out.
// Identity: output = input         (uses dfb_eltwise_copy.cpp;     all archs)
// Relu:     output = max(0, input) (uses dfb_eltwise_relu.cpp; all archs incl. Quasar)
enum class DfbTransform : uint8_t { Identity, Relu };

// Runs a NOC -> DM_producer -> DFB_in -> TRISC -> DFB_out -> DM_consumer -> NOC pipeline.
// In all modes the compute kernel uses the full unpack/math/pack pipeline so data physically
// flows through the TRISC. Golden is computed in bf16 space and compared with tolerance.
//
// On WH/BH: only num_producers == num_consumers == 1 is supported (BRISC = producer, NCRISC =
// consumer, 1 compute kernel). enable_implicit_sync is forced to false on WH/BH (read_in /
// write_out are Quasar-only).
void run_in_dfb_out_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::dfb::DataflowBufferConfig& dm2tensix_config,
    experimental::dfb::DataflowBufferConfig& tensix2dm_config,
    DfbTransform transform = DfbTransform::Identity) {
    TT_FATAL(
        dm2tensix_config.num_entries == tensix2dm_config.num_entries,
        "Num entries must be the same for in and out DFBs");
    TT_FATAL(
        dm2tensix_config.entry_size == tensix2dm_config.entry_size, "Entry size must be the same for in and out DFBs");

    const auto arch = mesh_device->get_devices()[0]->arch();
    const bool is_quasar = (arch == ARCH::QUASAR);

    if (!is_quasar) {
        // WH/BH DM: one BRISC (RISCV_0) producer + one NCRISC (RISCV_1) consumer + one compute kernel.
        if (dm2tensix_config.num_producers > 1 || dm2tensix_config.num_consumers > 1 ||
            tensix2dm_config.num_producers > 1 || tensix2dm_config.num_consumers > 1) {
            GTEST_SKIP() << "WH/BH DFB supports only 1 DM producer (BRISC) and 1 DM consumer (NCRISC)";
        }
        // read_in / write_out implicit-sync overloads are Quasar-only.
        dm2tensix_config.enable_implicit_sync = false;
        tensix2dm_config.enable_implicit_sync = false;
    }

    Program program = CreateProgram();

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    uint32_t buffer_size = dm2tensix_config.entry_size * dm2tensix_config.num_entries;
    distributed::DeviceLocalBufferConfig local_buffer_config{
        .page_size = dm2tensix_config.entry_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
    auto in_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    auto out_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());

    log_info(tt::LogTest, "In Buffer: [address: {} B, size: {} B]", in_buffer->address(), in_buffer->size());
    log_info(tt::LogTest, "Out Buffer: [address: {} B, size: {} B]", out_buffer->address(), out_buffer->size());

    CoreCoord logical_core = CoreCoord(0, 0);

    // --- Producer (DM, NOC -> DFB_in) ---
    const uint32_t entries_per_core = dm2tensix_config.num_entries;
    uint32_t num_entries_per_producer =
        (entries_per_core + dm2tensix_config.num_producers - 1) / dm2tensix_config.num_producers;
    std::vector<uint32_t> producer_cta = {
        (uint32_t)in_buffer->address(), num_entries_per_producer, (uint32_t)dm2tensix_config.enable_implicit_sync};
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(producer_cta);

    const std::string dm_producer_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp";
    KernelHandle producer_kernel;
    if (is_quasar) {
        producer_kernel = experimental::quasar::CreateKernel(
            program,
            dm_producer_kernel_path,
            logical_core,
            experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = dm2tensix_config.num_producers, .compile_args = producer_cta});
    } else {
        producer_kernel = CreateKernel(
            program,
            dm_producer_kernel_path,
            logical_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .compile_args = producer_cta});
    }

    // --- Compute (TRISC: DFB_in -> DFB_out via copy_tile + pack_tile) ---
    // dfb_eltwise_copy.cpp expects (per_core_tile_cnt, use_dfbs).
    // It hardcodes DFB ids 0 (input) and 1 (output); we register them in that order below.
    uint32_t num_entries_per_unpacker = dm2tensix_config.num_entries / dm2tensix_config.num_consumers;
    uint32_t num_entries_per_packer = tensix2dm_config.num_entries / tensix2dm_config.num_producers;
    TT_FATAL(
        num_entries_per_unpacker == num_entries_per_packer, "Num entries per unpacker and packer must be the same");
    std::vector<uint32_t> compute_cta = {num_entries_per_unpacker, /*use_dfbs=*/1u};

    const std::string compute_kernel_path = (transform == DfbTransform::Relu)
                                                ? "tests/tt_metal/tt_metal/test_kernels/compute/dfb_eltwise_relu.cpp"
                                                : "tests/tt_metal/tt_metal/test_kernels/compute/dfb_eltwise_copy.cpp";
    KernelHandle compute_kernel;
    if (is_quasar) {
        compute_kernel = CreateKernel(
            program,
            compute_kernel_path,
            logical_core,
            experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = 1, .compile_args = compute_cta});
    } else {
        compute_kernel =
            CreateKernel(program, compute_kernel_path, logical_core, ComputeConfig{.compile_args = compute_cta});
    }

    // --- Consumer (DM, DFB_out -> NOC) ---
    const bool out_is_all = (tensix2dm_config.cap == dfb::AccessPattern::ALL);
    uint32_t num_entries_per_consumer =
        out_is_all ? tensix2dm_config.num_entries
                   : (entries_per_core + tensix2dm_config.num_consumers - 1) / tensix2dm_config.num_consumers;
    std::vector<uint32_t> consumer_cta = {
        (uint32_t)out_buffer->address(),
        num_entries_per_consumer,
        (uint32_t)out_is_all,
        (uint32_t)tensix2dm_config.enable_implicit_sync};
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(consumer_cta);

    const std::string dm_consumer_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp";
    KernelHandle consumer_kernel;
    if (is_quasar) {
        consumer_kernel = experimental::quasar::CreateKernel(
            program,
            dm_consumer_kernel_path,
            logical_core,
            experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = tensix2dm_config.num_consumers, .compile_args = consumer_cta});
    } else {
        consumer_kernel = CreateKernel(
            program,
            dm_consumer_kernel_path,
            logical_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .compile_args = consumer_cta});
    }

    // --- DFBs + bindings ---
    // Order matters: input DFB must be id 0 and output DFB must be id 1 so dfb_eltwise_copy.cpp's
    // hardcoded DataflowBuffer dfb_in(0) / dfb_out(1) lines up.
    auto input_dfb_id = experimental::dfb::CreateDataflowBuffer(program, logical_core, dm2tensix_config);
    auto output_dfb_id = experimental::dfb::CreateDataflowBuffer(program, logical_core, tensix2dm_config);

    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, input_dfb_id, producer_kernel, compute_kernel);
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, output_dfb_id, compute_kernel, consumer_kernel);

    auto input_dfb = program.impl().get_dataflow_buffer(input_dfb_id);
    auto output_dfb = program.impl().get_dataflow_buffer(output_dfb_id);

    SetRuntimeArgs(
        program,
        producer_kernel,
        logical_core,
        {(uint32_t)input_dfb->config.producer_risc_mask, /*chunk_offset=*/0u, entries_per_core});
    // dfb_eltwise_copy.cpp takes no runtime args (PACK_RELU mode would, but we're not defining it).
    SetRuntimeArgs(
        program,
        consumer_kernel,
        logical_core,
        {(uint32_t)output_dfb->config.consumer_risc_mask,
         (uint32_t)output_dfb_id,
         /*chunk_offset=*/0u,
         entries_per_core});

    // Use packed-bfloat16 stimulus (2 bf16 values per uint32). Random uint32 values
    // would be denormal when reinterpreted as bf16 and get flushed to zero by the
    // unpack/pack pipeline. create_random_vector_of_bfloat16 produces values in a
    // normal-bf16 range (±1.0) that survive the unpack/copy_tile/pack_tile round-trip.
    auto input = create_random_vector_of_bfloat16(buffer_size, 1.0f, 0xCAFE);

    // Skip the default EXPECT_EQ inside execute_program_and_verify and run our own
    // bf16-tolerant comparison below — bytewise equality is not guaranteed through
    // the unpack/pack pipeline (rounding, denormal flush, etc).
    execute_program_and_verify(mesh_device, program, in_buffer, out_buffer, zero_coord, input, /*verify_output=*/false);

    std::vector<uint32_t> output;
    distributed::ReadShard(mesh_device->mesh_command_queue(), output, out_buffer, zero_coord, true);

    // Golden: identity (output == input) or relu (output == max(0, input)) per bf16 element.
    std::vector<uint32_t> golden = input;
    if (transform == DfbTransform::Relu) {
        for (size_t i = 0; i < input.size(); ++i) {
            auto [lo_bf, hi_bf] = unpack_two_bfloat16_from_uint32(input[i]);
            golden[i] = pack_two_bfloat16_into_uint32(
                {bfloat16(std::fmax(0.0f, static_cast<float>(lo_bf))),
                 bfloat16(std::fmax(0.0f, static_cast<float>(hi_bf)))});
        }
    }

    auto compare = [](float a, float b) {
        const float atol = 0.02f;
        const float rtol = 0.05f;
        float maxabs = std::fmax(std::fabs(a), std::fabs(b));
        return std::fabs(a - b) <= atol || std::fabs(a - b) <= rtol * maxabs;
    };
    int argfail = -1;
    bool pass = packed_uint32_t_vector_comparison(output, golden, compare, &argfail);
    const char* label = (transform == DfbTransform::Relu) ? "(relu transform)" : "(identity)";
    EXPECT_TRUE(pass) << "DFB pipeline output mismatch at position " << argfail << " " << label;
}

// =====================================================================================
// Single-DFB test macros
//
// WH/BH supports only 1 DM producer (BRISC) + 1 DM consumer (NCRISC) and no implicit_sync,
// so 1x1 configurations may run there with implicit_sync=false; everything else is Quasar-only.
// =====================================================================================

#define DFB_SKIP_IF_UNSUPPORTED(num_p, num_c)                                                           \
    if (devices_.at(0)->arch() != ARCH::QUASAR && (GetParam() || (num_p) > 1 || (num_c) > 1)) {        \
        GTEST_SKIP();                                                                                   \
    }

// DM -> ALL DM is unsupported with implicit_sync today.
#define DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC                                                             \
    if (GetParam()) {                                                                                \
        GTEST_SKIP() << "Skipping DM to ALL DM with implicit sync until support is added";          \
    }

#define DFB_NO_EXTRA_SKIP ((void)0)

constexpr uint32_t dfb_default_num_entries(uint32_t num_p, uint32_t num_c) {
    const uint32_t m = (num_p / std::gcd(num_p, num_c)) * num_c;
    return ((16u + m - 1u) / m) * m;
}

// Single-DFB test on a single core with default ring derived from (num_p, num_c)
// and entry_size=1024.
//   prefix       e.g. DM, DMTensix, TensixDM
//   suffix       e.g. 3Sx1S, 6Sx2B
//   p_kind/c_kind   DM | TENSIX
//   num_p/num_c     1..6 (DM); 1..4 (TENSIX)
//   pap_kind/cap_kind   STRIDED | ALL
//   extra_skip      DFB_NO_EXTRA_SKIP or DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC
#define DFB_TEST(prefix, suffix, p_kind, c_kind, num_p, pap_kind, num_c, cap_kind, extra_skip)          \
    TEST_P(DFBImplicitSyncParamFixture, prefix##Test1xDFB##suffix) {                                    \
        DFB_SKIP_IF_UNSUPPORTED((num_p), (num_c));                                                      \
        extra_skip;                                                                                     \
        experimental::dfb::DataflowBufferConfig config{                                                 \
            .entry_size = 1024,                                                                         \
            .num_entries = dfb_default_num_entries((num_p), (num_c)),                                   \
            .num_producers = (num_p),                                                                   \
            .pap = dfb::AccessPattern::pap_kind,                                                        \
            .num_consumers = (num_c),                                                                   \
            .cap = dfb::AccessPattern::cap_kind,                                                        \
            .enable_implicit_sync = GetParam()};                                                        \
        run_single_dfb_program(                                                                         \
            this->devices_.at(0), config, DFBPorCType::p_kind, DFBPorCType::c_kind);                    \
    }

// Variant for DM->DM tests that pass an explicit num_entries_in_buffer (forces wraparound
// when the requested total exceeds the ring size).
#define DFB_TEST_BUF(prefix, suffix, p_kind, c_kind, num_p, pap_kind, num_c, cap_kind, extra_skip, n_buf) \
    TEST_P(DFBImplicitSyncParamFixture, prefix##Test1xDFB##suffix) {                                    \
        DFB_SKIP_IF_UNSUPPORTED((num_p), (num_c));                                                      \
        extra_skip;                                                                                     \
        experimental::dfb::DataflowBufferConfig config{                                                 \
            .entry_size = 1024,                                                                         \
            .num_entries = dfb_default_num_entries((num_p), (num_c)),                                   \
            .num_producers = (num_p), .pap = dfb::AccessPattern::pap_kind,                              \
            .num_consumers = (num_c), .cap = dfb::AccessPattern::cap_kind,                              \
            .enable_implicit_sync = GetParam()};                                                        \
        CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));                       \
        run_single_dfb_program(                                                                         \
            this->devices_.at(0), config, DFBPorCType::p_kind, DFBPorCType::c_kind,                     \
            core_range_set, (n_buf));                                                                   \
    }

// =====================================================================================
// Strided
// =====================================================================================

// 1x1 (DM->DM uses num_entries_in_buffer=18 to exercise wraparound)
DFB_TEST_BUF(DM,       1Sx1S, DM,     DM,     1, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP, 18)
DFB_TEST    (DMTensix, 1Sx1S, DM,     TENSIX, 1, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 1Sx1S, TENSIX, DM,     1, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)

// NOC -> DM(BRISC) -> DFB_in -> TRISC(eltwise_copy) -> DFB_out -> DM(NCRISC) -> NOC.
// Enabled on WH/BH (1 producer + 1 consumer, explicit sync) and Quasar.
// entry_size = 2048 = bfloat16 32x32 tile = one logical tile per entry, which matches
// how dfb_eltwise_copy.cpp's copy_tile + pack_tile pipeline operates.
TEST_F(MeshDeviceFixture, DMTensixDMTest2xDFB1Sx1S) {
    experimental::dfb::DataflowBufferConfig dm2tensix_config{
        .entry_size = 2048,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16_b};

    experimental::dfb::DataflowBufferConfig tensix2dm_config{
        .entry_size = 2048,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16_b};

    run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config);
}

// Same pipeline but the compute kernel applies SFPU relu: output = max(0, input).
// relu_tile is the one SFPU op currently ported on Quasar, so this test runs on all three archs.
TEST_F(MeshDeviceFixture, DMTensixDMTest2xDFB1Sx1S_Relu) {
    experimental::dfb::DataflowBufferConfig dm2tensix_config{
        .entry_size = 2048,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16_b};

    experimental::dfb::DataflowBufferConfig tensix2dm_config{
        .entry_size = 2048,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16_b};

    run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config, DfbTransform::Relu);
}

// TEST_F(MeshDeviceFixture, DMTensixDMTest1xDFB2Sx1S1xDFB1Sx2S) {
//     if (devices_.at(0)->arch() != ARCH::QUASAR) {
//         GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
//     }
//     experimental::dfb::DataflowBufferConfig dm2tensix_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 2,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 1,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     experimental::dfb::DataflowBufferConfig tensix2dm_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 1,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 2,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config);
// }

// TEST_F(MeshDeviceFixture, DMTensixDMTest1xDFB4Sx1S1xDFB1Sx4S) {
//     if (devices_.at(0)->arch() != ARCH::QUASAR) {
//         GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
//     }
//     experimental::dfb::DataflowBufferConfig dm2tensix_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 4,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 1,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     experimental::dfb::DataflowBufferConfig tensix2dm_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 1,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 4,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config);
// }

DFB_TEST    (DM,       1Sx4S, DM,     DM,     1, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 1Sx4S, DM,     TENSIX, 1, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 1Sx4S, TENSIX, DM,     1, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       4Sx1S, DM,     DM,     4, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 4Sx1S, DM,     TENSIX, 4, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx1S, TENSIX, DM,     4, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)

DFB_TEST_BUF(DM,       4Sx4S, DM,     DM,     4, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP, 29)
DFB_TEST    (DMTensix, 4Sx4S, DM,     TENSIX, 4, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx4S, TENSIX, DM,     4, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)

DFB_TEST_BUF(DM,       2Sx4S, DM,     DM,     2, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP, 21)
DFB_TEST    (DMTensix, 2Sx4S, DM,     TENSIX, 2, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 2Sx4S, TENSIX, DM,     2, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       4Sx2S, DM,     DM,     4, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 4Sx2S, DM,     TENSIX, 4, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx2S, TENSIX, DM,     4, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)

// DM->DM strided: power-of-2 (1Sx2S, 2Sx1S, 2Sx2S)
DFB_TEST    (DM,       1Sx2S, DM,     DM,     1, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DM,       2Sx1S, DM,     DM,     2, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DM,       2Sx2S, DM,     DM,     2, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)

// DM->DM strided: 3-DM consumer column
DFB_TEST    (DM,       1Sx3S, DM,     DM,     1, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (DM,       2Sx3S, DM,     DM,     2, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP) // needs ALL access pattern

DFB_TEST    (DM,       3Sx1S, DM,     DM,     3, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (DM,       3Sx2S, DM,     DM,     3, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP) // needs ALL access pattern
DFB_TEST    (DM,       3Sx3S, DM,     DM,     3, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       1Sx5S, DM,     DM,     1, STRIDED, 5, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DM,       5Sx1S, DM,     DM,     5, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)

// DM->Tensix strided (Tensix consumers limited to {1,2,4})
// Power-of-2 gaps
DFB_TEST    (DMTensix, 1Sx2S, DM,     TENSIX, 1, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 2Sx1S, DM,     TENSIX, 2, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
// 3-DM producer
DFB_TEST    (DMTensix, 3Sx1S, DM,     TENSIX, 3, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (DMTensix, 3Sx2S, DM,     TENSIX, 3, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP) // needs BLOCKED access pattern
// DFB_TEST    (DMTensix, 3Sx4S, DM,     TENSIX, 3, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP) // needs BLOCKED access pattern
// 6-DM producer
DFB_TEST    (DMTensix, 6Sx1S, DM,     TENSIX, 6, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 6Sx2S, DM,     TENSIX, 6, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (DMTensix, 6Sx4S, DM,     TENSIX, 6, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP) // needs BLOCKED access pattern

// Tensix->DM strided (Tensix producers limited to {1,2,4})
// Power-of-2 gaps
DFB_TEST    (TensixDM, 2Sx1S, TENSIX, DM,     2, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 1Sx2S, TENSIX, DM,     1, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
// 3-DM consumer
DFB_TEST    (TensixDM, 1Sx3S, TENSIX, DM,     1, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 2Sx3S, TENSIX, DM,     2, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (TensixDM, 4Sx3S, TENSIX, DM,     4, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP) // needs BLOCKED access pattern
// 6-DM consumer
DFB_TEST    (TensixDM, 1Sx6S, TENSIX, DM,     1, STRIDED, 6, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 2Sx6S, TENSIX, DM,     2, STRIDED, 6, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (TensixDM, 4Sx6S, TENSIX, DM,     4, STRIDED, 6, STRIDED, DFB_NO_EXTRA_SKIP) // needs BLOCKED access pattern

// =====================================================================================
// ALL
// =====================================================================================

DFB_TEST    (DM,       1Sx4A, DM,     DM,     1, STRIDED, 4, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DMTensix, 1Sx4A, DM,     TENSIX, 1, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 1Sx4A, TENSIX, DM,     1, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       4Sx1A, DM,     DM,     4, STRIDED, 1, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DMTensix, 4Sx1A, DM,     TENSIX, 4, STRIDED, 1, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx1A, TENSIX, DM,     4, STRIDED, 1, ALL, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       4Sx4A, DM,     DM,     4, STRIDED, 4, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DMTensix, 4Sx4A, DM,     TENSIX, 4, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx4A, TENSIX, DM,     4, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       4Sx2A, DM,     DM,     4, STRIDED, 2, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DMTensix, 4Sx2A, DM,     TENSIX, 4, STRIDED, 2, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx2A, TENSIX, DM,     4, STRIDED, 2, ALL, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       2Sx4A, DM,     DM,     2, STRIDED, 4, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DMTensix, 2Sx4A, DM,     TENSIX, 2, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 2Sx4A, TENSIX, DM,     2, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)

// DM->DM ALL: 3-DM producer
DFB_TEST    (DM,       3Sx1A, DM,     DM,     3, STRIDED, 1, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DM,       3Sx2A, DM,     DM,     3, STRIDED, 2, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DM,       3Sx3A, DM,     DM,     3, STRIDED, 3, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)

// DM->DM ALL: 3-DM consumer
DFB_TEST    (DM,       1Sx3A, DM,     DM,     1, STRIDED, 3, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)
DFB_TEST    (DM,       2Sx3A, DM,     DM,     2, STRIDED, 3, ALL, DFB_SKIP_DM_DM_ALL_IMPLICIT_SYNC)

// DM->Tensix ALL (Tensix consumers limited to {1,2,4})
DFB_TEST    (DMTensix, 3Sx1A, DM,     TENSIX, 3, STRIDED, 1, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 3Sx2A, DM,     TENSIX, 3, STRIDED, 2, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 3Sx4A, DM,     TENSIX, 3, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 6Sx1A, DM,     TENSIX, 6, STRIDED, 1, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 6Sx2A, DM,     TENSIX, 6, STRIDED, 2, ALL, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 6Sx4A, DM,     TENSIX, 6, STRIDED, 4, ALL, DFB_NO_EXTRA_SKIP)

// Tensix->DM ALL (Tensix producers limited to {1,2,4}).
// DFB_TEST    (TensixDM, 1Sx3A, TENSIX, DM,     1, STRIDED, 3, ALL, DFB_NO_EXTRA_SKIP) // revisit the 3A Tensix consumer
// DFB_TEST    (TensixDM, 2Sx3A, TENSIX, DM,     2, STRIDED, 3, ALL, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (TensixDM, 4Sx3A, TENSIX, DM,     4, STRIDED, 3, ALL, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (TensixDM, 1Sx6A, TENSIX, DM,     1, STRIDED, 6, ALL, DFB_NO_EXTRA_SKIP) // revisit more than 4 ALL consumers
// DFB_TEST    (TensixDM, 2Sx6A, TENSIX, DM,     2, STRIDED, 6, ALL, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (TensixDM, 4Sx6A, TENSIX, DM,     4, STRIDED, 6, ALL, DFB_NO_EXTRA_SKIP)

// 1 strided DM producer, 1 strided DM consumer, num_entries=4.
// Ring wraps 16x; baseline to confirm wraparound logic with minimum participants.
TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB_RingPressure_1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB ring-pressure test for WH/BH until DFB is backported";
    }
    DFB_SKIP_IF_UNSUPPORTED(1, 1);
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 4,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(
        this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM,
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))), /*num_entries_in_buffer=*/64);
}

// 4 strided DM producers, 4 strided DM consumers, num_entries=4 -> capacity=1.
// Each producer stalls after every push; ring wraps 64x per producer.
TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB_RingPressure_4Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB ring-pressure test for WH/BH until DFB is backported";
    }
    DFB_SKIP_IF_UNSUPPORTED(4, 4);
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 4,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(
        this->devices_.at(0),
        config,
        DFBPorCType::DM,
        DFBPorCType::DM,
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))),
        /*num_entries_in_buffer=*/64);
}

// 4 DM producers (STRIDED), 4 Tensix consumers (ALL), num_entries=4 -> capacity=1.
// Every push stalls until all 4 ALL Tensix consumers ack; ring wraps 64x per producer.
// Exercises remapper fan-out on the DM->Tensix path under maximum ring pressure.
TEST_P(DFBImplicitSyncParamFixture, DMTensixTest1xDFB_RingPressure_4Sx4A) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB ring-pressure test for WH/BH until DFB is backported";
    }
    DFB_SKIP_IF_UNSUPPORTED(4, 4);
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 4,  // tight ring: capacity = num_entries / num_producers = 1
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::ALL,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(
        this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::TENSIX,
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))), /*num_entries_in_buffer=*/64);
}

// 2 Tensix producers (STRIDED), 4 DM consumers (STRIDED), num_entries=4 -> capacity=1.
// Ring wraps 64x per producer; exercises the Tensix->DM path with asymmetric P:C ratio
// under tight ring pressure (num_consumers > num_producers).
TEST_P(DFBImplicitSyncParamFixture, TensixDMTest1xDFB_RingPressure_2Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB ring-pressure test for WH/BH until DFB is backported";
    }
    DFB_SKIP_IF_UNSUPPORTED(2, 4);
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 4,  // tight ring: capacity = num_entries / max(num_p, num_c) = 1
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(
        this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM,
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))), /*num_entries_in_buffer=*/64);
}

TEST_P(DFBImplicitSyncParamFixture, MultiCoreDMTest2Core_1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set);
}

TEST_P(DFBImplicitSyncParamFixture, MultiCoreDMTest2Core_2Sx2S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set);
}

TEST_P(DFBImplicitSyncParamFixture, MultiCoreDMTest2Core_1Sx4A) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::ALL,
        .enable_implicit_sync = GetParam()};

    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set);
}

// =====================================================================================
// Concurrent DFBs on Same Core (TC Allocator Stress)
//
//
// TensixDMTest4xDFB_1Sx1S
//   4 concurrent 1Sx1S Tensix→DM DFBs.  Single Neo thread fills DFBs sequentially;
//   4 DM consumer threads drain each DFB as soon as entries arrive.
//   Harness: run_concurrent_tensix_dm_dfbs_program
//            + dfb_t6_seq_producer.cpp + dfb_multi_consumer_sep.cpp
//
// DMTest4xDFB_3Sx3S
//   4× 3Sx3S DFBs = 12 TCs total: stresses TC allocator across 6 DM threads.
//   DM threads 0..2 cooperate on each DFB in sequence as producers; DM threads 3..5 consume.
//   All 12 TCs allocated simultaneously in one Program.
//   Harness: run_sequential_dfbs_program + dfb_seq_producer.cpp + dfb_seq_consumer.cpp
//
// DMTest4xDFB_Mixed
//   2× 3Sx3S (strided) + 2× 3Sx3ALL (ALL) on the same core.
//   Mixed TC allocation (strided TCs + remapper-backed ALL TCs) across 6 DM threads.
//   Harness: run_sequential_dfbs_program + dfb_seq_producer.cpp + dfb_seq_consumer.cpp
// =====================================================================================

TEST_P(DFBImplicitSyncParamFixture, DMTest3xDFB_1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping: concurrent DFB test requires Quasar multi-threaded DM";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size    = 1024,
        .num_entries   = 16,
        .num_producers = 1,
        .pap           = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap           = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_concurrent_dfbs_program(this->devices_.at(0), /*num_dfbs=*/3, config);
}

TEST_P(DFBImplicitSyncParamFixture, TensixDMTest4xDFB_1Sx1S) {
    // 4 concurrent 1Sx1S DFBs: single Neo thread produces to all 4 sequentially while
    // 4 DM consumer threads drain their DFBs as soon as entries become available.
    // Uses dfb_t6_seq_producer.cpp + dfb_multi_consumer_sep.cpp.
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping: Tensix concurrent DFB test requires Quasar";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size    = 1024,
        .num_entries   = 16,
        .num_producers = 1,
        .pap           = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap           = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_concurrent_tensix_dm_dfbs_program(this->devices_.at(0), /*num_dfbs=*/4, config);
}

TEST_P(DFBImplicitSyncParamFixture, DMTest4xDFB_3Sx3S) {
    // 4 DFBs × 3Sx3S = 12 TCs: stresses TC allocator across 6 DM threads.
    // DM threads 0..2 cooperatively produce all 4 DFBs in sequence; DM threads 3..5
    // cooperatively consume all 4 DFBs in sequence.
    // Uses dfb_seq_producer.cpp + dfb_seq_consumer.cpp.
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping: sequential multi-DFB TC exhaustion test requires Quasar";
    }
    // num_entries must be divisible by max(num_producers, num_consumers) per
    // the host-side DFB validator. With 3 producers/consumers, 18 is the
    // smallest multiple of 3 that's >= 16 (matches dfb_default_num_entries).
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 18,
        .num_producers = 3,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 3,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_sequential_dfbs_program(this->devices_.at(0), {config, config, config, config});
}

TEST_P(DFBImplicitSyncParamFixture, DMTest4xDFB_Mixed) {
    // 2× 3Sx3S (strided consumers) + 2× 3Sx3B (ALL consumers) on the same core.
    // Mixed TC allocation exercises both plain strided TCs and remapper-backed ALL TCs
    // simultaneously across 6 DM threads (DM threads 0..2 produce, 3..5 consume).
    // Uses dfb_seq_producer.cpp + dfb_seq_consumer.cpp with per-DFB is_all flags.
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping: mixed multi-DFB TC exhaustion test requires Quasar";
    }
    // DM→DM ALL with implicit sync is a known runtime gap: the Remapper only
    // fans credits between Tensix and DM TCs (no DM↔DM path), and the
    // implicit-sync ISR's descriptor for ALL consumers doesn't account for
    // the fan-out factor (tiles_to_post=1 per producer TC, but tiles_to_ack=3
    // per consumer TC across 9 fanned-out TCs, so total_acked = 3 × total_posted
    // and finish_impl AAW hangs on acked == posted permanently).
    // The explicit-sync (broadcast_tc) path handles this in software.
    if (GetParam()) {
        GTEST_SKIP() << "DM→DM ALL with implicit sync not yet supported (known runtime gap)";
    }
    // num_entries must be divisible by max(num_producers, num_consumers). With
    // 3 producers/consumers, use 18 (smallest valid multiple of 3 >= 16).
    experimental::dfb::DataflowBufferConfig strided_cfg{
        .entry_size = 1024,
        .num_entries = 18,
        .num_producers = 3,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 3,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    experimental::dfb::DataflowBufferConfig all_cfg{
        .entry_size = 1024,
        .num_entries = 18,
        .num_producers = 3,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 3,
        .cap = dfb::AccessPattern::ALL,
        .enable_implicit_sync = GetParam()};
    run_sequential_dfbs_program(
        this->devices_.at(0), {strided_cfg, strided_cfg, all_cfg, all_cfg});
}

// =====================================================================================
// B-batch regression tests (runtime tests pinning bug fixes shipped on this branch)
// =====================================================================================

// B1: DM0-no-kernel regression
// -------------------------------------------------------------------------------------
// Pins the fix in tt_metal/hw/firmware/src/tt-2xx/dm.cc where
// start_subordinate_kernel_run_early() now reports whether *any* subordinate
// DM runs a user kernel. If subordinates do but DM0 itself doesn't (DM0 is
// reserved for dispatch in slow-dispatch mode), DM0 must still call
// setup_local_dfb_interfaces() so the implicit-sync ISR is programmed on its
// side — otherwise the threshold register never fires and the subordinates
// hang forever waiting for credits.
//
// Configuration: Tensix→DM 1Sx1S with implicit sync enabled.
//   * Producer is a Tensix TRISC → no DM produces.
//   * Consumer is one DM (placed by the Quasar runtime — typically DM1/etc, not DM0).
//   * DM0 has no user kernel of its own.
// If the dm.cc fix were missing, the implicit-sync ISR setup would be skipped
// on DM0 and the test would hang at "Writing DFB config to core".
//
// Note: every existing TensixDM*ImplicitSyncTrue test already exercises this
// path; this is the minimal direct test that names what it pins.
TEST_F(MeshDeviceFixture, B1_DM0NoKernel_TensixDMImplicitSync) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Implicit sync is Quasar-only; the dm.cc fix is Quasar-specific";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = true};
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM);
}

// B1b: DM0-idle-with-subordinate regression (the *original* PR fix scenario)
// -------------------------------------------------------------------------------------
// Companion to B1. The classic CreateKernel auto-allocator (GetProcessorsPerClusterQuasar)
// fills DMs starting from DM0, so B1 — despite its name and original intent — actually
// places the consumer on DM0 and exercises the "DM0 has a kernel, no subordinates run"
// path through dm.cc. The *other* path the dm.cc fix needs to keep working is
// "DM0 is idle, a subordinate DM runs a user kernel that touches DFBs" — which is the
// scenario Metal 2.0 produces in production (it reserves DM0+DM1 for internal use).
//
// To reach that state from the classic API we have to bypass the auto-allocator and
// hand-pick DM1 as the consumer's home; that's what consumer_dm_processors_override does.
// If the dm.cc condition were ever narrowed back to (enables & DM0) only — dropping the
// subordinate case — this test would hang at dfb.finish() because DM0 would never run
// setup_local_dfb_interfaces() and the implicit-sync ISR registers would not be armed.
TEST_F(MeshDeviceFixture, B1b_DM0IdleSubordinateRuns_TensixDMImplicitSync) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Implicit sync is Quasar-only; the dm.cc fix is Quasar-specific";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = true};
    // Force consumer onto DM1 so DM0 is idle but a subordinate runs the kernel.
    std::set<DataMovementProcessor> dm1_only{DataMovementProcessor::RISCV_1};
    run_single_dfb_program(
        this->devices_.at(0),
        config,
        DFBPorCType::TENSIX,
        DFBPorCType::DM,
        /*core_range_set=*/CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))),
        /*num_entries_in_buffer=*/std::nullopt,
        /*consumer_dm_processors_override=*/dm1_only);
}

// B3: Tail-credit race exposure
// -------------------------------------------------------------------------------------
// Pins the rewritten handle_final_credits() in
// tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:215 — the old code gated
// the rendezvous barrier on a counter value, which raced with the ISR firing
// concurrently. The fix is an unconditional sync_threads() before deciding
// whether to manually post tail credits.
//
// handle_final_credits runs at the END of every implicit-sync session
// (DataflowBuffer::finish), so simply re-running the same 1Sx1S DM-DM
// implicit-sync workload several times exercises the barrier path repeatedly.
// If the race ever re-appears, one of the iterations will hang or fail.
//
// Iteration count is a deliberate compromise: 3 iterations on the Quasar
// emulator costs ~6 minutes; on real Quasar silicon (when available) bump
// kNumIters higher to expose lower-probability windows of the race.
TEST_F(MeshDeviceFixture, B3_TailCreditRace_RepeatedImplicitSync_DMDM) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Implicit sync is Quasar-only; tail-credit race fix is Quasar-specific";
    }
    constexpr int kNumIters = 3;
    for (int i = 0; i < kNumIters; ++i) {
        SCOPED_TRACE(::testing::Message() << "iteration " << (i + 1) << "/" << kNumIters);
        experimental::dfb::DataflowBufferConfig config{
            .entry_size = 1024,
            .num_entries = 16,
            .num_producers = 1,
            .pap = dfb::AccessPattern::STRIDED,
            .num_consumers = 1,
            .cap = dfb::AccessPattern::STRIDED,
            .enable_implicit_sync = true};
        run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM);
    }
}

// =====================================================================================
// C2: DM → TRISC (inter) → TRISC self-loop (intra) → TRISC → DM (inter)
//
//   DRAM in_buffer
//      ↓ (DM producer, NoC)
//   DFB 0 (inter, DM → TRISC)
//      ↓ (TRISC unpack via copy_tile)
//   [SFPU relu_tile]                ← stage A
//      ↓ (TRISC pack)
//   DFB 1 (intra, TRISC self-loop)
//      ↓ (TRISC unpack via copy_tile)
//   [SFPU relu_tile]                ← stage B
//      ↓ (TRISC pack)
//   DFB 2 (inter, TRISC → DM)
//      ↓ (DM consumer, NoC)
//   DRAM out_buffer
//
// Both stages use the SFPU (relu); for positive bf16 inputs double-relu is
// identity, so output ≈ input (bf16 tolerance).
//
// Mirrors C1's DM↔TRISC bookending but routes data through the real
// unpack/SFPU/pack pipeline via the DFB API (vs. C1's raw-L1 mutation).
//
// Side benefit: this test demonstrates that the **unary** compute pipeline
// (copy_tile / relu_tile / pack_tile) works correctly against an INTRA-scope
// DFB — narrowing the "INTRA DFB + compute pipeline" gap historically called
// out in C1's comment. The remaining LLK gap on Quasar is the **unary↔binary
// unpack reconfig** (not specific to INTRA DFBs): if a kernel calls
// `binary_op_init_common(...)` and then `copy_tile_to_dst_init_short(...)` to
// switch back to unary, the subsequent `copy_tile` does not load srcA from
// L1, dst stays at 0, and the pipeline cascades zeros. Diagnosed by host-
// side L1 readback of dfb_self + dfb_out after an SFPU+FPU variant of this
// kernel — both rings were zero post-run. The unary-only pipeline used here
// avoids that switch and works fine. When the LLK reconfig is fixed, an
// SFPU+FPU variant of this same pipeline should become testable.
// =====================================================================================
static void run_c2_dm_trisc_selfloop_dm_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t entry_size, uint32_t num_entries) {
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "C2 INTRA-scope DFB self-loop requires Quasar";
    }

    Program program = CreateProgram();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    CoreCoord logical_core(0, 0);

    const uint32_t buffer_size = entry_size * num_entries;
    distributed::DeviceLocalBufferConfig local_buffer_config{.page_size = entry_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
    auto in_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    auto out_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());

    log_info(tt::LogTest, "In Buffer:  [address: {} B, size: {} B]", in_buffer->address(), in_buffer->size());
    log_info(tt::LogTest, "Out Buffer: [address: {} B, size: {} B]", out_buffer->address(), out_buffer->size());

    // --- DFB 0: inter, DM → TRISC (DM producer fills, compute kernel unpacks) ---
    experimental::dfb::DataflowBufferConfig dfb_in_config{
        .entry_size = entry_size,
        .num_entries = num_entries,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16_b};

    // --- DFB 1: intra, TRISC self-loop (compute kernel both produces and consumes) ---
    experimental::dfb::DataflowBufferConfig dfb_self_config{
        .entry_size = entry_size,
        .num_entries = num_entries,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16_b,
        .tensix_scope = experimental::dfb::TensixScope::INTRA};

    // --- DFB 2: inter, TRISC → DM (compute kernel packs, DM consumer drains) ---
    experimental::dfb::DataflowBufferConfig dfb_out_config{
        .entry_size = entry_size,
        .num_entries = num_entries,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16_b};

    // --- DM producer kernel (reuse the standard producer; it binds to logical DFB id 0) ---
    std::vector<uint32_t> producer_cta = {
        (uint32_t)in_buffer->address(),
        num_entries,
        /*implicit_sync=*/0u};
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(producer_cta);
    auto producer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
        logical_core,
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .compile_args = producer_cta});

    // --- Compute kernel: 3-DFB pipeline with two SFPU relu stages per tile ---
    std::vector<uint32_t> compute_cta = {num_entries};
    auto compute_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/dfb_c2_pipeline.cpp",
        logical_core,
        experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = 1, .compile_args = compute_cta});

    // --- DM consumer kernel (reuse the standard consumer; takes logical DFB id as a runtime arg) ---
    std::vector<uint32_t> consumer_cta = {
        (uint32_t)out_buffer->address(),
        num_entries,
        /*blocked_consumer=*/0u,
        /*implicit_sync=*/0u};
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(consumer_cta);
    auto consumer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp",
        logical_core,
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .compile_args = consumer_cta});

    // --- Create DFBs in id order (0, 1, 2) so the compute kernel's hardcoded ids line up ---
    auto dfb_in_id = experimental::dfb::CreateDataflowBuffer(program, logical_core, dfb_in_config);
    auto dfb_self_id = experimental::dfb::CreateDataflowBuffer(program, logical_core, dfb_self_config);
    auto dfb_out_id = experimental::dfb::CreateDataflowBuffer(program, logical_core, dfb_out_config);

    // --- Bind: DFB 0 = DM→T; DFB 1 = T→T (intra, same kernel both sides); DFB 2 = T→DM ---
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(program, dfb_in_id, producer_kernel, compute_kernel);
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, dfb_self_id, compute_kernel, compute_kernel);
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, dfb_out_id, compute_kernel, consumer_kernel);

    auto dfb_in = program.impl().get_dataflow_buffer(dfb_in_id);
    auto dfb_out = program.impl().get_dataflow_buffer(dfb_out_id);

    SetRuntimeArgs(
        program,
        producer_kernel,
        logical_core,
        {(uint32_t)dfb_in->config.producer_risc_mask, /*chunk_offset=*/0u, num_entries});
    SetRuntimeArgs(
        program,
        consumer_kernel,
        logical_core,
        {(uint32_t)dfb_out->config.consumer_risc_mask,
         (uint32_t)dfb_out_id,
         /*chunk_offset=*/0u,
         num_entries});
    // Compute kernel takes no runtime args (per_core_tile_cnt is in CTA).

    // bf16 stimulus in ±1.0 range — survives unpack/copy_tile/pack_tile round-trip.
    auto input = create_random_vector_of_bfloat16(buffer_size, 1.0f, 0xC0FE);

    execute_program_and_verify(mesh_device, program, in_buffer, out_buffer, zero_coord, input, /*verify_output=*/false);

    std::vector<uint32_t> output;
    distributed::ReadShard(mesh_device->mesh_command_queue(), output, out_buffer, zero_coord, true);

    // Golden: relu(relu(input)) per bf16 element. relu is idempotent and is
    // identity for non-negative inputs; for negative inputs, both relus clamp
    // to 0. So golden = max(0, input) per element.
    std::vector<uint32_t> golden(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        auto [lo_bf, hi_bf] = unpack_two_bfloat16_from_uint32(input[i]);
        const float lo_out = std::fmax(0.0f, static_cast<float>(lo_bf));
        const float hi_out = std::fmax(0.0f, static_cast<float>(hi_bf));
        golden[i] = pack_two_bfloat16_into_uint32({bfloat16(lo_out), bfloat16(hi_out)});
    }

    auto compare = [](float a, float b) {
        const float atol = 0.02f;
        const float rtol = 0.05f;
        float maxabs = std::fmax(std::fabs(a), std::fabs(b));
        return std::fabs(a - b) <= atol || std::fabs(a - b) <= rtol * maxabs;
    };
    int argfail = -1;
    bool pass = packed_uint32_t_vector_comparison(output, golden, compare, &argfail);
    EXPECT_TRUE(pass) << "C2 DM→TRISC→intra-selfloop→TRISC→DM (SFPU relu×2) mismatch at position " << argfail;
}

TEST_F(MeshDeviceFixture, C2_DMTriscSelfLoopDM_DoubleRelu) {
    run_c2_dm_trisc_selfloop_dm_program(this->devices_.at(0), /*entry_size=*/2048, /*num_entries=*/16);
}

INSTANTIATE_TEST_SUITE_P(
    ImplicitSync,
    DFBImplicitSyncParamFixture,
    ::testing::Bool(),
    ImplicitSyncParamName);

// Runs an intra-tensix DFB program on one core.
static void run_intra_tensix_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t num_threads) {
    IDevice* device = mesh_device->get_devices()[0];

    experimental::dfb::DataflowBufferConfig dfb_config{
        .entry_size = entry_size,
        .num_entries = num_entries,
        .num_producers = num_threads,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = num_threads,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .tensix_scope = experimental::dfb::TensixScope::INTRA};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    CoreRangeSet core_range_set(CoreRange(logical_core, logical_core));

    const uint32_t words_per_entry = entry_size / sizeof(uint32_t);

    TT_FATAL(
        num_entries % num_threads == 0,
        "num_entries ({}) must be divisible by num_threads ({}) for intra-tensix block partitioning",
        num_entries, num_threads);
    const uint32_t entries_per_neo = num_entries / num_threads;

    const std::string intra_kernel_path = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_intra.cpp";
    std::vector<uint32_t> cta = {entries_per_neo, words_per_entry};

    KernelHandle compute_kernel = CreateKernel(
        program,
        intra_kernel_path,
        core_range_set,
        experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = num_threads, .compile_args = cta});

    auto logical_dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_range_set, dfb_config);
    // Bind the same kernel as both producer and consumer: packer TRISC2 and unpacker TRISC0
    // on each Neo share the Tensix-only TC allocated for that Neo.
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, logical_dfb_id, compute_kernel, compute_kernel);

    const uint32_t total_size = num_entries * entry_size;
    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 100, total_size / sizeof(uint32_t));

    const uint32_t dfb_l1_addr =
        static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));

    detail::WriteToDeviceL1(device, logical_core, dfb_l1_addr, input);

    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    // Packer increments each word by 1, then unpacker increments it by 1 → +2 per word.
    // This holds for every Neo's ring independently, so the entire L1 region is input + 2.
    std::vector<uint32_t> expected(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        expected[i] = input[i] + 2;
    }

    std::vector<uint32_t> l1_data;
    detail::ReadFromDeviceL1(device, logical_core, dfb_l1_addr, total_size, l1_data);
    EXPECT_EQ(expected, l1_data) << "Intra-tensix DFB L1 mismatch";
}

TEST_F(MeshDeviceFixture, TensixIntraTest1xDFB1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping intra-tensix DFB test for WH/BH until DFB is backported";
    }
    run_intra_tensix_dfb_program(this->devices_.at(0), /*entry_size=*/1024, /*num_entries=*/16, /*num_threads=*/1);
}

TEST_F(MeshDeviceFixture, TensixIntraTest1xDFB4Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping intra-tensix DFB test for WH/BH until DFB is backported";
    }
    run_intra_tensix_dfb_program(this->devices_.at(0), /*entry_size=*/1024, /*num_entries=*/16, /*num_threads=*/4);
}

TEST_F(MeshDeviceFixture, TensixIntraAndRemapperTest_4Neo_DM1Sx4A) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping combined intra-tensix + remapper DFB test for WH/BH until DFB is backported";
    }

    IDevice* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core(0, 0);
    CoreRangeSet core_range_set(CoreRange(logical_core, logical_core));

    constexpr uint32_t entry_size  = 1024;
    constexpr uint32_t num_entries = 16;
    constexpr uint32_t num_neos    = 4;
    const uint32_t words_per_entry = entry_size / sizeof(uint32_t);
    const uint32_t entries_per_neo = num_entries;

    // dfb(0): DM->Tensix, 1Sx4A with remapper, implicit sync enabled.
    experimental::dfb::DataflowBufferConfig remapper_dfb_config{
        .entry_size           = entry_size,
        .num_entries          = num_entries,
        .num_producers        = 1,
        .pap                  = dfb::AccessPattern::STRIDED,
        .num_consumers        = num_neos,
        .cap                  = dfb::AccessPattern::ALL,
        .enable_implicit_sync = true};

    // dfb(1): intra-tensix, 4 packer->unpacker pairs, hidden TCs.
    experimental::dfb::DataflowBufferConfig intra_dfb_config{
        .entry_size           = entry_size,
        .num_entries          = num_entries,
        .num_producers        = num_neos,
        .pap                  = dfb::AccessPattern::STRIDED,
        .num_consumers        = num_neos,
        .cap                  = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .tensix_scope         = experimental::dfb::TensixScope::INTRA};

    const uint32_t buf_size = num_entries * entry_size;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    distributed::DeviceLocalBufferConfig local_cfg{.page_size = entry_size, .buffer_type = BufferType::DRAM};
    auto in_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = buf_size}, local_cfg, this->devices_.at(0).get());

    Program program = CreateProgram();

    // DM producer (1 thread, implicit sync enabled): reads DRAM in_buffer -> dfb(0) via NOC.
    std::vector<uint32_t> dm_cta = {
        (uint32_t)in_buffer->address(),
        num_entries,  // num_entries_per_producer: 1 producer owns all entries
        1u,           // implicit_sync = true
        0u};          // consume_all = false (producer is always STRIDED)
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(dm_cta);

    auto dm_producer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
        core_range_set,
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .compile_args = dm_cta});

    // Create DFBs: remapper first (-> logical ID 0), intra-tensix second (-> logical ID 1).
    auto remapper_dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_range_set, remapper_dfb_config);
    auto intra_dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_range_set, intra_dfb_config);

    // Combined compute kernel (4 Neo clusters):
    //   CTA[0] = num_entries      - ALL consumer loop count (each UNPACK sees all 16 entries)
    //   CTA[1] = entries_per_neo  - intra-tensix loop count per Neo (= 4)
    //   CTA[2] = words_per_entry  - words per dfb(1) entry for in-place increment
    std::vector<uint32_t> compute_cta = {num_entries, entries_per_neo, words_per_entry};

    auto compute_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/dfb_intra_and_consume_all.cpp",
        core_range_set,
        experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = num_neos, .compile_args = compute_cta});

    // dfb(0): DM producer -> compute ALL consumer (remapper fans out 1 TC post to 4 UNPACK TCs).
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, remapper_dfb_id, dm_producer_kernel, compute_kernel);
    // dfb(1): compute kernel is both producer (PACK TRISC) and consumer (UNPACK TRISC).
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, intra_dfb_id, compute_kernel, compute_kernel);

    // Runtime args for DM producer (dfb_producer.cpp: [0]=producer_mask, [1]=chunk_offset).
    auto remapper_dfb_impl = program.impl().get_dataflow_buffer(remapper_dfb_id);
    const uint32_t dm_producer_mask = remapper_dfb_impl->config.producer_risc_mask;
    SetRuntimeArgs(program, dm_producer_kernel, logical_core, {dm_producer_mask, 0u /*chunk_offset*/});

    // L1 layout follows DFB creation order:
    //   [l1_base + 0              ] -> dfb(0) remapper ring  (num_entries * entry_size bytes)
    //   [l1_base + remapper_size  ] -> dfb(1) intra ring     (num_entries * entry_size bytes)
    const uint32_t l1_base           = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    const uint32_t remapper_ring_size = num_entries * entry_size;
    const uint32_t intra_l1_addr     = l1_base + remapper_ring_size;

    // Pre-fill dfb(1)'s intra-tensix ring; kernel adds +2 per word.
    auto input_intra = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 100, num_entries * words_per_entry);
    detail::WriteToDeviceL1(device, logical_core, intra_l1_addr, input_intra);

    // Fill DRAM in_buffer; DM NOC-reads this into dfb(0)'s ring.
    auto input_remapper = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 100, num_entries * words_per_entry);
    distributed::WriteShard(this->devices_.at(0)->mesh_command_queue(), in_buffer, input_remapper, zero_coord, true);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    // Verify dfb(1): packer +1, unpacker +1 -> net +2 per word.
    {
        std::vector<uint32_t> expected(input_intra.size());
        for (size_t i = 0; i < input_intra.size(); i++) {
            expected[i] = input_intra[i] + 2;
        }
        std::vector<uint32_t> l1_data;
        detail::ReadFromDeviceL1(device, logical_core, intra_l1_addr, num_entries * entry_size, l1_data);
        EXPECT_EQ(expected, l1_data) << "Intra-tensix DFB L1 mismatch";
    }

    // Verify dfb(0): DM NOC-wrote input_remapper into L1; Tensix consumed but did not overwrite.
    {
        std::vector<uint32_t> l1_data;
        detail::ReadFromDeviceL1(device, logical_core, l1_base, num_entries * entry_size, l1_data);
        EXPECT_EQ(input_remapper, l1_data) << "DM->Tensix strided x all DFB L1 mismatch";
    }
}

// D1: TC `posted` / `acked` wrap-point exposure with implicit sync.
//
// The `posted` and `acked` TC bitfields are uint16; once `posted` reaches
// 65 535 and the ISR fires again, the hardware register rolls 0xFFFF→0x0000.
// We want to catch firmware code that uses a wider shadow accumulator than
// the 16-bit HW register — that bug fires the moment we cross the wrap, not
// over thousands of operations.
//
// Naive approach: push 65 536+ real tiles through NOC. Works on silicon
// (seconds) but takes many hours on the Quasar emulator (each NOC page is
// multi-second on emu).
//
// What this test does instead: launch DM producer + DM consumer kernels that
// each issue a single direct write to their TC's `posted` / `acked` HW
// register, advancing both counters by kPreloadValue (= 65 530) in one
// instruction *before* the normal kernel loop starts. The subsequent kPushTiles
// real tiles then cross the wrap point (counter goes 65 530 → 65 530+kPushTiles,
// wrapping at 65 536). The host's WriteShard / ReadShard only carries
// kPushTiles tiles' worth of data, so DRAM traffic is tiny.
//
// Counter trajectory (kPreloadValue=65528, kPushTiles=32, per_tc=8):
//   start (after preload):   posted = acked = 65 528,  ptxn/ctxn_loop_cnt = 8191
//   tile 8 (just past wrap): posted = acked = 0,       loop_cnt = 8192
//   tile 32 (end):           posted = acked = 24,      loop_cnt = 8195
//
// Catches the same bug class as the naive 65k-tile test, but runs in seconds
// on the emulator. Also requires the modular-comparison fix at
// dataflow_buffer.inl:319/349 (without that fix, prepare_implicit_read /
// prepare_implicit_write spin forever at the first post-wrap loop boundary
// because their absolute-counter check breaks at uint16 rollover).
TEST_F(MeshDeviceFixture, D1_LongImplicitSync_PostCounterWrap) {
    if (this->devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Implicit sync is Quasar-only";
    }

    // 65528 = 8191 * 8 — exact multiple of num_entries_per_txn_id_per_tc (=8 in
    // this config), so the kernel-side ptxn_id_loop_cnt_ lands on a clean
    // boundary (8191) when posted=65528 and increments to 8192 exactly as
    // posted wraps to 0. Picking a value that's not a multiple of per_tc would
    // desync kernel and HW by an inter-batch offset.
    constexpr uint32_t kPreloadValue = 65528;
    constexpr uint32_t kPushTiles = 32;
    constexpr uint32_t kEntrySize = 1024;
    constexpr uint32_t kRingEntries = 16;

    const auto& mesh_device = this->devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];
    Program program = CreateProgram();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    CoreCoord logical_core(0, 0);
    CoreRangeSet core_range_set(CoreRange(logical_core, logical_core));

    experimental::dfb::DataflowBufferConfig config{
        .entry_size = kEntrySize,
        .num_entries = kRingEntries,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = true};
    auto dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_range_set, config);
    ASSERT_EQ(dfb_id, 0u) << "Preload kernels hardcode dfb id 0";

    const uint32_t total_buffer_size = kPushTiles * kEntrySize;
    distributed::DeviceLocalBufferConfig local_buffer_config{.page_size = kEntrySize, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = total_buffer_size};
    auto in_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    auto out_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());

    std::vector<uint32_t> producer_cta = {
        (uint32_t)in_buffer->address(),
        kPushTiles,
        /*implicit_sync=*/1u,
        /*kPreloadPostedValue=*/kPreloadValue};
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(producer_cta);
    auto producer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_with_tc_preload.cpp",
        core_range_set,
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .compile_args = producer_cta});

    std::vector<uint32_t> consumer_cta = {
        (uint32_t)out_buffer->address(),
        kPushTiles,
        /*blocked_consumer=*/0u,
        /*implicit_sync=*/1u,
        /*kPreloadAckedValue=*/kPreloadValue};
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(consumer_cta);
    auto consumer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_with_tc_preload.cpp",
        core_range_set,
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .compile_args = consumer_cta});

    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(program, dfb_id, producer_kernel, consumer_kernel);

    // Bind populates the DFB's INTERNAL config.producer_risc_mask /
    // consumer_risc_mask; the local `config` variable here is a snapshot from
    // before Bind and still has those fields = 0. Always read the masks from
    // the DFB's internal copy via get_dataflow_buffer(id).
    auto dfb_impl = program.impl().get_dataflow_buffer(dfb_id);
    SetRuntimeArgs(
        program,
        producer_kernel,
        logical_core,
        {(uint32_t)dfb_impl->config.producer_risc_mask, /*chunk_offset=*/0u, kPushTiles});
    SetRuntimeArgs(
        program,
        consumer_kernel,
        logical_core,
        {(uint32_t)dfb_impl->config.consumer_risc_mask, (uint32_t)dfb_id, /*chunk_offset=*/0u, kPushTiles});

    auto input = create_random_vector_of_bfloat16(total_buffer_size, 1.0f, 0xD1D1);
    distributed::WriteShard(mesh_device->mesh_command_queue(), in_buffer, input, zero_coord, true);

    // TODO #38042 (mirroring execute_program_and_verify): WriteShard barrier
    // isn't yet uplifted on Quasar. Without this sleep + readback, only tile 0
    // makes it to DRAM before LaunchProgram fires, and the kernel reads zeros
    // for tiles 1-31 — producing exactly the failure pattern (output[256+]=0).
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::vector<uint32_t> rdback_in;
    distributed::ReadShard(mesh_device->mesh_command_queue(), rdback_in, in_buffer, zero_coord, true);
    tt_driver_atomics::mfence();
    ASSERT_EQ(rdback_in, input) << "WriteShard did not complete before LaunchProgram";

    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    std::vector<uint32_t> output;
    distributed::ReadShard(mesh_device->mesh_command_queue(), output, out_buffer, zero_coord, true);

    // On mismatch, find and log the first divergent uint32 + which tile / element
    // it belongs to. Each tile is 1024 B = 256 uint32; helps localize whether the
    // corruption is before-wrap (tile 0-7), at-wrap (tile 8), or after-wrap.
    if (input != output) {
        size_t first_diff = input.size();
        for (size_t i = 0; i < input.size(); ++i) {
            if (input[i] != output[i]) {
                first_diff = i;
                break;
            }
        }
        constexpr size_t kU32PerTile = 1024 / sizeof(uint32_t);
        size_t tile_idx = first_diff / kU32PerTile;
        size_t in_tile_offset = first_diff % kU32PerTile;
        log_info(
            tt::LogTest,
            "D1 mismatch at uint32 index {} (tile {}, element {} of {}). input[{}]=0x{:x} output[{}]=0x{:x}. "
            "Wrap is at tile 8 (after preload=65528 + 8 ISR fires).",
            first_diff,
            tile_idx,
            in_tile_offset,
            kU32PerTile,
            first_diff,
            input[first_diff],
            first_diff,
            output[first_diff]);
    }
    EXPECT_EQ(input, output) << "Identity copy across uint16 TC-counter wrap point failed";
}

// D2: All-DM 6-producer × 2-consumer with ring wraparound. The existing
// DMTensix 6Sx1S / 6Sx2S tests cover 6 DM producers, but only with a Tensix
// consumer and tile count <= ring size (no wraparound under contention). This
// test puts 6 producers and 2 consumers on DMs simultaneously (max concurrent
// DM occupancy on Quasar — 8 DMs per node) and forces ~4 ring wraparounds
// with 96 tiles through a 24-entry ring while all 8 DMs are hammering.
//
// Catches: NOC arbitration drops, TC-allocator clashes across 8 concurrent
// trids, IP-register fired_trids collisions under heavy load.
//
// Split into two TEST_Fs (ImplicitOff / ImplicitOn) rather than a single
// TEST_P with a bool param: TEST_P would run both variants back-to-back in
// the same process, and the in-process device close → reopen path between
// param values hangs on the Quasar emulator (the "cumulative-state flake"
// that tt_test's 20s cooldown is meant to avoid — but the cooldown only
// applies *between* tt_test invocations, not between TEST_P param values).
// With two TEST_Fs, each variant becomes its own tt_test pick and gets a
// fresh binary + cooldown.
//
// Tile count is kept small (96, not 1024) so the DRAM WriteShard / ReadShard
// of the input/output buffers stays under 1 minute on the Quasar emulator
// (each NOC page write is multi-second on emu). Concurrency stress comes from
// 8 DMs running at once, not from volume — 4 wraps is plenty to expose race
// bugs. Bump kTotalTiles when running on silicon for deeper stress.
static void run_d2_all_dms_concurrent(const std::shared_ptr<distributed::MeshDevice>& mesh_device, bool implicit_sync) {
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 24,
        .num_producers = 6,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = implicit_sync};
    constexpr uint32_t kTotalTiles = 96;
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    run_single_dfb_program(mesh_device, config, DFBPorCType::DM, DFBPorCType::DM, core_range_set, kTotalTiles);
}

TEST_F(MeshDeviceFixture, D2_AllDMsConcurrent_6Sx2S_ImplicitOff) {
    if (this->devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "6 DM producers + 2 DM consumers requires Quasar (8 DMs/node)";
    }
    run_d2_all_dms_concurrent(this->devices_.at(0), /*implicit_sync=*/false);
}

TEST_F(MeshDeviceFixture, D2_AllDMsConcurrent_6Sx2S_ImplicitOn) {
    if (this->devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "6 DM producers + 2 DM consumers requires Quasar (8 DMs/node)";
    }
    run_d2_all_dms_concurrent(this->devices_.at(0), /*implicit_sync=*/true);
}

// D3: Heterogeneous per-core HW config — forces a single DFB spanning two cores
// to bin into TWO DfbGroups instead of one.
//
// All existing multi-core DFB tests use identical configs everywhere, so the host
// always produces exactly one DfbGroup per DFB. If `hw_risc_configs_equal()` in
// finalize_single_dfb_config() were to accidentally return true for unequal
// per-core HW configs, no test would notice — every core would silently inherit
// group[0]'s TC/Remapper allocation.
//
// Trick: create a "decoy" DFB on core A only. The decoy consumes a TC slot on
// core A during finalize. Then create a "shared" DFB on cores A and B. The TC
// allocator gives core A's shared producer a slot LATER in the TC pool than
// core B's (because slot 0 on core A is already taken by the decoy), so the
// resulting `packed_tile_counter` differs between the two cores → bin_into_group
// splits them into two groups.
//
// Decoy is created first (becomes dfb id 0) so the standard dfb_consumer.cpp
// can still receive the shared dfb id (1) via runtime arg, and the shared
// producer uses dfb_producer_with_id.cpp (compile-time-configurable id).
TEST_F(MeshDeviceFixture, D3_MultiCoreDFB_TwoGroupsViaDecoy) {
    if (this->devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "TC-based grouping is Quasar-only";
    }

    const auto& mesh_device = this->devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    // The emu-quasar-1x3 simulator has only a single Tensix (logical (0,0)) —
    // this test needs at least two distinct Tensixes to drive divergent TC
    // allocation. Skip on any device with <2 worker cores.
    CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t num_workers = grid.x * grid.y;
    if (num_workers < 2) {
        GTEST_SKIP() << "Need >= 2 Tensix cores; device has " << num_workers
                     << " (single-Tensix emulator?). Run on silicon or a multi-Tensix sim.";
    }
    Program program = CreateProgram();
    auto zero_coord = distributed::MeshCoordinate(0, 0);

    // Two adjacent Tensixes — same convention as existing MultiCoreDMTest2Core_*.
    CoreCoord core_a(0, 0);
    CoreCoord core_b(1, 0);
    CoreRangeSet decoy_range(CoreRange(core_a, core_a));
    CoreRangeSet shared_range =
        CoreRangeSet(std::vector<CoreRange>{CoreRange(core_a, core_a), CoreRange(core_b, core_b)});

    // --- DFB-0: decoy on core A only (no kernel bindings; just claims a TC slot). ---
    // Without kernel binding the masks would stay 0 and finalize would reject the DFB,
    // so set them explicitly to DM0 producer / DM1 consumer.
    experimental::dfb::DataflowBufferConfig decoy_config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x2,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false};
    [[maybe_unused]] auto decoy_dfb_id = experimental::dfb::CreateDataflowBuffer(program, decoy_range, decoy_config);

    // --- DFB-1: shared across A and B (real producer/consumer kernels). ---
    experimental::dfb::DataflowBufferConfig shared_config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false};
    auto shared_dfb_id = experimental::dfb::CreateDataflowBuffer(program, shared_range, shared_config);

    // --- DRAM buffers: one slice per core. ---
    const uint32_t entries_per_core = 16;
    const uint32_t entry_size = shared_config.entry_size;
    const uint32_t total_buffer_size = 2 * entries_per_core * entry_size;
    distributed::DeviceLocalBufferConfig local_buffer_config{.page_size = entry_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = total_buffer_size};
    auto in_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    auto out_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());

    // --- Producer kernel: dfb_producer_with_id.cpp (uses shared_dfb_id, not the hardcoded 0). ---
    std::vector<uint32_t> producer_cta = {
        (uint32_t)in_buffer->address(),
        entries_per_core,
        /*implicit_sync=*/0u,
        (uint32_t)shared_dfb_id};
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(producer_cta);
    auto producer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_with_id.cpp",
        shared_range,
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .compile_args = producer_cta});

    // --- Consumer kernel: standard dfb_consumer.cpp (takes id via RTA). ---
    std::vector<uint32_t> consumer_cta = {
        (uint32_t)out_buffer->address(),
        entries_per_core,
        /*blocked_consumer=*/0u,
        /*implicit_sync=*/0u};
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(consumer_cta);
    auto consumer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp",
        shared_range,
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .compile_args = consumer_cta});

    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, shared_dfb_id, producer_kernel, consumer_kernel);

    // --- RTAs: each core gets a distinct chunk_offset so the two cores read/write
    //     disjoint slices of the global DRAM buffers.
    //     Producer/consumer masks must come from the DFB's INTERNAL config (set
    //     by Bind), not from the local `shared_config` variable (snapshotted
    //     before Bind and still 0). ---
    auto shared_dfb_impl = program.impl().get_dataflow_buffer(shared_dfb_id);
    const uint32_t shared_prod_mask = shared_dfb_impl->config.producer_risc_mask;
    const uint32_t shared_cons_mask = shared_dfb_impl->config.consumer_risc_mask;
    SetRuntimeArgs(program, producer_kernel, core_a, {shared_prod_mask, /*chunk_offset=*/0u, entries_per_core});
    SetRuntimeArgs(
        program, producer_kernel, core_b, {shared_prod_mask, /*chunk_offset=*/entries_per_core, entries_per_core});
    SetRuntimeArgs(
        program,
        consumer_kernel,
        core_a,
        {shared_cons_mask, (uint32_t)shared_dfb_id, /*chunk_offset=*/0u, entries_per_core});
    SetRuntimeArgs(
        program,
        consumer_kernel,
        core_b,
        {shared_cons_mask, (uint32_t)shared_dfb_id, /*chunk_offset=*/entries_per_core, entries_per_core});

    auto input = create_constant_vector_of_bfloat16(total_buffer_size, 1.0f);
    distributed::WriteShard(mesh_device->mesh_command_queue(), in_buffer, input, zero_coord, true);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    // --- Assertion 1 (the point of the test): the shared DFB binned into 2 groups. ---
    ASSERT_NE(shared_dfb_impl, nullptr);
    EXPECT_EQ(shared_dfb_impl->groups.size(), 2u)
        << "Expected the shared DFB to bin into 2 groups (one per core) because the decoy on core A "
        << "shifts the TC slot allocation; saw groups.size()=" << shared_dfb_impl->groups.size();

    // Each group should cover exactly one core.
    if (shared_dfb_impl->groups.size() == 2u) {
        for (const auto& group : shared_dfb_impl->groups) {
            EXPECT_EQ(group.core_ranges.num_cores(), 1u) << "Each group should contain exactly one core";
        }
    }

    // --- Assertion 2: end-to-end correctness on both cores. ---
    std::vector<uint32_t> output;
    distributed::ReadShard(mesh_device->mesh_command_queue(), output, out_buffer, zero_coord, true);
    EXPECT_EQ(input, output) << "Heterogeneous-group pipeline must still produce identity output";
}

}  // end namespace tt::tt_metal
