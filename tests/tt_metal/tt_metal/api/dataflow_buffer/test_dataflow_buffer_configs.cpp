// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
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
#include "tt_metal/impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"

namespace tt::tt_metal {

// These tests create a DFB and validate that the tile counter and remapper config is correct

// Validation structure for DFB tile counter configuration
struct DFBTileCounterExpectation {
    uint8_t expected_producer_tc_count;  // TCs per producer
    uint8_t expected_consumer_tc_count;  // TCs per consumer

    // Map of producer risc_id -> vector of (consumer risc_id, producer_tc_slot, consumer_tc_slot)
    std::map<uint8_t, std::vector<std::tuple<uint8_t, uint8_t, uint8_t>>> producer_to_consumer_pairings;
};

// Validates DFB tile counter configuration against expected pairings
void validate_dfb_tile_counters(
    Program& program,
    const CoreCoord& logical_core,
    const experimental::dfb::DataflowBufferConfig& config,
    const DFBTileCounterExpectation& expectation) {
    auto dfbs = program.impl().dataflow_buffers_on_core(logical_core);
    ASSERT_EQ(dfbs.size(), 1) << "Expected exactly 1 DFB on core";

    const auto& dfb = dfbs[0];

    ASSERT_EQ(dfb->risc_mask, config.producer_risc_mask | config.consumer_risc_mask);
    ASSERT_FALSE(dfb->groups.empty()) << "DFB has no groups (configs not finalized?)";
    // All single-core tests produce exactly one DfbGroup.
    const auto& hw_risc_configs = dfb->groups[0].hw_risc_configs;
    ASSERT_EQ(hw_risc_configs.size(), config.num_producers + config.num_consumers);

    // risc ID to risc config maps
    std::map<uint8_t, const experimental::dfb::detail::DFBRiscConfig*> producer_configs;
    std::map<uint8_t, const experimental::dfb::detail::DFBRiscConfig*> consumer_configs;

    for (const auto& rc : hw_risc_configs) {
        if (rc.is_producer) {
            producer_configs[rc.risc_id] = &rc;
        } else {
            consumer_configs[rc.risc_id] = &rc;
        }
    }

    for (const auto& [risc_id, rc] : producer_configs) {
        EXPECT_EQ(rc->config.num_tcs_to_rr, expectation.expected_producer_tc_count)
            << "Producer RISC " << (int)risc_id << " has wrong TC count";
    }

    for (const auto& [risc_id, rc] : consumer_configs) {
        EXPECT_EQ(rc->config.num_tcs_to_rr, expectation.expected_consumer_tc_count)
            << "Consumer RISC " << (int)risc_id << " has wrong TC count";
    }

    // Validate TC tensix_id for Tensix RISCs
    // Key constraint: Tensix RISCs can only access TCs from their own tensix_id
    for (const auto& [risc_id, rc] : producer_configs) {
        bool is_tensix_risc = risc_id >= 8;
        if (is_tensix_risc) {
            uint8_t expected_tensix_id = (risc_id - 8) % 4;
            for (uint8_t tc = 0; tc < rc->config.num_tcs_to_rr; tc++) {
                auto ptc = rc->config.packed_tile_counter[tc];
                uint8_t actual_tensix_id = ::dfb::get_tensix_id(ptc);
                EXPECT_EQ(actual_tensix_id, expected_tensix_id)
                    << "Tensix producer RISC " << (int)risc_id << " TC[" << (int)tc
                    << "] must use tensix_id=" << (int)expected_tensix_id << " but has " << (int)actual_tensix_id;
            }
        }
    }

    for (const auto& [risc_id, rc] : consumer_configs) {
        bool is_tensix_risc = risc_id >= 8;
        if (is_tensix_risc) {
            uint8_t expected_tensix_id = (risc_id - 8) % 4;
            for (uint8_t tc = 0; tc < rc->config.num_tcs_to_rr; tc++) {
                auto ptc = rc->config.packed_tile_counter[tc];
                uint8_t actual_tensix_id = ::dfb::get_tensix_id(ptc);
                EXPECT_EQ(actual_tensix_id, expected_tensix_id)
                    << "Tensix consumer RISC " << (int)risc_id << " TC[" << (int)tc
                    << "] must use tensix_id=" << (int)expected_tensix_id << " but has " << (int)actual_tensix_id;
            }
        }
    }

    // For ALL mode, validate remapper pair indices
    if (config.cap == dfb::AccessPattern::ALL) {
        std::set<uint8_t> seen_remapper_indices;
        for (const auto& [risc_id, rc] : producer_configs) {
            uint8_t remapper_idx = rc->config.remapper_pair_index;

            // Check valid range (0-63)
            EXPECT_LT(remapper_idx, 64) << "ALL: Producer RISC " << (int)risc_id
                                        << " has invalid remapper_pair_index " << (int)remapper_idx
                                        << " (must be 0-63)";

            // Check uniqueness among producers
            EXPECT_EQ(seen_remapper_indices.count(remapper_idx), 0)
                << "ALL: Producer RISC " << (int)risc_id << " has duplicate remapper_pair_index "
                << (int)remapper_idx;
            seen_remapper_indices.insert(remapper_idx);

            log_info(tt::LogTest, "ALL: Producer {} has remapper_pair_index {}", risc_id, remapper_idx);
        }
    }

    for (const auto& [producer_risc_id, pairings] : expectation.producer_to_consumer_pairings) {
        auto producer_it = producer_configs.find(producer_risc_id);
        ASSERT_NE(producer_it, producer_configs.end());

        const auto* producer_rc = producer_it->second;

        // For ALL mode, accumulate expected_consumer_tcs across all pairings for this producer
        uint32_t expected_consumer_tcs = 0;
        size_t consumer_idx = 0;

        for (const auto& [consumer_risc_id, producer_tc_slot, consumer_tc_slot] : pairings) {
            auto consumer_it = consumer_configs.find(consumer_risc_id);
            ASSERT_NE(consumer_it, consumer_configs.end());

            const auto* consumer_rc = consumer_it->second;

            ASSERT_LT(producer_tc_slot, 4) << "Max of 4 TCs allowed per producer";
            ASSERT_LT(consumer_tc_slot, 4) << "Max of 4 TCs allowed per consumer";

            auto producer_ptc = producer_rc->config.packed_tile_counter[producer_tc_slot];
            auto consumer_ptc = consumer_rc->config.packed_tile_counter[consumer_tc_slot];

            if (config.cap == dfb::AccessPattern::ALL) {
                // For ALL mode, consumer TCs are different from producer TC (remapper-based)
                // Accumulate the consumer TC IDs into expected_consumer_tcs
                if (consumer_idx < 4) {
                    uint8_t consumer_tc_id = ::dfb::get_counter_id(consumer_ptc);
                    expected_consumer_tcs |= (consumer_tc_id & 0x1F) << (consumer_idx * 5);
                    consumer_idx++;
                }

                log_info(
                    tt::LogTest,
                    "ALL: Producer {} TC[{}]=(tensix:{}, tc:{}) -> Consumer {} TC[{}]=(tensix:{}, tc:{})",
                    producer_risc_id,
                    producer_tc_slot,
                    ::dfb::get_tensix_id(producer_ptc),
                    ::dfb::get_counter_id(producer_ptc),
                    consumer_risc_id,
                    consumer_tc_slot,
                    ::dfb::get_tensix_id(consumer_ptc),
                    ::dfb::get_counter_id(consumer_ptc));
            } else {
                // For STRIDED mode, producer and consumer should share the exact same TC
                EXPECT_EQ(producer_ptc, consumer_ptc)
                    << "STRIDED: Producer " << (int)producer_risc_id << " TC[" << (int)producer_tc_slot
                    << "] should share TC with Consumer " << (int)consumer_risc_id << " TC[" << (int)consumer_tc_slot
                    << "]. Producer has (tensix:" << (int)::dfb::get_tensix_id(producer_ptc)
                    << ", tc:" << (int)::dfb::get_counter_id(producer_ptc)
                    << "), Consumer has (tensix:" << (int)::dfb::get_tensix_id(consumer_ptc)
                    << ", tc:" << (int)::dfb::get_counter_id(consumer_ptc) << ")";

                log_info(
                    tt::LogTest,
                    "STRIDED: Producer {} TC[{}] and Consumer {} TC[{}] share (tensix:{}, tc:{})",
                    producer_risc_id,
                    producer_tc_slot,
                    consumer_risc_id,
                    consumer_tc_slot,
                    ::dfb::get_tensix_id(producer_ptc),
                    ::dfb::get_counter_id(producer_ptc));
            }
        }

        if (config.cap == dfb::AccessPattern::ALL) {
            uint32_t actual_consumer_tcs = producer_rc->config.consumer_tcs;
            ASSERT_EQ(actual_consumer_tcs, expected_consumer_tcs)
                << "ALL: Producer " << (int)producer_risc_id << " consumer_tcs mismatch. "
                << "Expected: 0x" << std::hex << expected_consumer_tcs << ", Actual: 0x" << actual_consumer_tcs
                << std::dec;

            log_info(
                tt::LogTest,
                "ALL: Producer {} consumer_tcs validated: 0x{:x}",
                producer_risc_id,
                actual_consumer_tcs);
        }
    }
}

TEST_F(MeshDeviceFixture, DMTensixTest1xDFB1Sx1SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x10,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // 1 producer with 1 consumer -> 1 TC per producer
        .expected_consumer_tc_count = 1,  // 1 consumer with 1 producer -> 1 TC per consumer
        .producer_to_consumer_pairings = {
            {0, {{0, 0, 0}}},  // Producer 0 TC[0] pairs with Consumer risc 0 TC[0]
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB1Sx4SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x1E,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 4,  // 1 producer with 4 consumers -> 4 TCs per producer
        .expected_consumer_tc_count = 1,  // Each consumer pairs with 1 producer -> 1 TC per consumer
        .producer_to_consumer_pairings = {
            {0,
             {
                 {1, 0, 0},  // Producer 0 TC[0] pairs with Consumer risc 1 TC[0]
                 {2, 1, 0},  // Producer 0 TC[1] pairs with Consumer risc 2 TC[0]
                 {3, 2, 0},  // Producer 0 TC[2] pairs with Consumer risc 3 TC[0]
                 {4, 3, 0},  // Producer 0 TC[3] pairs with Consumer risc 4 TC[0]
             }}}};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTensixTest1xDFB4Sx1SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x10,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // Each producer pairs with 1 consumer -> 1 TC per producer
        .expected_consumer_tc_count = 4,  // 1 consumer with 4 producers -> 4 TCs per consumer
        .producer_to_consumer_pairings = {
            {0, {{4, 0, 0}}},  // Producer 0 TC[0] pairs with Consumer risc 4 TC[0]
            {1, {{4, 0, 1}}},  // Producer 1 TC[0] pairs with Consumer risc 4 TC[1]
            {2, {{4, 0, 2}}},  // Producer 2 TC[0] pairs with Consumer risc 4 TC[2]
            {3, {{4, 0, 3}}},  // Producer 3 TC[0] pairs with Consumer risc 4 TC[3]
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx1SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x10,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // Each producer pairs with 1 consumer -> 1 TC per producer
        .expected_consumer_tc_count = 4,  // 1 consumer with 4 producers -> 4 TCs per consumer
        .producer_to_consumer_pairings = {
            {0, {{4, 0, 0}}},  // Producer 0 TC[0] pairs with Consumer risc 4 TC[0]
            {1, {{4, 0, 1}}},  // Producer 1 TC[0] pairs with Consumer risc 4 TC[1]
            {2, {{4, 0, 2}}},  // Producer 2 TC[0] pairs with Consumer risc 4 TC[2]
            {3, {{4, 0, 3}}},  // Producer 3 TC[0] pairs with Consumer risc 4 TC[3]
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx4SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0xF0,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // Equal producers and consumers -> 1 TC per producer
        .expected_consumer_tc_count = 1,  // Equal producers and consumers -> 1 TC per consumer
        .producer_to_consumer_pairings = {
            {0, {{4, 0, 0}}},  // Producer 0 TC[0] pairs with Consumer risc 4 TC[0]
            {1, {{5, 0, 0}}},  // Producer 1 TC[0] pairs with Consumer risc 5 TC[0]
            {2, {{6, 0, 0}}},  // Producer 2 TC[0] pairs with Consumer risc 6 TC[0]
            {3, {{7, 0, 0}}},  // Producer 3 TC[0] pairs with Consumer risc 7 TC[0]
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB2Sx4SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x3,
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x3C,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 2,  // 4 consumers / 2 producers = 2 TCs per producer
        .expected_consumer_tc_count = 1,  // 2 producers / 4 consumers = 1 TC per consumer (min 1)
        .producer_to_consumer_pairings = {
            {0,
             {
                 {2, 0, 0},  // Producer 0 TC[0] pairs with Consumer risc 2 TC[0]
                 {4, 1, 0},  // Producer 0 TC[1] pairs with Consumer risc 4 TC[0]
             }},
            {1,
             {
                 {3, 0, 0},  // Producer 1 TC[0] pairs with Consumer risc 3 TC[0]
                 {5, 1, 0},  // Producer 1 TC[1] pairs with Consumer risc 5 TC[0]
             }},
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx2SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x30,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // 2 consumers / 4 producers = 1 TC per producer (min 1)
        .expected_consumer_tc_count = 2,  // 4 producers / 2 consumers = 2 TCs per consumer
        .producer_to_consumer_pairings = {
            {0, {{4, 0, 0}}},  // Producer 0 TC[0] pairs with Consumer risc 4 TC[0]
            {1, {{5, 0, 0}}},  // Producer 1 TC[0] pairs with Consumer risc 5 TC[0]
            {2, {{4, 0, 1}}},  // Producer 2 TC[0] pairs with Consumer risc 4 TC[1]
            {3, {{5, 0, 1}}},  // Producer 3 TC[0] pairs with Consumer risc 5 TC[1]
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB1Sx1BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x2,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::ALL,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // ALL: each producer has 1 TC
        .expected_consumer_tc_count = 1,  // ALL: each consumer has num_producers TCs = 1
        .producer_to_consumer_pairings = {
            {0, {{1, 0, 0}}},  // Producer 0 TC[0] maps to Consumer risc 1 TC[0] via remapper
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB1Sx4BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x1E,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::ALL,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // ALL: each producer has 1 TC
        .expected_consumer_tc_count = 1,  // ALL: each consumer has num_producers TCs = 1
        .producer_to_consumer_pairings = {
            {0,
             {
                 {1, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 1 TC[0] via remapper
                 {2, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 2 TC[0] via remapper
                 {3, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 3 TC[0] via remapper
                 {4, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 4 TC[0] via remapper
             }}}};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx1BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x10,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::ALL,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // ALL: each producer has 1 TC
        .expected_consumer_tc_count = 4,  // ALL: each consumer has num_producers TCs = 4
        .producer_to_consumer_pairings = {
            {0, {{4, 0, 0}}},  // Producer 0 TC[0] maps to Consumer risc 4 TC[0] via remapper
            {1, {{4, 0, 1}}},  // Producer 1 TC[0] maps to Consumer risc 4 TC[1] via remapper
            {2, {{4, 0, 2}}},  // Producer 2 TC[0] maps to Consumer risc 4 TC[2] via remapper
            {3, {{4, 0, 3}}},  // Producer 3 TC[0] maps to Consumer risc 4 TC[3] via remapper
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx4BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0xF0,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::ALL,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // ALL: each producer has 1 TC
        .expected_consumer_tc_count = 4,  // ALL: each consumer has num_producers TCs = 4
        .producer_to_consumer_pairings = {
            {0,
             {
                 {4, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 4 TC[0]
                 {5, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 5 TC[0]
                 {6, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 6 TC[0]
                 {7, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 7 TC[0]
             }},
            {1,
             {
                 {4, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 4 TC[1]
                 {5, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 5 TC[1]
                 {6, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 6 TC[1]
                 {7, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 7 TC[1]
             }},
            {2,
             {
                 {4, 0, 2},  // Producer 2 TC[0] maps to Consumer risc 4 TC[2]
                 {5, 0, 2},  // Producer 2 TC[0] maps to Consumer risc 5 TC[2]
                 {6, 0, 2},  // Producer 2 TC[0] maps to Consumer risc 6 TC[2]
                 {7, 0, 2},  // Producer 2 TC[0] maps to Consumer risc 7 TC[2]
             }},
            {3,
             {
                 {4, 0, 3},  // Producer 3 TC[0] maps to Consumer risc 4 TC[3]
                 {5, 0, 3},  // Producer 3 TC[0] maps to Consumer risc 5 TC[3]
                 {6, 0, 3},  // Producer 3 TC[0] maps to Consumer risc 6 TC[3]
                 {7, 0, 3},  // Producer 3 TC[0] maps to Consumer risc 7 TC[3]
             }},
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx2BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x30,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::ALL,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // ALL: each producer has 1 TC
        .expected_consumer_tc_count = 4,  // ALL: each consumer has num_producers TCs = 4
        .producer_to_consumer_pairings = {
            {0,
             {
                 {4, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 4 TC[0]
                 {5, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 5 TC[0]
             }},
            {1,
             {
                 {4, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 4 TC[1]
                 {5, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 5 TC[1]
             }},
            {2,
             {
                 {4, 0, 2},  // Producer 2 TC[0] maps to Consumer risc 4 TC[2]
                 {5, 0, 2},  // Producer 2 TC[0] maps to Consumer risc 5 TC[2]
             }},
            {3,
             {
                 {4, 0, 3},  // Producer 3 TC[0] maps to Consumer risc 4 TC[3]
                 {5, 0, 3},  // Producer 3 TC[0] maps to Consumer risc 5 TC[3]
             }},
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

// 2S x 4B: 2 producers (riscs 0,1) with 4 blocked consumers (riscs 2,3,4,5)
// Each producer has 1 TC, each consumer has 2 TCs (num_producers TCs)
// ALL: Each consumer's TC[i] pairs with producer[i]
TEST_F(MeshDeviceFixture, DMTest1xDFB2Sx4BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x3,
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x3C,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::ALL,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    // consumer_risc_mask 0x3C = riscs 2,3,4,5
    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // ALL: each producer has 1 TC
        .expected_consumer_tc_count = 2,  // ALL: each consumer has num_producers TCs = 2
        .producer_to_consumer_pairings = {
            {0,
             {
                 {2, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 2 TC[0]
                 {3, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 3 TC[0]
                 {4, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 4 TC[0]
                 {5, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 5 TC[0]
             }},
            {1,
             {
                 {2, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 2 TC[1]
                 {3, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 3 TC[1]
                 {4, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 4 TC[1]
                 {5, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 5 TC[1]
             }},
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

// ---------------------------------------------------------------------------
// Multi-core DFB tests
// ---------------------------------------------------------------------------

// Helper: validate that a multi-core DFB has the expected number of DfbGroups
// and that each core's hw_risc_configs matches a single reference group
// (i.e. all cores have identical TC/remapper assignments).
static void validate_multicore_dfb_groups(
    Program& program,
    const CoreRangeSet& core_range_set,
    uint32_t expected_num_groups,
    uint32_t expected_cores_per_group) {
    // Collect DFBs from the first core; they should all be on every core.
    CoreCoord first_core = core_range_set.ranges()[0].start_coord;
    auto dfbs = program.impl().dataflow_buffers_on_core(first_core);
    ASSERT_EQ(dfbs.size(), 1) << "Expected exactly 1 DFB on core";
    const auto& dfb = dfbs[0];

    ASSERT_EQ(dfb->groups.size(), expected_num_groups)
        << "Expected " << expected_num_groups << " DfbGroup(s)";

    for (const auto& grp : dfb->groups) {
        EXPECT_EQ(grp.l1_by_core.size(), expected_cores_per_group)
            << "DfbGroup should have " << expected_cores_per_group << " core(s)";
    }

    // All cores in the core_range_set should appear somewhere in l1_by_core.
    std::set<CoreCoord> accounted_cores;
    for (const auto& grp : dfb->groups) {
        for (const auto& [c, _] : grp.l1_by_core) {
            accounted_cores.insert(c);
        }
    }
    for (const CoreRange& cr : core_range_set.ranges()) {
        for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
            for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
                EXPECT_EQ(accounted_cores.count(CoreCoord(x, y)), 1u)
                    << "Core (" << x << "," << y << ") not found in any DfbGroup";
            }
        }
    }
}

// Multi-core DFB, no implicit sync: 2 cores, 1 producer, 1 consumer, STRIDED.
// Expected: 1 DfbGroup (homogeneous HW config) with 2 cores.
TEST_F(MeshDeviceFixture, MultiCoreDFB_1P1C_Strided_NoImplicitSync) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x2,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));  // 2 cores: (0,0) and (1,0)
    experimental::dfb::CreateDataflowBuffer(program, core_range_set, config);

    // Finalize configs explicitly (normally done during compile/ConfigureDeviceWithProgram).
    program.impl().finalize_dataflow_buffer_configs();

    // Both cores have identical TC config → 1 group with 2 cores.
    validate_multicore_dfb_groups(program, core_range_set, /*expected_num_groups=*/1, /*expected_cores_per_group=*/2);

    // Each core should have TC index 0 (independent per-core allocator starting from 0).
    for (const CoreRange& cr : core_range_set.ranges()) {
        for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
            for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
                CoreCoord core(x, y);
                auto dfbs = program.impl().dataflow_buffers_on_core(core);
                ASSERT_EQ(dfbs.size(), 1);
                const auto& dfb = dfbs[0];
                // Find this core's group
                const experimental::dfb::detail::DfbGroup* found_grp = nullptr;
                for (const auto& grp : dfb->groups) {
                    for (const auto& [c, _] : grp.l1_by_core) {
                        if (c == core) { found_grp = &grp; break; }
                    }
                    if (found_grp) {
                        break;
                    }
                }
                ASSERT_NE(found_grp, nullptr) << "Core (" << x << "," << y << ") not found in any DfbGroup";

                // Validate TC index is 0 (first allocation from fresh per-core allocator).
                for (const auto& rc : found_grp->hw_risc_configs) {
                    for (uint8_t tc = 0; tc < rc.config.num_tcs_to_rr; tc++) {
                        auto ptc = rc.config.packed_tile_counter[tc];
                        EXPECT_EQ(::dfb::get_counter_id(ptc), tc)
                            << "Core (" << x << "," << y << ") RISC " << (int)rc.risc_id
                            << " TC[" << (int)tc << "] should have counter_id=" << (int)tc;
                    }
                }
            }
        }
    }
}

// Multi-core DFB, with implicit sync: 2 cores, 1 producer, 1 consumer, STRIDED.
// Txn IDs should be allocated once (core-invariant) and identical across cores.
TEST_F(MeshDeviceFixture, MultiCoreDFB_1P1C_Strided_ImplicitSync) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x2,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = true};

    Program program = CreateProgram();
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));  // 2 cores
    experimental::dfb::CreateDataflowBuffer(program, core_range_set, config);

    program.impl().finalize_dataflow_buffer_configs();

    // Should still produce 1 group (identical HW config on both cores).
    validate_multicore_dfb_groups(program, core_range_set, /*expected_num_groups=*/1, /*expected_cores_per_group=*/2);

    // Txn ID descriptors are core-invariant: allocated once during finalization.
    CoreCoord first_core(0, 0);
    auto dfbs = program.impl().dataflow_buffers_on_core(first_core);
    ASSERT_EQ(dfbs.size(), 1);
    const auto& dfb = dfbs[0];

    EXPECT_EQ(dfb->producer_txn_descriptor.num_txn_ids, 2u)
        << "Expected 2 producer txn IDs (double-buffering)";
    EXPECT_EQ(dfb->consumer_txn_descriptor.num_txn_ids, 2u)
        << "Expected 2 consumer txn IDs (double-buffering)";
}

// Identical-config multi-core: assert one DfbGroup is produced (multicast-ready).
TEST_F(MeshDeviceFixture, MultiCoreDFB_HomogeneousGrid_SingleGroup) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 512,
        .num_entries = 8,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x2,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    // 4 cores in a 2x2 grid — all identical config → should produce 1 DfbGroup.
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 1)));
    experimental::dfb::CreateDataflowBuffer(program, core_range_set, config);

    program.impl().finalize_dataflow_buffer_configs();

    // All 4 cores have the same HW config → 1 DfbGroup with 4 cores.
    validate_multicore_dfb_groups(program, core_range_set, /*expected_num_groups=*/1, /*expected_cores_per_group=*/4);
}

// ---------------------------------------------------------------------------
// Intra-tensix DFB config test
// ---------------------------------------------------------------------------

// Validates an intra-tensix DFB (pack TRISC producer → unpack TRISC consumer, same Neo):
//   - Exactly one per-risc config entry (shared Neo bit) marked is_producer=true.
//   - The tensix-only TC (id ≥ TC_TENSIX_POOL_START) is assigned to Neo tensix_id derived from producer_risc_mask.
void validate_intra_tensix_dfb(
    Program& program,
    const CoreCoord& logical_core,
    const experimental::dfb::DataflowBufferConfig& config) {
    program.impl().finalize_dataflow_buffer_configs();

    auto dfbs = program.impl().dataflow_buffers_on_core(logical_core);
    ASSERT_EQ(dfbs.size(), 1u) << "Expected exactly 1 DFB on core";
    const auto& dfb = dfbs[0];

    ASSERT_EQ(dfb->risc_mask, config.producer_risc_mask)
        << "Intra-tensix risc_mask should equal producer_risc_mask (same Neo bit)";
    ASSERT_FALSE(dfb->use_remapper) << "Intra-tensix DFB must not use the remapper";
    ASSERT_FALSE(dfb->groups.empty()) << "DFB has no groups (configs not finalized?)";

    const auto& hw_risc_configs = dfb->groups[0].hw_risc_configs;
    ASSERT_EQ(hw_risc_configs.size(), 1u)
        << "Intra-tensix DFB should have exactly 1 per-risc config entry (shared Neo)";

    const auto& rc = hw_risc_configs[0];
    EXPECT_TRUE(rc.is_producer) << "Intra-tensix per-risc entry must be marked is_producer (pack TRISC inits TC)";

    uint8_t expected_tensix_id =
        static_cast<uint8_t>(__builtin_ctz(config.producer_risc_mask >> ::dfb::TENSIX_RISC_OFFSET));
    uint8_t expected_risc_id = static_cast<uint8_t>(::dfb::TENSIX_RISC_OFFSET + expected_tensix_id);
    EXPECT_EQ(rc.risc_id, expected_risc_id)
        << "Intra-tensix per-risc risc_id should match Neo bit in producer_risc_mask";

    ASSERT_EQ(rc.config.num_tcs_to_rr, 1u) << "Intra-tensix DFB should have exactly 1 TC";
    uint8_t tc_id = ::dfb::get_counter_id(rc.config.packed_tile_counter[0]);
    uint8_t actual_tensix_id = ::dfb::get_tensix_id(rc.config.packed_tile_counter[0]);
    EXPECT_EQ(actual_tensix_id, expected_tensix_id) << "TC tensix_id must match Neo";
    EXPECT_GE(tc_id, ::dfb::TC_TENSIX_POOL_START)
        << "Intra-tensix DFB must use a Tensix-only TC (id ≥ " << (int)::dfb::TC_TENSIX_POOL_START << ")";

    log_info(
        tt::LogTest,
        "Intra-tensix DFB: Neo{} Tensix-only TC (tensix_id={}, tc_id={})",
        expected_tensix_id, (int)actual_tensix_id, (int)tc_id);
}

TEST_F(MeshDeviceFixture, TensixIntraTest1xDFB1Sx1SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    // Intra-tensix: pack TRISC (producer) → unpack TRISC (consumer) on Neo0.
    // producer_risc_mask == consumer_risc_mask == bit 8 (Neo0).
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 4,
        .producer_risc_mask = 0x100,  // bit 8 = Neo0
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x100,  // bit 8 = Neo0 (same as producer — intentional for INTRA)
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .tensix_scope = experimental::dfb::TensixScope::INTRA};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    validate_intra_tensix_dfb(program, logical_core, config);
}

// ===========================================================================
// B-batch regression tests (host-side bug fixes shipped on this branch)
// ===========================================================================

// B2: Transaction-ID allocator boundaries
// ---------------------------------------------------------------------------
// Pins compute_optimal_txn_id_count() at tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:253.
// This branch replaced a hardcoded `TXN_IDS_PER_SIDE = 2` with a dynamic
// search that picks the smallest n in [2, NUM_TXN_IDS] such that
//   num_entries % (n * num_prods_or_cons * num_tcs_per_risc) == 0
// and falls back to 1 when no n in that range divides cleanly.
//
// For 1Sx1S configs with STRIDED access pattern, the divisor reduces to n,
// so the chosen n depends only on num_entries:
//   * even num_entries  → returns 2 (smallest match)
//   * odd, div by 3     → returns 3 (n=2 fails, n=3 wins)
//   * neither           → returns 1 (fallback)
//
// num_txn_ids lands in dfb->producer_txn_descriptor.num_txn_ids after finalize.
//
// All three cases run inside ONE TEST_F so the MeshDeviceFixture spawns aether
// (Quasar) only once for the whole suite instead of once per case — what the
// "host-only" computation needs is just a Program object, not a launched
// kernel, but the fixture still opens a device which on Quasar = aether spawn.

TEST_F(MeshDeviceFixture, B2_TxnIdAllocator_Boundaries_Config) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Implicit sync (and therefore the txn-id allocator) is Quasar-only";
    }
    struct Case {
        uint16_t num_entries;
        uint8_t expected_num_txn_ids;
        const char* rationale;
    };
    const Case cases[] = {
        {16, 2, "num_entries=16 → 16%2==0, smallest n in [2,4]"},
        {15, 3, "num_entries=15 → 15%2=1 (skip), 15%3=0 → pick n=3"},
        {7, 1, "num_entries=7 → no n in [2,4] divides cleanly → fallback 1"},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(
            ::testing::Message() << "case num_entries=" << c.num_entries << " expected=" << (int)c.expected_num_txn_ids
                                 << " (" << c.rationale << ")");
        experimental::dfb::DataflowBufferConfig config{
            .entry_size = 1024,
            .num_entries = c.num_entries,
            .producer_risc_mask = 0x1,
            .num_producers = 1,
            .pap = dfb::AccessPattern::STRIDED,
            .consumer_risc_mask = 0x2,
            .num_consumers = 1,
            .cap = dfb::AccessPattern::STRIDED,
            .enable_implicit_sync = true};
        Program program = CreateProgram();
        experimental::dfb::CreateDataflowBuffer(program, CoreCoord(0, 0), config);
        program.impl().finalize_dataflow_buffer_configs();

        auto dfbs = program.impl().dataflow_buffers_on_core(CoreCoord(0, 0));
        ASSERT_EQ(dfbs.size(), 1u);
        EXPECT_EQ(dfbs[0]->producer_txn_descriptor.num_txn_ids, c.expected_num_txn_ids);
    }
}

// B4: Cached threshold field value
// ---------------------------------------------------------------------------
// Pins compute_txn_descriptor() at tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:185
// which produces the `num_entries_to_process_threshold` value that gets cached
// in LocalDFBInterface::threshold on the device (replacing a per-call register
// read in the partial-tail spin loop, dataflow_buffer.inl:215).
//
// Formula:
//   * STRIDED consumer / any producer:  threshold = num_entries / num_txn_ids
//   * ALL consumer:                     threshold = num_consumers * (num_entries / num_txn_ids)
// (ALL is different because wr_sent is a single global counter; the ISR must
// not fire until *every* consumer in the fan-out has contributed its batch.)

// All cases in ONE TEST_F so the fixture opens a device (= aether on Quasar)
// only once.

TEST_F(MeshDeviceFixture, B4_CachedThreshold_Config) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Implicit sync (and therefore threshold caching) is Quasar-only";
    }

    // ----- case 1: 1Sx1S non-ALL, threshold = num_entries / num_txn_ids -----
    {
        SCOPED_TRACE("case 1: 1Sx1S non-ALL, num_entries=16 → threshold should be 16/2 = 8");
        experimental::dfb::DataflowBufferConfig config{
            .entry_size = 1024,
            .num_entries = 16,
            .producer_risc_mask = 0x1,
            .num_producers = 1,
            .pap = dfb::AccessPattern::STRIDED,
            .consumer_risc_mask = 0x2,
            .num_consumers = 1,
            .cap = dfb::AccessPattern::STRIDED,
            .enable_implicit_sync = true};
        Program program = CreateProgram();
        experimental::dfb::CreateDataflowBuffer(program, CoreCoord(0, 0), config);
        program.impl().finalize_dataflow_buffer_configs();

        auto dfbs = program.impl().dataflow_buffers_on_core(CoreCoord(0, 0));
        ASSERT_EQ(dfbs.size(), 1u);
        const auto& d = dfbs[0];
        ASSERT_EQ(d->producer_txn_descriptor.num_txn_ids, 2u);
        ASSERT_EQ(d->consumer_txn_descriptor.num_txn_ids, 2u);
        EXPECT_EQ(d->producer_txn_descriptor.num_entries_to_process_threshold, 8u);
        EXPECT_EQ(d->consumer_txn_descriptor.num_entries_to_process_threshold, 8u);
    }

    // ----- case 2: 1Sx3A ALL-consumer, threshold = num_consumers * (num_entries/num_txn_ids) -----
    // The ALL-consumer multiplier is the load-bearing piece the bug fix added.
    //
    // IMPORTANT: the host only computes the consumer_txn_descriptor when the
    // consumer is DM-side (see dataflow_buffer.cpp:1297, gated on
    // !consumer_is_tensix_only). For Tensix-side consumers the threshold
    // stays at its default-initialized value of 0 because there's no NoC ISR
    // threshold to program — Tensix uses its TC directly.
    // So we MUST use DM consumers here to actually exercise the ALL-consumer
    // threshold formula. DM→DM ALL implicit-sync is a runtime gap but the
    // host-side compute still runs and produces the formula value, which is
    // all this test inspects.
    {
        SCOPED_TRACE("case 2: 1S(DM)x3A(DM) ALL, num_entries=18 → producer 9, consumer 3*9=27");
        experimental::dfb::DataflowBufferConfig config{
            .entry_size = 1024,
            .num_entries = 18,
            .producer_risc_mask = 0x1,  // 1 DM producer
            .num_producers = 1,
            .pap = dfb::AccessPattern::STRIDED,
            .consumer_risc_mask = 0x2 | 0x4 | 0x8,  // 3 DM consumers (so the ALL-consumer formula path runs)
            .num_consumers = 3,
            .cap = dfb::AccessPattern::ALL,
            .enable_implicit_sync = true};
        Program program = CreateProgram();
        experimental::dfb::CreateDataflowBuffer(program, CoreCoord(0, 0), config);
        program.impl().finalize_dataflow_buffer_configs();

        auto dfbs = program.impl().dataflow_buffers_on_core(CoreCoord(0, 0));
        ASSERT_EQ(dfbs.size(), 1u);
        const auto& d = dfbs[0];
        EXPECT_EQ(d->producer_txn_descriptor.num_entries_to_process_threshold, 9u);
        EXPECT_EQ(d->consumer_txn_descriptor.num_entries_to_process_threshold, 27u);
    }
}

// B5: 5-TCs-per-RISC capacity (1Sx5S DM→Tensix)
// ---------------------------------------------------------------------------
// Pins MAX_NUM_TILE_COUNTERS_TO_RR raised from 4 → 6 in
// tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h.
//
// For 1Sx5S STRIDED, calculate_num_tile_counters() assigns the single producer
// 5 TCs (one per consumer pair). Pre-fix that exceeded the 4-TC ceiling and
// allocation would TT_FATAL. Post-fix the host allocator accepts 5 TCs cleanly.
//
// (DM→DM 6Sx1S or 1Sx6S would need 7 DM threads which Quasar can't provide;
// 1Sx5S DM→Tensix is the tightest equivalent we can validate.)

TEST_F(MeshDeviceFixture, B5_PerRiscTCCapacity_1Sx5S_DMTensix_Config) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Multi-DM TC stress test requires Quasar";
    }
    // 1 DM producer, 5 Tensix consumers. Producer carries 5 TCs (one per consumer).
    // num_entries=20 satisfies divisibility for n*1*5 = 5n with n=2 (20%10=0).
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 20,
        .producer_risc_mask = 0x1,  // 1 DM
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x100 | 0x200 | 0x400 | 0x800 | 0x1000,  // 5 Tensix (Neo0..Neo4)
        .num_consumers = 5,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    experimental::dfb::CreateDataflowBuffer(program, CoreCoord(0, 0), config);
    program.impl().finalize_dataflow_buffer_configs();

    auto dfbs = program.impl().dataflow_buffers_on_core(CoreCoord(0, 0));
    ASSERT_EQ(dfbs.size(), 1u);
    const auto& dfb = dfbs[0];
    ASSERT_FALSE(dfb->groups.empty());

    // Find the producer risc config and check it carries 5 TCs (>4, requires
    // post-fix MAX_NUM_TILE_COUNTERS_TO_RR >= 5).
    bool found_producer = false;
    for (const auto& rc : dfb->groups[0].hw_risc_configs) {
        if (rc.is_producer) {
            EXPECT_EQ(rc.config.num_tcs_to_rr, 5u) << "1Sx5S producer should carry 5 TCs (one per consumer pair); "
                                                      "requires MAX_NUM_TILE_COUNTERS_TO_RR >= 5";
            found_producer = true;
        }
    }
    EXPECT_TRUE(found_producer) << "Expected exactly 1 producer risc config";
}

// =====================================================================================
// B6 – B10: Negative DFB config tests.
//
// All five validations live in tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp and
// fire a TT_FATAL (which throws std::runtime_error) at host build time. These tests
// confirm the rejection fires AND that the error message matches the expected reason
// — a bare EXPECT_THROW would also catch unrelated throws (e.g., a device-init failure
// masquerading as success), so we substring-check what() to pin the actual call-site.
//
// All five skip on non-Quasar to match the rest of this file's convention. The
// TT_FATALs themselves are arch-agnostic (host-side validation, no device touch
// required), but DFBs aren't generally usable outside Quasar yet, so the skip
// keeps behavior consistent with the rest of the B-batch.
// =====================================================================================

namespace {
// Helper: run `stmt`, expect it to throw std::runtime_error, and verify the
// what() message contains `expected_substr`. The substring check is the
// important part — it guarantees we're rejecting for the right reason.
inline void expect_runtime_error_with(const std::function<void()>& stmt, const char* expected_substr) {
    try {
        stmt();
        FAIL() << "Expected std::runtime_error containing \"" << expected_substr << "\" but no throw occurred";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(expected_substr), std::string::npos)
            << "Caught std::runtime_error but message did not contain expected substring.\n"
            << "  expected substring: " << expected_substr << "\n"
            << "  actual what():      " << e.what();
    } catch (...) {
        FAIL() << "Expected std::runtime_error but a different exception type was thrown";
    }
}
}  // namespace

// B6 — Producer access pattern = ALL is rejected.
// Producers are always STRIDED in the current design; ALL is consumer-only.
TEST_F(MeshDeviceFixture, B6_AllProducer_Rejected) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "DFB config validation tested on Quasar (matches rest of B-batch)";
    }
    Program program = CreateProgram();
    CoreCoord logical_core(0, 0);
    experimental::dfb::DataflowBufferConfig bad{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::ALL,  // <-- the offense
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false};
    expect_runtime_error_with(
        [&]() { experimental::dfb::CreateDataflowBuffer(program, logical_core, bad); },
        "ALL producer pattern not supported");
}

// B7 — Mixing circular buffers and DFBs in the same Program is rejected.
// Direction tested: CB created first, then DFB rejected. The reverse direction
// (DFB-first → CB-second) is NOT symmetrically rejected by the host code —
// add_circular_buffer only checks compiled_.empty(), not dataflow_buffers_.empty()
// (see tt_metal/impl/program/program.cpp:1046–1048). That asymmetry is a real
// gap, separately noted; this test pins the confirmed direction.
TEST_F(MeshDeviceFixture, B7_CB_DFB_Mix_Rejected) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "DFB config validation tested on Quasar (matches rest of B-batch)";
    }
    Program program = CreateProgram();
    CoreCoord logical_core(0, 0);

    // Add a valid CB first.
    CircularBufferConfig cb_cfg(2048, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}});
    cb_cfg.set_page_size(tt::CBIndex::c_0, 1024);
    CreateCircularBuffer(program, logical_core, cb_cfg);

    // Now attempt a DFB — should be rejected because the program already has a CB.
    experimental::dfb::DataflowBufferConfig dfb_cfg{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false};
    expect_runtime_error_with(
        [&]() { experimental::dfb::CreateDataflowBuffer(program, logical_core, dfb_cfg); },
        "Cannot add dataflow buffer to a program with circular buffers");
}

// B8 — ALL consumer with num_consumers > 4 is rejected. The Remapper has 4
// clientR slots, so any 1×N ALL fanout with N>4 must be rejected at config time.
TEST_F(MeshDeviceFixture, B8_FiveAllConsumers_Rejected) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Remapper limit applies on Quasar (matches rest of B-batch)";
    }
    Program program = CreateProgram();
    CoreCoord logical_core(0, 0);
    experimental::dfb::DataflowBufferConfig bad{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 5,  // <-- 5 > Remapper's 4 clientR slots
        .cap = dfb::AccessPattern::ALL,
        .enable_implicit_sync = false};
    expect_runtime_error_with(
        [&]() { experimental::dfb::CreateDataflowBuffer(program, logical_core, bad); }, "at most 4 consumers");
}

// B9 — INTER tensix_scope is rejected. Only INTRA (same-Neo packer→unpacker
// self-loop) is supported today; INTER (cross-Neo) is explicitly not yet
// implemented.
TEST_F(MeshDeviceFixture, B9_InterTensixScope_Rejected) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "tensix_scope is Quasar-only";
    }
    Program program = CreateProgram();
    CoreCoord logical_core(0, 0);
    experimental::dfb::DataflowBufferConfig bad{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .tensix_scope = experimental::dfb::TensixScope::INTER};  // <-- unsupported
    expect_runtime_error_with(
        [&]() { experimental::dfb::CreateDataflowBuffer(program, logical_core, bad); },
        "Inter-tensix DFBs are not yet supported");
}

// B10 — num_entries divisibility against the txn_id allocator's fallback path.
//
//   compute_optimal_txn_id_count() scans n ∈ {2..NUM_TXN_IDS=4} for the smallest
//   n satisfying  num_entries % (n * num_prods * num_tcs_per_risc) == 0
//   and falls back to n=1 if none works. The TT_FATAL at compute_txn_descriptor()
//   line 198 fires only when even n=1 fails (i.e., num_entries is not divisible
//   by num_prods * num_tcs_per_risc).
//
// Two scopes:
//   (10a) num_entries=10, num_prods=3, num_tcs=1  →  10 % {3,6,9,12} ≠ 0 at any n
//         → fallback exhausted, TT_FATAL fires cleanly with the divisor in the
//         message. Pins the "clean failure path" guarantee.
//   (10b) num_entries=3,  num_prods=3, num_tcs=1  →  3 % 3 = 0 only at n=1
//         (3 % 6 ≠ 0, 3 % 9 ≠ 0, 3 % 12 ≠ 0). Verifies the fallback to n=1
//         actually succeeds and CreateDataflowBuffer does NOT throw.
TEST_F(MeshDeviceFixture, B10_NumEntriesDivisibility) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Txn-id allocator is Quasar-only";
    }

    // ---- 10a: pathological num_entries, no n ∈ {1..4} divides ----
    {
        SCOPED_TRACE("10a: num_entries=10, 3 producers, 3 consumers — no n divides");
        Program program = CreateProgram();
        CoreCoord logical_core(0, 0);
        experimental::dfb::DataflowBufferConfig bad{
            .entry_size = 1024,
            .num_entries = 10,  // 10 % (n * 3 * 1) ≠ 0 for any n
            .num_producers = 3,
            .pap = dfb::AccessPattern::STRIDED,
            .num_consumers = 3,
            .cap = dfb::AccessPattern::STRIDED,
            .enable_implicit_sync = false};
        expect_runtime_error_with(
            [&]() { experimental::dfb::CreateDataflowBuffer(program, logical_core, bad); }, "must be divisible by");
    }

    // ---- 10b: only n=1 divides — fallback path should succeed silently ----
    {
        SCOPED_TRACE("10b: num_entries=3, only n=1 satisfies divisibility — fallback should succeed");
        Program program = CreateProgram();
        CoreCoord logical_core(0, 0);
        experimental::dfb::DataflowBufferConfig ok{
            .entry_size = 1024,
            .num_entries = 3,  // 3 % (1*3*1) = 0; 3 % (n*3) ≠ 0 for n>1
            .num_producers = 3,
            .pap = dfb::AccessPattern::STRIDED,
            .num_consumers = 3,
            .cap = dfb::AccessPattern::STRIDED,
            .enable_implicit_sync = false};
        EXPECT_NO_THROW(experimental::dfb::CreateDataflowBuffer(program, logical_core, ok))
            << "Expected fallback to n=1 to succeed for num_entries=3 with 3 producers/3 consumers";
    }
}

}  // end namespace tt::tt_metal
