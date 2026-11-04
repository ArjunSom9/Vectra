# Copyright 2025 Texas A&M University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Vectra engine (Workstream A): the geo-distributed C++ vector search engine.
#
# Targets:
#   vectra_core      routing, index, memory, io; C++20 standard library only
#   vectra_proto     generated protobuf/gRPC stubs        (VECTRA_BUILD_SERVER)
#   vectra_server    async gRPC server, handlers, config  (VECTRA_BUILD_SERVER)
#   vectra_client    channel pool, async client, router   (VECTRA_BUILD_SERVER)
#   vectra_shard     the shard server binary              (VECTRA_BUILD_SERVER)
#   indexer, stress_client, shard_inspector               (VECTRA_BUILD_SERVER)
#
# Pinned toolchain (see docs/decisions/adr-008-toolchain-pins.md): GCC 14 or
# Clang 19, CMake >= 3.28, gRPC 1.71.0 with its vendored protobuf 29.0 and
# Abseil 20240722.0, GoogleTest 1.17.0.

cmake_minimum_required(VERSION 3.28)
project(vectra_engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VECTRA_BUILD_SERVER
       "Build the gRPC server, client library, and tools (needs gRPC)" ON)
option(VECTRA_BUILD_TESTS "Build the GoogleTest suites" ON)
option(VECTRA_WARNINGS_AS_ERRORS "Treat warnings in first-party code as errors"
       ON)
option(VECTRA_ENABLE_LTO "Link-time optimization for optimized builds" ON)
set(VECTRA_SANITIZE "" CACHE STRING
    "Sanitizers to enable, e.g. address,undefined or thread")
set(VECTRA_ARCH "native" CACHE STRING
    "Target CPU: native, znver5, x86-64-v4, x86-64-v3, or none")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(DetectSIMD)
include(LtoFlags)
include(Sanitizers)

find_package(Threads REQUIRED)

function(vectra_configure_target target)
  vectra_set_warnings(${target})
  vectra_apply_simd(${target})
  vectra_apply_lto(${target})
  vectra_apply_sanitizers(${target})
endfunction()

# ---------------------------------------------------------------------------
# Core: everything the hot path computes, with no third-party dependencies.
# ---------------------------------------------------------------------------
add_library(vectra_core STATIC
  src/common/latency_histogram.cc
  src/io/binary_file.cc
  src/io/npy.cc
  src/memory/aligned_buffer.cc
  src/memory/arena.cc
  src/memory/ram_lock.cc
  src/routing/fanout_policy.cc
  src/routing/geo_math.cc
  src/routing/geohash.cc
  src/routing/geohash_neighbors.cc
  src/routing/macro_region.cc
  src/routing/routing_table.cc
  src/index/flat_store.cc
  src/index/geofilter.cc
  src/index/hnsw_builder.cc
  src/index/hnsw_graph.cc
  src/index/hnsw_searcher.cc
  src/index/pq_codec.cc
  src/index/property_store.cc
  src/index/result_merge.cc
  src/index/search_scratch.cc
  src/index/shard_index.cc
  src/index/simd_distance.cc)
target_include_directories(vectra_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(vectra_core PUBLIC Threads::Threads)
vectra_configure_target(vectra_core)

# ---------------------------------------------------------------------------
# Server, client, and tools: gRPC asynchronous API.
# ---------------------------------------------------------------------------
if(VECTRA_BUILD_SERVER)
  find_package(absl CONFIG REQUIRED)
  find_package(Protobuf CONFIG REQUIRED)
  find_package(gRPC CONFIG REQUIRED)
  include(GrpcProtoGen)

  vectra_grpc_proto_library(vectra_proto
    PROTO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../contracts/proto"
    PROTOS
      vectra/common.proto
      vectra/config.proto
      vectra/health.proto
      vectra/ingest.proto
      vectra/pricing.proto
      vectra/search.proto)

  add_library(vectra_server STATIC
    src/config/engine_config.cc
    src/server/async_server.cc
    src/server/call_data.cc
    src/server/core_pin.cc
    src/server/cq_worker.cc
    src/service/health_handler.cc
    src/service/ingest_handler.cc
    src/service/proto_convert.cc
    src/service/search_handler.cc)
  target_link_libraries(vectra_server PUBLIC
    vectra_core vectra_proto gRPC::grpc++ protobuf::libprotobuf
    absl::flags absl::log absl::status absl::statusor absl::strings
    absl::synchronization)
  vectra_configure_target(vectra_server)

  add_library(vectra_client STATIC
    client/async_search.cc
    client/channel_pool.cc
    client/geo_router.cc)
  target_link_libraries(vectra_client PUBLIC
    vectra_core vectra_proto vectra_server gRPC::grpc++ absl::check
    absl::log absl::status absl::statusor absl::synchronization)
  vectra_configure_target(vectra_client)

  add_executable(vectra_shard src/main.cc)
  target_link_libraries(vectra_shard PRIVATE
    vectra_server absl::flags absl::flags_parse absl::log
    absl::log_initialize)
  vectra_configure_target(vectra_shard)

  add_subdirectory(apps/indexer)
  add_subdirectory(apps/shard_inspector)
  add_subdirectory(apps/stress_client)

  # `cmake --install` (make dist): the binaries and the configuration they
  # read by default.
  include(GNUInstallDirs)
  install(TARGETS vectra_shard indexer stress_client shard_inspector
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(DIRECTORY configs/
          DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/vectra)
endif()

# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
if(VECTRA_BUILD_TESTS)
  enable_testing()
  # GoogleTest 1.17.0 (2025-04-30): an installed copy only if it is exactly
  # that release, otherwise the pinned tag is fetched.
  find_package(GTest 1.17.0 EXACT CONFIG QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.17.0)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  include(GoogleTest)

  add_library(vectra_test_util STATIC
    tests/testing/in_process_cluster.cc
    tests/testing/synthetic_data.cc)
  target_include_directories(vectra_test_util PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  target_link_libraries(vectra_test_util PUBLIC vectra_core)
  vectra_configure_target(vectra_test_util)

  function(vectra_add_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
    add_executable(${name} ${ARG_SOURCES})
    target_link_libraries(${name} PRIVATE ${ARG_LIBS} vectra_test_util
                          GTest::gtest_main)
    vectra_configure_target(${name})
    gtest_discover_tests(${name} DISCOVERY_TIMEOUT 60)
  endfunction()

  foreach(unit
      arena_test
      binary_file_test
      fanout_policy_test
      flat_store_test
      geofilter_test
      geohash_neighbors_test
      geohash_test
      hnsw_test
      latency_histogram_test
      npy_test
      pq_codec_test
      routing_table_test
      shard_index_test
      simd_parity_test)
    vectra_add_test(${unit} SOURCES tests/unit/${unit}.cc)
  endforeach()

  foreach(boundary adjacent_recall_test bisecting_lines_test)
    vectra_add_test(${boundary} SOURCES tests/boundary/${boundary}.cc)
  endforeach()

  # Parameter-study tool (docs/research/hnsw-parameter-study.md); not a test.
  add_executable(index_study tests/tools/index_study.cc)
  target_link_libraries(index_study PRIVATE vectra_test_util)
  vectra_configure_target(index_study)

  if(VECTRA_BUILD_SERVER)
    foreach(integration ingest_roundtrip_test search_roundtrip_test)
      vectra_add_test(${integration}
        SOURCES tests/integration/${integration}.cc
        LIBS vectra_server vectra_client)
    endforeach()
  endif()
endif()
