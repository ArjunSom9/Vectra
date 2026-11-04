// Copyright 2025 Texas A&M University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Loads engine/configs/*.textproto (schema: contracts/proto/vectra/
// config.proto) and converts it into the core library's types.

#ifndef VECTRA_CONFIG_ENGINE_CONFIG_H_
#define VECTRA_CONFIG_ENGINE_CONFIG_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "google/protobuf/message.h"
#include "vectra/config.pb.h"
#include "vectra/index/shard_index.h"
#include "vectra/routing/fanout_policy.h"
#include "vectra/routing/macro_region.h"
#include "vectra/routing/routing_table.h"

namespace vectra::config {

// The four engine configuration files, parsed and validated.
struct EngineConfig {
  v1::IndexParams index;
  v1::GeohashPolicy geohash;
  v1::ShardMap shard_map;
  v1::ClusterTopology topology;
};

// Reads hnsw_params, geohash_policy, shard_map and cluster_topology
// (.textproto) from `directory`.
absl::StatusOr<EngineConfig> LoadEngineConfig(const std::string& directory);

// Parses one text-format message from `path`.
absl::Status ParseTextProtoFile(const std::string& path,
                                google::protobuf::Message* message);

// Index construction options from IndexParams.
index::IndexOptions ToIndexOptions(const v1::IndexParams& params);

// The routing table described by the shard map, keyed at the policy's
// shard precision.
absl::StatusOr<std::unique_ptr<routing::RoutingTable>> BuildRoutingTable(
    const EngineConfig& config);

absl::StatusOr<routing::MacroRegionMap> BuildMacroRegions(
    const EngineConfig& config);

// Parses Linux cpulist syntax ("0-3,8,10-11") into core ids.
absl::StatusOr<std::vector<int>> ParseCpuList(const std::string& text);

}  // namespace vectra::config

#endif  // VECTRA_CONFIG_ENGINE_CONFIG_H_
