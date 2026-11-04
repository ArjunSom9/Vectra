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

#include "vectra/config/engine_config.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "google/protobuf/text_format.h"

namespace vectra::config {

absl::Status ParseTextProtoFile(const std::string& path,
                                google::protobuf::Message* message) {
  std::ifstream input(path);
  if (!input) return absl::NotFoundError(absl::StrCat("cannot read ", path));
  std::stringstream contents;
  contents << input.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(contents.str(), message)) {
    return absl::InvalidArgumentError(
        absl::StrCat(path, " is not a valid ", message->GetTypeName()));
  }
  return absl::OkStatus();
}

absl::StatusOr<EngineConfig> LoadEngineConfig(const std::string& directory) {
  EngineConfig config;
  const std::string base = directory.empty() ? "." : directory;
  if (absl::Status status = ParseTextProtoFile(
          absl::StrCat(base, "/hnsw_params.textproto"), &config.index);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = ParseTextProtoFile(
          absl::StrCat(base, "/geohash_policy.textproto"), &config.geohash);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = ParseTextProtoFile(
          absl::StrCat(base, "/shard_map.textproto"), &config.shard_map);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = ParseTextProtoFile(
          absl::StrCat(base, "/cluster_topology.textproto"), &config.topology);
      !status.ok()) {
    return status;
  }
  const v1::IndexParams& index = config.index;
  if (index.dimension() == 0 || index.pq_subspaces() == 0 ||
      index.dimension() % index.pq_subspaces() != 0) {
    return absl::InvalidArgumentError(
        "hnsw_params: dimension must be a positive multiple of pq_subspaces");
  }
  if (index.m() < 2 || index.m0() < index.m()) {
    return absl::InvalidArgumentError("hnsw_params: need m >= 2, m0 >= m");
  }
  const v1::GeohashPolicy& geohash = config.geohash;
  if (geohash.shard_chars() < 1 ||
      geohash.shard_chars() > routing::RoutingTable::kMaxPrecision ||
      geohash.max_radius_km() <= 0.0) {
    return absl::InvalidArgumentError(
        "geohash_policy: shard_chars must be 1..5 and max_radius_km > 0");
  }
  return config;
}

index::IndexOptions ToIndexOptions(const v1::IndexParams& params) {
  index::IndexOptions options;
  options.dimension = static_cast<int>(params.dimension());
  options.m = static_cast<int>(params.m());
  options.m0 = static_cast<int>(params.m0());
  options.ef_construction = static_cast<int>(params.ef_construction());
  options.ef_search = static_cast<int>(params.ef_search());
  options.rerank_depth = static_cast<int>(params.rerank_depth());
  options.max_elements = params.max_elements();
  options.pq_subspaces = static_cast<int>(params.pq_subspaces());
  options.pq_training_samples = params.pq_training_samples();
  options.pq_kmeans_iterations =
      static_cast<int>(params.pq_kmeans_iterations());
  options.exhaustive_scan_threshold = params.exhaustive_scan_threshold();
  options.build_threads = static_cast<int>(params.build_threads());
  options.seed = params.seed();
  return options;
}

absl::StatusOr<std::unique_ptr<routing::RoutingTable>> BuildRoutingTable(
    const EngineConfig& config) {
  std::vector<routing::PrefixAssignment> assignments;
  for (const v1::PrefixAssignment& entry : config.shard_map.assignments()) {
    assignments.push_back({entry.prefix(), entry.shard_id()});
  }
  std::string error;
  std::optional<routing::RoutingTable> table = routing::RoutingTable::Build(
      static_cast<int>(config.geohash.shard_chars()), assignments, &error);
  if (!table.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat("shard_map: ", error));
  }
  return std::make_unique<routing::RoutingTable>(std::move(*table));
}

absl::StatusOr<routing::MacroRegionMap> BuildMacroRegions(
    const EngineConfig& config) {
  std::vector<routing::MacroRegion> regions;
  for (const v1::MacroRegion& region : config.topology.regions()) {
    regions.push_back({region.name(),
                       region.datacenter(),
                       {region.prefixes().begin(), region.prefixes().end()}});
  }
  std::string error;
  std::optional<routing::MacroRegionMap> map =
      routing::MacroRegionMap::Build(std::move(regions), &error);
  if (!map.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("cluster_topology: ", error));
  }
  return std::move(*map);
}

absl::StatusOr<std::vector<int>> ParseCpuList(const std::string& text) {
  std::vector<int> cores;
  if (text.empty()) return cores;
  for (absl::string_view part : absl::StrSplit(text, ',')) {
    std::vector<absl::string_view> bounds = absl::StrSplit(part, '-');
    int first = 0;
    int last = 0;
    if (bounds.size() > 2 || !absl::SimpleAtoi(bounds.front(), &first) ||
        !absl::SimpleAtoi(bounds.back(), &last) || first < 0 || last < first) {
      return absl::InvalidArgumentError(
          absl::StrCat("invalid cpulist entry '", part, "'"));
    }
    for (int core = first; core <= last; ++core) cores.push_back(core);
  }
  return cores;
}

}  // namespace vectra::config
