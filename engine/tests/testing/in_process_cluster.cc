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

#include "testing/in_process_cluster.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vectra/index/result_merge.h"
#include "vectra/index/search_scratch.h"

namespace vectra::testing {

InProcessCluster::InProcessCluster(
    const routing::RoutingTable* table, const index::IndexOptions& options,
    const std::vector<float>& embeddings,
    const std::vector<index::PropertyRecord>& records, double max_radius_km)
    : policy_(table, /*default_radius_km=*/1.0, max_radius_km) {
  const size_t dimension = static_cast<size_t>(options.dimension);
  std::map<uint32_t,
           std::pair<std::vector<float>, std::vector<index::PropertyRecord>>>
      partitions;
  for (size_t i = 0; i < records.size(); ++i) {
    const std::optional<uint32_t> shard =
        table->ShardForPoint(records[i].location);
    if (!shard.has_value()) continue;
    auto& [vectors, rows] = partitions[*shard];
    vectors.insert(vectors.end(), embeddings.begin() + i * dimension,
                   embeddings.begin() + (i + 1) * dimension);
    rows.push_back(records[i]);
  }
  for (auto& [shard, partition] : partitions) {
    std::string error;
    shards_[shard] = index::ShardIndex::Build(options, partition.first.data(),
                                              partition.second.data(),
                                              partition.second.size(), &error);
  }
}

InProcessCluster::Result InProcessCluster::Search(const float* query,
                                                  const routing::LatLng& center,
                                                  double radius_km, int k,
                                                  bool exact, int ef) const {
  Result result;
  const routing::RoutePlan plan = policy_.Plan(center, radius_km);
  if (!plan.ok()) return result;
  result.cells_intersected = plan.cells_intersected;
  index::SearchScratch scratch;
  std::vector<std::vector<index::RoutedHit>> per_shard;
  for (const routing::ShardTarget& target : plan.targets) {
    auto it = shards_.find(target.shard_id);
    if (it == shards_.end()) continue;
    ++result.shards_touched;
    index::IndexQuery request;
    request.embedding = query;
    request.center = center;
    request.radius_km = plan.radius_km;
    request.k = k;
    request.ef = ef;
    request.exact = exact;
    std::vector<index::IndexHit> hits;
    index::IndexSearchStats stats;
    it->second->Search(request, &scratch, &hits, &stats);
    std::vector<index::RoutedHit>& routed = per_shard.emplace_back();
    for (const index::IndexHit& hit : hits) {
      index::RoutedHit entry;
      entry.property_id = it->second->properties().id(hit.node);
      entry.squared_l2 = hit.squared_l2;
      entry.geo_distance_km = hit.geo_distance_km;
      entry.shard_id = target.shard_id;
      routed.push_back(entry);
    }
  }
  for (const index::RoutedHit& hit :
       index::MergeTopK(per_shard, static_cast<size_t>(k))) {
    result.property_ids.push_back(hit.property_id);
  }
  return result;
}

}  // namespace vectra::testing
