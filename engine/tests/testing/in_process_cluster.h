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

#ifndef ENGINE_TESTS_TESTING_IN_PROCESS_CLUSTER_H_
#define ENGINE_TESTS_TESTING_IN_PROCESS_CLUSTER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "vectra/index/property_store.h"
#include "vectra/index/shard_index.h"
#include "vectra/routing/fanout_policy.h"
#include "vectra/routing/routing_table.h"

namespace vectra::testing {

// Shards a corpus by a routing table into one in-memory ShardIndex per shard
// and answers queries the way the production router does: plan the fan-out,
// search every target shard, merge the top k. Exercises routing and merging
// without the network layer.
class InProcessCluster {
 public:
  // `table` must outlive the cluster. Every record must be routable.
  InProcessCluster(const routing::RoutingTable* table,
                   const index::IndexOptions& options,
                   const std::vector<float>& embeddings,
                   const std::vector<index::PropertyRecord>& records,
                   double max_radius_km);

  struct Result {
    std::vector<int64_t> property_ids;
    int shards_touched = 0;
    int cells_intersected = 0;
  };

  Result Search(const float* query, const routing::LatLng& center,
                double radius_km, int k, bool exact, int ef = 0) const;

 private:
  routing::FanoutPolicy policy_;
  std::map<uint32_t, std::unique_ptr<index::ShardIndex>> shards_;
};

}  // namespace vectra::testing

#endif  // ENGINE_TESTS_TESTING_IN_PROCESS_CLUSTER_H_
