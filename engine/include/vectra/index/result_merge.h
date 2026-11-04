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

#ifndef VECTRA_INDEX_RESULT_MERGE_H_
#define VECTRA_INDEX_RESULT_MERGE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vectra::index {

// A shard-qualified search hit, as merged by the router.
struct RoutedHit {
  int64_t property_id = 0;
  float squared_l2 = 0.0f;
  float geo_distance_km = 0.0f;
  uint32_t shard_id = 0;
  // Position of the hit in its shard's response.
  uint32_t rank = 0;
};

// Merges per-shard hit lists into the global top `k` by exact squared L2,
// breaking ties by property id. Shards own disjoint cells, so a property can
// only appear once; duplicates are dropped defensively anyway. In the common
// single-shard case the list passes through unchanged.
std::vector<RoutedHit> MergeTopK(
    const std::vector<std::vector<RoutedHit>>& per_shard, size_t k);

}  // namespace vectra::index

#endif  // VECTRA_INDEX_RESULT_MERGE_H_
