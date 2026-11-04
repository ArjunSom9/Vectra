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

#include "vectra/index/result_merge.h"

#include <algorithm>
#include <cstddef>
#include <unordered_set>
#include <vector>

namespace vectra::index {

std::vector<RoutedHit> MergeTopK(
    const std::vector<std::vector<RoutedHit>>& per_shard, size_t k) {
  if (per_shard.size() == 1 && per_shard.front().size() <= k) {
    return per_shard.front();
  }
  std::vector<RoutedHit> merged;
  for (const std::vector<RoutedHit>& hits : per_shard) {
    merged.insert(merged.end(), hits.begin(), hits.end());
  }
  std::sort(
      merged.begin(), merged.end(), [](const RoutedHit& a, const RoutedHit& b) {
        return a.squared_l2 < b.squared_l2 ||
               (a.squared_l2 == b.squared_l2 && a.property_id < b.property_id);
      });
  std::unordered_set<int64_t> seen;
  std::vector<RoutedHit> top;
  top.reserve(std::min(k, merged.size()));
  for (const RoutedHit& hit : merged) {
    if (top.size() == k) break;
    if (seen.insert(hit.property_id).second) top.push_back(hit);
  }
  return top;
}

}  // namespace vectra::index
