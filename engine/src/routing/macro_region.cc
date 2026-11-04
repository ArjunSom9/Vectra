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

#include "vectra/routing/macro_region.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vectra::routing {

std::optional<MacroRegionMap> MacroRegionMap::Build(
    std::vector<MacroRegion> regions, std::string* error) {
  std::vector<PrefixAssignment> assignments;
  for (size_t i = 0; i < regions.size(); ++i) {
    for (const std::string& prefix : regions[i].prefixes) {
      assignments.push_back({prefix, static_cast<uint32_t>(i)});
    }
  }
  std::optional<RoutingTable> table =
      RoutingTable::Build(kPrecision, assignments, error);
  if (!table.has_value()) return std::nullopt;
  return MacroRegionMap(std::move(regions), std::move(*table));
}

const MacroRegion* MacroRegionMap::RegionFor(const LatLng& point) const {
  std::optional<uint32_t> index = table_.ShardForPoint(point);
  if (!index.has_value()) return nullptr;
  return &regions_[*index];
}

}  // namespace vectra::routing
