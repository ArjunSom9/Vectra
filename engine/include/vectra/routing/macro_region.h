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

#ifndef VECTRA_ROUTING_MACRO_REGION_H_
#define VECTRA_ROUTING_MACRO_REGION_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vectra/routing/geo_math.h"
#include "vectra/routing/routing_table.h"

namespace vectra::routing {

// A geographic cluster served by one data center.
struct MacroRegion {
  std::string name;
  std::string datacenter;
  // GeoHash prefixes of at most three characters (~156 km cells).
  std::vector<std::string> prefixes;
};

// Selects the macro-region (data center) for a coordinate from its
// 3-character GeoHash, by longest-prefix match. Immutable and thread-safe.
class MacroRegionMap {
 public:
  // Precision of the macro-region key.
  static constexpr int kPrecision = 3;

  // Fails on invalid or conflicting prefixes (see RoutingTable::Build).
  static std::optional<MacroRegionMap> Build(std::vector<MacroRegion> regions,
                                             std::string* error);

  // Returns the region containing `point`, or nullptr when none does.
  const MacroRegion* RegionFor(const LatLng& point) const;

  const std::vector<MacroRegion>& regions() const { return regions_; }

 private:
  MacroRegionMap(std::vector<MacroRegion> regions, RoutingTable table)
      : regions_(std::move(regions)), table_(std::move(table)) {}

  std::vector<MacroRegion> regions_;
  // Maps 3-character cells to an index into regions_.
  RoutingTable table_;
};

}  // namespace vectra::routing

#endif  // VECTRA_ROUTING_MACRO_REGION_H_
