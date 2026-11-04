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

#include "vectra/routing/fanout_policy.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

#include "vectra/routing/geohash_neighbors.h"

namespace vectra::routing {

FanoutPolicy::FanoutPolicy(const RoutingTable* table, double default_radius_km,
                           double max_radius_km)
    : table_(table),
      default_radius_km_(default_radius_km),
      max_radius_km_(max_radius_km) {}

RoutePlan FanoutPolicy::Plan(const LatLng& center, double radius_km) const {
  RoutePlan plan;
  if (!IsValidLatLng(center)) {
    plan.status = PlanStatus::kInvalidCoordinate;
    plan.message = "latitude must be in [-90, 90] and longitude in [-180, 180]";
    return plan;
  }
  const double radius = radius_km == 0.0 ? default_radius_km_ : radius_km;
  if (!std::isfinite(radius) || radius < 0.0 || radius > max_radius_km_) {
    plan.status = PlanStatus::kInvalidRadius;
    plan.message =
        "radius_km must be in [0, " + std::to_string(max_radius_km_) + "]";
    return plan;
  }
  plan.radius_km = radius;
  const int precision = table_->precision();
  std::optional<DiskCover> cover =
      CoverDisk(center, radius, precision, /*max_rings=*/1);
  if (!cover.has_value()) {
    plan.status = PlanStatus::kRadiusExceedsBound;
    plan.message =
        "radius_km reaches beyond the cells adjacent to the query's shard "
        "cell; the router bounds fan-out at nine cells";
    return plan;
  }
  plan.home_cell = cover->cells.front();
  plan.cells_intersected = static_cast<int>(cover->cells.size());
  for (const GeohashCell& cell : cover->cells) {
    std::optional<uint32_t> owner = table_->ShardForCell(cell);
    if (!owner.has_value()) continue;  // No properties are stored there.
    ShardTarget* target = nullptr;
    for (ShardTarget& existing : plan.targets) {
      if (existing.shard_id == *owner) target = &existing;
    }
    if (target == nullptr) {
      plan.targets.push_back(ShardTarget{*owner, {}});
      target = &plan.targets.back();
    }
    target->cells.push_back(cell);
  }
  return plan;
}

}  // namespace vectra::routing
