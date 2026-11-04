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

#ifndef VECTRA_ROUTING_FANOUT_POLICY_H_
#define VECTRA_ROUTING_FANOUT_POLICY_H_

#include <cstdint>
#include <string>
#include <vector>

#include "vectra/routing/geo_math.h"
#include "vectra/routing/geohash.h"
#include "vectra/routing/routing_table.h"

namespace vectra::routing {

// Outcome categories, mapped to gRPC status codes by the callers.
enum class PlanStatus {
  kOk,
  kInvalidCoordinate,   // INVALID_ARGUMENT
  kInvalidRadius,       // INVALID_ARGUMENT
  kRadiusExceedsBound,  // INVALID_ARGUMENT: would leave the 3x3 block
};

// One shard that must see the query, with the shard-key cells of the search
// disk that it owns.
struct ShardTarget {
  uint32_t shard_id = 0;
  std::vector<GeohashCell> cells;
};

// Where a query goes. In the common case `targets` holds exactly one shard.
struct RoutePlan {
  PlanStatus status = PlanStatus::kOk;
  std::string message;
  GeohashCell home_cell;
  double radius_km = 0.0;
  // Shard-key cells intersected by the disk (1..9).
  int cells_intersected = 0;
  // Home shard first (when owned), then the other owners in cell order.
  std::vector<ShardTarget> targets;

  bool ok() const { return status == PlanStatus::kOk; }
};

// Routes a (location, radius) query to the shards whose cells the disk
// intersects: exactly one shard while the disk stays inside its cell, at most
// nine when it crosses into the eight adjacent cells. A radius that would
// reach past the adjacent ring is rejected rather than broadcast, which keeps
// the fan-out bound known at route time. Immutable and thread-safe.
class FanoutPolicy {
 public:
  // `table` must outlive the policy. Radii are in kilometers; a request radius
  // of zero selects `default_radius_km`.
  FanoutPolicy(const RoutingTable* table, double default_radius_km,
               double max_radius_km);

  RoutePlan Plan(const LatLng& center, double radius_km) const;

  double default_radius_km() const { return default_radius_km_; }
  double max_radius_km() const { return max_radius_km_; }

 private:
  const RoutingTable* table_;
  double default_radius_km_;
  double max_radius_km_;
};

}  // namespace vectra::routing

#endif  // VECTRA_ROUTING_FANOUT_POLICY_H_
