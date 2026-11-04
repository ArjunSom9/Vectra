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

#ifndef VECTRA_ROUTING_ROUTING_TABLE_H_
#define VECTRA_ROUTING_ROUTING_TABLE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vectra/routing/geo_math.h"
#include "vectra/routing/geohash.h"

namespace vectra::routing {

// One shard-map entry: every cell whose GeoHash starts with `prefix` belongs
// to `shard_id`.
struct PrefixAssignment {
  std::string prefix;
  uint32_t shard_id = 0;
};

// The deterministic map from shard-key cells to shards. Assignments are
// resolved by longest prefix and expanded into a dense array with one slot
// per cell (32^4 = 1,048,576 four-byte slots at the production shard
// precision), so every lookup is a single array read. The table is immutable
// after Build and safe to share across threads.
class RoutingTable {
 public:
  // Largest supported shard-key precision (32^5 slots).
  static constexpr int kMaxPrecision = 5;

  // Builds the table for cells of `precision` characters. Fails (returning
  // nullopt and filling `error`) on an invalid prefix, a prefix longer than
  // `precision`, or one prefix assigned to two different shards.
  static std::optional<RoutingTable> Build(
      int precision, const std::vector<PrefixAssignment>& assignments,
      std::string* error);

  RoutingTable(RoutingTable&&) noexcept = default;
  RoutingTable& operator=(RoutingTable&&) noexcept = default;
  RoutingTable(const RoutingTable&) = delete;
  RoutingTable& operator=(const RoutingTable&) = delete;

  // Owner of `cell`, which may be of any precision at least the table's.
  // Returns nullopt for unassigned cells.
  std::optional<uint32_t> ShardForCell(const GeohashCell& cell) const;

  // Owner of the shard-key cell containing `point` (which must be valid).
  std::optional<uint32_t> ShardForPoint(const LatLng& point) const;

  // Prefixes assigned to `shard_id`, in shard-map order.
  std::vector<std::string> PrefixesForShard(uint32_t shard_id) const;

  int precision() const { return precision_; }

 private:
  static constexpr int32_t kUnassigned = -1;

  RoutingTable(int precision, std::vector<int32_t> owners,
               std::vector<PrefixAssignment> assignments)
      : precision_(precision),
        owners_(std::move(owners)),
        assignments_(std::move(assignments)) {}

  int precision_ = 0;
  // owners_[cell.bits] is the owning shard id or kUnassigned.
  std::vector<int32_t> owners_;
  std::vector<PrefixAssignment> assignments_;
};

}  // namespace vectra::routing

#endif  // VECTRA_ROUTING_ROUTING_TABLE_H_
