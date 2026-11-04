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

#include "vectra/routing/routing_table.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vectra::routing {

std::optional<RoutingTable> RoutingTable::Build(
    int precision, const std::vector<PrefixAssignment>& assignments,
    std::string* error) {
  auto fail = [error](std::string message) -> std::optional<RoutingTable> {
    if (error != nullptr) *error = std::move(message);
    return std::nullopt;
  };
  if (precision < 1 || precision > kMaxPrecision) {
    return fail("routing precision must be in [1, " +
                std::to_string(kMaxPrecision) + "]");
  }
  std::map<std::string, uint32_t> seen;
  std::vector<std::pair<GeohashCell, uint32_t>> parsed;
  parsed.reserve(assignments.size());
  for (const PrefixAssignment& assignment : assignments) {
    std::optional<GeohashCell> cell = ParseGeohash(assignment.prefix);
    if (!cell.has_value()) {
      return fail("invalid GeoHash prefix '" + assignment.prefix + "'");
    }
    if (cell->precision > precision) {
      return fail("prefix '" + assignment.prefix + "' is longer than the " +
                  std::to_string(precision) + "-character shard key");
    }
    auto [it, inserted] =
        seen.emplace(CellToString(*cell), assignment.shard_id);
    if (!inserted && it->second != assignment.shard_id) {
      return fail("prefix '" + assignment.prefix +
                  "' is assigned to more than one shard");
    }
    parsed.emplace_back(*cell, assignment.shard_id);
  }
  // Shorter prefixes first, so longer (more specific) ones overwrite them.
  std::stable_sort(parsed.begin(), parsed.end(),
                   [](const auto& a, const auto& b) {
                     return a.first.precision < b.first.precision;
                   });
  std::vector<int32_t> owners(size_t{1} << (5 * precision), kUnassigned);
  for (const auto& [cell, shard_id] : parsed) {
    const int shift = 5 * (precision - cell.precision);
    const uint64_t first = cell.bits << shift;
    const uint64_t count = uint64_t{1} << shift;
    std::fill_n(owners.begin() + static_cast<std::ptrdiff_t>(first), count,
                static_cast<int32_t>(shard_id));
  }
  return RoutingTable(precision, std::move(owners), assignments);
}

std::optional<uint32_t> RoutingTable::ShardForCell(
    const GeohashCell& cell) const {
  if (cell.precision < precision_) return std::nullopt;
  const GeohashCell key = CellPrefix(cell, precision_);
  const int32_t owner = owners_[key.bits];
  if (owner == kUnassigned) return std::nullopt;
  return static_cast<uint32_t>(owner);
}

std::optional<uint32_t> RoutingTable::ShardForPoint(const LatLng& point) const {
  return ShardForCell(EncodeCell(point, precision_));
}

std::vector<std::string> RoutingTable::PrefixesForShard(
    uint32_t shard_id) const {
  std::vector<std::string> prefixes;
  for (const PrefixAssignment& assignment : assignments_) {
    if (assignment.shard_id == shard_id) prefixes.push_back(assignment.prefix);
  }
  return prefixes;
}

}  // namespace vectra::routing
