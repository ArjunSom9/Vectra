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

#include <cstdint>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "vectra/routing/geo_math.h"
#include "vectra/routing/geohash.h"
#include "vectra/routing/routing_table.h"

namespace vectra::routing {
namespace {

// Assigns every 4-character cell in a 2-character prefix to shard
// (row + column) % 3, a checkerboard that maximizes boundary crossings.
RoutingTable CheckerboardTable(const std::string& prefix) {
  const GeohashCell parent = *ParseGeohash(prefix);
  std::vector<PrefixAssignment> assignments;
  for (uint64_t child = 0; child < 1024; ++child) {
    GeohashCell cell{(parent.bits << 10) | child, 4};
    const CellIndex index = Deinterleave(cell);
    assignments.push_back({CellToString(cell), (index.column + index.row) % 3});
  }
  return *RoutingTable::Build(4, assignments, nullptr);
}

TEST(FanoutPolicyTest, InteriorQueryTouchesExactlyOneShard) {
  const RoutingTable table = CheckerboardTable("9q");
  const FanoutPolicy policy(&table, 2.0, 15.0);
  const GeohashCell home = EncodeCell({37.7749, -122.4194}, 4);
  const RoutePlan plan = policy.Plan(CellCenter(home), 2.0);
  ASSERT_TRUE(plan.ok()) << plan.message;
  EXPECT_EQ(plan.cells_intersected, 1);
  ASSERT_EQ(plan.targets.size(), 1u);
  EXPECT_EQ(plan.targets[0].shard_id, *table.ShardForCell(home));
}

TEST(FanoutPolicyTest, DefaultRadiusAppliesWhenZero) {
  const RoutingTable table = CheckerboardTable("9q");
  const FanoutPolicy policy(&table, 3.5, 15.0);
  const RoutePlan plan = policy.Plan({37.7749, -122.4194}, 0.0);
  ASSERT_TRUE(plan.ok());
  EXPECT_DOUBLE_EQ(plan.radius_km, 3.5);
}

TEST(FanoutPolicyTest, BoundaryQueryDuplicatesOnlyToIntersectingShards) {
  const RoutingTable table = CheckerboardTable("9q");
  const FanoutPolicy policy(&table, 2.0, 15.0);
  const GeohashCell home = EncodeCell({37.7749, -122.4194}, 4);
  const GeoBox box = CellBox(home);
  // 0.5 km south of the northern edge, mid-cell in longitude.
  const LatLng near_edge{box.max_lat - 0.5 / 111.195,
                         (box.min_lng + box.max_lng) / 2};
  const RoutePlan plan = policy.Plan(near_edge, 1.0);
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(plan.cells_intersected, 2);
  EXPECT_EQ(plan.targets.size(), 2u);  // Checkerboard neighbors differ.
  EXPECT_EQ(plan.targets[0].shard_id, *table.ShardForCell(home));
}

TEST(FanoutPolicyTest, FanOutNeverExceedsNineCells) {
  const RoutingTable table = CheckerboardTable("9q");
  const FanoutPolicy policy(&table, 2.0, 30.0);
  std::mt19937_64 rng(3);
  int rejected = 0;
  for (int trial = 0; trial < 5000; ++trial) {
    const LatLng center{
        36.0 + 3.0 * static_cast<double>(rng() % 10000) / 1e4,
        -123.0 + 3.0 * static_cast<double>(rng() % 10000) / 1e4};
    const double radius = 25.0 * static_cast<double>(rng() % 10000) / 1e4;
    const RoutePlan plan = policy.Plan(center, radius);
    if (!plan.ok()) {
      EXPECT_EQ(plan.status, PlanStatus::kRadiusExceedsBound);
      ++rejected;
      continue;
    }
    EXPECT_GE(plan.cells_intersected, 1);
    EXPECT_LE(plan.cells_intersected, 9);
    EXPECT_LE(plan.targets.size(), 3u);  // Only three shards exist.
    std::set<uint32_t> distinct;
    for (const ShardTarget& target : plan.targets) {
      EXPECT_TRUE(distinct.insert(target.shard_id).second);
    }
  }
  // Radii above the 19.5 km cell height must have been rejected sometimes.
  EXPECT_GT(rejected, 0);
}

TEST(FanoutPolicyTest, RejectsInvalidInput) {
  const RoutingTable table = CheckerboardTable("9q");
  const FanoutPolicy policy(&table, 2.0, 15.0);
  EXPECT_EQ(policy.Plan({91.0, 0.0}, 1.0).status,
            PlanStatus::kInvalidCoordinate);
  EXPECT_EQ(policy.Plan({37.0, -122.0}, -1.0).status,
            PlanStatus::kInvalidRadius);
  EXPECT_EQ(policy.Plan({37.0, -122.0}, 16.0).status,
            PlanStatus::kInvalidRadius);
}

TEST(FanoutPolicyTest, UnownedCellsAreSkipped) {
  const RoutingTable table = *RoutingTable::Build(4, {{"9q8y", 7}}, nullptr);
  const FanoutPolicy policy(&table, 2.0, 15.0);
  const GeoBox box = CellBox(*ParseGeohash("9q8y"));
  const LatLng corner{box.max_lat - 1e-4, box.max_lng - 1e-4};
  const RoutePlan plan = policy.Plan(corner, 1.0);
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(plan.cells_intersected, 4);
  ASSERT_EQ(plan.targets.size(), 1u);
  EXPECT_EQ(plan.targets[0].shard_id, 7u);
}

}  // namespace
}  // namespace vectra::routing
