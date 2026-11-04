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

#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "vectra/routing/geohash.h"
#include "vectra/routing/macro_region.h"

namespace vectra::routing {
namespace {

TEST(RoutingTableTest, LongestPrefixWins) {
  std::string error;
  std::optional<RoutingTable> table = RoutingTable::Build(
      4, {{"9q", 1}, {"9q8", 2}, {"9q8y", 3}, {"dr5", 4}}, &error);
  ASSERT_TRUE(table.has_value()) << error;
  EXPECT_EQ(table->ShardForCell(*ParseGeohash("9q8y")), 3u);
  EXPECT_EQ(table->ShardForCell(*ParseGeohash("9q8z")), 2u);
  EXPECT_EQ(table->ShardForCell(*ParseGeohash("9q5c")), 1u);
  EXPECT_EQ(table->ShardForCell(*ParseGeohash("dr5r")), 4u);
  EXPECT_FALSE(table->ShardForCell(*ParseGeohash("u4pr")).has_value());
  // Finer cells resolve through their shard-key prefix.
  EXPECT_EQ(table->ShardForCell(*ParseGeohash("9q8yyk8")), 3u);
  // Coarser cells are ambiguous and therefore unrouted.
  EXPECT_FALSE(table->ShardForCell(*ParseGeohash("9q")).has_value());
  // San Francisco (37.7749, -122.4194) lies in 9q8y.
  EXPECT_EQ(table->ShardForPoint({37.7749, -122.4194}), 3u);
}

TEST(RoutingTableTest, RejectsInvalidAndConflictingAssignments) {
  std::string error;
  EXPECT_FALSE(RoutingTable::Build(4, {{"9qa", 1}}, &error).has_value());
  EXPECT_NE(error.find("invalid"), std::string::npos);
  EXPECT_FALSE(RoutingTable::Build(4, {{"9q8yy", 1}}, &error).has_value());
  EXPECT_FALSE(
      RoutingTable::Build(4, {{"9q", 1}, {"9Q", 2}}, &error).has_value());
  EXPECT_NE(error.find("more than one shard"), std::string::npos);
  EXPECT_FALSE(RoutingTable::Build(0, {}, &error).has_value());
  // The same prefix repeated for the same shard is harmless.
  EXPECT_TRUE(RoutingTable::Build(4, {{"9q", 1}, {"9q", 1}}, &error));
}

TEST(RoutingTableTest, ListsPrefixesPerShard) {
  std::optional<RoutingTable> table =
      RoutingTable::Build(4, {{"9q", 1}, {"dr", 2}, {"9r", 1}}, nullptr);
  ASSERT_TRUE(table.has_value());
  EXPECT_EQ(table->PrefixesForShard(1), (std::vector<std::string>{"9q", "9r"}));
  EXPECT_EQ(table->PrefixesForShard(2), (std::vector<std::string>{"dr"}));
  EXPECT_TRUE(table->PrefixesForShard(9).empty());
}

TEST(MacroRegionMapTest, SelectsRegionByThreeCharacterPrefix) {
  std::string error;
  std::optional<MacroRegionMap> regions =
      MacroRegionMap::Build({{"us-west", "dc-sfo", {"9q", "9r", "c2"}},
                             {"us-east", "dc-iad", {"dr", "dq"}}},
                            &error);
  ASSERT_TRUE(regions.has_value()) << error;
  const MacroRegion* west = regions->RegionFor({37.7749, -122.4194});
  ASSERT_NE(west, nullptr);
  EXPECT_EQ(west->datacenter, "dc-sfo");
  const MacroRegion* east = regions->RegionFor({40.7128, -74.0060});
  ASSERT_NE(east, nullptr);
  EXPECT_EQ(east->name, "us-east");
  EXPECT_EQ(regions->RegionFor({51.5074, -0.1278}), nullptr);  // London.
}

}  // namespace
}  // namespace vectra::routing
