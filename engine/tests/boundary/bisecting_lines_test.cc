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

// Properties that are physically adjacent but sit on opposite sides of a
// GeoHash bisecting line share no prefix [17], so they live on different
// shards. These tests place comparables on both sides of the hardest lines
// and require the router to find every one of them.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "testing/in_process_cluster.h"
#include "testing/synthetic_data.h"
#include "vectra/index/property_store.h"
#include "vectra/index/shard_index.h"
#include "vectra/routing/geo_math.h"
#include "vectra/routing/geohash.h"
#include "vectra/routing/routing_table.h"

namespace vectra {
namespace {

constexpr int kDimension = 16;

index::IndexOptions TinyOptions() {
  index::IndexOptions options;
  options.dimension = kDimension;
  options.pq_subspaces = 4;
  options.ef_construction = 64;
  options.pq_kmeans_iterations = 5;
  return options;
}

struct BisectingLine {
  std::string name;
  routing::LatLng on_line;
  // Bearing that crosses the line.
  double crossing_bearing_deg;
};

class BisectingLinesTest : public ::testing::TestWithParam<BisectingLine> {};

TEST_P(BisectingLinesTest, FindsComparablesOnBothSides) {
  const BisectingLine& line = GetParam();
  const std::vector<routing::LatLng> places =
      testing::PointsAround(600, line.on_line, 2.0, 99);
  const std::vector<float> embeddings =
      testing::GeoAwareEmbeddings(places, kDimension, 3);
  const std::vector<index::PropertyRecord> records =
      testing::RecordsAt(places, 0);

  // One shard per 4-character cell present in the corpus.
  std::vector<routing::PrefixAssignment> assignments;
  std::set<std::string> cells;
  for (const routing::LatLng& place : places) {
    cells.insert(routing::EncodeGeohash(place, 4));
  }
  ASSERT_GE(cells.size(), 2u) << "the corpus must straddle the line";
  uint32_t shard = 0;
  for (const std::string& cell : cells) assignments.push_back({cell, shard++});
  const std::optional<routing::RoutingTable> table =
      routing::RoutingTable::Build(4, assignments, nullptr);
  ASSERT_TRUE(table.has_value());
  const testing::InProcessCluster cluster(&*table, TinyOptions(), embeddings,
                                          records, 15.0);

  // Query 300 m from the line, with a radius that crosses it.
  const routing::LatLng center =
      testing::DestinationPoint(line.on_line, line.crossing_bearing_deg, 0.3);
  for (size_t q = 0; q < 40; ++q) {
    const float* query = embeddings.data() + (q * 13) * kDimension;
    std::vector<bool> allowed(places.size());
    for (size_t i = 0; i < places.size(); ++i) {
      allowed[i] = routing::HaversineKm(center, places[i]) <= 1.5;
    }
    std::vector<int64_t> truth;
    for (uint32_t row :
         testing::BruteForceKnn(embeddings, kDimension, query, 10, allowed)) {
      truth.push_back(records[row].id);
    }
    const testing::InProcessCluster::Result result =
        cluster.Search(query, center, 1.5, 10, /*exact=*/true);
    EXPECT_GE(result.shards_touched, 2) << line.name;
    EXPECT_EQ(result.property_ids, truth) << line.name;
  }
}

INSTANTIATE_TEST_SUITE_P(
    HardestLines, BisectingLinesTest,
    ::testing::Values(
        // The equator splits the first latitude bit.
        BisectingLine{"equator", {0.0, 10.0}, 0.0},
        // The prime meridian splits the first longitude bit.
        BisectingLine{"prime_meridian", {45.0, 0.0}, 90.0},
        // The antimeridian: longitudes 180 and -180 are neighbors.
        BisectingLine{"antimeridian", {-17.0, 180.0}, 270.0},
        // An ordinary 4-character cell edge inside San Francisco.
        BisectingLine{
            "cell_edge",
            {routing::CellBox(routing::EncodeCell({37.7749, -122.4194}, 4))
                 .max_lat,
             -122.4194},
            0.0}),
    [](const ::testing::TestParamInfo<BisectingLine>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace vectra
