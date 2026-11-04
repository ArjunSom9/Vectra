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

#include "vectra/routing/geohash_neighbors.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "testing/synthetic_data.h"
#include "vectra/routing/geo_math.h"
#include "vectra/routing/geohash.h"

namespace vectra::routing {
namespace {

std::set<std::string> Names(const std::vector<GeohashCell>& cells) {
  std::set<std::string> names;
  for (const GeohashCell& cell : cells) names.insert(CellToString(cell));
  return names;
}

TEST(GeohashNeighborsTest, MatchesIndependentlyComputedNeighbors) {
  // Ground truth from a separate implementation (decode, shift the center by
  // one cell, re-encode).
  const std::set<std::string> expected = {"ezs48", "ezs49", "ezs43", "ezs41",
                                          "ezs40", "ezefp", "ezefr", "ezefx"};
  EXPECT_EQ(Names(NeighborCells(*ParseGeohash("ezs42"))), expected);
}

TEST(GeohashNeighborsTest, LongitudeWrapsAcrossTheAntimeridian) {
  const GeohashCell west = EncodeCell({10.0, 179.99}, 4);
  EXPECT_EQ(CellToString(west), "xczb");
  const std::optional<GeohashCell> east = OffsetCell(west, 1, 0);
  ASSERT_TRUE(east.has_value());
  EXPECT_EQ(CellToString(*east), "81b0");
  EXPECT_EQ(*east, EncodeCell({10.0, -179.99}, 4));
}

TEST(GeohashNeighborsTest, PolarCellsHaveNoNeighborsBeyondThePole) {
  const GeohashCell top = EncodeCell({89.99, 0.0}, 4);
  EXPECT_FALSE(OffsetCell(top, 0, 1).has_value());
  EXPECT_EQ(NeighborCells(top).size(), 5u);
}

TEST(DistanceToBoxTest, AgreesWithDenseBoundarySampling) {
  std::mt19937_64 rng(11);
  const GeoBox box = CellBox(EncodeCell({40.73, -73.99}, 4));
  for (int trial = 0; trial < 200; ++trial) {
    const LatLng point{39.9 + 1.6 * static_cast<double>(rng() % 10000) / 1e4,
                       -74.9 + 1.8 * static_cast<double>(rng() % 10000) / 1e4};
    double sampled = HaversineKm(point, {box.min_lat, box.min_lng});
    constexpr int kSteps = 2000;
    for (int i = 0; i <= kSteps; ++i) {
      const double t = static_cast<double>(i) / kSteps;
      const double lat = box.min_lat + t * (box.max_lat - box.min_lat);
      const double lng = box.min_lng + t * (box.max_lng - box.min_lng);
      sampled = std::min({sampled, HaversineKm(point, {lat, box.min_lng}),
                          HaversineKm(point, {lat, box.max_lng}),
                          HaversineKm(point, {box.min_lat, lng}),
                          HaversineKm(point, {box.max_lat, lng})});
    }
    const bool inside = point.lat >= box.min_lat && point.lat <= box.max_lat &&
                        point.lng >= box.min_lng && point.lng <= box.max_lng;
    const double exact = DistanceToBoxKm(point, box);
    if (inside) {
      EXPECT_EQ(exact, 0.0);
    } else {
      // The exact distance never exceeds a sampled one, and sampling at this
      // density is within 30 m of the true minimum.
      EXPECT_LE(exact, sampled + 1e-9);
      EXPECT_GE(exact, sampled - 0.03);
    }
  }
}

TEST(CoverDiskTest, ZeroRadiusAndCellCenterStayHome) {
  const GeohashCell home = EncodeCell({37.7749, -122.4194}, 4);
  const LatLng center = CellCenter(home);
  std::optional<DiskCover> cover = CoverDisk(center, 0.0, 4, 1);
  ASSERT_TRUE(cover.has_value());
  EXPECT_EQ(cover->cells.size(), 1u);
  cover = CoverDisk(center, 5.0, 4, 1);  // Half-height is ~9.8 km.
  ASSERT_TRUE(cover.has_value());
  EXPECT_EQ(cover->cells.size(), 1u);
  EXPECT_EQ(cover->rings, 0);
}

TEST(CoverDiskTest, CornerDiskTouchesFourCells) {
  const GeohashCell home = EncodeCell({37.7749, -122.4194}, 4);
  const GeoBox box = CellBox(home);
  // Just inside the north-east corner.
  const LatLng corner{box.max_lat - 1e-4, box.max_lng - 1e-4};
  const std::optional<DiskCover> cover = CoverDisk(corner, 1.0, 4, 1);
  ASSERT_TRUE(cover.has_value());
  EXPECT_EQ(cover->cells.size(), 4u);
  EXPECT_EQ(cover->cells.front(), home);
}

TEST(CoverDiskTest, RejectsDisksBeyondTheRingLimit) {
  // Precision-4 cells are 19.5 km tall, so a 25 km disk always reaches past
  // the adjacent ring.
  EXPECT_FALSE(CoverDisk({37.7749, -122.4194}, 25.0, 4, 1).has_value());
  EXPECT_TRUE(CoverDisk({37.7749, -122.4194}, 25.0, 4, 3).has_value());
}

TEST(CoverDiskTest, EverySampledPointOfTheDiskLiesInACoveredCell) {
  std::mt19937_64 rng(5);
  for (int trial = 0; trial < 300; ++trial) {
    const LatLng center{
        25.0 + 25.0 * static_cast<double>(rng() % 10000) / 1e4,
        -120.0 + 50.0 * static_cast<double>(rng() % 10000) / 1e4};
    const double radius = 0.1 + 9.0 * static_cast<double>(rng() % 1000) / 1e3;
    const int precision = 4 + static_cast<int>(rng() % 3);
    const std::optional<DiskCover> cover =
        CoverDisk(center, radius, precision, 16);
    ASSERT_TRUE(cover.has_value());
    const std::set<std::string> covered = Names(cover->cells);
    for (const LatLng& point :
         testing::PointsAround(200, center, radius, rng())) {
      EXPECT_TRUE(covered.count(EncodeGeohash(point, precision)))
          << "point " << point.lat << "," << point.lng << " radius " << radius;
    }
    // And every covered cell really intersects the disk.
    for (const GeohashCell& cell : cover->cells) {
      EXPECT_LE(DistanceToBoxKm(center, CellBox(cell)), radius + 1e-6);
    }
  }
}

}  // namespace
}  // namespace vectra::routing
