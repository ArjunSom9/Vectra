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

#include "vectra/routing/geohash.h"

#include <optional>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "vectra/routing/geo_math.h"

namespace vectra::routing {
namespace {

TEST(GeohashTest, EncodesReferenceExamples) {
  // Both examples are from the GeoHash article [17].
  EXPECT_EQ(EncodeGeohash({42.6, -5.6}, 5), "ezs42");
  EXPECT_EQ(EncodeGeohash({57.64911, 10.40744}, 11), "u4pruydqqvj");
}

TEST(GeohashTest, DecodesReferenceCenter) {
  const std::optional<GeohashCell> cell = ParseGeohash("ezs42");
  ASSERT_TRUE(cell.has_value());
  const LatLng center = CellCenter(*cell);
  EXPECT_DOUBLE_EQ(center.lat, 42.60498046875);
  EXPECT_DOUBLE_EQ(center.lng, -5.60302734375);
}

TEST(GeohashTest, ExtremeCoordinatesMapToCornerCells) {
  EXPECT_EQ(EncodeGeohash({-90.0, -180.0}, 4), "0000");
  EXPECT_EQ(EncodeGeohash({90.0, 180.0}, 4), "zzzz");
  EXPECT_EQ(EncodeGeohash({0.0, 0.0}, 5), "s0000");
}

TEST(GeohashTest, ParseIsCaseInsensitiveAndRejectsInvalidText) {
  EXPECT_EQ(ParseGeohash("EZS42"), ParseGeohash("ezs42"));
  EXPECT_FALSE(ParseGeohash("").has_value());
  EXPECT_FALSE(ParseGeohash("ezs4a").has_value());  // 'a' is not base-32.
  EXPECT_FALSE(ParseGeohash("ilo").has_value());
  EXPECT_FALSE(ParseGeohash("0123456789bcd").has_value());  // 13 > 12.
}

TEST(GeohashTest, RandomPointsRoundTripThroughBitsAndText) {
  std::mt19937_64 rng(7);
  for (int trial = 0; trial < 2000; ++trial) {
    const LatLng point{static_cast<double>(rng() % 180000) / 1000.0 - 90.0,
                       static_cast<double>(rng() % 360000) / 1000.0 - 180.0};
    const int precision = 1 + static_cast<int>(rng() % kMaxGeohashPrecision);
    const GeohashCell cell = EncodeCell(point, precision);
    EXPECT_EQ(ParseGeohash(CellToString(cell)), cell);
    EXPECT_EQ(Interleave(Deinterleave(cell), precision), cell);
    const GeoBox box = CellBox(cell);
    EXPECT_LE(box.min_lat, point.lat);
    EXPECT_GE(box.max_lat, point.lat);
    EXPECT_LE(box.min_lng, point.lng);
    EXPECT_GE(box.max_lng, point.lng);
    // Coarser encodings are prefixes of finer ones.
    const std::string fine = CellToString(EncodeCell(point, 12));
    EXPECT_EQ(fine.substr(0, static_cast<size_t>(precision)),
              CellToString(cell));
    EXPECT_EQ(CellPrefix(EncodeCell(point, 12), precision), cell);
  }
}

TEST(GeohashTest, CellDimensionsMatchPublishedTable) {
  // Cell sizes at the equator as tabulated in [17] (the Elasticsearch table
  // [18] gives the same values rounded to two significant digits).
  struct Row {
    int precision;
    double width_km;
    double height_km;
  };
  for (const Row& row : {Row{3, 156.0, 156.0}, Row{4, 39.1, 19.5},
                         Row{5, 4.89, 4.89}, Row{6, 1.22, 0.61}}) {
    EXPECT_NEAR(CellWidthKm(row.precision, 0.0), row.width_km,
                0.01 * row.width_km)
        << "precision " << row.precision;
    EXPECT_NEAR(CellHeightKm(row.precision), row.height_km,
                0.01 * row.height_km)
        << "precision " << row.precision;
  }
  // Width shrinks with the cosine of latitude; height does not change.
  EXPECT_NEAR(CellWidthKm(4, 60.0), CellWidthKm(4, 0.0) / 2.0, 1e-9);
}

TEST(GeoMathTest, HaversineMatchesKnownDistance) {
  // One degree of latitude on the mean-radius sphere.
  EXPECT_NEAR(HaversineKm({0.0, 0.0}, {1.0, 0.0}), 111.195, 0.001);
  // Crossing the antimeridian takes the short way around.
  EXPECT_NEAR(HaversineKm({0.0, 179.5}, {0.0, -179.5}), 111.195, 0.001);
}

TEST(GeoMathTest, ChordOrderingMatchesGreatCircleOrdering) {
  for (double km : {0.001, 0.5, 5.0, 19.5, 500.0}) {
    const auto a = ToUnitVector({10.0, 20.0});
    const LatLng far{10.0 + km / 111.195, 20.0};
    const auto b = ToUnitVector(far);
    const double chord2 = (a[0] - b[0]) * (a[0] - b[0]) +
                          (a[1] - b[1]) * (a[1] - b[1]) +
                          (a[2] - b[2]) * (a[2] - b[2]);
    EXPECT_NEAR(chord2,
                ChordSquaredForDistanceKm(HaversineKm({10.0, 20.0}, far)),
                1e-12 * (1.0 + chord2));
  }
}

}  // namespace
}  // namespace vectra::routing
