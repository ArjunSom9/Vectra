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

#include "vectra/index/geofilter.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <set>
#include <vector>

#include "gtest/gtest.h"
#include "testing/synthetic_data.h"
#include "vectra/index/property_store.h"
#include "vectra/routing/geo_math.h"
#include "vectra/routing/geohash.h"
#include "vectra/routing/geohash_neighbors.h"

namespace vectra::index {
namespace {

constexpr routing::LatLng kSanFrancisco{37.7749, -122.4194};

PropertyStore StoreAround(const routing::LatLng& center, size_t count,
                          double radius_km) {
  PropertyStore store(count);
  for (const PropertyRecord& record : testing::RecordsAt(
           testing::PointsAround(count, center, radius_km, 17), 0)) {
    store.Append(record);
  }
  return store;
}

TEST(GeoFilterTest, RangesMatchBruteForceMembership) {
  const PropertyStore store = StoreAround(kSanFrancisco, 5000, 25.0);
  const GeoFilter filter = GeoFilter::Build(store.cell6_data(), store.size());
  EXPECT_EQ(filter.size(), 5000u);
  std::set<uint64_t> cells5;
  for (uint32_t i = 0; i < 5000; ++i) cells5.insert(store.cell6(i) >> 5);
  size_t total = 0;
  for (uint64_t bits : cells5) {
    const routing::GeohashCell cell{bits, 5};
    const IdRange range = filter.Range(cell);
    total += range.size();
    std::set<uint32_t> expected;
    for (uint32_t i = 0; i < 5000; ++i) {
      if ((store.cell6(i) >> 5) == bits) expected.insert(i);
    }
    const std::set<uint32_t> actual(filter.ids() + range.begin,
                                    filter.ids() + range.end);
    EXPECT_EQ(actual, expected);
  }
  EXPECT_EQ(total, 5000u);
  // A 6-character cell is a sub-range of its parent's range.
  const routing::GeohashCell child{store.cell6(0), 6};
  const IdRange parent = filter.Range(routing::CellPrefix(child, 5));
  const IdRange inner = filter.Range(child);
  EXPECT_GE(inner.begin, parent.begin);
  EXPECT_LE(inner.end, parent.end);
}

TEST(SearchPredicateTest, AdmitsExactlyThePointsWithinTheRadius) {
  const PropertyStore store = StoreAround(kSanFrancisco, 4000, 15.0);
  std::mt19937_64 rng(23);
  for (int trial = 0; trial < 50; ++trial) {
    const routing::LatLng center =
        testing::PointsAround(1, kSanFrancisco, 10.0, rng())[0];
    const double radius = 0.5 + 4.5 * static_cast<double>(rng() % 1000) / 1e3;
    const std::optional<routing::DiskCover> cover =
        routing::CoverDisk(center, radius, kPrefilterPrecision, 16);
    ASSERT_TRUE(cover.has_value());
    const SearchPredicate predicate(&store, center, radius, cover->cells,
                                    /*sold_before_epoch_day=*/0);
    for (uint32_t i = 0; i < 4000; ++i) {
      const double distance = routing::HaversineKm(center, store.location(i));
      if (std::fabs(distance - radius) < 1e-6) continue;  // On the boundary.
      EXPECT_EQ(predicate.Admits(i), distance <= radius) << "node " << i;
      EXPECT_NEAR(predicate.DistanceKm(i), distance, 1e-6);
    }
  }
}

TEST(SearchPredicateTest, SaleDateBoundAdmitsOnlyEarlierSales) {
  const PropertyStore store = StoreAround(kSanFrancisco, 3000, 15.0);
  constexpr int32_t kBound = 18000;
  // No radius and no cells: the sale date alone decides.
  const SearchPredicate predicate(&store, kSanFrancisco, 0.0, {}, kBound);
  int admitted = 0;
  for (uint32_t i = 0; i < 3000; ++i) {
    const bool earlier = store.attributes(i).last_sale_epoch_day < kBound;
    EXPECT_EQ(store.last_sale_epoch_day(i),
              store.attributes(i).last_sale_epoch_day);
    EXPECT_EQ(predicate.Admits(i), earlier) << "node " << i;
    admitted += earlier ? 1 : 0;
  }
  EXPECT_GT(admitted, 0);
  EXPECT_LT(admitted, 3000);
}

TEST(SearchPredicateTest, SaleDateAndRadiusCombine) {
  const PropertyStore store = StoreAround(kSanFrancisco, 3000, 15.0);
  constexpr int32_t kBound = 19000;
  const std::optional<routing::DiskCover> cover =
      routing::CoverDisk(kSanFrancisco, 4.0, kPrefilterPrecision, 16);
  ASSERT_TRUE(cover.has_value());
  const SearchPredicate predicate(&store, kSanFrancisco, 4.0, cover->cells,
                                  kBound);
  for (uint32_t i = 0; i < 3000; ++i) {
    const double distance =
        routing::HaversineKm(kSanFrancisco, store.location(i));
    if (std::fabs(distance - 4.0) < 1e-6) continue;
    EXPECT_EQ(
        predicate.Admits(i),
        distance <= 4.0 && store.attributes(i).last_sale_epoch_day < kBound);
  }
}

}  // namespace
}  // namespace vectra::index
