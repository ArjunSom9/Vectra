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

#include "testing/synthetic_data.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "vectra/index/simd_distance.h"

namespace vectra::testing {
namespace {

constexpr double kRadians = std::numbers::pi / 180.0;

// Uniform double in [0, 1) that does not depend on the standard library's
// distribution implementations.
double Uniform(std::mt19937_64& rng) {
  return static_cast<double>(rng() >> 11) * 0x1.0p-53;
}

// Standard normal by Box-Muller.
double Normal(std::mt19937_64& rng) {
  const double u1 = 1.0 - Uniform(rng);
  const double u2 = Uniform(rng);
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
}

}  // namespace

std::vector<float> ClusteredVectors(size_t count, int dimension, int clusters,
                                    float spread, uint64_t seed) {
  std::mt19937_64 rng(seed);
  const size_t dim = static_cast<size_t>(dimension);
  std::vector<float> centers(static_cast<size_t>(clusters) * dim);
  for (float& value : centers) {
    value = static_cast<float>(2.0 * Uniform(rng) - 1.0);
  }
  std::vector<float> data(count * dim);
  for (size_t i = 0; i < count; ++i) {
    const size_t cluster = static_cast<size_t>(rng() % clusters);
    for (size_t d = 0; d < dim; ++d) {
      data[i * dim + d] =
          centers[cluster * dim + d] + spread * static_cast<float>(Normal(rng));
    }
  }
  return data;
}

routing::LatLng DestinationPoint(const routing::LatLng& start,
                                 double bearing_deg, double distance_km) {
  const double angle = distance_km / routing::kEarthRadiusKm;
  const double bearing = bearing_deg * kRadians;
  const double lat1 = start.lat * kRadians;
  const double lng1 = start.lng * kRadians;
  const double lat2 =
      std::asin(std::sin(lat1) * std::cos(angle) +
                std::cos(lat1) * std::sin(angle) * std::cos(bearing));
  const double lng2 =
      lng1 + std::atan2(std::sin(bearing) * std::sin(angle) * std::cos(lat1),
                        std::cos(angle) - std::sin(lat1) * std::sin(lat2));
  double lng = lng2 / kRadians;
  while (lng > 180.0) lng -= 360.0;
  while (lng < -180.0) lng += 360.0;
  return {lat2 / kRadians, lng};
}

std::vector<routing::LatLng> PointsAround(size_t count,
                                          const routing::LatLng& center,
                                          double radius_km, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<routing::LatLng> points;
  points.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const double distance = radius_km * std::sqrt(Uniform(rng));
    const double bearing = 360.0 * Uniform(rng);
    points.push_back(DestinationPoint(center, bearing, distance));
  }
  return points;
}

std::vector<index::PropertyRecord> RecordsAt(
    const std::vector<routing::LatLng>& locations, int64_t first_id) {
  std::vector<index::PropertyRecord> records(locations.size());
  for (size_t i = 0; i < locations.size(); ++i) {
    records[i].id = first_id + static_cast<int64_t>(i);
    records[i].location = locations[i];
    records[i].attributes.living_area_sqft =
        1000.0f + static_cast<float>(i % 900);
    records[i].attributes.bedrooms = static_cast<int32_t>(1 + i % 5);
    records[i].attributes.last_sale_price_usd = 250000.0 + 10.0 * i;
    // Sale days spread over 2015-2024, deterministic per index.
    records[i].attributes.last_sale_epoch_day =
        16436 + static_cast<int32_t>((i * 7919) % 3650);
  }
  return records;
}

std::vector<float> GeoAwareEmbeddings(
    const std::vector<routing::LatLng>& locations, int dimension,
    uint64_t seed) {
  const size_t dim = static_cast<size_t>(dimension);
  std::vector<float> data =
      ClusteredVectors(locations.size(), dimension, 32, 0.15f, seed);
  for (size_t i = 0; i < locations.size(); ++i) {
    const double lat = locations[i].lat * kRadians;
    const double lng = locations[i].lng * kRadians;
    // Smooth positional features at a ~10 km wavelength.
    for (size_t f = 0; f < 4 && 2 * f + 1 < dim; ++f) {
      const double scale = 600.0 * (1.0 + static_cast<double>(f));
      data[i * dim + 2 * f] = static_cast<float>(std::sin(scale * lat));
      data[i * dim + 2 * f + 1] = static_cast<float>(std::cos(scale * lng));
    }
  }
  return data;
}

std::vector<uint32_t> BruteForceKnn(const std::vector<float>& data,
                                    int dimension, const float* query, size_t k,
                                    const std::vector<bool>& allowed) {
  const size_t dim = static_cast<size_t>(dimension);
  const size_t rows = data.size() / dim;
  std::vector<std::pair<double, uint32_t>> scored;
  scored.reserve(rows);
  for (size_t i = 0; i < rows; ++i) {
    if (!allowed.empty() && !allowed[i]) continue;
    scored.emplace_back(index::L2SquaredDistanceReference(
                            query, data.data() + i * dim, dimension),
                        static_cast<uint32_t>(i));
  }
  const size_t take = std::min(k, scored.size());
  std::partial_sort(scored.begin(),
                    scored.begin() + static_cast<std::ptrdiff_t>(take),
                    scored.end());
  std::vector<uint32_t> ids;
  ids.reserve(take);
  for (size_t i = 0; i < take; ++i) ids.push_back(scored[i].second);
  return ids;
}

double Recall(const std::vector<uint32_t>& truth,
              const std::vector<uint32_t>& found) {
  if (truth.empty()) return 1.0;
  const std::set<uint32_t> found_set(found.begin(), found.end());
  size_t hits = 0;
  for (uint32_t id : truth) hits += found_set.count(id);
  return static_cast<double>(hits) / static_cast<double>(truth.size());
}

}  // namespace vectra::testing
