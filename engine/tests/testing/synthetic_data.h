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

// Deterministic synthetic corpora and brute-force oracles for engine tests.

#ifndef ENGINE_TESTS_TESTING_SYNTHETIC_DATA_H_
#define ENGINE_TESTS_TESTING_SYNTHETIC_DATA_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vectra/index/property_store.h"
#include "vectra/routing/geo_math.h"

namespace vectra::testing {

// Gaussian-mixture vectors: `clusters` random centers in [-1, 1]^dimension,
// each point a center plus N(0, spread^2) noise. Row-major.
std::vector<float> ClusteredVectors(size_t count, int dimension, int clusters,
                                    float spread, uint64_t seed);

// Points uniformly distributed (by area) within `radius_km` of `center`.
std::vector<routing::LatLng> PointsAround(size_t count,
                                          const routing::LatLng& center,
                                          double radius_km, uint64_t seed);

// Records with sequential ids starting at `first_id` at the given locations.
std::vector<index::PropertyRecord> RecordsAt(
    const std::vector<routing::LatLng>& locations, int64_t first_id);

// Embeddings that mix location with latent attributes, the way property
// embeddings do: dimensions 0..7 are smooth functions of position (so
// nearby properties are near in embedding space) and the rest come from a
// Gaussian mixture.
std::vector<float> GeoAwareEmbeddings(
    const std::vector<routing::LatLng>& locations, int dimension,
    uint64_t seed);

// The point reached by traveling `distance_km` from `start` on the given
// initial bearing (degrees clockwise from north).
routing::LatLng DestinationPoint(const routing::LatLng& start,
                                 double bearing_deg, double distance_km);

// Indices of the `k` rows of `data` nearest to `query` by squared L2 among
// rows for which `allowed[i]` is true (all rows if `allowed` is empty),
// nearest first.
std::vector<uint32_t> BruteForceKnn(const std::vector<float>& data,
                                    int dimension, const float* query, size_t k,
                                    const std::vector<bool>& allowed);

// |truth intersect found| / |truth|; 1 when truth is empty.
double Recall(const std::vector<uint32_t>& truth,
              const std::vector<uint32_t>& found);

}  // namespace vectra::testing

#endif  // ENGINE_TESTS_TESTING_SYNTHETIC_DATA_H_
