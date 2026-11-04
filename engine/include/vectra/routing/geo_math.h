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

// Spherical geometry used by routing and radius filtering.

#ifndef VECTRA_ROUTING_GEO_MATH_H_
#define VECTRA_ROUTING_GEO_MATH_H_

#include <array>

namespace vectra::routing {

// Mean Earth radius (IUGG), in kilometers.
inline constexpr double kEarthRadiusKm = 6371.0088;

// A WGS-84 coordinate in decimal degrees.
struct LatLng {
  double lat = 0.0;
  double lng = 0.0;
};

// A latitude/longitude-aligned box in decimal degrees. GeoHash cells never
// straddle the antimeridian, so min_lng <= max_lng always holds.
struct GeoBox {
  double min_lat = 0.0;
  double max_lat = 0.0;
  double min_lng = 0.0;
  double max_lng = 0.0;
};

// Returns true when the latitude is in [-90, 90] and the longitude is in
// [-180, 180] (both finite).
bool IsValidLatLng(const LatLng& point);

// Great-circle distance between two points by the haversine formula.
double HaversineKm(const LatLng& a, const LatLng& b);

// Shortest great-circle distance from `point` to the box (zero inside it).
// Exact for boxes smaller than a hemisphere: edges of constant latitude are
// handled as parallels and edges of constant longitude as meridian arcs.
double DistanceToBoxKm(const LatLng& point, const GeoBox& box);

// Unit vector of `point` on the sphere (x toward 0E, z toward the north pole).
std::array<double, 3> ToUnitVector(const LatLng& point);

// Squared chord length of a great-circle arc of `distance_km`. Comparing
// squared chords between unit vectors is equivalent to comparing great-circle
// distances and needs no trigonometry per candidate.
double ChordSquaredForDistanceKm(double distance_km);

}  // namespace vectra::routing

#endif  // VECTRA_ROUTING_GEO_MATH_H_
