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

#include "vectra/routing/geo_math.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace vectra::routing {
namespace {

constexpr double kDegreesToRadians = std::numbers::pi / 180.0;

// Wraps a longitude difference into (-180, 180].
double WrapDegrees(double delta) {
  double wrapped = std::fmod(delta, 360.0);
  if (wrapped > 180.0) wrapped -= 360.0;
  if (wrapped <= -180.0) wrapped += 360.0;
  return wrapped;
}

bool LongitudeWithin(double lng, const GeoBox& box) {
  if (lng >= box.min_lng && lng <= box.max_lng) return true;
  // -180 and 180 denote the same meridian.
  const double alternate = lng > 0.0 ? lng - 360.0 : lng + 360.0;
  return alternate >= box.min_lng && alternate <= box.max_lng;
}

// The longitude inside [min_lng, max_lng] nearest to `lng` along a parallel.
double NearestLongitude(double lng, const GeoBox& box) {
  if (LongitudeWithin(lng, box)) return lng;
  const double to_min = std::fabs(WrapDegrees(lng - box.min_lng));
  const double to_max = std::fabs(WrapDegrees(lng - box.max_lng));
  return to_min <= to_max ? box.min_lng : box.max_lng;
}

// Distance from `point` to the meridian arc at `edge_lng` spanning
// [min_lat, max_lat]. The nearest point of the full meridian half-circle
// lies at latitude atan2(sin(lat), cos(lat) * cos(dlng)); if that falls
// outside the arc, the nearer endpoint is the answer because the distance
// along a great circle is unimodal.
double DistanceToMeridianArcKm(const LatLng& point, double edge_lng,
                               double min_lat, double max_lat) {
  double best = std::min(HaversineKm(point, {min_lat, edge_lng}),
                         HaversineKm(point, {max_lat, edge_lng}));
  const double dlng = WrapDegrees(point.lng - edge_lng) * kDegreesToRadians;
  if (std::fabs(dlng) <= std::numbers::pi / 2) {
    const double lat = point.lat * kDegreesToRadians;
    const double foot_lat =
        std::atan2(std::sin(lat), std::cos(lat) * std::cos(dlng)) /
        kDegreesToRadians;
    if (foot_lat >= min_lat && foot_lat <= max_lat) {
      best = std::min(best, HaversineKm(point, {foot_lat, edge_lng}));
    }
  }
  return best;
}

}  // namespace

bool IsValidLatLng(const LatLng& point) {
  return std::isfinite(point.lat) && std::isfinite(point.lng) &&
         point.lat >= -90.0 && point.lat <= 90.0 && point.lng >= -180.0 &&
         point.lng <= 180.0;
}

double HaversineKm(const LatLng& a, const LatLng& b) {
  const double lat_a = a.lat * kDegreesToRadians;
  const double lat_b = b.lat * kDegreesToRadians;
  const double dlat = lat_b - lat_a;
  const double dlng = WrapDegrees(b.lng - a.lng) * kDegreesToRadians;
  const double sin_dlat = std::sin(dlat / 2);
  const double sin_dlng = std::sin(dlng / 2);
  const double h = sin_dlat * sin_dlat +
                   std::cos(lat_a) * std::cos(lat_b) * sin_dlng * sin_dlng;
  return 2.0 * kEarthRadiusKm * std::asin(std::min(1.0, std::sqrt(h)));
}

double DistanceToBoxKm(const LatLng& point, const GeoBox& box) {
  const bool lat_inside = point.lat >= box.min_lat && point.lat <= box.max_lat;
  if (lat_inside && LongitudeWithin(point.lng, box)) return 0.0;
  // Along a parallel the distance grows monotonically with the longitude
  // offset, so each latitude edge is closest at the nearest longitude.
  const double edge_lng = NearestLongitude(point.lng, box);
  double best = std::min(HaversineKm(point, {box.min_lat, edge_lng}),
                         HaversineKm(point, {box.max_lat, edge_lng}));
  best = std::min(best, DistanceToMeridianArcKm(point, box.min_lng, box.min_lat,
                                                box.max_lat));
  best = std::min(best, DistanceToMeridianArcKm(point, box.max_lng, box.min_lat,
                                                box.max_lat));
  return best;
}

std::array<double, 3> ToUnitVector(const LatLng& point) {
  const double lat = point.lat * kDegreesToRadians;
  const double lng = point.lng * kDegreesToRadians;
  return {std::cos(lat) * std::cos(lng), std::cos(lat) * std::sin(lng),
          std::sin(lat)};
}

double ChordSquaredForDistanceKm(double distance_km) {
  if (distance_km <= 0.0) return 0.0;
  const double angle = std::min(distance_km / kEarthRadiusKm, std::numbers::pi);
  const double chord = 2.0 * std::sin(angle / 2);
  return chord * chord;
}

}  // namespace vectra::routing
