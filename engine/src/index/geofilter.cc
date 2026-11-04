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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace vectra::index {

GeoFilter GeoFilter::Build(const uint32_t* cell6, size_t count) {
  std::vector<std::pair<uint32_t, uint32_t>> entries(count);
  for (size_t i = 0; i < count; ++i) {
    entries[i] = {cell6[i], static_cast<uint32_t>(i)};
  }
  std::sort(entries.begin(), entries.end());
  GeoFilter filter;
  filter.cells_.resize(count);
  filter.ids_.resize(count);
  for (size_t i = 0; i < count; ++i) {
    filter.cells_[i] = entries[i].first;
    filter.ids_[i] = entries[i].second;
  }
  return filter;
}

IdRange GeoFilter::Range(const routing::GeohashCell& cell) const {
  const int shift = 5 * (kNodeCellPrecision - cell.precision);
  const uint32_t first = static_cast<uint32_t>(cell.bits << shift);
  const uint32_t last = first + ((uint32_t{1} << shift) - 1);
  const auto begin = std::lower_bound(cells_.begin(), cells_.end(), first);
  const auto end = std::upper_bound(begin, cells_.end(), last);
  return IdRange{static_cast<size_t>(begin - cells_.begin()),
                 static_cast<size_t>(end - cells_.begin())};
}

size_t GeoFilter::Population(
    const std::vector<routing::GeohashCell>& cells) const {
  size_t population = 0;
  for (const routing::GeohashCell& cell : cells)
    population += Range(cell).size();
  return population;
}

SearchPredicate::SearchPredicate(
    const PropertyStore* properties, const routing::LatLng& center,
    double radius_km, const std::vector<routing::GeohashCell>& allowed_cells,
    int32_t sold_before_epoch_day)
    : properties_(properties),
      center_(routing::ToUnitVector(center)),
      geographic_(radius_km > 0.0),
      // A relative slack of 1e-12 absorbs rounding in the chord comparison.
      limit_(geographic_
                 ? routing::ChordSquaredForDistanceKm(radius_km) * (1.0 + 1e-12)
                 : 0.0),
      sold_before_(sold_before_epoch_day) {
  allowed_.reserve(allowed_cells.size());
  for (const routing::GeohashCell& cell : allowed_cells) {
    allowed_.push_back(static_cast<uint32_t>(cell.bits));
  }
  std::sort(allowed_.begin(), allowed_.end());
}

double SearchPredicate::DistanceKm(uint32_t node) const {
  const double chord = std::sqrt(ChordSquared(node));
  return 2.0 * routing::kEarthRadiusKm * std::asin(std::min(1.0, chord / 2));
}

}  // namespace vectra::index
