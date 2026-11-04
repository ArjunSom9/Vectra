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

// The in-node GeoHash prefilter applied before graph traversal.

#ifndef VECTRA_INDEX_GEOFILTER_H_
#define VECTRA_INDEX_GEOFILTER_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "vectra/index/property_store.h"
#include "vectra/routing/geo_math.h"
#include "vectra/routing/geohash.h"

namespace vectra::index {

// Precision of the prefilter cells (~4.9 km x 4.9 km).
inline constexpr int kPrefilterPrecision = 5;

// A run [begin, end) of entries in GeoFilter::ids().
struct IdRange {
  size_t begin = 0;
  size_t end = 0;

  size_t size() const { return end - begin; }
};

// Inverted index from GeoHash cells to node ids. Entries are sorted by
// 6-character cell, so every 5-character prefilter cell, and every
// 6-character cell inside it, is one contiguous run of ids. Immutable after
// Build and safe to share across threads.
class GeoFilter {
 public:
  // Indexes `count` nodes whose 6-character cells are `cell6[0..count)`.
  static GeoFilter Build(const uint32_t* cell6, size_t count);

  // Entries whose cell starts with `cell` (precision 1..6).
  IdRange Range(const routing::GeohashCell& cell) const;

  // Total entries in the given cells.
  size_t Population(const std::vector<routing::GeohashCell>& cells) const;

  const uint32_t* ids() const { return ids_.data(); }
  size_t size() const { return ids_.size(); }
  size_t ResidentBytes() const { return 8 * ids_.size(); }

 private:
  std::vector<uint32_t> cells_;
  std::vector<uint32_t> ids_;
};

// The admission test of a filtered search. Admits a node when
//   * its last sale precedes `sold_before_epoch_day` (when positive),
//   * its 5-character cell is one of `allowed_cells` (when not empty), and
//   * its great-circle distance from `center` is at most `radius_km` (when
//     positive).
// The distance test compares squared chords between precomputed unit
// vectors, which orders points exactly as the great-circle distance does.
class SearchPredicate {
 public:
  // `properties` must outlive the predicate. `allowed_cells` may be in any
  // order.
  SearchPredicate(const PropertyStore* properties,
                  const routing::LatLng& center, double radius_km,
                  const std::vector<routing::GeohashCell>& allowed_cells,
                  int32_t sold_before_epoch_day);

  bool Admits(uint32_t node) const {
    if (sold_before_ > 0 &&
        properties_->last_sale_epoch_day(node) >= sold_before_) {
      return false;
    }
    if (!allowed_.empty()) {
      const uint32_t cell5 = properties_->cell6(node) >> 5;
      if (!std::binary_search(allowed_.begin(), allowed_.end(), cell5)) {
        return false;
      }
    }
    return !geographic_ || ChordSquared(node) <= limit_;
  }

  // Great-circle distance from the center to `node`, in kilometers.
  double DistanceKm(uint32_t node) const;

 private:
  double ChordSquared(uint32_t node) const {
    const std::array<double, 3>& p = properties_->unit_vector(node);
    const double dx = p[0] - center_[0];
    const double dy = p[1] - center_[1];
    const double dz = p[2] - center_[2];
    return dx * dx + dy * dy + dz * dz;
  }

  const PropertyStore* properties_;
  std::array<double, 3> center_;
  bool geographic_;
  double limit_;
  std::vector<uint32_t> allowed_;
  int32_t sold_before_;
};

}  // namespace vectra::index

#endif  // VECTRA_INDEX_GEOFILTER_H_
