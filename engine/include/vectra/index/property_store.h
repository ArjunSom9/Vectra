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

#ifndef VECTRA_INDEX_PROPERTY_STORE_H_
#define VECTRA_INDEX_PROPERTY_STORE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "vectra/routing/geo_math.h"

namespace vectra::index {

// Precision of the per-node GeoHash cell kept for filtering and for the
// pricing model's micro-neighborhood features (~1.2 km x 0.6 km).
inline constexpr int kNodeCellPrecision = 6;

// Hedonic attributes; mirrors vectra.common.v1.PropertyAttributes.
struct PropertyAttributes {
  float living_area_sqft = 0.0f;
  float lot_area_sqft = 0.0f;
  int32_t bedrooms = 0;
  float bathrooms = 0.0f;
  int32_t year_built = 0;
  double last_sale_price_usd = 0.0;
  int32_t last_sale_epoch_day = 0;
};

// Everything a shard stores about a property besides its embedding.
struct PropertyRecord {
  int64_t id = 0;
  routing::LatLng location;
  PropertyAttributes attributes;
};

// Column-oriented metadata for every node, indexed by node id. Columns are
// sized to capacity up front, so addresses never move and one writer may
// Append while readers access rows the owner has already published.
class PropertyStore {
 public:
  explicit PropertyStore(size_t capacity);

  PropertyStore(PropertyStore&&) noexcept = default;
  PropertyStore& operator=(PropertyStore&&) noexcept = default;
  PropertyStore(const PropertyStore&) = delete;
  PropertyStore& operator=(const PropertyStore&) = delete;

  // Stores `record` in the next row, deriving its unit vector and GeoHash
  // cell. Returns the row id, or nullopt when full or the location is
  // invalid. Not thread-safe with respect to other appends.
  std::optional<uint32_t> Append(const PropertyRecord& record);

  int64_t id(uint32_t row) const { return ids_[row]; }
  routing::LatLng location(uint32_t row) const { return locations_[row]; }
  const PropertyAttributes& attributes(uint32_t row) const {
    return attributes_[row];
  }
  // Unit vector on the sphere (see routing::ToUnitVector).
  const std::array<double, 3>& unit_vector(uint32_t row) const {
    return unit_vectors_[row];
  }
  // GeoHash cell at kNodeCellPrecision, as interleaved bits.
  uint32_t cell6(uint32_t row) const { return cells_[row]; }
  // attributes(row).last_sale_epoch_day, from a dense column for filters.
  int32_t last_sale_epoch_day(uint32_t row) const { return sale_days_[row]; }
  const uint32_t* cell6_data() const { return cells_.data(); }

  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }
  size_t ResidentBytes() const;

 private:
  size_t capacity_ = 0;
  size_t size_ = 0;
  std::vector<int64_t> ids_;
  std::vector<routing::LatLng> locations_;
  std::vector<std::array<double, 3>> unit_vectors_;
  std::vector<uint32_t> cells_;
  std::vector<int32_t> sale_days_;
  std::vector<PropertyAttributes> attributes_;
};

}  // namespace vectra::index

#endif  // VECTRA_INDEX_PROPERTY_STORE_H_
