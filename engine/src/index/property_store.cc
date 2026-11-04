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

#include "vectra/index/property_store.h"

#include <optional>

#include "vectra/routing/geohash.h"

namespace vectra::index {

PropertyStore::PropertyStore(size_t capacity)
    : capacity_(capacity),
      ids_(capacity),
      locations_(capacity),
      unit_vectors_(capacity),
      cells_(capacity),
      sale_days_(capacity),
      attributes_(capacity) {}

std::optional<uint32_t> PropertyStore::Append(const PropertyRecord& record) {
  if (size_ >= capacity_ || !routing::IsValidLatLng(record.location)) {
    return std::nullopt;
  }
  const uint32_t row = static_cast<uint32_t>(size_);
  ids_[row] = record.id;
  locations_[row] = record.location;
  unit_vectors_[row] = routing::ToUnitVector(record.location);
  cells_[row] = static_cast<uint32_t>(
      routing::EncodeCell(record.location, kNodeCellPrecision).bits);
  sale_days_[row] = record.attributes.last_sale_epoch_day;
  attributes_[row] = record.attributes;
  ++size_;
  return row;
}

size_t PropertyStore::ResidentBytes() const {
  return capacity_ * (sizeof(int64_t) + sizeof(routing::LatLng) +
                      sizeof(std::array<double, 3>) + sizeof(uint32_t) +
                      sizeof(int32_t) + sizeof(PropertyAttributes));
}

}  // namespace vectra::index
