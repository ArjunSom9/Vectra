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

#include "vectra/index/flat_store.h"

#include <cstring>
#include <optional>
#include <utility>

#include "vectra/index/simd_distance.h"

namespace vectra::index {

std::optional<FlatStore> FlatStore::Create(int dimension, size_t capacity) {
  if (dimension <= 0) return std::nullopt;
  constexpr size_t kFloatsPerLine = memory::kCacheLineBytes / sizeof(float);
  const size_t stride = (static_cast<size_t>(dimension) + kFloatsPerLine - 1) /
                        kFloatsPerLine * kFloatsPerLine;
  std::optional<memory::AlignedBuffer> vectors =
      memory::AlignedBuffer::Allocate(capacity * stride * sizeof(float));
  std::optional<memory::AlignedBuffer> norms =
      memory::AlignedBuffer::Allocate(capacity * sizeof(float));
  if (!vectors.has_value() || !norms.has_value()) return std::nullopt;
  return FlatStore(dimension, capacity, stride, std::move(*vectors),
                   std::move(*norms));
}

std::optional<uint32_t> FlatStore::Append(const float* vector) {
  if (size_ >= capacity_) return std::nullopt;
  const uint32_t id = static_cast<uint32_t>(size_);
  float* row = reinterpret_cast<float*>(vectors_.data()) +
               static_cast<size_t>(id) * stride_floats_;
  std::memcpy(row, vector, sizeof(float) * dimension_);
  reinterpret_cast<float*>(norms_.data())[id] =
      index::SquaredNorm(row, dimension_);
  ++size_;
  return id;
}

}  // namespace vectra::index
