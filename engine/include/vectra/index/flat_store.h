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

#ifndef VECTRA_INDEX_FLAT_STORE_H_
#define VECTRA_INDEX_FLAT_STORE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "vectra/memory/aligned_buffer.h"

namespace vectra::index {

// Full-precision (FP32) embeddings in one contiguous, cache-line-aligned
// block, plus each row's squared norm.
//
// The HNSW graph navigates on compact PQ codes; this store supplies the exact
// distances for the final re-rank of the best candidates (the compressed
// traversal / exact re-rank split used by DiskANN [12] and LVQ [30]), for
// graph construction, and for exhaustive scans of small radius populations.
// The squared norms are the precomputed ||k||^2 shipped to the inference tier.
//
// One writer may Append while readers access rows below the count the owner
// has published; addresses never move.
class FlatStore {
 public:
  // Reserves room for `capacity` vectors of `dimension` floats.
  static std::optional<FlatStore> Create(int dimension, size_t capacity);

  FlatStore(FlatStore&&) noexcept = default;
  FlatStore& operator=(FlatStore&&) noexcept = default;
  FlatStore(const FlatStore&) = delete;
  FlatStore& operator=(const FlatStore&) = delete;

  // Copies `vector` into the next row and returns its id, or nullopt when the
  // store is full. Not thread-safe with respect to other appends.
  std::optional<uint32_t> Append(const float* vector);

  const float* Row(uint32_t id) const {
    return reinterpret_cast<const float*>(vectors_.data()) +
           static_cast<size_t>(id) * stride_floats_;
  }
  float SquaredNorm(uint32_t id) const {
    return reinterpret_cast<const float*>(norms_.data())[id];
  }

  int dimension() const { return dimension_; }
  // Floats between consecutive rows (dimension rounded up to a cache line).
  size_t stride_floats() const { return stride_floats_; }
  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }
  size_t ResidentBytes() const { return vectors_.size() + norms_.size(); }

  const void* vectors_data() const { return vectors_.data(); }
  size_t vectors_bytes() const { return vectors_.size(); }
  const void* norms_data() const { return norms_.data(); }
  size_t norms_bytes() const { return norms_.size(); }

 private:
  FlatStore(int dimension, size_t capacity, size_t stride_floats,
            memory::AlignedBuffer vectors, memory::AlignedBuffer norms)
      : dimension_(dimension),
        capacity_(capacity),
        stride_floats_(stride_floats),
        vectors_(std::move(vectors)),
        norms_(std::move(norms)) {}

  int dimension_ = 0;
  size_t capacity_ = 0;
  size_t stride_floats_ = 0;
  size_t size_ = 0;
  memory::AlignedBuffer vectors_;
  memory::AlignedBuffer norms_;
};

}  // namespace vectra::index

#endif  // VECTRA_INDEX_FLAT_STORE_H_
