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

#ifndef VECTRA_INDEX_PQ_CODEC_H_
#define VECTRA_INDEX_PQ_CODEC_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vectra/index/simd_distance.h"

namespace vectra::index {

struct PqTrainOptions {
  // Number of sub-quantizers m; the code is m bytes long.
  int num_subspaces = 16;
  int kmeans_iterations = 25;
  // Training uses a seeded uniform sample of at most this many vectors.
  size_t max_training_vectors = 65536;
  // Sub-spaces are trained concurrently on up to this many threads.
  int threads = 1;
  uint64_t seed = 1;
};

// Product quantization (Jegou, Douze & Schmid [22]): a vector is split into
// m sub-vectors, each replaced by the index of its nearest centroid in a
// 256-entry codebook, so a 128-dimensional FP32 vector (512 bytes) becomes an
// m-byte code. At m = 16 that is 32x compression; at m = 8, 64x.
//
// Query time uses asymmetric distance computation: the query stays
// uncompressed and a per-query table of sub-space distances turns every code
// distance into m table lookups (see PqAdcDistance).
//
// Immutable after construction and safe to share across threads.
class ProductQuantizer {
 public:
  // Trains codebooks by Lloyd's k-means in every sub-space over `count`
  // vectors of `dimension` floats laid out `stride_floats` apart. Fails when
  // `dimension` is not divisible by the number of sub-spaces or no vectors
  // are given.
  static std::optional<ProductQuantizer> Train(
      const float* vectors, size_t count, size_t stride_floats, int dimension,
      const PqTrainOptions& options, std::string* error);

  // Wraps previously trained codebooks laid out as
  // [subspace][centroid][subspace dimension].
  static std::optional<ProductQuantizer> FromCodebooks(
      int dimension, int num_subspaces, std::vector<float> codebooks,
      std::string* error);

  ProductQuantizer(const ProductQuantizer&) = default;
  ProductQuantizer& operator=(const ProductQuantizer&) = default;
  ProductQuantizer(ProductQuantizer&&) noexcept = default;
  ProductQuantizer& operator=(ProductQuantizer&&) noexcept = default;

  int dimension() const { return dimension_; }
  int num_subspaces() const { return num_subspaces_; }
  int subspace_dimension() const { return subspace_dimension_; }
  int code_bytes() const { return num_subspaces_; }
  const std::vector<float>& codebooks() const { return codebooks_; }

  // Writes the code_bytes()-byte code of `vector` to `code`.
  void Encode(const float* vector, uint8_t* code) const;

  // Writes the reconstruction of `code` (dimension() floats) to `vector`.
  void Decode(const uint8_t* code, float* vector) const;

  // Fills `table` (num_subspaces() * kPqCentroids floats) with the squared
  // distance from each query sub-vector to each centroid of its sub-space.
  void ComputeDistanceTable(const float* query, float* table) const;

 private:
  ProductQuantizer(int dimension, int num_subspaces,
                   std::vector<float> codebooks);

  const float* Centroid(int subspace, int centroid) const {
    return codebooks_.data() +
           (static_cast<size_t>(subspace) * kPqCentroids + centroid) *
               subspace_dimension_;
  }

  int dimension_ = 0;
  int num_subspaces_ = 0;
  int subspace_dimension_ = 0;
  std::vector<float> codebooks_;
};

}  // namespace vectra::index

#endif  // VECTRA_INDEX_PQ_CODEC_H_
