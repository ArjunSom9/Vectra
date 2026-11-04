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

#include "vectra/index/pq_codec.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "vectra/index/simd_distance.h"

namespace vectra::index {
namespace {

// Symmetric perturbation used when splitting a cluster to refill an empty
// one, as in FAISS [29].
constexpr float kSplitEpsilon = 1.0f / 1024.0f;

// Uniform index in [0, bound) from a 64-bit engine. std::uniform_*
// distributions are implementation-defined, which would make trained
// codebooks differ between libstdc++ and libc++; this does not.
size_t UniformIndex(std::mt19937_64& rng, size_t bound) {
  return static_cast<size_t>(rng() % bound);
}

// Lloyd's k-means over `count` points of `dim` floats (row-major), writing
// `k` centroids. Fewer points than centroids simply repeats points.
void TrainKMeans(const float* points, size_t count, int dim, int k,
                 int iterations, uint64_t seed, float* centroids) {
  std::mt19937_64 rng(seed);
  std::vector<size_t> order(count);
  std::iota(order.begin(), order.end(), size_t{0});
  for (size_t i = 0; i < std::min(count, static_cast<size_t>(k)); ++i) {
    std::swap(order[i], order[i + UniformIndex(rng, count - i)]);
  }
  for (int c = 0; c < k; ++c) {
    std::memcpy(centroids + static_cast<size_t>(c) * dim,
                points + order[static_cast<size_t>(c) % count] * dim,
                sizeof(float) * dim);
  }
  if (count <= static_cast<size_t>(k)) return;

  std::vector<int> assignment(count, 0);
  std::vector<double> sums(static_cast<size_t>(k) * dim);
  std::vector<size_t> sizes(static_cast<size_t>(k));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    for (size_t i = 0; i < count; ++i) {
      const float* point = points + i * dim;
      float best = std::numeric_limits<float>::max();
      int best_index = 0;
      for (int c = 0; c < k; ++c) {
        const float d = L2SquaredDistance(
            point, centroids + static_cast<size_t>(c) * dim, dim);
        if (d < best) {
          best = d;
          best_index = c;
        }
      }
      assignment[i] = best_index;
    }
    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(sizes.begin(), sizes.end(), size_t{0});
    for (size_t i = 0; i < count; ++i) {
      const size_t c = static_cast<size_t>(assignment[i]);
      ++sizes[c];
      for (int d = 0; d < dim; ++d) sums[c * dim + d] += points[i * dim + d];
    }
    for (size_t c = 0; c < static_cast<size_t>(k); ++c) {
      if (sizes[c] == 0) continue;
      for (int d = 0; d < dim; ++d) {
        centroids[c * dim + d] = static_cast<float>(
            sums[c * dim + d] / static_cast<double>(sizes[c]));
      }
    }
    // Refill each empty cluster by splitting the currently largest one.
    for (size_t c = 0; c < static_cast<size_t>(k); ++c) {
      if (sizes[c] != 0) continue;
      const size_t largest = static_cast<size_t>(
          std::max_element(sizes.begin(), sizes.end()) - sizes.begin());
      for (int d = 0; d < dim; ++d) {
        const float value = centroids[largest * dim + d];
        centroids[c * dim + d] = value * (1.0f + kSplitEpsilon);
        centroids[largest * dim + d] = value * (1.0f - kSplitEpsilon);
      }
      sizes[c] = sizes[largest] / 2;
      sizes[largest] -= sizes[c];
    }
  }
}

}  // namespace

ProductQuantizer::ProductQuantizer(int dimension, int num_subspaces,
                                   std::vector<float> codebooks)
    : dimension_(dimension),
      num_subspaces_(num_subspaces),
      subspace_dimension_(dimension / num_subspaces),
      codebooks_(std::move(codebooks)) {}

std::optional<ProductQuantizer> ProductQuantizer::Train(
    const float* vectors, size_t count, size_t stride_floats, int dimension,
    const PqTrainOptions& options, std::string* error) {
  const int m = options.num_subspaces;
  if (m <= 0 || dimension <= 0 || dimension % m != 0) {
    if (error != nullptr) {
      *error =
          "PQ requires the dimension to be a positive multiple of the "
          "number of sub-spaces";
    }
    return std::nullopt;
  }
  if (count == 0) {
    if (error != nullptr) *error = "PQ training needs at least one vector";
    return std::nullopt;
  }
  const int dsub = dimension / m;

  // Seeded uniform sample without replacement.
  std::mt19937_64 rng(options.seed);
  const size_t sample_size = std::min(count, options.max_training_vectors);
  std::vector<size_t> rows(count);
  std::iota(rows.begin(), rows.end(), size_t{0});
  for (size_t i = 0; i < sample_size; ++i) {
    std::swap(rows[i], rows[i + UniformIndex(rng, count - i)]);
  }

  std::vector<float> codebooks(static_cast<size_t>(m) * kPqCentroids * dsub);
  auto train_subspace = [&](int s) {
    std::vector<float> points(sample_size * dsub);
    for (size_t i = 0; i < sample_size; ++i) {
      std::memcpy(
          points.data() + i * dsub,
          vectors + rows[i] * stride_floats + static_cast<size_t>(s) * dsub,
          sizeof(float) * dsub);
    }
    TrainKMeans(
        points.data(), sample_size, dsub, kPqCentroids,
        options.kmeans_iterations, options.seed + 1 + s,
        codebooks.data() + static_cast<size_t>(s) * kPqCentroids * dsub);
  };
  const int threads = std::clamp(options.threads, 1, m);
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(threads));
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      for (int s = t; s < m; s += threads) train_subspace(s);
    });
  }
  for (std::thread& worker : workers) worker.join();
  return ProductQuantizer(dimension, m, std::move(codebooks));
}

std::optional<ProductQuantizer> ProductQuantizer::FromCodebooks(
    int dimension, int num_subspaces, std::vector<float> codebooks,
    std::string* error) {
  if (num_subspaces <= 0 || dimension <= 0 || dimension % num_subspaces != 0 ||
      codebooks.size() != static_cast<size_t>(dimension) * kPqCentroids) {
    if (error != nullptr) *error = "PQ codebooks do not match the geometry";
    return std::nullopt;
  }
  return ProductQuantizer(dimension, num_subspaces, std::move(codebooks));
}

void ProductQuantizer::Encode(const float* vector, uint8_t* code) const {
  for (int s = 0; s < num_subspaces_; ++s) {
    const float* sub = vector + s * subspace_dimension_;
    float best = std::numeric_limits<float>::max();
    int best_index = 0;
    for (int c = 0; c < kPqCentroids; ++c) {
      const float d =
          L2SquaredDistance(sub, Centroid(s, c), subspace_dimension_);
      if (d < best) {
        best = d;
        best_index = c;
      }
    }
    code[s] = static_cast<uint8_t>(best_index);
  }
}

void ProductQuantizer::Decode(const uint8_t* code, float* vector) const {
  for (int s = 0; s < num_subspaces_; ++s) {
    std::memcpy(vector + s * subspace_dimension_, Centroid(s, code[s]),
                sizeof(float) * subspace_dimension_);
  }
}

void ProductQuantizer::ComputeDistanceTable(const float* query,
                                            float* table) const {
  for (int s = 0; s < num_subspaces_; ++s) {
    const float* sub = query + s * subspace_dimension_;
    for (int c = 0; c < kPqCentroids; ++c) {
      table[s * kPqCentroids + c] =
          L2SquaredDistance(sub, Centroid(s, c), subspace_dimension_);
    }
  }
}

}  // namespace vectra::index
