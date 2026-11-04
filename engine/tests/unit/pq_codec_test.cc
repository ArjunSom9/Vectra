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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "testing/synthetic_data.h"
#include "vectra/index/simd_distance.h"

namespace vectra::index {
namespace {

constexpr int kDimension = 32;

ProductQuantizer TrainOn(const std::vector<float>& data, int subspaces,
                         int threads = 1) {
  PqTrainOptions options;
  options.num_subspaces = subspaces;
  options.kmeans_iterations = 12;
  options.threads = threads;
  options.seed = 9;
  std::string error;
  std::optional<ProductQuantizer> pq =
      ProductQuantizer::Train(data.data(), data.size() / kDimension, kDimension,
                              kDimension, options, &error);
  EXPECT_TRUE(pq.has_value()) << error;
  return *pq;
}

TEST(PqCodecTest, CodeSizeIsOneBytePerSubspace) {
  const std::vector<float> data =
      testing::ClusteredVectors(2000, kDimension, 16, 0.05f, 1);
  const ProductQuantizer pq = TrainOn(data, 8);
  EXPECT_EQ(pq.code_bytes(), 8);
  EXPECT_EQ(pq.subspace_dimension(), 4);
  // 32 FP32 dimensions = 128 bytes, compressed to 8: 16x.
  EXPECT_EQ(kDimension * sizeof(float) / pq.code_bytes(), 16u);
}

TEST(PqCodecTest, ReconstructionErrorIsSmallForClusteredData) {
  const std::vector<float> data =
      testing::ClusteredVectors(4000, kDimension, 16, 0.02f, 2);
  const ProductQuantizer pq = TrainOn(data, 8);
  std::vector<uint8_t> code(8);
  std::vector<float> decoded(kDimension);
  double error = 0.0;
  double energy = 0.0;
  for (size_t i = 0; i < 4000; ++i) {
    const float* vector = data.data() + i * kDimension;
    pq.Encode(vector, code.data());
    pq.Decode(code.data(), decoded.data());
    error += L2SquaredDistanceReference(vector, decoded.data(), kDimension);
    energy += InnerProductReference(vector, vector, kDimension);
  }
  EXPECT_LT(error / energy, 0.01);
}

TEST(PqCodecTest, AsymmetricDistanceEqualsDistanceToReconstruction) {
  const std::vector<float> data =
      testing::ClusteredVectors(3000, kDimension, 8, 0.1f, 3);
  const ProductQuantizer pq = TrainOn(data, 16, /*threads=*/4);
  const std::vector<float> queries =
      testing::ClusteredVectors(20, kDimension, 8, 0.1f, 4);
  std::vector<float> table(16 * kPqCentroids);
  std::vector<uint8_t> code(16);
  std::vector<float> decoded(kDimension);
  for (size_t q = 0; q < 20; ++q) {
    const float* query = queries.data() + q * kDimension;
    pq.ComputeDistanceTable(query, table.data());
    for (size_t i = 0; i < 100; ++i) {
      pq.Encode(data.data() + i * kDimension, code.data());
      pq.Decode(code.data(), decoded.data());
      const double expected =
          L2SquaredDistanceReference(query, decoded.data(), kDimension);
      EXPECT_NEAR(PqAdcDistance(table.data(), code.data(), 16), expected,
                  1e-4 * (1.0 + expected));
    }
  }
}

TEST(PqCodecTest, TrainingIsDeterministicAcrossThreadCounts) {
  const std::vector<float> data =
      testing::ClusteredVectors(1500, kDimension, 8, 0.1f, 5);
  EXPECT_EQ(TrainOn(data, 8, 1).codebooks(), TrainOn(data, 8, 4).codebooks());
}

TEST(PqCodecTest, HandlesFewerVectorsThanCentroids) {
  const std::vector<float> data =
      testing::ClusteredVectors(10, kDimension, 2, 0.1f, 6);
  const ProductQuantizer pq = TrainOn(data, 4);
  std::vector<uint8_t> code(4);
  std::vector<float> decoded(kDimension);
  pq.Encode(data.data(), code.data());
  pq.Decode(code.data(), decoded.data());
  // Every training vector is itself a centroid, so it is reproduced exactly.
  EXPECT_DOUBLE_EQ(
      L2SquaredDistanceReference(data.data(), decoded.data(), kDimension), 0.0);
}

TEST(PqCodecTest, RejectsIndivisibleGeometryAndRoundTripsCodebooks) {
  const std::vector<float> data =
      testing::ClusteredVectors(100, kDimension, 2, 0.1f, 7);
  PqTrainOptions options;
  options.num_subspaces = 5;  // 32 % 5 != 0.
  std::string error;
  EXPECT_FALSE(ProductQuantizer::Train(data.data(), 100, kDimension, kDimension,
                                       options, &error)
                   .has_value());
  EXPECT_FALSE(error.empty());

  const ProductQuantizer pq = TrainOn(data, 8);
  std::optional<ProductQuantizer> copy =
      ProductQuantizer::FromCodebooks(kDimension, 8, pq.codebooks(), &error);
  ASSERT_TRUE(copy.has_value()) << error;
  EXPECT_EQ(copy->codebooks(), pq.codebooks());
  EXPECT_FALSE(ProductQuantizer::FromCodebooks(kDimension, 8, {1.0f}, &error));
}

}  // namespace
}  // namespace vectra::index
