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

#include <cmath>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "testing/synthetic_data.h"
#include "vectra/index/simd_distance.h"

namespace vectra::index {
namespace {

// The vectorized kernels reorder the floating-point reduction; they must
// agree with a double-precision scalar loop to within float accumulation
// error at every dimension, including tails that are not a multiple of the
// sixteen-lane width.
TEST(SimdParityTest, KernelsMatchScalarReferenceAtEveryDimension) {
  for (int dimension = 1; dimension <= 300; ++dimension) {
    const std::vector<float> data =
        testing::ClusteredVectors(2, dimension, 1, 1.0f, dimension);
    const float* a = data.data();
    const float* b = data.data() + dimension;
    const double l2 = L2SquaredDistanceReference(a, b, dimension);
    const double dot = InnerProductReference(a, b, dimension);
    const double tolerance = 1e-5 * (1.0 + dimension);
    EXPECT_NEAR(L2SquaredDistance(a, b, dimension), l2, tolerance * (1.0 + l2))
        << "dimension " << dimension;
    EXPECT_NEAR(InnerProduct(a, b, dimension), dot,
                tolerance * (1.0 + std::fabs(dot)))
        << "dimension " << dimension;
    EXPECT_NEAR(SquaredNorm(a, dimension),
                InnerProductReference(a, a, dimension),
                tolerance * (1.0 + InnerProductReference(a, a, dimension)));
  }
}

TEST(SimdParityTest, AdcSumsTableEntries) {
  for (int subspaces : {1, 3, 4, 8, 16, 17, 32}) {
    std::vector<float> table(static_cast<size_t>(subspaces) * kPqCentroids);
    for (size_t i = 0; i < table.size(); ++i) {
      table[i] = static_cast<float>(i % 97) * 0.25f;
    }
    std::vector<uint8_t> code(static_cast<size_t>(subspaces));
    double expected = 0.0;
    for (int s = 0; s < subspaces; ++s) {
      code[static_cast<size_t>(s)] = static_cast<uint8_t>((s * 37 + 11) % 256);
      expected += table[static_cast<size_t>(s) * kPqCentroids +
                        code[static_cast<size_t>(s)]];
    }
    EXPECT_NEAR(PqAdcDistance(table.data(), code.data(), subspaces), expected,
                1e-4 * (1.0 + expected));
  }
}

}  // namespace
}  // namespace vectra::index
