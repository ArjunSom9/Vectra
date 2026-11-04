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

#include <cstdint>
#include <optional>
#include <vector>

#include "gtest/gtest.h"
#include "testing/synthetic_data.h"
#include "vectra/index/simd_distance.h"
#include "vectra/memory/aligned_buffer.h"

namespace vectra::index {
namespace {

TEST(FlatStoreTest, AppendsRowsWithCacheLineStrideAndNorms) {
  constexpr int kDimension = 20;
  std::optional<FlatStore> store = FlatStore::Create(kDimension, 50);
  ASSERT_TRUE(store.has_value());
  EXPECT_EQ(store->stride_floats(), 32u);  // 20 floats rounded to 2 lines.
  const std::vector<float> data =
      testing::ClusteredVectors(50, kDimension, 4, 0.5f, 1);
  for (uint32_t i = 0; i < 50; ++i) {
    ASSERT_EQ(store->Append(data.data() + i * kDimension), i);
  }
  EXPECT_EQ(store->size(), 50u);
  EXPECT_FALSE(store->Append(data.data()).has_value());  // Full.
  for (uint32_t i = 0; i < 50; ++i) {
    const float* row = store->Row(i);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(row) % memory::kCacheLineBytes, 0u);
    for (int d = 0; d < kDimension; ++d) {
      EXPECT_EQ(row[d], data[i * kDimension + static_cast<size_t>(d)]);
    }
    EXPECT_NEAR(store->SquaredNorm(i),
                InnerProductReference(row, row, kDimension), 1e-4);
  }
}

TEST(FlatStoreTest, RejectsNonPositiveDimension) {
  EXPECT_FALSE(FlatStore::Create(0, 10).has_value());
}

}  // namespace
}  // namespace vectra::index
