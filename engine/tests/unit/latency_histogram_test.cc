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

#include "vectra/common/latency_histogram.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace vectra {
namespace {

TEST(LatencyHistogramTest, SmallValuesAreExact) {
  LatencyHistogram histogram;
  for (uint64_t v = 1; v <= 50; ++v) histogram.Record(v);
  EXPECT_EQ(histogram.count(), 50u);
  EXPECT_EQ(histogram.Quantile(0.5), 25u);
  EXPECT_EQ(histogram.Quantile(1.0), 50u);
  EXPECT_EQ(histogram.max(), 50u);
  EXPECT_DOUBLE_EQ(histogram.Mean(), 25.5);
}

TEST(LatencyHistogramTest, QuantilesWithinOneAndAHalfPercent) {
  std::mt19937_64 rng(1);
  std::vector<uint64_t> samples;
  LatencyHistogram a;
  LatencyHistogram b;
  for (int i = 0; i < 200000; ++i) {
    // Log-normal-ish latencies around 300 microseconds, in nanoseconds.
    const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53;
    const uint64_t value = static_cast<uint64_t>(
        300000.0 * std::exp(0.5 * std::sqrt(-2.0 * std::log(1.0 - u))));
    samples.push_back(value);
    (i % 2 == 0 ? a : b).Record(value);
  }
  a.Merge(b);
  std::sort(samples.begin(), samples.end());
  for (double q : {0.5, 0.9, 0.99, 0.999}) {
    const uint64_t exact =
        samples[static_cast<size_t>(
                    std::ceil(q * static_cast<double>(samples.size()))) -
                1];
    const uint64_t estimate = a.Quantile(q);
    EXPECT_GE(estimate, exact);  // Upper bounds are conservative.
    EXPECT_LE(static_cast<double>(estimate),
              static_cast<double>(exact) * (1.0 + 1.0 / 64.0) + 1.0);
  }
}

TEST(LatencyHistogramTest, EmptyHistogramReportsZero) {
  const LatencyHistogram histogram;
  EXPECT_EQ(histogram.Quantile(0.99), 0u);
  EXPECT_EQ(histogram.Mean(), 0.0);
}

}  // namespace
}  // namespace vectra
