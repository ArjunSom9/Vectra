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

#ifndef VECTRA_COMMON_LATENCY_HISTOGRAM_H_
#define VECTRA_COMMON_LATENCY_HISTOGRAM_H_

#include <array>
#include <cstdint>

namespace vectra {

// A fixed-size log-linear histogram of non-negative integer latencies (for
// example nanoseconds). Values below 64 are counted exactly; above that, each
// power of two is split into 64 linear sub-buckets, bounding the relative
// quantile error at 1/64 (about 1.6%). Recording is O(1) and allocation-free,
// so a load generator can keep one histogram per thread for a 30-minute,
// 50,000 QPS soak and merge them at the end.
//
// Not thread-safe; give each thread its own instance and Merge.
class LatencyHistogram {
 public:
  LatencyHistogram() = default;

  void Record(uint64_t value);
  void Merge(const LatencyHistogram& other);

  uint64_t count() const { return count_; }
  uint64_t max() const { return max_; }
  double Mean() const;

  // The smallest recorded bucket value v such that at least `quantile` of the
  // samples are <= v, reported as the bucket's upper bound (conservative).
  // `quantile` is in [0, 1]; returns 0 for an empty histogram.
  uint64_t Quantile(double quantile) const;

 private:
  static constexpr int kSubBucketBits = 6;
  static constexpr int kSubBuckets = 1 << kSubBucketBits;
  static constexpr int kBuckets = kSubBuckets * (64 - kSubBucketBits + 1);

  static int BucketOf(uint64_t value);
  static uint64_t UpperBoundOf(int bucket);

  std::array<uint64_t, kBuckets> counts_{};
  uint64_t count_ = 0;
  uint64_t max_ = 0;
  // Sum of samples, for the mean; saturates rather than overflowing.
  uint64_t sum_ = 0;
};

}  // namespace vectra

#endif  // VECTRA_COMMON_LATENCY_HISTOGRAM_H_
