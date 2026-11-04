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
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace vectra {

int LatencyHistogram::BucketOf(uint64_t value) {
  if (value < kSubBuckets) return static_cast<int>(value);
  // Position of the highest set bit, at least kSubBucketBits.
  const int exponent = 63 - std::countl_zero(value);
  const int shift = exponent - kSubBucketBits;
  const int sub_bucket = static_cast<int>((value >> shift) & (kSubBuckets - 1));
  return (shift + 1) * kSubBuckets + sub_bucket;
}

uint64_t LatencyHistogram::UpperBoundOf(int bucket) {
  if (bucket < kSubBuckets) return static_cast<uint64_t>(bucket);
  const int shift = bucket / kSubBuckets - 1;
  const uint64_t sub_bucket = static_cast<uint64_t>(bucket % kSubBuckets);
  const uint64_t lower = (uint64_t{kSubBuckets} | sub_bucket) << shift;
  return lower + ((uint64_t{1} << shift) - 1);
}

void LatencyHistogram::Record(uint64_t value) {
  ++counts_[static_cast<size_t>(BucketOf(value))];
  ++count_;
  max_ = std::max(max_, value);
  sum_ = value > std::numeric_limits<uint64_t>::max() - sum_
             ? std::numeric_limits<uint64_t>::max()
             : sum_ + value;
}

void LatencyHistogram::Merge(const LatencyHistogram& other) {
  for (size_t i = 0; i < counts_.size(); ++i) counts_[i] += other.counts_[i];
  count_ += other.count_;
  max_ = std::max(max_, other.max_);
  sum_ = other.sum_ > std::numeric_limits<uint64_t>::max() - sum_
             ? std::numeric_limits<uint64_t>::max()
             : sum_ + other.sum_;
}

double LatencyHistogram::Mean() const {
  return count_ == 0 ? 0.0
                     : static_cast<double>(sum_) / static_cast<double>(count_);
}

uint64_t LatencyHistogram::Quantile(double quantile) const {
  if (count_ == 0) return 0;
  const double clamped = std::clamp(quantile, 0.0, 1.0);
  const uint64_t rank = std::max<uint64_t>(
      1,
      static_cast<uint64_t>(std::ceil(clamped * static_cast<double>(count_))));
  uint64_t seen = 0;
  for (size_t bucket = 0; bucket < counts_.size(); ++bucket) {
    seen += counts_[bucket];
    if (seen >= rank) {
      return std::min(UpperBoundOf(static_cast<int>(bucket)), max_);
    }
  }
  return max_;
}

}  // namespace vectra
