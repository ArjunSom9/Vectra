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

#include "vectra/index/simd_distance.h"

#include <cstdint>

namespace vectra::index {
namespace {

// Sixteen float lanes: one AVX-512 register, two AVX2 registers, or four NEON
// registers per accumulator group.
constexpr int kLanes = 16;

// Pairwise tree reduction keeps the rounding error at O(log kLanes).
float ReduceLanes(float (&lanes)[kLanes]) {
  for (int width = kLanes / 2; width > 0; width /= 2) {
    for (int i = 0; i < width; ++i) lanes[i] += lanes[i + width];
  }
  return lanes[0];
}

}  // namespace

float L2SquaredDistance(const float* a, const float* b, int dimension) {
  float lanes[kLanes] = {};
  int i = 0;
  for (; i + kLanes <= dimension; i += kLanes) {
    for (int j = 0; j < kLanes; ++j) {
      const float d = a[i + j] - b[i + j];
      lanes[j] += d * d;
    }
  }
  float sum = ReduceLanes(lanes);
  for (; i < dimension; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float InnerProduct(const float* a, const float* b, int dimension) {
  float lanes[kLanes] = {};
  int i = 0;
  for (; i + kLanes <= dimension; i += kLanes) {
    for (int j = 0; j < kLanes; ++j) lanes[j] += a[i + j] * b[i + j];
  }
  float sum = ReduceLanes(lanes);
  for (; i < dimension; ++i) sum += a[i] * b[i];
  return sum;
}

float SquaredNorm(const float* v, int dimension) {
  return InnerProduct(v, v, dimension);
}

float PqAdcDistance(const float* table, const uint8_t* code, int subspaces) {
  // Four independent chains hide the latency of the dependent table loads.
  float lanes[4] = {};
  int s = 0;
  for (; s + 4 <= subspaces; s += 4) {
    lanes[0] += table[(s + 0) * kPqCentroids + code[s + 0]];
    lanes[1] += table[(s + 1) * kPqCentroids + code[s + 1]];
    lanes[2] += table[(s + 2) * kPqCentroids + code[s + 2]];
    lanes[3] += table[(s + 3) * kPqCentroids + code[s + 3]];
  }
  float sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
  for (; s < subspaces; ++s) sum += table[s * kPqCentroids + code[s]];
  return sum;
}

double L2SquaredDistanceReference(const float* a, const float* b,
                                  int dimension) {
  double sum = 0.0;
  for (int i = 0; i < dimension; ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum += d * d;
  }
  return sum;
}

double InnerProductReference(const float* a, const float* b, int dimension) {
  double sum = 0.0;
  for (int i = 0; i < dimension; ++i) {
    sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return sum;
}

}  // namespace vectra::index
