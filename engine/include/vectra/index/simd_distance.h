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

// Distance kernels for the CPU search path.
//
// The kernels are written in standard C++ with sixteen independent
// accumulator lanes. That fixed-order reduction is exactly what GCC and Clang
// vectorize into packed SIMD (AVX2 or AVX-512 on x86-64, NEON on AArch64, as
// selected by cmake/DetectSIMD.cmake) without -ffast-math and without
// nonstandard intrinsics, which the Google style guide disallows outside a
// portability wrapper.

#ifndef VECTRA_INDEX_SIMD_DISTANCE_H_
#define VECTRA_INDEX_SIMD_DISTANCE_H_

#include <cstdint>

namespace vectra::index {

// Number of centroids per product-quantization sub-space (8-bit codes [22]).
inline constexpr int kPqCentroids = 256;

// Squared Euclidean distance between two `dimension`-float vectors.
float L2SquaredDistance(const float* a, const float* b, int dimension);

// Dot product of two `dimension`-float vectors.
float InnerProduct(const float* a, const float* b, int dimension);

// Squared Euclidean norm of a `dimension`-float vector.
float SquaredNorm(const float* v, int dimension);

// Asymmetric distance computation [22]: the sum over `subspaces` of
// table[s * kPqCentroids + code[s]], where `table` holds the squared
// distances from the query's sub-vectors to every centroid.
float PqAdcDistance(const float* table, const uint8_t* code, int subspaces);

// Double-precision scalar references used by the SIMD parity tests.
double L2SquaredDistanceReference(const float* a, const float* b,
                                  int dimension);
double InnerProductReference(const float* a, const float* b, int dimension);

}  // namespace vectra::index

#endif  // VECTRA_INDEX_SIMD_DISTANCE_H_
