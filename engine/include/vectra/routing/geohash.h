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

// GeoHash (Niemeyer, 2008 [17]): the bisected bits of longitude and latitude
// are interleaved along a Z-order curve, longitude first, and rendered five
// bits at a time in base 32. Each extra character divides a cell into 32.

#ifndef VECTRA_ROUTING_GEOHASH_H_
#define VECTRA_ROUTING_GEOHASH_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "vectra/routing/geo_math.h"

namespace vectra::routing {

inline constexpr int kMaxGeohashPrecision = 12;
inline constexpr std::string_view kGeohashAlphabet =
    "0123456789bcdefghjkmnpqrstuvwxyz";

// A GeoHash cell in integer form: `bits` holds 5 * precision interleaved
// bits, right-aligned, most significant bit first (the first character).
struct GeohashCell {
  uint64_t bits = 0;
  int precision = 0;

  friend bool operator==(const GeohashCell& a, const GeohashCell& b) = default;
};

// Column (longitude) and row (latitude) indices of a cell in the
// 2^LongitudeBits x 2^LatitudeBits grid of its precision. Column 0 is at
// -180 degrees and row 0 at -90 degrees.
struct CellIndex {
  uint32_t column = 0;
  uint32_t row = 0;
};

// Bits spent on each axis at `precision` (longitude gets the odd extra bit).
int LongitudeBits(int precision);
int LatitudeBits(int precision);

// Encodes a valid coordinate (see IsValidLatLng) at a precision in
// [1, kMaxGeohashPrecision] by the standard bisection.
GeohashCell EncodeCell(const LatLng& point, int precision);

// Same as EncodeCell, rendered as base-32 text.
std::string EncodeGeohash(const LatLng& point, int precision);

std::string CellToString(const GeohashCell& cell);

// Parses base-32 text (case-insensitive). Returns nullopt for empty, too long,
// or out-of-alphabet input.
std::optional<GeohashCell> ParseGeohash(std::string_view text);

CellIndex Deinterleave(const GeohashCell& cell);
GeohashCell Interleave(const CellIndex& index, int precision);

// The cell's bounding box and center.
GeoBox CellBox(const GeohashCell& cell);
LatLng CellCenter(const GeohashCell& cell);

// The ancestor of `cell` at a coarser (or equal) precision.
GeohashCell CellPrefix(const GeohashCell& cell, int precision);

// North-south extent of a cell, which is the same at every latitude.
double CellHeightKm(int precision);

// East-west extent of a cell along the parallel at `latitude_deg`.
double CellWidthKm(int precision, double latitude_deg);

}  // namespace vectra::routing

#endif  // VECTRA_ROUTING_GEOHASH_H_
