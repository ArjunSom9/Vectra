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

#include "vectra/routing/geohash.h"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>

namespace vectra::routing {
namespace {

// Index of `value` among 2^bits equal intervals of [lo, hi], computed by
// repeated bisection exactly as the reference algorithm does: a value equal
// to a midpoint goes to the upper half, and `hi` itself to the last interval.
uint32_t BisectionIndex(double value, double lo, double hi, int bits) {
  uint32_t index = 0;
  for (int i = 0; i < bits; ++i) {
    const double mid = (lo + hi) / 2;
    index <<= 1;
    if (value >= mid) {
      index |= 1;
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return index;
}

int CharToValue(char c) {
  if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  const size_t position = kGeohashAlphabet.find(c);
  return position == std::string_view::npos ? -1 : static_cast<int>(position);
}

}  // namespace

int LongitudeBits(int precision) { return (5 * precision + 1) / 2; }

int LatitudeBits(int precision) { return 5 * precision / 2; }

GeohashCell EncodeCell(const LatLng& point, int precision) {
  CellIndex index;
  index.column =
      BisectionIndex(point.lng, -180.0, 180.0, LongitudeBits(precision));
  index.row = BisectionIndex(point.lat, -90.0, 90.0, LatitudeBits(precision));
  return Interleave(index, precision);
}

std::string EncodeGeohash(const LatLng& point, int precision) {
  return CellToString(EncodeCell(point, precision));
}

std::string CellToString(const GeohashCell& cell) {
  std::string text(static_cast<size_t>(cell.precision), '0');
  for (int i = 0; i < cell.precision; ++i) {
    const int shift = 5 * (cell.precision - 1 - i);
    text[static_cast<size_t>(i)] = kGeohashAlphabet[(cell.bits >> shift) & 31];
  }
  return text;
}

std::optional<GeohashCell> ParseGeohash(std::string_view text) {
  if (text.empty() || text.size() > kMaxGeohashPrecision) return std::nullopt;
  GeohashCell cell;
  cell.precision = static_cast<int>(text.size());
  for (char c : text) {
    const int value = CharToValue(c);
    if (value < 0) return std::nullopt;
    cell.bits = (cell.bits << 5) | static_cast<uint64_t>(value);
  }
  return cell;
}

CellIndex Deinterleave(const GeohashCell& cell) {
  CellIndex index;
  const int total = 5 * cell.precision;
  for (int i = 0; i < total; ++i) {
    const uint32_t bit =
        static_cast<uint32_t>((cell.bits >> (total - 1 - i)) & 1);
    if (i % 2 == 0) {
      index.column = (index.column << 1) | bit;
    } else {
      index.row = (index.row << 1) | bit;
    }
  }
  return index;
}

GeohashCell Interleave(const CellIndex& index, int precision) {
  GeohashCell cell;
  cell.precision = precision;
  int column_bits = LongitudeBits(precision);
  int row_bits = LatitudeBits(precision);
  const int total = 5 * precision;
  for (int i = 0; i < total; ++i) {
    uint64_t bit;
    if (i % 2 == 0) {
      --column_bits;
      bit = (index.column >> column_bits) & 1;
    } else {
      --row_bits;
      bit = (index.row >> row_bits) & 1;
    }
    cell.bits = (cell.bits << 1) | bit;
  }
  return cell;
}

GeoBox CellBox(const GeohashCell& cell) {
  const CellIndex index = Deinterleave(cell);
  const double column_span =
      360.0 / std::ldexp(1.0, LongitudeBits(cell.precision));
  const double row_span = 180.0 / std::ldexp(1.0, LatitudeBits(cell.precision));
  GeoBox box;
  box.min_lng = -180.0 + index.column * column_span;
  box.max_lng = box.min_lng + column_span;
  box.min_lat = -90.0 + index.row * row_span;
  box.max_lat = box.min_lat + row_span;
  return box;
}

LatLng CellCenter(const GeohashCell& cell) {
  const GeoBox box = CellBox(cell);
  return {(box.min_lat + box.max_lat) / 2, (box.min_lng + box.max_lng) / 2};
}

GeohashCell CellPrefix(const GeohashCell& cell, int precision) {
  GeohashCell prefix;
  prefix.precision = precision;
  prefix.bits = cell.bits >> (5 * (cell.precision - precision));
  return prefix;
}

double CellHeightKm(int precision) {
  const double degrees = 180.0 / std::ldexp(1.0, LatitudeBits(precision));
  return degrees * std::numbers::pi / 180.0 * kEarthRadiusKm;
}

double CellWidthKm(int precision, double latitude_deg) {
  const double degrees = 360.0 / std::ldexp(1.0, LongitudeBits(precision));
  return degrees * std::numbers::pi / 180.0 * kEarthRadiusKm *
         std::cos(latitude_deg * std::numbers::pi / 180.0);
}

}  // namespace vectra::routing
