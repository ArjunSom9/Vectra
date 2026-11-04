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

#include "vectra/routing/geohash_neighbors.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <vector>

namespace vectra::routing {
namespace {

// Rounding slack for the disk/box test: one millimeter.
constexpr double kIntersectionSlackKm = 1e-6;

bool Contains(const std::vector<GeohashCell>& cells, const GeohashCell& cell) {
  return std::find(cells.begin(), cells.end(), cell) != cells.end();
}

}  // namespace

std::optional<GeohashCell> OffsetCell(const GeohashCell& cell, int d_column,
                                      int d_row) {
  const CellIndex index = Deinterleave(cell);
  const int64_t columns = int64_t{1} << LongitudeBits(cell.precision);
  const int64_t rows = int64_t{1} << LatitudeBits(cell.precision);
  const int64_t row = int64_t{index.row} + d_row;
  if (row < 0 || row >= rows) return std::nullopt;
  int64_t column = (int64_t{index.column} + d_column) % columns;
  if (column < 0) column += columns;
  CellIndex moved;
  moved.column = static_cast<uint32_t>(column);
  moved.row = static_cast<uint32_t>(row);
  return Interleave(moved, cell.precision);
}

std::vector<GeohashCell> NeighborCells(const GeohashCell& cell) {
  std::vector<GeohashCell> neighbors;
  neighbors.reserve(8);
  for (int d_row = 1; d_row >= -1; --d_row) {
    for (int d_column = -1; d_column <= 1; ++d_column) {
      if (d_row == 0 && d_column == 0) continue;
      std::optional<GeohashCell> neighbor = OffsetCell(cell, d_column, d_row);
      // Coarse grids can wrap onto the cell itself or repeat a neighbor.
      if (neighbor.has_value() && !(*neighbor == cell) &&
          !Contains(neighbors, *neighbor)) {
        neighbors.push_back(*neighbor);
      }
    }
  }
  return neighbors;
}

std::optional<DiskCover> CoverDisk(const LatLng& center, double radius_km,
                                   int precision, int max_rings) {
  const GeohashCell home = EncodeCell(center, precision);
  const int64_t columns = int64_t{1} << LongitudeBits(precision);
  DiskCover cover;
  cover.cells.push_back(home);
  if (radius_km <= 0.0) return cover;
  for (int ring = 1;; ++ring) {
    // Rings revisit cells only once they wrap all the way around a parallel.
    const bool may_repeat = 2 * int64_t{ring} + 1 >= columns;
    bool ring_contributed = false;
    for (int d_row = -ring; d_row <= ring; ++d_row) {
      for (int d_column = -ring; d_column <= ring; ++d_column) {
        if (std::max(std::abs(d_row), std::abs(d_column)) != ring) continue;
        std::optional<GeohashCell> cell = OffsetCell(home, d_column, d_row);
        if (!cell.has_value()) continue;
        if (may_repeat && Contains(cover.cells, *cell)) continue;
        if (DistanceToBoxKm(center, CellBox(*cell)) >
            radius_km + kIntersectionSlackKm) {
          continue;
        }
        if (ring > max_rings) return std::nullopt;
        cover.cells.push_back(*cell);
        cover.rings = ring;
        ring_contributed = true;
      }
    }
    // The disk is connected and contains the home cell, so the set of cells
    // it touches is connected: an empty ring means no farther ring can
    // contribute either.
    if (!ring_contributed) return cover;
  }
}

}  // namespace vectra::routing
