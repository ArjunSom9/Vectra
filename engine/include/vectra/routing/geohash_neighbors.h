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

// Cell adjacency and disk covering. GeoHash is a rigid grid, so two physically
// adjacent properties can sit on opposite sides of a bisecting line with no
// common prefix [17]; recall is preserved by visiting exactly the neighboring
// cells that a search disk intersects.

#ifndef VECTRA_ROUTING_GEOHASH_NEIGHBORS_H_
#define VECTRA_ROUTING_GEOHASH_NEIGHBORS_H_

#include <optional>
#include <vector>

#include "vectra/routing/geo_math.h"
#include "vectra/routing/geohash.h"

namespace vectra::routing {

// Returns the cell displaced by `d_column` columns east and `d_row` rows
// north. Longitude wraps across the antimeridian; there is no cell beyond a
// pole, in which case nullopt is returned.
std::optional<GeohashCell> OffsetCell(const GeohashCell& cell, int d_column,
                                      int d_row);

// Returns the up-to-eight cells sharing an edge or a corner with `cell`
// (fewer in the polar rows).
std::vector<GeohashCell> NeighborCells(const GeohashCell& cell);

// The cells covering a search disk.
struct DiskCover {
  // Every cell at the requested precision that the disk intersects, home cell
  // first, then in ring order.
  std::vector<GeohashCell> cells;
  // Chebyshev ring index of the farthest contributing cell: 0 means only the
  // home cell, 1 means the cover stays within the 3x3 block.
  int rings = 0;
};

// Covers the geodesic disk of `radius_km` around `center` with cells of
// `precision`, expanding square rings around the home cell until a ring adds
// nothing. Returns nullopt when the disk reaches beyond `max_rings`.
std::optional<DiskCover> CoverDisk(const LatLng& center, double radius_km,
                                   int precision, int max_rings);

}  // namespace vectra::routing

#endif  // VECTRA_ROUTING_GEOHASH_NEIGHBORS_H_
