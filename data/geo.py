# Copyright 2025 Texas A&M University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GeoHash cells, adjacency, disk covering and great-circle distance.

The Python mirror of engine/src/routing: the gateway's router and the corpus
pipelines must make exactly the decisions the C++ engine makes, so both
implementations encode by the same bisection and are tested against the same
reference values.

A cell is an integer holding 5 * precision interleaved bits (longitude first,
most significant bit first), as in the GeoHash definition [17].

Typical usage:

  cell = geo.encode_cell(37.7749, -122.4194, 4)
  geo.cell_to_string(cell, 4)  # '9q8y'
  geo.cover_disk(37.7749, -122.4194, 2.0, 4, max_rings=1)
"""

from collections.abc import Sequence
import dataclasses
import math

import numpy as np

ALPHABET = '0123456789bcdefghjkmnpqrstuvwxyz'
MAX_PRECISION = 12
EARTH_RADIUS_KM = 6371.0088

_INTERSECTION_SLACK_KM = 1e-6


@dataclasses.dataclass(frozen=True)
class Box:
    """A latitude/longitude-aligned box in decimal degrees."""

    min_lat: float
    max_lat: float
    min_lng: float
    max_lng: float


def longitude_bits(precision: int) -> int:
    """Bits of longitude in a cell of `precision` characters."""
    return (5 * precision + 1) // 2


def latitude_bits(precision: int) -> int:
    """Bits of latitude in a cell of `precision` characters."""
    return 5 * precision // 2


def _bisect(value: float, lo: float, hi: float, bits: int) -> int:
    index = 0
    for _ in range(bits):
        mid = (lo + hi) / 2
        index <<= 1
        if value >= mid:
            index |= 1
            lo = mid
        else:
            hi = mid
    return index


def interleave(column: int, row: int, precision: int) -> int:
    """Merges longitude (column) and latitude (row) indices into a cell."""
    column_bits = longitude_bits(precision)
    row_bits = latitude_bits(precision)
    cell = 0
    for i in range(5 * precision):
        if i % 2 == 0:
            column_bits -= 1
            bit = (column >> column_bits) & 1
        else:
            row_bits -= 1
            bit = (row >> row_bits) & 1
        cell = (cell << 1) | bit
    return cell


def deinterleave(cell: int, precision: int) -> tuple[int, int]:
    """Splits a cell into its (column, row) grid indices."""
    total = 5 * precision
    column = row = 0
    for i in range(total):
        bit = (cell >> (total - 1 - i)) & 1
        if i % 2 == 0:
            column = (column << 1) | bit
        else:
            row = (row << 1) | bit
    return column, row


def encode_cell(lat: float, lng: float, precision: int) -> int:
    """Encodes a valid coordinate as a cell of `precision` characters."""
    column = _bisect(lng, -180.0, 180.0, longitude_bits(precision))
    row = _bisect(lat, -90.0, 90.0, latitude_bits(precision))
    return interleave(column, row, precision)


def encode(lat: float, lng: float, precision: int) -> str:
    """Encodes a valid coordinate as base-32 GeoHash text."""
    return cell_to_string(encode_cell(lat, lng, precision), precision)


def cell_to_string(cell: int, precision: int) -> str:
    """Renders a cell as base-32 text."""
    return ''.join(
        ALPHABET[(cell >> (5 * (precision - 1 - i))) & 31]
        for i in range(precision)
    )


def parse(text: str) -> tuple[int, int]:
    """Parses GeoHash text into (cell, precision).

    Raises:
      ValueError: If the text is empty, too long, or not base-32.
    """
    if not text or len(text) > MAX_PRECISION:
        raise ValueError(
            f'GeoHash must have 1..{MAX_PRECISION} chars: {text!r}'
        )
    cell = 0
    for char in text.lower():
        value = ALPHABET.find(char)
        if value < 0:
            raise ValueError(f'Not a GeoHash character: {char!r} in {text!r}')
        cell = (cell << 5) | value
    return cell, len(text)


def cell_box(cell: int, precision: int) -> Box:
    """The bounding box of a cell."""
    column, row = deinterleave(cell, precision)
    column_span = 360.0 / (1 << longitude_bits(precision))
    row_span = 180.0 / (1 << latitude_bits(precision))
    min_lng = -180.0 + column * column_span
    min_lat = -90.0 + row * row_span
    return Box(min_lat, min_lat + row_span, min_lng, min_lng + column_span)


def offset_cell(
    cell: int, precision: int, d_column: int, d_row: int
) -> int | None:
    """The cell displaced by (d_column east, d_row north); None past a pole."""
    column, row = deinterleave(cell, precision)
    row += d_row
    if row < 0 or row >= (1 << latitude_bits(precision)):
        return None
    column = (column + d_column) % (1 << longitude_bits(precision))
    return interleave(column, row, precision)


def neighbors(cell: int, precision: int) -> list[int]:
    """The up-to-eight cells sharing an edge or corner with `cell`."""
    found = []
    for d_row in (1, 0, -1):
        for d_column in (-1, 0, 1):
            if d_row == 0 and d_column == 0:
                continue
            other = offset_cell(cell, precision, d_column, d_row)
            if other is not None and other != cell and other not in found:
                found.append(other)
    return found


def _wrap_degrees(delta: float) -> float:
    wrapped = math.fmod(delta, 360.0)
    if wrapped > 180.0:
        wrapped -= 360.0
    if wrapped <= -180.0:
        wrapped += 360.0
    return wrapped


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance by the haversine formula."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(_wrap_degrees(lng2 - lng1))
    h = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def haversine_km_array(
    lat1: np.ndarray, lng1: np.ndarray, lat2: np.ndarray, lng2: np.ndarray
) -> np.ndarray:
    """Vectorized haversine_km (broadcasting)."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlmb = np.radians(
        (np.asarray(lng2) - np.asarray(lng1) + 180.0) % 360.0 - 180.0
    )
    h = (
        np.sin((phi2 - phi1) / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(h)))


def _longitude_within(lng: float, box: Box) -> bool:
    if box.min_lng <= lng <= box.max_lng:
        return True
    alternate = lng - 360.0 if lng > 0 else lng + 360.0
    return box.min_lng <= alternate <= box.max_lng


def _meridian_arc_distance_km(
    lat: float, lng: float, edge_lng: float, min_lat: float, max_lat: float
) -> float:
    best = min(
        haversine_km(lat, lng, min_lat, edge_lng),
        haversine_km(lat, lng, max_lat, edge_lng),
    )
    dlng = math.radians(_wrap_degrees(lng - edge_lng))
    if abs(dlng) <= math.pi / 2:
        phi = math.radians(lat)
        foot = math.degrees(
            math.atan2(math.sin(phi), math.cos(phi) * math.cos(dlng))
        )
        if min_lat <= foot <= max_lat:
            best = min(best, haversine_km(lat, lng, foot, edge_lng))
    return best


def distance_to_box_km(lat: float, lng: float, box: Box) -> float:
    """Shortest great-circle distance from a point to a box (0 inside)."""
    if box.min_lat <= lat <= box.max_lat and _longitude_within(lng, box):
        return 0.0
    if _longitude_within(lng, box):
        edge_lng = lng
    else:
        to_min = abs(_wrap_degrees(lng - box.min_lng))
        to_max = abs(_wrap_degrees(lng - box.max_lng))
        edge_lng = box.min_lng if to_min <= to_max else box.max_lng
    return min(
        haversine_km(lat, lng, box.min_lat, edge_lng),
        haversine_km(lat, lng, box.max_lat, edge_lng),
        _meridian_arc_distance_km(
            lat, lng, box.min_lng, box.min_lat, box.max_lat
        ),
        _meridian_arc_distance_km(
            lat, lng, box.max_lng, box.min_lat, box.max_lat
        ),
    )


def cover_disk(
    lat: float, lng: float, radius_km: float, precision: int, max_rings: int
) -> list[int] | None:
    """Cells of `precision` that intersect the disk, home cell first.

    Square rings around the home cell are added until a ring contributes
    nothing (the touched cells form a connected set).

    Returns:
      The cells, or None when the disk reaches beyond `max_rings` rings.
    """
    home = encode_cell(lat, lng, precision)
    cells = [home]
    if radius_km <= 0:
        return cells
    columns = 1 << longitude_bits(precision)
    ring = 0
    while True:
        ring += 1
        may_repeat = 2 * ring + 1 >= columns
        contributed = False
        for d_row in range(-ring, ring + 1):
            for d_column in range(-ring, ring + 1):
                if max(abs(d_row), abs(d_column)) != ring:
                    continue
                cell = offset_cell(home, precision, d_column, d_row)
                if cell is None or (may_repeat and cell in cells):
                    continue
                box = cell_box(cell, precision)
                if distance_to_box_km(lat, lng, box) > (
                    radius_km + _INTERSECTION_SLACK_KM
                ):
                    continue
                if ring > max_rings:
                    return None
                cells.append(cell)
                contributed = True
        if not contributed:
            return cells


def encode_cells(
    lat: np.ndarray, lng: np.ndarray, precision: int
) -> np.ndarray:
    """Vectorized encode_cell; returns uint64 cells.

    Uses the same bisection comparisons as encode_cell, so results agree
    bit-for-bit with the scalar version and the C++ engine.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lng = np.asarray(lng, dtype=np.float64)

    def bisect(values, lo, hi, bits):
        lo = np.full(values.shape, lo)
        hi = np.full(values.shape, hi)
        index = np.zeros(values.shape, dtype=np.uint64)
        for _ in range(bits):
            mid = (lo + hi) / 2
            upper = values >= mid
            index = (index << np.uint64(1)) | upper.astype(np.uint64)
            lo = np.where(upper, mid, lo)
            hi = np.where(upper, hi, mid)
        return index

    columns = bisect(lng, -180.0, 180.0, longitude_bits(precision))
    rows = bisect(lat, -90.0, 90.0, latitude_bits(precision))
    cells = np.zeros(lat.shape, dtype=np.uint64)
    column_bits = longitude_bits(precision)
    row_bits = latitude_bits(precision)
    for i in range(5 * precision):
        if i % 2 == 0:
            column_bits -= 1
            bit = (columns >> np.uint64(column_bits)) & np.uint64(1)
        else:
            row_bits -= 1
            bit = (rows >> np.uint64(row_bits)) & np.uint64(1)
        cells = (cells << np.uint64(1)) | bit
    return cells


def cell_prefix(
    cells: np.ndarray | int, precision: int, coarser: int
) -> np.ndarray | int:
    """Ancestors at `coarser` precision of cells at `precision`."""
    shift = 5 * (precision - coarser)
    if isinstance(cells, np.ndarray):
        return cells >> np.uint64(shift)
    return cells >> shift


def cell_height_km(precision: int) -> float:
    """North-south extent of a cell."""
    return (
        180.0
        / (1 << latitude_bits(precision))
        * math.pi
        / 180.0
        * EARTH_RADIUS_KM
    )


def cell_width_km(precision: int, latitude: float) -> float:
    """East-west extent of a cell along the parallel at `latitude`."""
    return (
        360.0
        / (1 << longitude_bits(precision))
        * math.pi
        / 180.0
        * EARTH_RADIUS_KM
        * math.cos(math.radians(latitude))
    )


def destination(
    lat: float, lng: float, bearing_deg: float, distance_km: float
) -> tuple[float, float]:
    """The point `distance_km` from (lat, lng) along an initial bearing."""
    angle = distance_km / EARTH_RADIUS_KM
    bearing = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lmb1 = math.radians(lng)
    phi2 = math.asin(
        math.sin(phi1) * math.cos(angle)
        + math.cos(phi1) * math.sin(angle) * math.cos(bearing)
    )
    lmb2 = lmb1 + math.atan2(
        math.sin(bearing) * math.sin(angle) * math.cos(phi1),
        math.cos(angle) - math.sin(phi1) * math.sin(phi2),
    )
    lng2 = (math.degrees(lmb2) + 180.0) % 360.0 - 180.0
    return math.degrees(phi2), lng2


def strings(cells: Sequence[int], precision: int) -> list[str]:
    """Renders many cells as text."""
    return [cell_to_string(int(cell), precision) for cell in cells]
