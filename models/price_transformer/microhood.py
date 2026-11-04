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

"""Geometry and micro-neighborhood features of query/comparable pairs.

The 6-character GeoHash cell (~1.2 km x 0.6 km [17] [18]) bounds the
micro-neighborhood used for economic isolation: comparables in the subject's
own cell, its 5-character cell, or its 4-character shard cell are flagged so
the model can weight them differently. All operations are branch-free
tensor arithmetic.
"""

import math

import torch

EARTH_RADIUS_KM = 6371.0088
_DEGREES_TO_RADIANS = math.pi / 180.0


def haversine_km(
    query_location: torch.Tensor, candidate_location: torch.Tensor
) -> torch.Tensor:
    """Great-circle distances from each query to its candidates.

    Args:
      query_location: [B, 2] (latitude, longitude) in degrees.
      candidate_location: [B, K, 2] in degrees.

    Returns:
      [B, K] distances in kilometers.
    """
    lat1 = query_location[:, None, 0] * _DEGREES_TO_RADIANS
    lng1 = query_location[:, None, 1] * _DEGREES_TO_RADIANS
    lat2 = candidate_location[..., 0] * _DEGREES_TO_RADIANS
    lng2 = candidate_location[..., 1] * _DEGREES_TO_RADIANS
    h = (
        torch.sin((lat2 - lat1) / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lng2 - lng1) / 2) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(h.clamp(0.0, 1.0)))


def relative_offsets_km(
    query_location: torch.Tensor, candidate_location: torch.Tensor
) -> torch.Tensor:
    """East and north offsets of candidates from the query, [B, K, 2] km.

    Local equirectangular projection; accurate to well under 1% within the
    routing policy's 15 km radius.
    """
    km_per_degree = EARTH_RADIUS_KM * _DEGREES_TO_RADIANS
    delta = candidate_location - query_location[:, None, :]
    east = (
        delta[..., 1]
        * km_per_degree
        * torch.cos(query_location[:, None, 0] * _DEGREES_TO_RADIANS)
    )
    north = delta[..., 0] * km_per_degree
    return torch.stack([east, north], dim=-1)


def same_cell_flags(
    query_cell6: torch.Tensor, candidate_cell6: torch.Tensor
) -> torch.Tensor:
    """Whether each candidate shares the query's 6-, 5- and 4-char cell.

    Args:
      query_cell6: [B] integer 6-character cells.
      candidate_cell6: [B, K] integer 6-character cells.

    Returns:
      [B, K, 3] float flags (6-, 5-, 4-character match).
    """
    query = query_cell6[:, None]
    return torch.stack(
        [
            candidate_cell6 == query,
            torch.bitwise_right_shift(candidate_cell6, 5)
            == torch.bitwise_right_shift(query, 5),
            torch.bitwise_right_shift(candidate_cell6, 10)
            == torch.bitwise_right_shift(query, 10),
        ],
        dim=-1,
    ).to(torch.float32)
