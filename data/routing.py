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

"""Shard routing shared by the corpus pipelines and the gateway.

Mirrors engine/src/routing/{routing_table,fanout_policy}.cc and reads the
same engine/configs/*.textproto files (schema: contracts/proto/vectra/
config.proto), so Python and C++ always agree on which shard owns a cell.

Typical usage:

  config = routing.load_engine_config('engine/configs')
  table = routing.RoutingTable.from_config(config)
  plan = routing.FanoutPolicy.from_config(config, table).plan(lat, lng, 2.0)
"""

import dataclasses
import os

from google.protobuf import text_format
import numpy as np

from data import geo
from vectra import config_pb2

_UNASSIGNED = -1


@dataclasses.dataclass
class EngineConfig:
    """The four engine configuration files."""

    index: config_pb2.IndexParams
    geohash: config_pb2.GeohashPolicy
    shard_map: config_pb2.ShardMap
    topology: config_pb2.ClusterTopology


def load_engine_config(directory: str) -> EngineConfig:
    """Parses engine/configs/*.textproto from `directory`."""

    def parse(name, message):
        with open(
            os.path.join(directory, f'{name}.textproto'), encoding='utf-8'
        ) as f:
            return text_format.Parse(f.read(), message)

    return EngineConfig(
        index=parse('hnsw_params', config_pb2.IndexParams()),
        geohash=parse('geohash_policy', config_pb2.GeohashPolicy()),
        shard_map=parse('shard_map', config_pb2.ShardMap()),
        topology=parse('cluster_topology', config_pb2.ClusterTopology()),
    )


class RoutingTable:
    """Longest-prefix map from shard-key cells to shard ids.

    Expanded into a dense array with one slot per cell, exactly like the C++
    RoutingTable, so lookups are vectorizable.
    """

    def __init__(self, precision: int, assignments: list[tuple[str, int]]):
        """Builds the table.

        Args:
          precision: Shard-key precision in characters (1..5).
          assignments: (prefix, shard id) pairs; longest prefix wins.

        Raises:
          ValueError: On an invalid, too long, or conflicting prefix.
        """
        if not 1 <= precision <= 5:
            raise ValueError(f'routing precision must be 1..5: {precision}')
        self.precision = precision
        self.assignments = list(assignments)
        seen = {}
        parsed = []
        for prefix, shard in assignments:
            cell, length = geo.parse(prefix)
            if length > precision:
                raise ValueError(f'prefix {prefix!r} longer than shard key')
            key = geo.cell_to_string(cell, length)
            if seen.setdefault(key, shard) != shard:
                raise ValueError(f'prefix {prefix!r} assigned twice')
            parsed.append((length, cell, shard))
        parsed.sort(key=lambda entry: entry[0])
        self._owners = np.full(1 << (5 * precision), _UNASSIGNED, np.int32)
        for length, cell, shard in parsed:
            shift = 5 * (precision - length)
            first = cell << shift
            self._owners[first : first + (1 << shift)] = shard

    @classmethod
    def from_config(cls, config: EngineConfig) -> 'RoutingTable':
        return cls(
            config.geohash.shard_chars,
            [(a.prefix, a.shard_id) for a in config.shard_map.assignments],
        )

    def shard_for_cell(self, cell: int, precision: int) -> int | None:
        """Owner of a cell of at least the table's precision, or None."""
        if precision < self.precision:
            return None
        owner = int(self._owners[cell >> (5 * (precision - self.precision))])
        return None if owner == _UNASSIGNED else owner

    def shard_for_point(self, lat: float, lng: float) -> int | None:
        return self.shard_for_cell(
            geo.encode_cell(lat, lng, self.precision), self.precision
        )

    def shards_for_cells(self, cells: np.ndarray, precision: int) -> np.ndarray:
        """Vectorized owners (-1 when unassigned) of uint64 cells."""
        keys = np.asarray(cells, np.uint64) >> np.uint64(
            5 * (precision - self.precision)
        )
        return self._owners[keys.astype(np.int64)]


@dataclasses.dataclass
class RoutePlan:
    """Shards that must see a query; the home shard comes first."""

    radius_km: float
    cells_intersected: int
    shard_ids: list[int]


class FanoutPolicy:
    """Routes (location, radius) queries; mirrors routing::FanoutPolicy."""

    def __init__(
        self,
        table: RoutingTable,
        default_radius_km: float,
        max_radius_km: float,
    ):
        self._table = table
        self.default_radius_km = default_radius_km
        self.max_radius_km = max_radius_km

    @classmethod
    def from_config(
        cls, config: EngineConfig, table: RoutingTable
    ) -> 'FanoutPolicy':
        return cls(
            table,
            config.geohash.default_radius_km,
            config.geohash.max_radius_km,
        )

    def plan(self, lat: float, lng: float, radius_km: float) -> RoutePlan:
        """Plans the fan-out.

        Raises:
          ValueError: For an invalid coordinate or radius, or a radius that
            would reach past the eight cells adjacent to the query's cell.
        """
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
            raise ValueError(f'invalid coordinate: ({lat}, {lng})')
        radius = self.default_radius_km if radius_km == 0 else radius_km
        if not 0 <= radius <= self.max_radius_km:
            raise ValueError(
                f'radius_km must be in [0, {self.max_radius_km}]: {radius}'
            )
        cells = geo.cover_disk(
            lat, lng, radius, self._table.precision, max_rings=1
        )
        if cells is None:
            raise ValueError(
                'radius reaches beyond the cells adjacent to the query cell; '
                'fan-out is bounded at nine cells'
            )
        shard_ids = []
        for cell in cells:
            owner = self._table.shard_for_cell(cell, self._table.precision)
            if owner is not None and owner not in shard_ids:
                shard_ids.append(owner)
        return RoutePlan(radius, len(cells), shard_ids)
