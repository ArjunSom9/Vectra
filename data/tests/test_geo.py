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

"""The Python routing mirror must agree with the C++ engine exactly.

These are the same reference values engine/tests/unit/geohash*_test.cc use.
"""

import numpy as np
import pytest

from data import geo
from data import routing


def test_encodes_reference_examples():
    assert geo.encode(42.6, -5.6, 5) == 'ezs42'
    assert geo.encode(57.64911, 10.40744, 11) == 'u4pruydqqvj'
    assert geo.encode(0.0, 0.0, 5) == 's0000'
    assert geo.encode(-90.0, -180.0, 4) == '0000'
    assert geo.encode(90.0, 180.0, 4) == 'zzzz'


def test_neighbors_match_reference():
    cell, precision = geo.parse('ezs42')
    assert set(geo.strings(geo.neighbors(cell, precision), precision)) == {
        'ezs48',
        'ezs49',
        'ezs43',
        'ezs41',
        'ezs40',
        'ezefp',
        'ezefr',
        'ezefx',
    }


def test_antimeridian_wraps():
    west = geo.encode_cell(10.0, 179.99, 4)
    assert geo.cell_to_string(west, 4) == 'xczb'
    assert geo.cell_to_string(geo.offset_cell(west, 4, 1, 0), 4) == '81b0'


def test_parse_rejects_invalid_text():
    for text in ('', 'ezs4a', '0123456789bcd'):
        with pytest.raises(ValueError):
            geo.parse(text)
    assert geo.parse('EZS42') == geo.parse('ezs42')


def test_vectorized_encoding_matches_scalar():
    rng = np.random.default_rng(1)
    lat = rng.uniform(-90, 90, 3000)
    lng = rng.uniform(-180, 180, 3000)
    for precision in (1, 4, 6, 12):
        cells = geo.encode_cells(lat, lng, precision)
        for i in range(0, 3000, 97):
            assert int(cells[i]) == geo.encode_cell(lat[i], lng[i], precision)


def test_cell_dimensions_match_published_table():
    assert geo.cell_width_km(4, 0.0) == pytest.approx(39.1, rel=0.01)
    assert geo.cell_height_km(4) == pytest.approx(19.5, rel=0.01)
    assert geo.cell_width_km(6, 0.0) == pytest.approx(1.22, rel=0.01)
    assert geo.cell_height_km(6) == pytest.approx(0.61, rel=0.01)


def test_cover_disk_contains_every_sampled_point():
    rng = np.random.default_rng(2)
    for _ in range(100):
        lat = rng.uniform(25, 48)
        lng = rng.uniform(-123, -70)
        radius = rng.uniform(0.1, 9.0)
        cells = geo.cover_disk(lat, lng, radius, 4, max_rings=8)
        for _ in range(100):
            point = geo.destination(
                lat, lng, rng.uniform(0, 360), radius * np.sqrt(rng.uniform())
            )
            assert geo.encode_cell(*point, 4) in cells


def test_cover_disk_respects_ring_limit():
    assert geo.cover_disk(37.7749, -122.4194, 25.0, 4, max_rings=1) is None
    assert len(geo.cover_disk(37.7749, -122.4194, 0.0, 4, max_rings=1)) == 1


def test_routing_table_longest_prefix():
    table = routing.RoutingTable(4, [('9q', 1), ('9q8', 2), ('9q8y', 3)])
    assert table.shard_for_cell(geo.parse('9q8y')[0], 4) == 3
    assert table.shard_for_cell(geo.parse('9q8z')[0], 4) == 2
    assert table.shard_for_cell(geo.parse('9q5c')[0], 4) == 1
    assert table.shard_for_cell(geo.parse('dr5r')[0], 4) is None
    assert table.shard_for_point(37.7749, -122.4194) == 3
    cells = geo.encode_cells(
        np.array([37.7749, 40.7128]), np.array([-122.4194, -74.0060]), 6
    )
    assert table.shards_for_cells(cells, 6).tolist() == [3, -1]
    with pytest.raises(ValueError):
        routing.RoutingTable(4, [('9q', 1), ('9Q', 2)])


def test_fanout_policy_bounds_and_errors():
    table = routing.RoutingTable(4, [('9q', 1)])
    policy = routing.FanoutPolicy(table, 2.0, 15.0)
    box = geo.cell_box(geo.encode_cell(37.7749, -122.4194, 4), 4)
    plan = policy.plan(box.max_lat - 1e-4, box.max_lng - 1e-4, 1.0)
    assert plan.cells_intersected == 4
    assert plan.shard_ids == [1]
    assert policy.plan(37.7749, -122.4194, 0).radius_km == 2.0
    with pytest.raises(ValueError):
        policy.plan(37.7749, -122.4194, 16.0)
    with pytest.raises(ValueError):
        policy.plan(95.0, 0.0, 1.0)


def test_engine_configs_parse_and_route():
    config = routing.load_engine_config('engine/configs')
    table = routing.RoutingTable.from_config(config)
    assert table.shard_for_point(37.7749, -122.4194) == 1  # San Francisco.
    assert table.shard_for_point(41.8781, -87.6298) == 2  # Chicago.
    assert table.shard_for_point(40.7128, -74.0060) == 3  # New York.
    assert config.index.dimension % config.index.pq_subspaces == 0
