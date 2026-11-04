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

"""Partitions a corpus into per-shard directories for the indexer.

  python -m data.pipelines.split_shards --corpus_dir=data/corpus/dev

Rows are assigned by the shard map in engine/configs (the same longest-prefix
routing table the engine and gateway use). With --rebalance, a new shard map
is computed first: every 4-character cell of each macro-region is assigned to
one of that region's --shards_per_region shards, largest cells first, onto
the least-loaded shard, so shard sizes stay balanced while a region's cells
never leave its data center.
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
from google.protobuf import text_format
import numpy as np

from data import geo
from data import routing
from data.pipelines import corpus_io
from vectra import config_pb2

# Flags shared by several pipeline binaries are defined with allow_override so
# that tests may import more than one binary into a single process.
_CORPUS_DIR = flags.DEFINE_string(
    'corpus_dir',
    'data/corpus/dev',
    'Corpus directory (with the geohash6 column).',
    allow_override=True,
)
_CONFIG_DIR = flags.DEFINE_string(
    'config_dir',
    'engine/configs',
    'Engine configuration directory.',
    allow_override=True,
)
_SHARDS_DIR = flags.DEFINE_string(
    'shards_dir', '', 'Output root; defaults to <corpus>/shards.'
)
_REBALANCE = flags.DEFINE_bool(
    'rebalance', False, 'Compute a balanced 4-character shard map.'
)
_SHARDS_PER_REGION = flags.DEFINE_integer(
    'shards_per_region', 2, 'Shards per macro-region.'
)
_SHARD_MAP_OUT = flags.DEFINE_string(
    'shard_map_out', '', 'Where --rebalance writes its map.'
)
_ENDPOINT_PORT = flags.DEFINE_integer(
    'endpoint_port', 7101, 'Port of generated endpoints.'
)


def rebalance(
    cells6: np.ndarray,
    config: routing.EngineConfig,
    shards_per_region: int,
    port: int,
) -> config_pb2.ShardMap:
    """Balanced shard map over the populated shard-key cells.

    Args:
      cells6: 6-character cells of every record.
      config: Engine configuration (supplies regions and precision).
      shards_per_region: Shards to create in every macro-region.
      port: Port used in generated endpoints.

    Returns:
      A ShardMap whose assignments are full shard-key cells.
    """
    precision = config.geohash.shard_chars
    keys, counts = np.unique(
        geo.cell_prefix(cells6, 6, precision), return_counts=True
    )
    region_table = routing.RoutingTable(
        min(3, precision),
        [
            (prefix, index)
            for index, region in enumerate(config.topology.regions)
            for prefix in region.prefixes
        ],
    )
    regions = region_table.shards_for_cells(keys, precision)
    shard_map = config_pb2.ShardMap()
    next_id = 1
    for index, region in enumerate(config.topology.regions):
        in_region = np.flatnonzero(regions == index)
        if in_region.size == 0:
            continue
        ids = list(range(next_id, next_id + shards_per_region))
        next_id += shards_per_region
        for n, shard_id in enumerate(ids):
            shard_map.shards.add(
                id=shard_id,
                macro_region=region.name,
                endpoints=[f'{region.name}-node{n + 1}:{port}'],
            )
        loads = np.zeros(shards_per_region, dtype=np.int64)
        for row in in_region[np.argsort(-counts[in_region], kind='stable')]:
            target = int(np.argmin(loads))
            loads[target] += counts[row]
            shard_map.assignments.add(
                prefix=geo.cell_to_string(int(keys[row]), precision),
                shard_id=ids[target],
            )
        logging.info(
            '%s: %d cells over %d shards, loads %s',
            region.name,
            len(in_region),
            shards_per_region,
            loads.tolist(),
        )
    return shard_map


def write_shards(
    corpus_dir: str, out_dir: str, table: routing.RoutingTable
) -> dict[int, int]:
    """Writes one directory per shard; returns {shard id: record count}."""
    cells6 = corpus_io.read_column(corpus_dir, corpus_io.GEOHASH_COLUMN)
    owners = table.shards_for_cells(cells6, 6)
    unassigned = int(np.sum(owners < 0))
    if unassigned:
        logging.warning(
            '%d records fall in unassigned cells and are skipped', unassigned
        )
    embeddings = corpus_io.read_column(
        corpus_dir, corpus_io.EMBEDDING_COLUMN, mmap=True
    )
    sizes = {}
    for shard_id in sorted(set(owners[owners >= 0].tolist())):
        rows = np.flatnonzero(owners == shard_id)
        shard_dir = os.path.join(out_dir, f'shard-{shard_id}')
        for name in corpus_io.RECORD_COLUMNS:
            corpus_io.write_column(
                shard_dir, name, corpus_io.read_column(corpus_dir, name)[rows]
            )
        corpus_io.write_column(
            shard_dir, corpus_io.EMBEDDING_COLUMN, np.asarray(embeddings[rows])
        )
        corpus_io.write_manifest(
            shard_dir,
            list(corpus_io.RECORD_COLUMNS) + [corpus_io.EMBEDDING_COLUMN],
            {'shard_id': shard_id, 'num_properties': len(rows)},
        )
        sizes[shard_id] = len(rows)
        logging.info(
            'shard %d: %d records -> %s', shard_id, len(rows), shard_dir
        )
    return sizes


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    config = routing.load_engine_config(_CONFIG_DIR.value)
    if _REBALANCE.value:
        cells6 = corpus_io.read_column(
            _CORPUS_DIR.value, corpus_io.GEOHASH_COLUMN
        )
        shard_map = rebalance(
            cells6, config, _SHARDS_PER_REGION.value, _ENDPOINT_PORT.value
        )
        out = _SHARD_MAP_OUT.value or os.path.join(
            _CORPUS_DIR.value, 'shard_map.textproto'
        )
        with open(out, 'w', encoding='utf-8') as f:
            f.write(
                '# proto-file: contracts/proto/vectra/config.proto\n'
                '# proto-message: vectra.config.v1.ShardMap\n'
                '# Generated by data.pipelines.split_shards --rebalance.\n'
            )
            f.write(text_format.MessageToString(shard_map))
        logging.info('Wrote balanced shard map to %s', out)
        config.shard_map = shard_map
    out_dir = _SHARDS_DIR.value or os.path.join(_CORPUS_DIR.value, 'shards')
    write_shards(
        _CORPUS_DIR.value, out_dir, routing.RoutingTable.from_config(config)
    )


if __name__ == '__main__':
    app.run(main)
