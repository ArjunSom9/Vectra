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

"""Adds each record's 6-character GeoHash cell to a corpus.

  python -m data.pipelines.assign_geohash --corpus_dir=data/corpus/dev

The geohash6 column holds 30-bit integer cells; 3-, 4- and 5-character
cells (macro-region, shard key, prefilter) are right shifts of it.
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
import numpy as np

from data import geo
from data.pipelines import corpus_io

_CORPUS_DIR = flags.DEFINE_string(
    'corpus_dir',
    'data/corpus/dev',
    'Corpus directory to update.',
    allow_override=True,
)


def assign(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """6-character cells (uint32) of each coordinate."""
    return geo.encode_cells(latitude, longitude, 6).astype(np.uint32)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    directory = _CORPUS_DIR.value
    cells = assign(
        corpus_io.read_column(directory, 'latitude'),
        corpus_io.read_column(directory, 'longitude'),
    )
    corpus_io.write_column(directory, corpus_io.GEOHASH_COLUMN, cells)
    manifest = corpus_io.read_manifest(directory)
    columns = list(manifest['columns']) + [corpus_io.GEOHASH_COLUMN]
    extra = {k: v for k, v in manifest.items() if k != 'columns'}
    corpus_io.write_manifest(directory, sorted(set(columns)), extra)
    shard_keys, counts = np.unique(cells >> 10, return_counts=True)
    logging.info(
        '%d records in %d shard-key cells (largest %d)',
        len(cells),
        len(shard_keys),
        counts.max(),
    )


if __name__ == '__main__':
    app.run(main)
