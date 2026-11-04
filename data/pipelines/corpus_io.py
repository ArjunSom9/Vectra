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

"""Reading and writing the columnar corpus format.

A corpus directory holds one NumPy .npy file per column (the format the C++
indexer reads with engine/src/io/npy.cc) and a manifest.json recording the
schema version, row count, and a SHA-256 checksum per column, so an index is
always a reproducible function of checksum-pinned inputs.
"""

from collections.abc import Iterable
import hashlib
import json
import os

import numpy as np

SCHEMA_VERSION = 1
MANIFEST = 'manifest.json'

# Column name -> NumPy dtype descriptor, as consumed by the indexer.
RECORD_COLUMNS = {
    'property_id': '<i8',
    'latitude': '<f8',
    'longitude': '<f8',
    'living_area_sqft': '<f4',
    'lot_area_sqft': '<f4',
    'bedrooms': '<i4',
    'bathrooms': '<f4',
    'year_built': '<i4',
    'last_sale_price_usd': '<f8',
    'last_sale_epoch_day': '<i4',
}
EMBEDDING_COLUMN = 'embeddings'
GEOHASH_COLUMN = 'geohash6'


def column_path(directory: str, name: str) -> str:
    return os.path.join(directory, f'{name}.npy')


def write_column(directory: str, name: str, values: np.ndarray) -> None:
    """Writes one column (C order, little-endian)."""
    os.makedirs(directory, exist_ok=True)
    np.save(column_path(directory, name), np.ascontiguousarray(values))


def read_column(directory: str, name: str, mmap: bool = False) -> np.ndarray:
    """Reads one column, optionally memory-mapped."""
    return np.load(
        column_path(directory, name), mmap_mode='r' if mmap else None
    )


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    directory: str, columns: Iterable[str], extra: dict | None = None
) -> dict:
    """Records shapes, dtypes and checksums of `columns`."""
    manifest = {'schema_version': SCHEMA_VERSION, 'columns': {}}
    if extra:
        manifest.update(extra)
    for name in columns:
        path = column_path(directory, name)
        array = np.load(path, mmap_mode='r')
        manifest['columns'][name] = {
            'dtype': array.dtype.str,
            'shape': list(array.shape),
            'sha256': sha256_of(path),
        }
    with open(os.path.join(directory, MANIFEST), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')
    return manifest


def read_manifest(directory: str) -> dict:
    with open(os.path.join(directory, MANIFEST), encoding='utf-8') as f:
        return json.load(f)
