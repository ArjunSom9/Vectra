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

"""Validates a corpus directory against data/schemas and its manifest.

  python -m data.pipelines.validate_corpus --corpus_dir=data/corpus/dev

Checks every column's presence, dtype, length and value range against
property_record.json, the embedding matrix against embedding_schema.json,
id uniqueness, and every SHA-256 recorded in manifest.json. Exits non-zero on
the first class of failure so CI and the index-rebuild runbook can gate on it.
"""

from collections.abc import Sequence
import json
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np

from data.pipelines import corpus_io

_CORPUS_DIR = flags.DEFINE_string(
    'corpus_dir',
    'data/corpus/dev',
    'Corpus directory to validate.',
    allow_override=True,
)
_SCHEMA_DIR = flags.DEFINE_string(
    'schema_dir', 'data/schemas', 'Directory of the JSON schemas.'
)
_VERIFY_CHECKSUMS = flags.DEFINE_bool(
    'verify_checksums', True, 'Recompute and compare SHA-256 sums.'
)


def _load(schema_dir: str, name: str) -> dict:
    with open(os.path.join(schema_dir, name), encoding='utf-8') as f:
        return json.load(f)


def validate(
    corpus_dir: str, schema_dir: str, verify_checksums: bool
) -> list[str]:
    """Returns a list of problems (empty when the corpus is valid)."""
    problems = []
    record = _load(schema_dir, 'property_record.json')
    embedding = _load(schema_dir, 'embedding_schema.json')
    rows = None
    for name, spec in record['properties'].items():
        path = corpus_io.column_path(corpus_dir, name)
        if not os.path.exists(path):
            if name in record['required']:
                problems.append(f'missing column {name}')
            continue
        values = np.load(path, mmap_mode='r')
        expected = spec['x-dtype']
        if values.dtype.str != expected:
            problems.append(
                f'{name}: dtype {values.dtype.str}, expected {expected}'
            )
        if rows is None:
            rows = len(values)
        elif len(values) != rows:
            problems.append(f'{name}: {len(values)} rows, expected {rows}')
        if values.size == 0:
            continue
        if values.dtype.kind == 'f' and not np.all(np.isfinite(values)):
            problems.append(f'{name}: non-finite values')
        low, high = values.min(), values.max()
        minimum = spec.get('minimum')
        exclusive_minimum = spec.get('exclusiveMinimum')
        maximum = spec.get('maximum')
        if minimum is not None and low < minimum:
            problems.append(f'{name}: minimum {low} < {minimum}')
        if exclusive_minimum is not None and low <= exclusive_minimum:
            problems.append(f'{name}: minimum {low} <= {exclusive_minimum}')
        if maximum is not None and high > maximum:
            problems.append(f'{name}: maximum {high} > {maximum}')
        if spec.get('x-unique') and len(np.unique(values)) != len(values):
            problems.append(f'{name}: values are not unique')

    path = corpus_io.column_path(corpus_dir, embedding['x-column'])
    if not os.path.exists(path):
        problems.append('missing embeddings column')
    else:
        matrix = np.load(path, mmap_mode='r')
        if matrix.dtype.str != embedding['x-dtype']:
            problems.append(f'embeddings: dtype {matrix.dtype.str}')
        if matrix.ndim != 2 or matrix.shape[1] != embedding['x-dimension']:
            problems.append(f'embeddings: shape {matrix.shape}')
        elif rows is not None and matrix.shape[0] != rows:
            problems.append(
                f'embeddings: {matrix.shape[0]} rows, expected {rows}'
            )
        for start in range(0, len(matrix), 1 << 20):
            block = np.asarray(matrix[start : start + (1 << 20)])
            if not np.all(np.isfinite(block)):
                problems.append('embeddings: non-finite values')
                break
            if (
                block.min() < embedding['items']['minimum']
                or block.max() > embedding['items']['maximum']
            ):
                problems.append('embeddings: values outside [-1, 1]')
                break

    if verify_checksums:
        manifest = corpus_io.read_manifest(corpus_dir)
        for name, entry in manifest['columns'].items():
            path = corpus_io.column_path(corpus_dir, name)
            if not os.path.exists(path):
                problems.append(f'manifest lists missing column {name}')
            elif corpus_io.sha256_of(path) != entry['sha256']:
                problems.append(f'{name}: checksum mismatch')
    return problems


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    problems = validate(
        _CORPUS_DIR.value, _SCHEMA_DIR.value, _VERIFY_CHECKSUMS.value
    )
    for problem in problems:
        logging.error('%s', problem)
    if problems:
        raise SystemExit(1)
    logging.info('%s is valid', _CORPUS_DIR.value)


if __name__ == '__main__':
    app.run(main)
