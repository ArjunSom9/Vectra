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

"""Generates the property corpus, its embeddings, and held-out queries.

  python -m data.pipelines.generate_embeddings \
      --out_dir=data/corpus/dev --num_properties=200000 --num_queries=2000

The production run uses --num_properties=10000000. Embeddings are written in
chunks to a memory-mapped .npy, so the 5.1 GB embedding matrix of the full
corpus never has to fit in memory at once.
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np

from data import embedding_model
from data import synthetic
from data.pipelines import corpus_io

_OUT_DIR = flags.DEFINE_string(
    'out_dir', 'data/corpus/dev', 'Corpus directory to create.'
)
_NUM_PROPERTIES = flags.DEFINE_integer(
    'num_properties', 200_000, 'Records in the corpus.'
)
_NUM_QUERIES = flags.DEFINE_integer(
    'num_queries', 2000, 'Held-out query properties.'
)
_SEED = flags.DEFINE_integer('seed', 2025, 'Generator seed.')
_CHUNK = flags.DEFINE_integer('chunk', 262_144, 'Rows embedded per batch.')


def write_embeddings(
    directory: str,
    name: str,
    corpus: synthetic.Corpus,
    embedder: embedding_model.PropertyEmbedder,
    chunk: int,
) -> None:
    """Embeds `corpus` into `<directory>/<name>.npy` in chunks."""
    os.makedirs(directory, exist_ok=True)
    out = np.lib.format.open_memmap(
        corpus_io.column_path(directory, name),
        mode='w+',
        dtype=np.float32,
        shape=(len(corpus), embedder.dimension),
    )
    attributes = corpus.attributes()
    for start in range(0, len(corpus), chunk):
        rows = slice(start, start + chunk)
        out[rows] = embedder.embed(
            {key: value[rows] for key, value in attributes.items()},
            corpus.latitude[rows],
            corpus.longitude[rows],
        )
    out.flush()
    del out


def write_corpus(
    directory: str,
    corpus: synthetic.Corpus,
    embedder: embedding_model.PropertyEmbedder,
    chunk: int,
    prefix: str = '',
) -> list[str]:
    """Writes every record column and the embeddings; returns column names."""
    names = []
    for name in corpus_io.RECORD_COLUMNS:
        corpus_io.write_column(directory, prefix + name, getattr(corpus, name))
        names.append(prefix + name)
    embedding_name = (
        prefix + 'embedding' if prefix else corpus_io.EMBEDDING_COLUMN
    )
    write_embeddings(directory, embedding_name, corpus, embedder, chunk)
    names.append(embedding_name)
    return names


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    total = _NUM_PROPERTIES.value + _NUM_QUERIES.value
    logging.info('Drawing %d properties (seed %d)', total, _SEED.value)
    everything = synthetic.generate(total, _SEED.value)
    rows = np.random.default_rng(_SEED.value + 2).permutation(total)
    corpus = everything.take(np.sort(rows[: _NUM_PROPERTIES.value]))
    queries = everything.take(np.sort(rows[_NUM_PROPERTIES.value :]))

    embedder = embedding_model.PropertyEmbedder()
    directory = _OUT_DIR.value
    columns = write_corpus(directory, corpus, embedder, _CHUNK.value)
    corpus_io.write_column(directory, 'metro_index', corpus.metro_index)
    columns.append('metro_index')
    query_dir = os.path.join(directory, 'queries')
    query_columns = write_corpus(
        query_dir, queries, embedder, _CHUNK.value, prefix='query_'
    )
    corpus_io.write_manifest(
        query_dir, query_columns, {'num_queries': len(queries)}
    )
    corpus_io.write_manifest(
        directory,
        columns,
        {
            'num_properties': len(corpus),
            'seed': _SEED.value,
            'dimension': embedder.dimension,
        },
    )
    logging.info(
        'Wrote %d records and %d queries to %s',
        len(corpus),
        len(queries),
        directory,
    )


if __name__ == '__main__':
    app.run(main)
