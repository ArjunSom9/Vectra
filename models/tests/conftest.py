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

"""Shared fixtures: a small corpus and the examples drawn from it."""

import pytest

from data import embedding_model
from data import synthetic
from data.pipelines import assign_geohash
from data.pipelines import corpus_io
from data.pipelines import generate_embeddings
from models.training import dataset

NUM_PROPERTIES = 6000
NUM_CANDIDATES = 16


@pytest.fixture(scope='session', name='corpus_dir')
def _corpus_dir(tmp_path_factory) -> str:
    corpus = synthetic.generate(NUM_PROPERTIES, seed=11)
    directory = str(tmp_path_factory.mktemp('corpus'))
    columns = generate_embeddings.write_corpus(
        directory, corpus, embedding_model.PropertyEmbedder(), chunk=4096
    )
    corpus_io.write_column(
        directory,
        corpus_io.GEOHASH_COLUMN,
        assign_geohash.assign(corpus.latitude, corpus.longitude),
    )
    corpus_io.write_column(directory, 'metro_index', corpus.metro_index)
    corpus_io.write_manifest(
        directory, columns + [corpus_io.GEOHASH_COLUMN, 'metro_index']
    )
    return directory


@pytest.fixture(scope='session', name='data_config')
def _data_config(corpus_dir) -> dataset.DatasetConfig:
    return dataset.DatasetConfig(
        corpus_dir=corpus_dir, num_candidates=NUM_CANDIDATES, max_subjects=1500
    )


@pytest.fixture(scope='session', name='examples')
def _examples(data_config) -> dataset.Examples:
    return dataset.build_examples(data_config)
