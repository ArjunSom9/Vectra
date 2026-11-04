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

"""End-to-end run of the corpus pipelines on a small corpus."""

import os

import numpy as np

from data import embedding_model
from data import geo
from data import routing
from data import synthetic
from data.pipelines import assign_geohash
from data.pipelines import corpus_io
from data.pipelines import generate_embeddings
from data.pipelines import split_shards
from data.pipelines import validate_corpus


def _build(tmp_path, count=3000):
    corpus = synthetic.generate(count, seed=7)
    directory = str(tmp_path / 'corpus')
    columns = generate_embeddings.write_corpus(
        directory, corpus, embedding_model.PropertyEmbedder(), chunk=1000
    )
    cells = assign_geohash.assign(corpus.latitude, corpus.longitude)
    corpus_io.write_column(directory, corpus_io.GEOHASH_COLUMN, cells)
    corpus_io.write_manifest(directory, columns + [corpus_io.GEOHASH_COLUMN])
    return corpus, directory


def test_generator_is_deterministic_and_plausible():
    a = synthetic.generate(2000, seed=3)
    b = synthetic.generate(2000, seed=3)
    np.testing.assert_array_equal(a.last_sale_price_usd, b.last_sale_price_usd)
    assert np.all(a.last_sale_price_usd > 20_000)
    assert 200_000 < np.median(a.last_sale_price_usd) < 1_500_000
    assert np.all((a.bedrooms >= 1) & (a.bedrooms <= 7))


def test_embeddings_are_deterministic_and_location_aware():
    corpus = synthetic.generate(500, seed=4)
    embedder = embedding_model.PropertyEmbedder()
    first = embedder.embed(
        corpus.attributes(), corpus.latitude, corpus.longitude
    )
    again = embedding_model.PropertyEmbedder().embed(
        corpus.attributes(), corpus.latitude, corpus.longitude
    )
    np.testing.assert_array_equal(first, again)
    assert first.shape == (500, embedding_model.EMBEDDING_DIM)
    assert first.dtype == np.float32
    # Moving a property 50 km changes its embedding more than moving it 50 m.
    one = {key: value[:1] for key, value in corpus.attributes().items()}
    lat, lng = corpus.latitude[:1], corpus.longitude[:1]
    near = embedder.embed(one, lat + 0.00045, lng)
    far = embedder.embed(one, lat + 0.45, lng)
    base = embedder.embed(one, lat, lng)
    assert np.linalg.norm(near - base) < np.linalg.norm(far - base)


def test_pipeline_validates_and_splits(tmp_path):
    corpus, directory = _build(tmp_path)
    assert not validate_corpus.validate(directory, 'data/schemas', True)

    config = routing.load_engine_config('engine/configs')
    table = routing.RoutingTable.from_config(config)
    out = str(tmp_path / 'shards')
    sizes = split_shards.write_shards(directory, out, table)
    assert sum(sizes.values()) == len(corpus)
    for shard_id in sizes:
        shard_dir = os.path.join(out, f'shard-{shard_id}')
        lat = corpus_io.read_column(shard_dir, 'latitude')
        lng = corpus_io.read_column(shard_dir, 'longitude')
        owners = table.shards_for_cells(geo.encode_cells(lat, lng, 6), 6)
        assert set(owners.tolist()) == {shard_id}
        assert corpus_io.read_column(shard_dir, 'embeddings').shape[1] == 128


def test_validation_reports_corruption(tmp_path):
    _, directory = _build(tmp_path, count=500)
    prices = corpus_io.read_column(directory, 'last_sale_price_usd')
    prices[0] = -5.0
    corpus_io.write_column(directory, 'last_sale_price_usd', prices)
    problems = validate_corpus.validate(directory, 'data/schemas', True)
    assert any('last_sale_price_usd' in problem for problem in problems)


def test_rebalance_keeps_regions_and_balances(tmp_path):
    corpus, directory = _build(tmp_path, count=6000)
    config = routing.load_engine_config('engine/configs')
    cells = corpus_io.read_column(directory, corpus_io.GEOHASH_COLUMN)
    shard_map = split_shards.rebalance(cells, config, 2, 7101)
    assert len(shard_map.shards) == 6
    config.shard_map = shard_map
    table = routing.RoutingTable.from_config(config)
    owners = table.shards_for_cells(cells, 6)
    assert np.all(owners > 0)  # Every record is routable.
    counts = np.bincount(owners)[1:]
    # Within each region, the two shards are within 2x of each other.
    for first in range(0, 6, 2):
        pair = counts[first : first + 2]
        assert pair.max() <= 2 * pair.min() + 50
    assert len(corpus) == counts.sum()
