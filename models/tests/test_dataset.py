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

"""Training examples are leak-free and match the serving retrieval rule."""

import numpy as np

from data import geo
from data.pipelines import corpus_io
from models.training import dataset


def test_comparables_sold_strictly_before_the_subject(examples):
    inputs = examples.inputs
    earlier = inputs.cand_sale_day < inputs.valuation_day[:, None]
    assert np.all(earlier | ~inputs.cand_valid)
    np.testing.assert_array_equal(inputs.valuation_day, examples.sale_day)


def test_comparables_lie_inside_the_radius(examples, data_config):
    inputs = examples.inputs
    lat, lng = inputs.query_location[:, 0], inputs.query_location[:, 1]
    distance = geo.haversine_km_array(
        lat[:, None].astype(np.float64),
        lng[:, None].astype(np.float64),
        inputs.cand_location[..., 0].astype(np.float64),
        inputs.cand_location[..., 1].astype(np.float64),
    )
    # float32 coordinates carry ~1 m of rounding.
    assert np.all(
        (distance <= data_config.radius_km + 1e-2) | ~inputs.cand_valid
    )


def test_comparables_are_ranked_by_embedding_distance(examples):
    inputs = examples.inputs
    distance = np.sum(
        (inputs.cand_embedding - inputs.query_embedding[:, None, :]) ** 2, -1
    )
    for row in range(len(examples)):
        valid = distance[row][inputs.cand_valid[row]]
        assert np.all(np.diff(valid) >= -1e-5)
        # Valid slots come first, padding after.
        count = int(inputs.cand_valid[row].sum())
        assert inputs.cand_valid[row, :count].all()


def test_subject_is_never_its_own_comparable(examples, corpus_dir):
    cells = corpus_io.read_column(corpus_dir, corpus_io.GEOHASH_COLUMN)
    np.testing.assert_array_equal(
        examples.inputs.query_cell6, cells[examples.subject_row]
    )
    same = (
        examples.inputs.cand_embedding
        == examples.inputs.query_embedding[:, None, :]
    ).all(-1) & examples.inputs.cand_valid
    assert not same.any()


def test_targets_are_the_subject_sale_prices(examples, corpus_dir):
    prices = corpus_io.read_column(corpus_dir, 'last_sale_price_usd')
    np.testing.assert_allclose(
        examples.target_log_price,
        np.log(prices[examples.subject_row]),
        rtol=1e-6,
    )


def test_split_by_day_partitions_by_sale_date(examples, data_config):
    train_rows, validation_rows = dataset.split_by_day(
        examples, data_config.split_day
    )
    assert len(train_rows) + len(validation_rows) == len(examples)
    assert np.all(examples.sale_day[train_rows] < data_config.split_day)
    assert np.all(examples.sale_day[validation_rows] >= data_config.split_day)
    assert len(train_rows) and len(validation_rows)


def test_examples_round_trip_through_npz(examples, tmp_path):
    path = str(tmp_path / 'examples.npz')
    examples.save(path)
    loaded = dataset.Examples.load(path)
    for name in examples.inputs._fields:
        np.testing.assert_array_equal(
            getattr(loaded.inputs, name), getattr(examples.inputs, name)
        )
    np.testing.assert_array_equal(loaded.subject_row, examples.subject_row)


def test_baseline_is_the_mean_of_valid_comparables(examples):
    baseline = dataset.baseline_log_price(examples)
    row = 0
    valid = examples.inputs.cand_valid[row]
    assert np.isclose(
        baseline[row], examples.inputs.cand_log_price[row][valid].mean()
    )
