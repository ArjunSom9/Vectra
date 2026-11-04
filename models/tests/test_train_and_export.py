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

"""Train -> checkpoint -> deployment compile -> serving, on CPU."""

import json
import os

import numpy as np
import pytest
import torch

from models.evaluation import backtest
from models.evaluation import metrics
from models.export import compile_entry
from models.training import checkpoint
from models.training import dataset
from models.training import train


@pytest.fixture(scope='module', name='trained')
def _trained(examples, data_config, tmp_path_factory):
    model_config, config = train.load_configs(
        'models/training/configs/smoke.yaml'
    )
    train_rows, validation_rows = dataset.split_by_day(
        examples, data_config.split_day
    )
    model, history = train.train(
        model_config, config, examples, train_rows, validation_rows
    )
    directory = str(tmp_path_factory.mktemp('checkpoint'))
    checkpoint.save(directory, model, {'history': history})
    return model, history, directory, validation_rows


def test_training_beats_the_comparable_mean(trained, examples):
    model, history, _, validation_rows = trained
    assert len(history) == 2
    assert history[-1]['train_loss'] < history[0]['train_loss'] * 1.5
    baseline = metrics.price_errors(
        dataset.baseline_log_price(examples)[validation_rows],
        examples.target_log_price[validation_rows],
    )
    predicted = train.predict(model, examples, validation_rows)
    assert np.isfinite(predicted).all()
    assert (
        metrics.price_errors(
            predicted, examples.target_log_price[validation_rows]
        )['mape']
        < baseline['mape']
    )


def test_checkpoint_round_trip(trained, examples):
    model, _, directory, rows = trained
    loaded = checkpoint.load(directory)
    inputs, _ = examples.tensors(rows[:9])
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(*inputs), model(*inputs), rtol=0, atol=0
        )
    with open(
        os.path.join(directory, checkpoint.CONFIG), encoding='utf-8'
    ) as f:
        assert json.load(f)['model']['num_candidates'] == 16


def test_backtest_reports_per_metro(trained, examples, corpus_dir):
    model, _, _, rows = trained
    metro = np.load(os.path.join(corpus_dir, 'metro_index.npy'))
    report = backtest.backtest(model, examples, rows, metro)
    assert report['model']['count'] == len(rows)
    assert sum(m['count'] for m in report['per_metro'].values()) == len(rows)


def test_export_bundle_compiles_verifies_and_serves(
    trained, examples, tmp_path
):
    model, _, directory, rows = trained
    out = str(tmp_path / 'bundle')
    spec = compile_entry.ServingSpec('cpu', batch_size=8, block_dtype='float32')
    manifest = compile_entry.export_bundle(
        directory, out, spec, numerics_batches=1
    )
    assert manifest['compile']['sites'] == model.config.num_layers
    assert manifest['compile']['graph_break_count'] == 0
    assert manifest['numerics']['max_abs_error_vs_eager'] < 1e-3

    serving = compile_entry.load_bundle(out)
    for count in (1, 8, 13):
        inputs, _ = examples.tensors(rows[:count])
        with torch.no_grad():
            want = model(*inputs)
        got = serving.predict(inputs)
        assert got.shape == (count,)
        torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)


def test_tampered_bundle_is_rejected(trained, tmp_path):
    _, _, directory, _ = trained
    out = str(tmp_path / 'bundle')
    spec = compile_entry.ServingSpec('cpu', batch_size=4, block_dtype='float32')
    compile_entry.export_bundle(directory, out, spec, numerics_batches=1)
    with open(os.path.join(out, checkpoint.WEIGHTS), 'r+b') as f:
        f.seek(-4, os.SEEK_END)
        f.write(b'\x00\x00\x80\x7f')
    with pytest.raises(ValueError, match='checksum'):
        compile_entry.load_bundle(out)
