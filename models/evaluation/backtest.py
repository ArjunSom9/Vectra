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

"""Temporal backtest of a trained checkpoint.

  python -m models.evaluation.backtest \
      --checkpoint_dir=checkpoints/price_transformer_v1

Scores every sale on or after the split day (whose comparables are all
earlier sales) overall and per metro, against the comparable-mean baseline.
"""

from collections.abc import Sequence
import json
import os

from absl import app
from absl import flags
import numpy as np

from data import synthetic
from data.pipelines import corpus_io
from models.evaluation import metrics
from models.price_transformer import model as model_lib
from models.training import checkpoint
from models.training import dataset
from models.training import train

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    'checkpoints/price_transformer_v1',
    'Checkpoint to evaluate.',
    allow_override=True,
)
_DATA_CONFIG = flags.DEFINE_string(
    'data_config',
    'models/training/configs/dataset_v1.yaml',
    'YAML with a `dataset` section.',
    allow_override=True,
)


def backtest(
    model: model_lib.PriceTransformer,
    examples: dataset.Examples,
    rows: np.ndarray,
    metro: np.ndarray | None = None,
) -> dict:
    """Accuracy of `model` and of the comparable-mean baseline on `rows`.

    Args:
      model: The model to score (run eagerly).
      examples: All examples.
      rows: Which examples to score.
      metro: Optional corpus `metro_index` column, for a per-metro breakdown.

    Returns:
      {'model': ..., 'comparable_mean': ..., 'per_metro': {...}} of
      metrics.price_errors dictionaries.
    """
    predicted = train.predict(model, examples, rows)
    actual = examples.target_log_price[rows]
    baseline = dataset.baseline_log_price(examples)[rows]
    report = {
        'model': metrics.price_errors(predicted, actual),
        'comparable_mean': metrics.price_errors(baseline, actual),
    }
    if metro is not None:
        metro_of_row = metro[examples.subject_row[rows]]
        report['per_metro'] = {
            m.name: metrics.price_errors(
                predicted[metro_of_row == i], actual[metro_of_row == i]
            )
            for i, m in enumerate(synthetic.METROS)
            if np.any(metro_of_row == i)
        }
    return report


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    data_config = dataset.DatasetConfig.from_yaml(_DATA_CONFIG.value)
    cache = os.path.join(_CHECKPOINT_DIR.value, 'examples.npz')
    examples = (
        dataset.Examples.load(cache)
        if os.path.exists(cache)
        else dataset.build_examples(data_config)
    )
    _, rows = dataset.split_by_day(examples, data_config.split_day)
    metro = None
    if os.path.exists(
        corpus_io.column_path(data_config.corpus_dir, 'metro_index')
    ):
        metro = corpus_io.read_column(data_config.corpus_dir, 'metro_index')
    model = checkpoint.load(_CHECKPOINT_DIR.value)
    print(json.dumps(backtest(model, examples, rows, metro), indent=2))


if __name__ == '__main__':
    app.run(main)
