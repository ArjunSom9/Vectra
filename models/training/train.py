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

"""Trains the price Transformer in eager mode.

  python -m models.training.train \
      --model_config=models/training/configs/base.yaml \
      --data_config=models/training/configs/dataset_v1.yaml \
      --checkpoint_dir=checkpoints/price_transformer_v1

Training and evaluation always run eager; only deployment compiles.
"""

from collections.abc import Sequence
import dataclasses
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
import yaml

from models.evaluation import metrics
from models.price_transformer import model as model_lib
from models.training import checkpoint
from models.training import dataset

_MODEL_CONFIG = flags.DEFINE_string(
    'model_config',
    'models/training/configs/base.yaml',
    'YAML with `model` and `training` sections.',
    allow_override=True,
)
_DATA_CONFIG = flags.DEFINE_string(
    'data_config',
    'models/training/configs/dataset_v1.yaml',
    'YAML with a `dataset` section.',
    allow_override=True,
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    'checkpoints/price_transformer_v1',
    'Where to write the checkpoint.',
    allow_override=True,
)
_DEVICE = flags.DEFINE_string(
    'device', 'cpu', 'Training device.', allow_override=True
)


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 0


def load_configs(path: str) -> tuple[model_lib.ModelConfig, TrainingConfig]:
    with open(path, encoding='utf-8') as f:
        values = yaml.safe_load(f)
    names = {field.name for field in dataclasses.fields(TrainingConfig)}
    training = {
        k: v for k, v in values.get('training', {}).items() if k in names
    }
    return (
        model_lib.ModelConfig.from_dict(values['model']),
        TrainingConfig(**training),
    )


def predict(
    model: model_lib.PriceTransformer,
    examples: dataset.Examples,
    rows: np.ndarray,
    batch_size: int = 1024,
    device: str = 'cpu',
) -> np.ndarray:
    """Predicted log prices for `rows` (eager, no gradients)."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            inputs, _ = examples.tensors(rows[start : start + batch_size])
            outputs.append(model(*inputs.to(device)).float().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, np.float32)


def train(
    model_config: model_lib.ModelConfig,
    config: TrainingConfig,
    examples: dataset.Examples,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    device: str = 'cpu',
) -> tuple[model_lib.PriceTransformer, list]:
    """Fits a model with AdamW and a cosine schedule on Huber loss.

    Returns:
      (model, history), where history holds per-epoch validation metrics.
    """
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = model_lib.PriceTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps = config.epochs * dataset.num_batches(
        len(train_rows), config.batch_size
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, steps)
    )
    history = []
    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(train_rows)
        losses = []
        for start in range(0, len(order), config.batch_size):
            inputs, target = examples.tensors(
                order[start : start + config.batch_size]
            )
            prediction = model(*inputs.to(device))
            loss = torch.nn.functional.huber_loss(
                prediction, target.to(device), delta=0.2
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            losses.append(loss.item())
        report = metrics.price_errors(
            predict(model, examples, validation_rows, device=device),
            examples.target_log_price[validation_rows],
        )
        report['epoch'] = epoch
        report['train_loss'] = float(np.mean(losses)) if losses else 0.0
        history.append(report)
        logging.info(
            'epoch %d: loss %.4f, validation MAPE %.3f',
            epoch,
            report['train_loss'],
            report['mape'],
        )
    return model.eval(), history


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    model_config, config = load_configs(_MODEL_CONFIG.value)
    data_config = dataset.DatasetConfig.from_yaml(_DATA_CONFIG.value)
    if data_config.num_candidates != model_config.num_candidates:
        raise app.UsageError('dataset and model num_candidates differ')
    cache = os.path.join(_CHECKPOINT_DIR.value, 'examples.npz')
    if os.path.exists(cache):
        examples = dataset.Examples.load(cache)
    else:
        examples = dataset.build_examples(data_config)
        os.makedirs(_CHECKPOINT_DIR.value, exist_ok=True)
        examples.save(cache)
    train_rows, validation_rows = dataset.split_by_day(
        examples, data_config.split_day
    )
    logging.info(
        '%d train / %d validation examples',
        len(train_rows),
        len(validation_rows),
    )
    model, history = train(
        model_config,
        config,
        examples,
        train_rows,
        validation_rows,
        _DEVICE.value,
    )
    baseline = metrics.price_errors(
        dataset.baseline_log_price(examples)[validation_rows],
        examples.target_log_price[validation_rows],
    )
    checkpoint.save(
        _CHECKPOINT_DIR.value,
        model,
        {'history': history, 'comparable_mean_baseline': baseline},
    )
    logging.info(
        'validation MAPE %.3f (comparable-mean baseline %.3f)',
        history[-1]['mape'] if history else float('nan'),
        baseline['mape'],
    )


if __name__ == '__main__':
    app.run(main)
