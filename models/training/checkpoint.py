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

"""Checkpoints: weights in safetensors, architecture in JSON.

safetensors stores raw tensors only, so loading a checkpoint never executes
code from the file (unlike pickle-based formats).
"""

import dataclasses
import json
import os

from safetensors import torch as safetensors_torch
import torch

from models.price_transformer import model as model_lib

WEIGHTS = 'model.safetensors'
CONFIG = 'model_config.json'


def save(
    directory: str,
    model: model_lib.PriceTransformer,
    metadata: dict | None = None,
) -> None:
    """Writes `directory`/{model.safetensors, model_config.json}."""
    os.makedirs(directory, exist_ok=True)
    state = {
        name: tensor.detach().float().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    safetensors_torch.save_file(state, os.path.join(directory, WEIGHTS))
    with open(os.path.join(directory, CONFIG), 'w', encoding='utf-8') as f:
        json.dump(
            {
                'model': dataclasses.asdict(model.config),
                'metadata': metadata or {},
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write('\n')


def load(
    directory: str, device: torch.device | str = 'cpu'
) -> model_lib.PriceTransformer:
    """Rebuilds the model from `directory` in eval mode."""
    with open(os.path.join(directory, CONFIG), encoding='utf-8') as f:
        config = model_lib.ModelConfig.from_dict(json.load(f)['model'])
    model = model_lib.PriceTransformer(config)
    model.load_state_dict(
        safetensors_torch.load_file(os.path.join(directory, WEIGHTS))
    )
    return model.to(device).eval()
