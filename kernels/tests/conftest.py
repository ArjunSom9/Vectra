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

"""Implementations under test for the kernel suites.

* 'tiled': the PyTorch emulation of the kernel algorithm (always runs).
* 'matmul': the operator's CPU implementation, the kernel's arithmetic as
  whole-tensor operations (always runs).
* 'triton': the real Triton kernel, on a CUDA GPU, or on CPU under Triton's
  interpreter when TRITON_INTERPRET=1 (how kernel-correctness.yml exercises
  the kernel on runners without GPUs).
"""

import os

import pytest
import torch

from vectra_kernels import custom_ops
from vectra_kernels import reference

INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'


def triton_device() -> str | None:
    """Device on which the Triton kernel can run here, if any."""
    if custom_ops.spatial_rbf_attention is None:
        return None
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu' if INTERPRET else None


IMPLEMENTATIONS = ['tiled', 'matmul'] + (['triton'] if triton_device() else [])


def run(implementation: str, q, k, v, scale, key_bias=None):
    """Runs one implementation on its device; returns a CPU float64 result."""
    if implementation == 'tiled':
        out = reference.tiled_rbf_attention(q, k, v, scale, key_bias)
    elif implementation == 'matmul':
        out = reference.matmul_rbf_attention(q, k, v, scale, key_bias)
    else:
        device = triton_device()

        def move(t):
            return None if t is None else t.to(device)

        out = custom_ops.spatial_rbf_attention.rbf_attention_forward(
            move(q), move(k), move(v), move(scale), move(key_bias)
        )
    return out.double().cpu()


@pytest.fixture(params=IMPLEMENTATIONS, name='implementation')
def _implementation(request):
    return request.param
