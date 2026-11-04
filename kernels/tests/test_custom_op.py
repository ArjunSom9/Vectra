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

"""Deliverable B4: vectra::rbf_attention is a well-formed PyTorch operator."""

import pytest
import torch

from vectra_kernels import custom_ops
from vectra_kernels import reference


def _inputs(device='cpu', dtype=torch.float32, bias=True):
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(2, 3, 5, 16, generator=gen)
    k = torch.randn(2, 3, 9, 16, generator=gen)
    v = torch.randn(2, 3, 9, 8, generator=gen)
    scale = torch.rand(1, 3, 1, 1, generator=gen) * 0.1 + 0.02
    key_bias = torch.zeros(2, 1, 1, 9)
    key_bias[1, ..., 7:] = float('-inf')
    args = [t.to(device=device, dtype=dtype) for t in (q, k, v)]
    args += [scale.to(device), key_bias.to(device, dtype) if bias else None]
    return tuple(args)


@pytest.mark.parametrize('bias', [True, False])
def test_opcheck(bias):
    torch.library.opcheck(
        torch.ops.vectra.rbf_attention.default,
        _inputs(bias=bias),
        test_utils=(
            'test_schema',
            'test_faketensor',
            'test_aot_dispatch_static',
            'test_aot_dispatch_dynamic',
        ),
    )


def test_operator_matches_the_float64_oracle():
    args = _inputs()
    out = torch.ops.vectra.rbf_attention(*args)
    torch.testing.assert_close(
        out.double(),
        reference.expanded_rbf_attention(*args),
        rtol=1e-5,
        atol=1e-6,
    )


def test_fake_implementation_reports_the_output_shape():
    with torch._subclasses.fake_tensor.FakeTensorMode():  # pylint: disable=protected-access
        q = torch.empty(4, 2, 7, 32, dtype=torch.float16)
        k = torch.empty(4, 2, 11, 32, dtype=torch.float16)
        v = torch.empty(4, 2, 11, 24, dtype=torch.float16)
        out = torch.ops.vectra.rbf_attention(
            q, k, v, torch.empty(1, 2, 1, 1), None
        )
    assert out.shape == (4, 2, 7, 24) and out.dtype == torch.float16


@pytest.mark.gpu
def test_cuda_dispatches_to_the_triton_kernel():
    if not custom_ops.triton_available():
        pytest.skip('Triton is not installed on this host')
    args = _inputs(device='cuda', dtype=torch.float16)
    out = torch.ops.vectra.rbf_attention(*args)
    want = reference.expanded_rbf_attention(*args)
    torch.testing.assert_close(
        out.double().cpu(), want.cpu(), rtol=1e-3, atol=1e-3
    )
