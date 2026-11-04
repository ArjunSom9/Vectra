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

"""Numerical parity of the fused kernel with the float64 oracle (B2).

The acceptance gate is allclose with rtol 1e-3. Outputs are weighted sums
that can be arbitrarily close to zero, so a relative tolerance alone is not
meaningful; every comparison pairs rtol with an absolute tolerance scaled to
the magnitude of V. Half-precision runs round the attention weights to the
input dtype before the second tensor-core product (as FlashAttention does
[3]), which bounds their error at about 2^-11 (float16) or 2^-8 (bfloat16) of
max|V|.
"""

import pytest
import torch

from kernels.tests import conftest
from vectra_kernels import reference

# (dtype, rtol, atol relative to max|V|)
TOLERANCES = {
    torch.float32: (1e-4, 1e-5),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}

SHAPES = [
    # (B, H, Nq, Nk, D, Dv): the serving shape (query + 64 candidates,
    # 8 heads of 32), a long sequence, and assorted others.
    (8, 8, 65, 65, 32, 32),
    (2, 4, 257, 257, 64, 64),
    (1, 1, 16, 16, 16, 16),
    (4, 2, 33, 129, 64, 32),
]


def _inputs(shape, dtype, seed, with_bias):
    batch, heads, n_q, n_k, dim, value_dim = shape
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, n_q, dim, generator=gen).to(dtype)
    k = torch.randn(batch, heads, n_k, dim, generator=gen).to(dtype)
    v = torch.randn(batch, heads, n_k, value_dim, generator=gen).to(dtype)
    # sigma^2 near the typical squared distance (2 * dim), as the model learns.
    scale = 1.0 / (
        2.0 * dim * (0.5 + torch.rand(1, heads, 1, 1, generator=gen))
    )
    bias = None
    if with_bias:
        bias = torch.zeros(batch, 1, 1, n_k)
        bias[:, :, :, n_k // 2 + 1 :] = float('-inf')  # Padded candidates.
    return q, k, v, scale, bias


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('dtype', list(TOLERANCES))
@pytest.mark.parametrize('with_bias', [False, True])
def test_matches_float64_oracle(implementation, shape, dtype, with_bias):
    q, k, v, scale, bias = _inputs(shape, dtype, 0, with_bias)
    expected = reference.expanded_rbf_attention(q, k, v, scale, bias)
    actual = conftest.run(implementation, q, k, v, scale, bias)
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(
        actual, expected, rtol=rtol, atol=atol * v.abs().max().item()
    )


def test_eager_baseline_matches_oracle():
    q, k, v, scale, bias = _inputs(SHAPES[0], torch.float32, 1, True)
    torch.testing.assert_close(
        reference.eager_rbf_attention(q, k, v, scale, bias).double(),
        reference.expanded_rbf_attention(q, k, v, scale, bias),
        rtol=1e-4,
        atol=1e-5,
    )


def test_custom_op_dispatches_and_matches():
    q, k, v, scale, bias = _inputs(SHAPES[0], torch.float32, 2, True)
    out = torch.ops.vectra.rbf_attention(q, k, v, scale, bias)
    assert out.shape == (8, 8, 65, 32)
    torch.testing.assert_close(
        out.double(),
        reference.expanded_rbf_attention(q, k, v, scale, bias),
        rtol=1e-4,
        atol=1e-5,
    )
