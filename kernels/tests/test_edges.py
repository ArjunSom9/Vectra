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

"""Edge shapes and values for the fused kernel (B2, risk register row 4).

Masked tails (counts that are not multiples of the block sizes), head
dimensions that are not powers of two, single queries, fully and partially
masked candidate sets, extreme distances, and strided inputs.
"""

import pytest
import torch

from kernels.tests import conftest
from vectra_kernels import reference


def _check(implementation, q, k, v, scale, bias=None, rtol=1e-4, atol=1e-5):
    expected = reference.expanded_rbf_attention(q, k, v, scale, bias)
    actual = conftest.run(implementation, q, k, v, scale, bias)
    torch.testing.assert_close(
        actual, expected, rtol=rtol, atol=atol, equal_nan=True
    )


@pytest.mark.parametrize(
    'n_q,n_k',
    [(1, 1), (1, 65), (17, 3), (63, 65), (65, 63), (129, 1), (130, 200)],
)
def test_tails_not_multiple_of_block(implementation, n_q, n_k):
    gen = torch.Generator().manual_seed(n_q * 1000 + n_k)
    q = torch.randn(2, 3, n_q, 32, generator=gen)
    k = torch.randn(2, 3, n_k, 32, generator=gen)
    v = torch.randn(2, 3, n_k, 32, generator=gen)
    _check(implementation, q, k, v, torch.tensor(0.02))


@pytest.mark.parametrize(
    'dim,value_dim', [(8, 8), (48, 48), (80, 24), (100, 128), (128, 7)]
)
def test_head_dims_padded_in_registers(implementation, dim, value_dim):
    gen = torch.Generator().manual_seed(dim)
    q = torch.randn(1, 2, 40, dim, generator=gen)
    k = torch.randn(1, 2, 70, dim, generator=gen)
    v = torch.randn(1, 2, 70, value_dim, generator=gen)
    _check(implementation, q, k, v, torch.tensor(1.0 / (4.0 * dim)))


def test_all_but_one_candidate_masked(implementation):
    gen = torch.Generator().manual_seed(5)
    q = torch.randn(3, 2, 9, 16, generator=gen)
    k = torch.randn(3, 2, 70, 16, generator=gen)
    v = torch.randn(3, 2, 70, 16, generator=gen)
    bias = torch.full((3, 1, 1, 70), float('-inf'))
    bias[..., 0] = 0.0
    out = conftest.run(implementation, q, k, v, torch.tensor(0.05), bias)
    # Every query attends only to candidate 0.
    torch.testing.assert_close(
        out, v[:, :, :1, :].expand_as(out).double(), rtol=1e-5, atol=1e-6
    )


def test_fully_masked_rows_match_eager_nan(implementation):
    q = torch.randn(1, 1, 4, 16)
    k = torch.randn(1, 1, 20, 16)
    v = torch.randn(1, 1, 20, 16)
    bias = torch.full((1, 1, 1, 20), float('-inf'))
    eager = reference.eager_rbf_attention(q, k, v, torch.tensor(0.1), bias)
    out = conftest.run(implementation, q, k, v, torch.tensor(0.1), bias)
    assert torch.isnan(eager).all() and torch.isnan(out).all()


def test_far_candidates_do_not_underflow_the_row(implementation):
    # Distances of ~1e4 give logits near -1e3: exp underflows everywhere
    # without the running-maximum subtraction.
    gen = torch.Generator().manual_seed(6)
    q = torch.randn(2, 2, 10, 32, generator=gen) * 20
    k = torch.randn(2, 2, 50, 32, generator=gen) * 20
    v = torch.randn(2, 2, 50, 32, generator=gen)
    _check(implementation, q, k, v, torch.tensor(0.1), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('scale', [0.0, -0.01, 3.0])
def test_degenerate_and_negative_scales(implementation, scale):
    gen = torch.Generator().manual_seed(7)
    q = torch.randn(1, 2, 12, 16, generator=gen)
    k = torch.randn(1, 2, 30, 16, generator=gen)
    v = torch.randn(1, 2, 30, 16, generator=gen)
    _check(implementation, q, k, v, torch.tensor(scale), atol=1e-4)


def test_head_transposed_views_and_per_head_bias(implementation):
    # The model produces q, k, v as views of [B, N, H, D] projections.
    gen = torch.Generator().manual_seed(8)
    projected = torch.randn(4, 65, 3 * 4 * 32, generator=gen)
    q, k, v = (
        t.view(4, 65, 4, 32).transpose(1, 2)
        for t in projected.split(4 * 32, dim=-1)
    )
    assert not q.is_contiguous()
    bias = torch.zeros(4, 4, 1, 65)
    bias[1, 2, 0, 40:] = float('-inf')
    _check(implementation, q, k, v, torch.full((1, 4, 1, 1), 0.02), bias)
