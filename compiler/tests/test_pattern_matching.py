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

"""Deliverable C2 (a) and (b): what the pass must and must not rewrite.

(a) The same RBF attention written in different, idiomatic styles is found
    and rewritten.
(b) Look-alikes that compute something else are left alone, and still
    produce their own (eager) results after compilation.
"""

import pytest
import torch
from torch import fx

from compiler.tests import conftest
from vectra_compiler import pattern

F = torch.nn.functional


# --- (a) Authoring styles that must be rewritten. ----------------------------


def method_chain(q, k, v, sigma):
    sq_dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).pow(2).sum(-1)
    return torch.softmax(-sq_dist / (2 * sigma**2), dim=-1) @ v


def functional_calls(q, k, v, sigma):
    diff = torch.sub(torch.unsqueeze(q, 3), torch.unsqueeze(k, 2))
    sq_dist = torch.sum(torch.square(diff), dim=-1)
    gamma = torch.reciprocal(2.0 * sigma * sigma)
    return torch.matmul(F.softmax(torch.mul(sq_dist, -gamma), dim=-1), v)


def indexing_and_products(q, k, v, sigma):
    diff = q[:, :, :, None, :] - k[:, :, None, :, :]
    sq_dist = (diff * diff).sum(dim=4)
    logits = -(sq_dist * (0.5 / sigma**2))
    weights = torch.exp(logits)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights @ v


def literal_temperature(q, k, v, sigma):
    del sigma  # A fixed bandwidth instead of a learned one.
    sq_dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).pow(2).sum(-1)
    return torch.softmax(sq_dist * -0.125, dim=-1) @ v


class MaskedHead(torch.nn.Module):
    """RBF attention with a key-padding mask, as in the price model."""

    def forward(self, q, k, v, sigma, valid):
        bias = torch.zeros_like(valid, dtype=q.dtype).masked_fill(
            ~valid, float('-inf')
        )[:, None, None, :]
        sq_dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).pow(2).sum(-1)
        return torch.softmax(-sq_dist / (2 * sigma.pow(2)) + bias, -1) @ v


@pytest.mark.parametrize(
    'fn',
    [
        method_chain,
        functional_calls,
        indexing_and_products,
        literal_temperature,
    ],
    ids=lambda fn: fn.__name__,
)
def test_authoring_styles_are_rewritten(fn):
    q, k, v, sigma = conftest.attention_inputs()
    out, report = conftest.compile_and_run(fn, q, k, v, sigma)
    assert report.sites == 1, report.matches
    torch.testing.assert_close(out, fn(q, k, v, sigma), rtol=1e-4, atol=1e-5)


def test_masked_attention_is_rewritten_with_its_bias():
    q, k, v, sigma = conftest.attention_inputs(n_k=10)
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[0, 7:] = False
    module = MaskedHead()
    out, report = conftest.compile_and_run(module, q, k, v, sigma, valid)
    assert report.matches == {'rbf_divisor_bias_softmax[fp32]': 1}
    torch.testing.assert_close(
        out, module(q, k, v, sigma, valid), rtol=1e-4, atol=1e-5
    )


def test_every_site_in_a_graph_is_rewritten():
    q, k, v, sigma = conftest.attention_inputs(n_q=9, n_k=9)

    def two_layers(q, k, v, sigma):
        h = method_chain(q, k, v, sigma)
        return indexing_and_products(h, k, v, sigma) + h

    out, report = conftest.compile_and_run(two_layers, q, k, v, sigma)
    assert report.sites == 2
    torch.testing.assert_close(
        out, two_layers(q, k, v, sigma), rtol=1e-4, atol=1e-5
    )


def test_value_layout_does_not_matter():
    q, k, v, sigma = conftest.attention_inputs(n_k=8, dim=8)
    v_transposed = v.transpose(-1, -2).contiguous().transpose(-1, -2)
    out, report = conftest.compile_and_run(
        method_chain, q, k, v_transposed, sigma
    )
    assert report.sites == 1
    torch.testing.assert_close(
        out, method_chain(q, k, v, sigma), rtol=1e-4, atol=1e-5
    )


# --- (b) Look-alikes that must not be rewritten. ----------------------------


def l1_distance(q, k, v, sigma):
    dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).abs().sum(-1)
    return torch.softmax(-dist / (2 * sigma**2), dim=-1) @ v


def cubed_difference(q, k, v, sigma):
    dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).pow(3).sum(-1)
    return torch.softmax(-dist / (2 * sigma**2), dim=-1) @ v


def reduced_over_keys(q, k, v, sigma):
    # Sums over the key axis instead of the feature axis (needs Nk == D).
    dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).pow(2).sum(-2)
    return torch.softmax(-dist / (2 * sigma**2), dim=-1) @ v


def normalized_over_queries(q, k, v, sigma):
    sq_dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).pow(2).sum(-1)
    return torch.softmax(-sq_dist / (2 * sigma**2), dim=-2) @ v


def no_negation(q, k, v, sigma):
    sq_dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).pow(2).sum(-1)
    return torch.softmax(sq_dist / (2 * sigma**2), dim=-1) @ v


def log_softmax_weights(q, k, v, sigma):
    sq_dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).pow(2).sum(-1)
    return torch.log_softmax(-sq_dist / (2 * sigma**2), dim=-1) @ v


def dot_product_attention(q, k, v, sigma):
    logits = q @ k.transpose(-1, -2) / sigma
    return torch.softmax(logits, dim=-1) @ v


def per_key_scale(q, k, v, sigma):
    del sigma
    sq_dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).pow(2).sum(-1)
    scale = torch.linspace(0.1, 1.0, k.shape[-2], dtype=q.dtype)
    return torch.softmax(-sq_dist * scale, dim=-1) @ v


def per_query_bias(q, k, v, sigma):
    sq_dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).pow(2).sum(-1)
    bias = torch.arange(q.shape[-2], dtype=q.dtype)[:, None] * 0.1
    return torch.softmax(-sq_dist / (2 * sigma**2) + bias, dim=-1) @ v


def weights_only(q, k, v, sigma):
    del v
    sq_dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).pow(2).sum(-1)
    return torch.softmax(-sq_dist / (2 * sigma**2), dim=-1)


@pytest.mark.parametrize(
    'fn',
    [
        l1_distance,
        cubed_difference,
        reduced_over_keys,
        normalized_over_queries,
        no_negation,
        log_softmax_weights,
        dot_product_attention,
        per_key_scale,
        per_query_bias,
        weights_only,
    ],
    ids=lambda fn: fn.__name__,
)
def test_lookalikes_are_not_rewritten(fn):
    # Square shapes so every variant is well-formed (Nq == Nk == D).
    q, k, v, sigma = conftest.attention_inputs(n_q=8, n_k=8, dim=8)
    out, report = conftest.compile_and_run(fn, q, k, v, sigma)
    assert report.sites == 0, report.matches
    torch.testing.assert_close(out, fn(q, k, v, sigma), rtol=1e-4, atol=1e-5)


# --- The catalog itself. ----------------------------------------------------


def test_catalog_covers_every_formulation_and_precision():
    catalog = pattern.load_catalog('cpu')
    assert {entry.spec for entry in catalog} == set(pattern.CATALOG)
    assert len({entry.name for entry in catalog}) == len(catalog)
    for entry in catalog:
        # torch.softmax upcasts half precision, so it needs per-dtype
        # templates; exp / sum decomposes identically for every dtype.
        if entry.spec.normalization == 'softmax':
            assert entry.dtype in pattern.DTYPES
        else:
            assert entry.dtype is None
    for entry in catalog:
        # Placeholders bind positionally; the value cut is always last.
        names = [
            n.name
            for n in entry.graph_module.graph.nodes
            if n.op == 'placeholder'
        ]
        assert len(names) == len(entry.spec.placeholder_names)
        assert names[-1] == pattern.V_BMM
        targets = {n.target for n in entry.graph_module.graph.nodes}
        assert torch.ops.aten.bmm.default in targets
        assert torch.ops.aten.pow.Tensor_Scalar in targets


def test_squares_are_canonicalized():
    graph = fx.Graph()
    x = graph.placeholder('x')
    square = graph.call_function(torch.ops.aten.mul.Tensor, (x, x))
    graph.output(square)
    assert pattern.canonicalize_squares(graph) == 1
    targets = [n.target for n in graph.nodes if n.op == 'call_function']
    assert targets == [torch.ops.aten.pow.Tensor_Scalar]
