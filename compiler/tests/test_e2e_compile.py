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

"""Deliverables C1 and C2 (c) on the real price model.

The production model compiles to one graph, every layer's attention is
rewritten, the compiled model matches eager on both radius-trimming branches,
and the deployment compile refuses models it cannot fully accelerate.
"""

import pytest
import torch

from models.price_transformer import model as model_lib
from models.price_transformer import spatial_attention
from vectra_compiler import backend
from vectra_compiler import inductor_hook

CONFIG = model_lib.ModelConfig(
    d_model=64, num_heads=2, num_layers=3, ffn_dim=128, num_candidates=24
)


@pytest.fixture(name='model')
def _model():
    torch.manual_seed(0)
    return model_lib.PriceTransformer(CONFIG).eval()


def _far_inputs(seed: int) -> model_lib.PricingInputs:
    """Inputs where one request has no comparable inside its radius."""
    inputs = model_lib.example_inputs(CONFIG, 8, seed=seed)
    radius = inputs.radius_km.clone()
    radius[3] = 1e-3
    return inputs._replace(radius_km=radius)


def test_backend_is_registered_by_name(model):
    assert backend.BACKEND_NAME in torch._dynamo.list_backends()  # pylint: disable=protected-access
    compiled = torch.compile(
        model, backend=backend.BACKEND_NAME, fullgraph=True
    )
    inputs = model_lib.example_inputs(CONFIG, 8)
    with torch.no_grad():
        torch.testing.assert_close(
            compiled(*inputs), model(*inputs), rtol=1e-3, atol=1e-4
        )


def test_serving_compile_rewrites_every_layer(model):
    inputs = model_lib.example_inputs(CONFIG, 8)
    compiled, report = backend.compile_for_serving(
        model, tuple(inputs), expected_sites=CONFIG.num_layers
    )
    assert report.graph_count == 1
    assert report.graph_break_count == 0
    assert report.sites == CONFIG.num_layers
    assert report.matches == {'rbf_divisor_bias_softmax[fp32]': 3}
    with torch.no_grad():
        for seed in range(3):
            fresh = model_lib.example_inputs(CONFIG, 8, seed=10 + seed)
            # The design's numerics gate: allclose at rtol 1e-3.
            torch.testing.assert_close(
                compiled(*fresh), model(*fresh), rtol=1e-3, atol=1e-4
            )


def test_both_radius_branches_match_eager(model):
    inputs = model_lib.example_inputs(CONFIG, 8)
    compiled, _ = backend.compile_for_serving(
        model, tuple(inputs), expected_sites=CONFIG.num_layers
    )
    with torch.no_grad():
        for seed in range(2):
            far = _far_inputs(seed)
            torch.testing.assert_close(
                compiled(*far), model(*far), rtol=1e-3, atol=1e-4
            )


@pytest.mark.parametrize(
    'dtype', [torch.float16, torch.bfloat16], ids=['float16', 'bfloat16']
)
def test_reduced_precision_blocks_are_rewritten(model, dtype):
    model.set_block_dtype(dtype)
    inputs = model_lib.example_inputs(CONFIG, 8)
    compiled, report = backend.compile_for_serving(
        model, tuple(inputs), expected_sites=CONFIG.num_layers
    )
    assert report.matches == {'rbf_divisor_bias_softmax[half]': 3}
    with torch.no_grad():
        torch.testing.assert_close(
            compiled(*inputs), model(*inputs), rtol=1e-3, atol=2e-2
        )


class _L1Attention(spatial_attention.SpatialAttention):
    """A look-alike layer: L1 instead of squared-L2 distances."""

    def forward(self, x, key_bias):
        batch, tokens, width = x.shape
        q, k, v = (
            t.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
            for t in self.qkv(x).split(width, dim=-1)
        )
        dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).abs().sum(-1)
        weights = torch.softmax(-dist / self.log_sigma.exp() + key_bias, -1)
        return self.out(
            (weights @ v).transpose(1, 2).reshape(batch, tokens, width)
        )


def test_unaccelerated_layers_fail_the_serving_compile(model):
    replaced = _L1Attention(CONFIG.d_model, CONFIG.num_heads)
    model.blocks[1].attention = replaced.eval()
    with pytest.raises(backend.PatternNotFoundError, match='2 of 3'):
        backend.compile_for_serving(
            model,
            tuple(model_lib.example_inputs(CONFIG, 8)),
            expected_sites=CONFIG.num_layers,
        )


class _BranchyModel(torch.nn.Module):
    """Data-dependent Python control flow: a graph break."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, *inputs):
        out = self.inner(*inputs)
        if out.mean() > 13.0:
            return out - 0.01
        return out


def test_graph_breaks_fail_the_serving_compile(model):
    inputs = model_lib.example_inputs(CONFIG, 8)
    report = inductor_hook.explain(_BranchyModel(model), *inputs)
    assert report.graph_break_count >= 1
    with pytest.raises(inductor_hook.GraphBreakError):
        backend.compile_for_serving(
            _BranchyModel(model),
            tuple(inputs),
            expected_sites=CONFIG.num_layers,
        )
