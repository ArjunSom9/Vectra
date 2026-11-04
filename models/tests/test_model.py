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

"""The eager price model: masking, radius trimming, precision, capture."""

import pytest
import torch

from models.price_transformer import model as model_lib
from models.price_transformer import spatial_attention
from vectra_compiler import inductor_hook

CONFIG = model_lib.ModelConfig(
    d_model=32, num_heads=2, ffn_dim=64, num_candidates=16
)


@pytest.fixture(name='model')
def _model():
    torch.manual_seed(0)
    return model_lib.PriceTransformer(CONFIG).eval()


def test_forward_returns_one_finite_log_price_per_request(model):
    inputs = model_lib.example_inputs(CONFIG, batch_size=5)
    with torch.no_grad():
        out = model(*inputs)
    assert out.shape == (5,)
    assert torch.isfinite(out).all()


def test_padding_candidates_do_not_affect_the_prediction(model):
    inputs = model_lib.example_inputs(CONFIG, batch_size=4)
    invalid = ~inputs.cand_valid
    garbage = inputs._replace(
        cand_embedding=inputs.cand_embedding.masked_fill(
            invalid[..., None], 9.0
        ),
        cand_log_price=inputs.cand_log_price.masked_fill(invalid, 99.0),
        cand_location=inputs.cand_location.masked_fill(
            invalid[..., None], -45.0
        ),
    )
    with torch.no_grad():
        torch.testing.assert_close(model(*garbage), model(*inputs))


def test_trim_keeps_comparables_inside_the_radius():
    distance = torch.tensor([[1.0, 2.0, 6.0], [0.5, 7.0, 8.0]])
    radius = torch.tensor([5.0, 5.0])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    mask = model_lib.trim_to_radius(distance, radius, valid, min_keep=2)
    assert mask.tolist() == [[True, True, False], [True, False, False]]


def test_trim_falls_back_to_the_nearest_comparables():
    distance = torch.tensor([[9.0, 6.0, 8.0, 7.0], [1.0, 9.0, 9.0, 9.0]])
    radius = torch.tensor([5.0, 5.0])
    valid = torch.tensor([[True, True, False, True], [True, True, True, True]])
    mask = model_lib.trim_to_radius(distance, radius, valid, min_keep=2)
    # Row 0 has nothing inside 5 km: keep its two nearest valid ones (6, 7).
    assert mask.tolist() == [
        [False, True, False, True],
        [True, False, False, False],
    ]


def test_reduced_precision_blocks_track_float32(model):
    inputs = model_lib.example_inputs(CONFIG, batch_size=8)
    with torch.no_grad():
        want = model(*inputs)
        got = model.set_block_dtype(torch.bfloat16)(*inputs)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, want, rtol=0, atol=5e-2)


def test_spatial_attention_is_rbf_attention():
    torch.manual_seed(1)
    layer = spatial_attention.SpatialAttention(d_model=16, num_heads=2).eval()
    x = torch.randn(2, 5, 16)
    bias = torch.zeros(2, 1, 1, 5)
    with torch.no_grad():
        q, k, v = layer.qkv(x).view(2, 5, 3, 2, 8).permute(2, 0, 3, 1, 4)
        sigma = layer.log_sigma.exp()
        weights = torch.softmax(
            -(torch.cdist(q, k) ** 2) / (2 * sigma**2), dim=-1
        )
        want = layer.out((weights @ v).transpose(1, 2).reshape(2, 5, 16))
        torch.testing.assert_close(layer(x, bias), want, rtol=1e-4, atol=1e-5)


def test_model_captures_as_a_single_graph(model):
    """Deliverable C1: no graph breaks anywhere in the forward pass."""
    torch._dynamo.reset()  # pylint: disable=protected-access
    report = inductor_hook.explain(
        model, *model_lib.example_inputs(CONFIG, batch_size=4)
    )
    assert report.graph_count == 1
    assert report.graph_break_count == 0, report.break_reasons
