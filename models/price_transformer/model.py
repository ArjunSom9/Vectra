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

"""The price Transformer.

The subject property and its K retrieved comparables become K + 1 tokens;
pre-norm Transformer blocks with spatial (RBF) attention mix them, and the
subject token is read out as a correction to the comparables' mean log price.

The one data-dependent branch, trimming comparables to the search radius
with a fallback to the nearest ones when none are inside it, is expressed
with torch.cond [35], so TorchDynamo captures the forward pass as a single
graph [24] and the attention pattern stays visible to the compiler pass.
"""

from collections.abc import Mapping
import dataclasses
import typing

import torch
from torch import nn

from models.price_transformer import candidate_encoder
from models.price_transformer import microhood
from models.price_transformer import spatial_attention


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters."""

    embedding_dim: int = 128
    attribute_dim: int = 5
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ffn_dim: int = 256
    num_candidates: int = 64
    # Comparables kept when none lies inside the requested radius.
    min_candidates: int = 4

    @classmethod
    def from_dict(cls, values: Mapping[str, typing.Any]) -> 'ModelConfig':
        names = {field.name for field in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names})


class PricingInputs(typing.NamedTuple):
    """One batch of pricing requests, in forward-argument order.

    Attributes:
      query_embedding: [B, E] float32.
      query_attributes: [B, A] float32, standardized.
      query_location: [B, 2] (latitude, longitude) degrees.
      query_cell6: [B] int64 6-character GeoHash cells.
      valuation_day: [B] float32 days since the epoch.
      radius_km: [B] float32 comparable search radius.
      cand_embedding: [B, K, E].
      cand_attributes: [B, K, A].
      cand_log_price: [B, K] natural log of sale price.
      cand_sale_day: [B, K] float32 days since the epoch.
      cand_location: [B, K, 2].
      cand_cell6: [B, K] int64.
      cand_valid: [B, K] bool; False for padding.
    """

    query_embedding: torch.Tensor
    query_attributes: torch.Tensor
    query_location: torch.Tensor
    query_cell6: torch.Tensor
    valuation_day: torch.Tensor
    radius_km: torch.Tensor
    cand_embedding: torch.Tensor
    cand_attributes: torch.Tensor
    cand_log_price: torch.Tensor
    cand_sale_day: torch.Tensor
    cand_location: torch.Tensor
    cand_cell6: torch.Tensor
    cand_valid: torch.Tensor

    def to(self, device: torch.device | str) -> 'PricingInputs':
        return PricingInputs(*(t.to(device) for t in self))  # pylint: disable=not-an-iterable


def trim_to_radius(
    distance_km: torch.Tensor,
    radius_km: torch.Tensor,
    valid: torch.Tensor,
    min_keep: int,
) -> torch.Tensor:
    """Keeps comparables inside the radius; falls back to the nearest ones.

    The common case, every request having at least one comparable inside
    its radius, takes the cheap branch. Otherwise affected rows keep their
    `min_keep` nearest valid comparables. The choice depends on tensor
    values, so it is a torch.cond rather than a Python `if`, which would
    break the graph.

    Args:
      distance_km: [B, K] great-circle distances.
      radius_km: [B] radii.
      valid: [B, K] candidate validity.
      min_keep: Comparables kept by the fallback (at most K).

    Returns:
      [B, K] bool mask of comparables to attend to.
    """

    def within_radius(distance, radius, mask):
        return mask & (distance <= radius.unsqueeze(-1))

    def with_fallback(distance, radius, mask):
        inside = mask & (distance <= radius.unsqueeze(-1))
        far = distance.masked_fill(~mask, float('inf'))
        nearest = far.topk(min_keep, dim=-1, largest=False).indices
        fallback = torch.zeros_like(mask).scatter(-1, nearest, True) & mask
        return torch.where(inside.any(-1, keepdim=True), inside, fallback)

    all_covered = (
        (valid & (distance_km <= radius_km.unsqueeze(-1))).any(-1).all()
    )
    return torch.cond(
        all_covered,
        within_radius,
        with_fallback,
        (distance_km, radius_km, valid),
    )


class _Block(nn.Module):
    """Pre-norm Transformer block with spatial attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = spatial_attention.SpatialAttention(
            config.d_model, config.num_heads
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.d_model),
        )

    def forward(self, x: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), key_bias)
        return x + self.ffn(self.norm2(x))


class PriceTransformer(nn.Module):
    """Predicts a property's log price from its retrieved comparables.

    Attributes:
      config: The architecture.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.query_encoder = candidate_encoder.QueryEncoder(
            config.embedding_dim, config.attribute_dim, config.d_model
        )
        self.candidate_encoder = candidate_encoder.CandidateEncoder(
            config.embedding_dim, config.attribute_dim, config.d_model
        )
        self.blocks = nn.ModuleList(
            [_Block(config) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, 1)
        self._block_dtype = torch.float32

    def set_block_dtype(self, dtype: torch.dtype) -> 'PriceTransformer':
        """Runs the Transformer blocks in `dtype` (e.g. float16 for serving).

        Encoders, geometry and the read-out stay in float32: coordinates in
        half precision would lose kilometers.
        """
        self.blocks.to(dtype)
        self._block_dtype = dtype
        return self

    def forward(
        self,
        query_embedding,
        query_attributes,
        query_location,
        query_cell6,
        valuation_day,
        radius_km,
        cand_embedding,
        cand_attributes,
        cand_log_price,
        cand_sale_day,
        cand_location,
        cand_cell6,
        cand_valid,
    ) -> torch.Tensor:
        """Returns predicted natural-log prices, [B] (see PricingInputs)."""
        distance = microhood.haversine_km(query_location, cand_location)
        valid = trim_to_radius(
            distance,
            radius_km,
            cand_valid,
            min(self.config.min_candidates, cand_valid.shape[-1]),
        )
        candidates = self.candidate_encoder(
            cand_embedding,
            cand_attributes,
            cand_log_price,
            valuation_day.unsqueeze(-1) - cand_sale_day,
            microhood.relative_offsets_km(query_location, cand_location),
            distance,
            radius_km,
            microhood.same_cell_flags(query_cell6, cand_cell6),
        )
        query = self.query_encoder(query_embedding, query_attributes)
        tokens = torch.cat([query.unsqueeze(1), candidates], dim=1).to(
            self._block_dtype
        )
        token_valid = torch.cat([torch.ones_like(valid[:, :1]), valid], dim=1)
        key_bias = torch.zeros_like(
            token_valid, dtype=tokens.dtype
        ).masked_fill(~token_valid, float('-inf'))[:, None, None, :]
        for block in self.blocks:
            tokens = block(tokens, key_bias)
        weights = valid.to(cand_log_price.dtype)
        baseline = (cand_log_price * weights).sum(-1) / weights.sum(
            -1
        ).clamp_min(1.0)
        readout = self.head(self.norm(tokens[:, 0].float()))
        return baseline + readout.squeeze(-1)


def example_inputs(
    config: ModelConfig,
    batch_size: int,
    device: torch.device | str = 'cpu',
    seed: int = 0,
) -> PricingInputs:
    """Random inputs of the serving shape, for compilation and tests."""
    gen = torch.Generator().manual_seed(seed)
    k = config.num_candidates

    def rand(*shape):
        return torch.randn(*shape, generator=gen)

    center = torch.tensor([37.7749, -122.4194])
    query_location = center + 0.02 * rand(batch_size, 2)
    cand_location = query_location[:, None, :] + 0.02 * rand(batch_size, k, 2)
    cells = torch.randint(0, 1 << 30, (batch_size,), generator=gen)
    valid = torch.ones(batch_size, k, dtype=torch.bool)
    valid[:, k - k // 8 :] = False  # Some padding.
    inputs = PricingInputs(
        query_embedding=torch.tanh(rand(batch_size, config.embedding_dim)),
        query_attributes=rand(batch_size, config.attribute_dim),
        query_location=query_location,
        query_cell6=cells,
        valuation_day=torch.full((batch_size,), 20000.0),
        radius_km=torch.full((batch_size,), 5.0),
        cand_embedding=torch.tanh(rand(batch_size, k, config.embedding_dim)),
        cand_attributes=rand(batch_size, k, config.attribute_dim),
        cand_log_price=13.0 + 0.5 * rand(batch_size, k),
        cand_sale_day=20000.0 - 900.0 * rand(batch_size, k).abs(),
        cand_location=cand_location,
        cand_cell6=cells[:, None]
        + torch.randint(0, 3, (batch_size, k), generator=gen),
        cand_valid=valid,
    )
    return inputs.to(device)
