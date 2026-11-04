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

"""Token encoders for the subject property and its comparables."""

import torch
from torch import nn

# Log-price normalization (log of $500,000, and a typical spread).
LOG_PRICE_CENTER = 13.12
LOG_PRICE_SCALE = 0.6
_DAYS_PER_YEAR = 365.25


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs)
    )


class QueryEncoder(nn.Module):
    """Encodes the subject property (its price is what is predicted)."""

    def __init__(self, embedding_dim: int, attribute_dim: int, d_model: int):
        super().__init__()
        self.mlp = _mlp(embedding_dim + attribute_dim, 2 * d_model, d_model)
        self.role = nn.Parameter(torch.zeros(d_model))

    def forward(
        self, embedding: torch.Tensor, attributes: torch.Tensor
    ) -> torch.Tensor:
        """[B, E], [B, A] -> [B, d_model]."""
        return self.mlp(torch.cat([embedding, attributes], dim=-1)) + self.role


class CandidateEncoder(nn.Module):
    """Encodes comparables with their sale price, age of sale and position.

    Per-candidate inputs: the embedding, standardized attributes, the
    normalized log sale price, years between that sale and the valuation
    date, east/north offsets and distance scaled by the search radius, and
    the micro-neighborhood flags.
    """

    def __init__(self, embedding_dim: int, attribute_dim: int, d_model: int):
        super().__init__()
        extra = 1 + 1 + 3 + 3  # price, age, geometry, cell flags
        self.mlp = _mlp(
            embedding_dim + attribute_dim + extra, 2 * d_model, d_model
        )

    def forward(
        self,
        embedding: torch.Tensor,
        attributes: torch.Tensor,
        log_price: torch.Tensor,
        sale_age_days: torch.Tensor,
        offsets_km: torch.Tensor,
        distance_km: torch.Tensor,
        radius_km: torch.Tensor,
        cell_flags: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes [B, K] comparables into [B, K, d_model] tokens."""
        radius = radius_km[:, None, None]
        features = [
            embedding,
            attributes,
            ((log_price - LOG_PRICE_CENTER) / LOG_PRICE_SCALE).unsqueeze(-1),
            (sale_age_days / _DAYS_PER_YEAR).unsqueeze(-1),
            offsets_km / radius,
            distance_km.unsqueeze(-1) / radius,
            cell_flags,
        ]
        return self.mlp(torch.cat(features, dim=-1))
