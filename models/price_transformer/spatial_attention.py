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

"""Spatial (RBF) multi-head attention over the query and its comparables.

Scores are a radial-basis kernel of the distance between projected tokens,

  score(q_i, k_j) = softmax_j(-||q_i - k_j||^2 / (2 sigma_h^2) + mask_j),

the RBF form of attention evaluated by Tsai et al. [14] alongside the
exponential kernel of standard attention, with a learned bandwidth per head.

This is written as the straightforward eager chain on purpose: it is the
pattern the deployment compiler recognizes and replaces with the fused
kernel (compiler/guides/authoring_for_dynamo.md).
"""

import math

import torch
from torch import nn


class SpatialAttention(nn.Module):
    """Multi-head RBF attention with a key-padding bias.

    Attributes:
      num_heads: Number of heads.
      head_dim: Width of each head.
    """

    def __init__(self, d_model: int, num_heads: int):
        """Creates the projections and per-head bandwidths.

        Args:
          d_model: Token width; must be divisible by `num_heads`.
          num_heads: Number of attention heads.

        Raises:
          ValueError: If d_model is not divisible by num_heads.
        """
        super().__init__()
        if d_model % num_heads:
            raise ValueError(
                f'd_model {d_model} not divisible by num_heads {num_heads}'
            )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        # sigma^2 starts at the head width, near the typical squared distance
        # between projected tokens.
        self.log_sigma = nn.Parameter(
            torch.full((1, num_heads, 1, 1), 0.5 * math.log(self.head_dim))
        )

    def forward(self, x: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
        """Attends every token to every unmasked token.

        Args:
          x: Tokens, [B, N, d_model].
          key_bias: Additive key bias, [B, 1, 1, N]; 0 for real tokens and
            -inf for padding.

        Returns:
          [B, N, d_model].
        """
        batch, tokens, width = x.shape
        q, k, v = (
            t.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
            for t in self.qkv(x).split(width, dim=-1)
        )
        sq_dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).pow(2).sum(-1)
        logits = -sq_dist / (2 * self.log_sigma.exp() ** 2) + key_bias
        weights = torch.softmax(logits, dim=-1)
        mixed = (weights @ v).transpose(1, 2).reshape(batch, tokens, width)
        return self.out(mixed)
