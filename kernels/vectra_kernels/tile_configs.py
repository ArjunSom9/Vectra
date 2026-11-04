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

"""Tile shapes for the fused RBF attention kernel, and the H100 budget.

Hardware figures are those of the H100 SXM5 in the architecture whitepaper
[16] (Table 3): 132 SMs, up to 228 KB of shared memory and 256 KB of
registers per SM, a 50 MB L2, and 3,352 GB/s of HBM3 bandwidth.

Tiles are chosen so that the live Q, K and V tiles of a program, multiplied
by the software-pipelining depth (num_stages), fit in shared memory with room
for a second resident program per SM [16] [33].
"""

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class GpuSpec:
    """Peak figures used for budgets and roofline placement."""

    name: str
    sm_count: int
    shared_memory_per_sm_bytes: int
    register_file_per_sm_bytes: int
    l2_bytes: int
    hbm_bytes_per_s: float
    fp16_tensor_flops: float  # Dense, FP32 accumulate.
    fp32_flops: float  # Non-tensor.

    @property
    def machine_balance(self) -> float:
        """FLOP per byte at which the FP16 tensor roofline bends [6]."""
        return self.fp16_tensor_flops / self.hbm_bytes_per_s


H100_SXM5 = GpuSpec(
    name='H100 SXM5',
    sm_count=132,
    shared_memory_per_sm_bytes=228 * 1024,
    register_file_per_sm_bytes=256 * 1024,
    l2_bytes=50 * 1024 * 1024,
    hbm_bytes_per_s=3352e9,
    fp16_tensor_flops=989.4e12,
    fp32_flops=66.9e12,
)


@dataclasses.dataclass(frozen=True)
class TileConfig:
    """Launch shape of one kernel program."""

    block_q: int
    block_k: int
    num_warps: int
    num_stages: int


# Candidates explored by kernels/benchmarks/throughput.py --sweep (see
# docs/research/triton-tiling-study.md).
SWEEP = (
    TileConfig(16, 64, 4, 2),
    TileConfig(32, 64, 4, 2),
    TileConfig(64, 32, 4, 2),
    TileConfig(64, 64, 4, 2),
    TileConfig(64, 64, 4, 3),
    TileConfig(64, 128, 8, 2),
    TileConfig(128, 64, 8, 2),
    TileConfig(128, 64, 8, 3),
    TileConfig(128, 128, 8, 2),
)


def padded_dim(dim: int) -> int:
    """Head dimension padded to a power of two >= 16 (tl.dot minimum)."""
    return max(16, 1 << max(0, math.ceil(math.log2(dim))))


def shared_memory_bytes(
    config: TileConfig, head_dim: int, value_dim: int, element_bytes: int
) -> int:
    """Shared memory a program stages: Q once, K and V per pipeline stage."""
    d = padded_dim(head_dim)
    dv = padded_dim(value_dim)
    per_stage = config.block_k * (d + dv) * element_bytes
    return config.block_q * d * element_bytes + config.num_stages * per_stage


def fits(
    config: TileConfig,
    head_dim: int,
    value_dim: int,
    element_bytes: int,
    spec: GpuSpec = H100_SXM5,
) -> bool:
    """True when two programs of `config` can be resident on one SM."""
    return 2 * shared_memory_bytes(
        config, head_dim, value_dim, element_bytes
    ) <= (spec.shared_memory_per_sm_bytes)


def select(
    n_q: int, n_k: int, head_dim: int, value_dim: int, element_bytes: int
) -> TileConfig:
    """Heuristic tile choice for a problem shape.

    Small query sets (the serving shape: one query property plus tens to a
    few hundred candidates) use a query tile that covers all queries, so each
    K/V tile is read from HBM exactly once per (batch, head). Larger
    problems use 128-row tiles for tensor-core efficiency.
    """
    del n_k  # The candidate axis is streamed; its length does not matter.
    block_q = 16
    while block_q < min(n_q, 128):
        block_q *= 2
    config = TileConfig(block_q, 64, 4 if block_q <= 64 else 8, 2)
    while not fits(config, head_dim, value_dim, element_bytes) and (
        config.block_k > 16
    ):
        config = dataclasses.replace(config, block_k=config.block_k // 2)
    return config
