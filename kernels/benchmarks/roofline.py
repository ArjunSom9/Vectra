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

"""Analytic HBM traffic, FLOPs and roofline placement of RBF attention.

  python -m kernels.benchmarks.roofline [--batch=64 --tokens=65 ...]

The eager model charges every operator of the chain one HBM round trip for
its inputs and outputs (one kernel per operator, the eager execution model
[3] [8]): the broadcast difference, its square, the feature reduction, the
scaling, the bias, softmax, and the value matrix multiply. The fused model
reads Q, K, V, the coefficient and the bias once and writes the output once.
Roofline placement [6] uses the H100 SXM5 figures of
vectra_kernels.tile_configs.

These are the budgets the Nsight Compute measurements of
benchmark/bandwidth are checked against, not measurements themselves.
"""

from collections.abc import Sequence
import dataclasses
import json

from absl import app
from absl import flags

from vectra_kernels import tile_configs

_BATCH = flags.DEFINE_integer(
    'batch', 64, 'Requests per batch.', allow_override=True
)
_TOKENS = flags.DEFINE_integer(
    'tokens', 65, 'Query + candidate tokens.', allow_override=True
)
_HEADS = flags.DEFINE_integer(
    'heads', 4, 'Attention heads.', allow_override=True
)
_HEAD_DIM = flags.DEFINE_integer(
    'head_dim', 32, 'Head width.', allow_override=True
)
_ELEMENT_BYTES = flags.DEFINE_integer(
    'element_bytes', 2, '2 for float16, 4 for float32.', allow_override=True
)


@dataclasses.dataclass(frozen=True)
class Shape:
    """One attention problem: [B, H, Nq, D] x [B, H, Nk, D]."""

    batch: int
    heads: int
    n_q: int
    n_k: int
    head_dim: int
    value_dim: int
    element_bytes: int

    @property
    def planes(self) -> int:
        return self.batch * self.heads


# The production serving shape: 64 requests, the subject plus 64
# comparables as tokens, 4 heads of width 32, float16 blocks.
SERVING = Shape(
    batch=64,
    heads=4,
    n_q=65,
    n_k=65,
    head_dim=32,
    value_dim=32,
    element_bytes=2,
)


def fused_bytes(s: Shape) -> int:
    """Q, K, V, coefficient and bias read once; output written once."""
    reads = (
        s.planes
        * (s.n_q * s.head_dim + s.n_k * s.head_dim + s.n_k * s.value_dim)
        * s.element_bytes
    )
    reads += s.heads * 4 + s.batch * s.n_k * s.element_bytes
    return reads + s.planes * s.n_q * s.value_dim * s.element_bytes


def eager_bytes(s: Shape) -> dict[str, int]:
    """Per-operator HBM traffic of the eager chain."""
    e = s.element_bytes
    q = s.planes * s.n_q * s.head_dim * e
    k = s.planes * s.n_k * s.head_dim * e
    v = s.planes * s.n_k * s.value_dim * e
    diff = s.planes * s.n_q * s.n_k * s.head_dim * e
    scores = s.planes * s.n_q * s.n_k * e
    bias = s.batch * s.n_k * e
    out = s.planes * s.n_q * s.value_dim * e
    return {
        'sub': q + k + diff,
        'pow': 2 * diff,
        'sum': diff + scores,
        'neg': 2 * scores,
        'scale': 2 * scores,
        'bias': 2 * scores + bias,
        'softmax': 2 * scores,
        'bmm': scores + v + out,
    }


def fused_flops(s: Shape) -> int:
    """Tensor-core dots (QK^T and PV) plus norms and the online softmax."""
    per_plane = (
        2 * s.n_q * s.n_k * s.head_dim
        + 2 * s.n_q * s.n_k * s.value_dim
        + 2 * (s.n_q + s.n_k) * s.head_dim
        + 6 * s.n_q * s.n_k
    )
    return s.planes * per_plane


def eager_flops(s: Shape) -> int:
    """SIMT difference, square and reduction, then softmax and PV."""
    per_plane = (
        3 * s.n_q * s.n_k * s.head_dim
        + 3 * s.n_q * s.n_k
        + 5 * s.n_q * s.n_k
        + 2 * s.n_q * s.n_k * s.value_dim
    )
    return s.planes * per_plane


def attainable_flops(
    intensity: float, peak_flops: float, spec: tile_configs.GpuSpec
) -> float:
    """The roofline: min(compute peak, intensity x HBM bandwidth) [6]."""
    return min(peak_flops, intensity * spec.hbm_bytes_per_s)


def analyze(
    s: Shape, spec: tile_configs.GpuSpec = tile_configs.H100_SXM5
) -> dict:
    """Budgets and roofline placement of both formulations at `s`."""
    eager = eager_bytes(s)
    eager_total = sum(eager.values())
    fused = fused_bytes(s)
    fused_ai = fused_flops(s) / fused
    eager_ai = eager_flops(s) / eager_total
    working_set = fused
    return {
        'shape': dataclasses.asdict(s),
        'eager_bytes': eager_total,
        'eager_bytes_by_op': eager,
        'fused_bytes': fused,
        'bytes_ratio': fused / eager_total,
        'eager_intensity_flop_per_byte': eager_ai,
        'fused_intensity_flop_per_byte': fused_ai,
        'machine_balance_flop_per_byte': spec.machine_balance,
        'fused_is_memory_bound': fused_ai < spec.machine_balance,
        'eager_attainable_tflops': attainable_flops(
            eager_ai, spec.fp32_flops, spec
        )
        / 1e12,
        'fused_attainable_tflops': attainable_flops(
            fused_ai, spec.fp16_tensor_flops, spec
        )
        / 1e12,
        # Time to stream the fused kernel's bytes at full HBM bandwidth.
        'fused_hbm_floor_us': fused / spec.hbm_bytes_per_s * 1e6,
        # A working set below L2 capacity can be served from L2, so DRAM
        # counters understate the traffic the kernel actually issues.
        'fused_working_set_fits_l2': working_set < spec.l2_bytes,
    }


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    shape = Shape(
        _BATCH.value,
        _HEADS.value,
        _TOKENS.value,
        _TOKENS.value,
        _HEAD_DIM.value,
        _HEAD_DIM.value,
        _ELEMENT_BYTES.value,
    )
    print(json.dumps(analyze(shape), indent=2))


if __name__ == '__main__':
    app.run(main)
