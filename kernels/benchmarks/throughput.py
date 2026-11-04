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

"""Kernel-level timing: eager chain versus the fused Triton kernel.

  python -m kernels.benchmarks.throughput --batches=64,256,1024 [--sweep]

Times each implementation with CUDA events over many iterations after
warm-up, at the serving shape and larger batches, and checks numerics
against the float64 reference before reporting any time (the design's
numerics-first rule). --sweep also times every tile configuration of
vectra_kernels.tile_configs.SWEEP that fits in shared memory
(docs/research/triton-tiling-study.md). Needs a CUDA GPU.
"""

from collections.abc import Callable, Sequence
import csv
import dataclasses
import json
import os

from absl import app
from absl import flags
import torch

from kernels.benchmarks import roofline
from vectra_kernels import reference
from vectra_kernels import tile_configs

_BATCHES = flags.DEFINE_list(
    'batches', ['64', '256', '1024'], 'Requests per batch.', allow_override=True
)
_TOKENS = flags.DEFINE_integer(
    'tokens', 65, 'Query + candidate tokens.', allow_override=True
)
_HEADS = flags.DEFINE_integer('heads', 4, 'Heads.', allow_override=True)
_HEAD_DIM = flags.DEFINE_integer(
    'head_dim', 32, 'Head width.', allow_override=True
)
_DTYPE = flags.DEFINE_enum(
    'dtype',
    'float16',
    ['float16', 'bfloat16', 'float32'],
    'Element type.',
    allow_override=True,
)
_ITERS = flags.DEFINE_integer(
    'iters', 200, 'Timed iterations.', allow_override=True
)
_SWEEP = flags.DEFINE_bool(
    'sweep', False, 'Time every tile config.', allow_override=True
)
_OUT = flags.DEFINE_string(
    'out',
    'kernels/benchmarks/results',
    'Directory for results.',
    allow_override=True,
)


@dataclasses.dataclass
class Timing:
    implementation: str
    batch: int
    milliseconds: float
    tflops: float
    effective_gbps: float
    max_abs_error: float


def time_cuda(fn: Callable[[], torch.Tensor], iters: int) -> float:
    """Mean milliseconds per call, measured with CUDA events."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def problem(shape: roofline.Shape, dtype: torch.dtype, seed: int = 0):
    gen = torch.Generator(device='cuda').manual_seed(seed)

    def randn(*size):
        return torch.randn(*size, generator=gen, device='cuda').to(dtype)

    q = randn(shape.batch, shape.heads, shape.n_q, shape.head_dim)
    k = randn(shape.batch, shape.heads, shape.n_k, shape.head_dim)
    v = randn(shape.batch, shape.heads, shape.n_k, shape.value_dim)
    scale = torch.full(
        (1, shape.heads, 1, 1), 1.0 / (2 * shape.head_dim), device='cuda'
    )
    bias = torch.zeros(shape.batch, 1, 1, shape.n_k, device='cuda', dtype=dtype)
    bias[:, ..., -8:] = float('-inf')  # Padding, as in serving.
    return q, k, v, scale, bias


def run(
    shape: roofline.Shape,
    dtype: torch.dtype,
    iters: int,
    configs: Sequence[tile_configs.TileConfig | None],
) -> list[Timing]:
    # Imported here: the module needs Triton and a GPU.
    from vectra_kernels import spatial_rbf_attention  # pylint: disable=import-outside-toplevel

    q, k, v, scale, bias = problem(shape, dtype)
    truth = reference.expanded_rbf_attention(
        q.double(), k.double(), v.double(), scale.double(), bias.double()
    )
    model = roofline.analyze(shape)
    candidates = {
        'eager': lambda: reference.eager_rbf_attention(
            q, k, v, scale.to(dtype), bias
        )
    }
    for config in configs:
        name = (
            'fused'
            if config is None
            else f'fused[{config.block_q}x{config.block_k},'
            f'w{config.num_warps},s{config.num_stages}]'
        )
        candidates[name] = (
            lambda c=config: spatial_rbf_attention.rbf_attention_forward(
                q, k, v, scale, bias, config=c
            )
        )
    timings = []
    for name, fn in candidates.items():
        error = (fn().double() - truth).abs().max().item()
        milliseconds = time_cuda(fn, iters)
        eager = name == 'eager'
        # Effective bandwidth: the analytic traffic of the formulation over
        # its measured time (Nsight Compute measures the real traffic).
        traffic = model['eager_bytes' if eager else 'fused_bytes']
        flops = (
            roofline.eager_flops(shape)
            if eager
            else roofline.fused_flops(shape)
        )
        timings.append(
            Timing(
                name,
                shape.batch,
                milliseconds,
                flops / milliseconds / 1e9,
                traffic / milliseconds / 1e6,
                error,
            )
        )
    return timings


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    if not torch.cuda.is_available():
        raise app.UsageError('kernels.benchmarks.throughput needs a CUDA GPU')
    dtype = getattr(torch, _DTYPE.value)
    element_bytes = torch.empty((), dtype=dtype).element_size()
    configs = [None]
    if _SWEEP.value:
        configs += [
            c
            for c in tile_configs.SWEEP
            if tile_configs.fits(
                c, _HEAD_DIM.value, _HEAD_DIM.value, element_bytes
            )
        ]
    rows = []
    for batch in (int(b) for b in _BATCHES.value):
        shape = roofline.Shape(
            batch,
            _HEADS.value,
            _TOKENS.value,
            _TOKENS.value,
            _HEAD_DIM.value,
            _HEAD_DIM.value,
            element_bytes,
        )
        rows += run(shape, dtype, _ITERS.value, configs)
    os.makedirs(_OUT.value, exist_ok=True)
    path = os.path.join(_OUT.value, f'throughput_{_DTYPE.value}.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=[field.name for field in dataclasses.fields(Timing)]
        )
        writer.writeheader()
        writer.writerows(dataclasses.asdict(r) for r in rows)
    print(
        json.dumps(
            {
                'device': torch.cuda.get_device_name(),
                'results': [dataclasses.asdict(r) for r in rows],
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    app.run(main)
