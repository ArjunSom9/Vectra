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

"""Serving-model throughput, eager versus compiled (B3, C3).

  python -m benchmark.gpu_throughput.run_throughput --mode=eager \
      --checkpoint_dir=checkpoints/price_transformer_v1 \
      --out=benchmark/reports/throughput_eager.json
  ... --mode=vectra --out=benchmark/reports/throughput_vectra.json
  python -m benchmark.gpu_throughput.compare_runs \
      --baseline=benchmark/reports/throughput_eager.json \
      --candidate=benchmark/reports/throughput_vectra.json

Throughput is measured on the full model at the production batch, as the
design requires, not on the kernel alone. Modes: `eager` (the baseline),
`inductor` (torch.compile without the rewrite, to separate Inductor's own
fusion from the custom kernel) and `vectra` (the deployment compile). Each
run first checks its numerics against the eager model (rtol 1e-3) and
reports them with the timing. --profile_once runs one forward pass inside
the NVTX range `vectra_model` for Nsight Compute.
"""

from collections.abc import Sequence
import json
import os
import time

from absl import app
from absl import flags
import torch

from models.export import compile_entry
from models.price_transformer import model as model_lib
from models.training import checkpoint

_MODE = flags.DEFINE_enum(
    'mode',
    'vectra',
    ['eager', 'inductor', 'vectra'],
    'What to run.',
    allow_override=True,
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'Trained weights (random base.yaml model if empty).',
    allow_override=True,
)
_DEVICE = flags.DEFINE_string(
    'device',
    'cuda' if torch.cuda.is_available() else 'cpu',
    'Device.',
    allow_override=True,
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 64, 'Serving batch.', allow_override=True
)
_BLOCK_DTYPE = flags.DEFINE_enum(
    'block_dtype',
    None,
    ['float32', 'float16', 'bfloat16'],
    'Block precision (default: float16 on CUDA).',
    allow_override=True,
)
_ITERS = flags.DEFINE_integer(
    'iters', 500, 'Timed forward passes.', allow_override=True
)
_WARMUP = flags.DEFINE_integer(
    'warmup', 50, 'Untimed forward passes.', allow_override=True
)
_PROFILE_ONCE = flags.DEFINE_bool(
    'profile_once',
    False,
    'One NVTX-wrapped pass, no timing.',
    allow_override=True,
)
_OUT = flags.DEFINE_string('out', '', 'Output JSON.', allow_override=True)


def _model(spec: compile_entry.ServingSpec) -> model_lib.PriceTransformer:
    if _CHECKPOINT_DIR.value:
        model = checkpoint.load(_CHECKPOINT_DIR.value)
    else:
        torch.manual_seed(0)
        model = model_lib.PriceTransformer(model_lib.ModelConfig()).eval()
    return compile_entry.prepare(model, spec)


def _timed(fn, inputs, device: str, iters: int, warmup: int) -> float:
    """Seconds per call."""
    with torch.no_grad():
        for _ in range(warmup):
            fn(*inputs)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn(*inputs)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / 1000.0 / iters
        begin = time.perf_counter()
        for _ in range(iters):
            fn(*inputs)
        return (time.perf_counter() - begin) / iters


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    device = _DEVICE.value
    spec = compile_entry.ServingSpec(
        device,
        _BATCH_SIZE.value,
        _BLOCK_DTYPE.value or compile_entry.default_block_dtype(device),
    )
    eager = _model(spec)
    inputs = tuple(
        model_lib.example_inputs(eager.config, spec.batch_size, device)
    )
    sites = 0
    if _MODE.value == 'eager':
        fn = eager
    elif _MODE.value == 'inductor':
        fn = torch.compile(eager, fullgraph=True, dynamic=False)
    else:
        fn, report, _ = compile_entry.compile_model(eager, spec)
        sites = report.sites
    worst, _ = compile_entry.max_abs_error(
        fn, eager, eager.config, spec, batches=2
    )
    if _PROFILE_ONCE.value:
        with torch.no_grad():
            fn(*inputs)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push('vectra_model')
            fn(*inputs)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
        return
    seconds = _timed(fn, inputs, device, _ITERS.value, _WARMUP.value)
    result = {
        'mode': _MODE.value,
        'device': (
            torch.cuda.get_device_name(device)
            if device.startswith('cuda')
            else 'cpu'
        ),
        'batch_size': spec.batch_size,
        'block_dtype': spec.block_dtype,
        'num_candidates': eager.config.num_candidates,
        'rewritten_sites': sites,
        'ms_per_batch': seconds * 1e3,
        'requests_per_s': spec.batch_size / seconds,
        'max_abs_error_vs_eager': worst,
        'numerics_rtol': compile_entry.RTOL,
        'versions': compile_entry.toolchain_versions(),
    }
    if _OUT.value:
        os.makedirs(os.path.dirname(_OUT.value) or '.', exist_ok=True)
        with open(_OUT.value, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    app.run(main)
