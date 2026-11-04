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

"""One profiled launch of the eager chain or the fused kernel.

  ncu --nvtx --nvtx-include "vectra_region/" \
      --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
      python -m kernels.benchmarks.bandwidth --impl=fused --batch=1024

The measured region is wrapped in the NVTX range `vectra_region`, so
Nsight Compute profiles exactly the kernels of one forward pass of the
attention region and nothing else (warm-up and setup run outside it).
benchmark/bandwidth/run_bandwidth_proof.py drives this module for both
implementations and compares the counters. Needs a CUDA GPU.
"""

from collections.abc import Sequence

from absl import app
from absl import flags
import torch

from kernels.benchmarks import roofline
from kernels.benchmarks import throughput
from vectra_kernels import reference

REGION = 'vectra_region'

_IMPL = flags.DEFINE_enum(
    'impl',
    'fused',
    ['eager', 'fused'],
    'Implementation to launch.',
    allow_override=True,
)
_BATCH = flags.DEFINE_integer(
    'batch', 1024, 'Requests per batch.', allow_override=True
)
_TOKENS = flags.DEFINE_integer(
    'tokens', 65, 'Query + candidate tokens.', allow_override=True
)
_HEADS = flags.DEFINE_integer('heads', 4, 'Heads.', allow_override=True)
_HEAD_DIM = flags.DEFINE_integer(
    'head_dim', 32, 'Head width.', allow_override=True
)


def launch(impl: str, shape: roofline.Shape) -> torch.Tensor:
    """Warms up, then runs one forward pass inside the NVTX region."""
    from vectra_kernels import spatial_rbf_attention  # pylint: disable=import-outside-toplevel

    q, k, v, scale, bias = throughput.problem(shape, torch.float16)

    def fn():
        if impl == 'eager':
            return reference.eager_rbf_attention(q, k, v, scale.half(), bias)
        return spatial_rbf_attention.rbf_attention_forward(q, k, v, scale, bias)

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(REGION)
    out = fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    return out


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    if not torch.cuda.is_available():
        raise app.UsageError('kernels.benchmarks.bandwidth needs a CUDA GPU')
    shape = roofline.Shape(
        _BATCH.value,
        _HEADS.value,
        _TOKENS.value,
        _TOKENS.value,
        _HEAD_DIM.value,
        _HEAD_DIM.value,
        2,
    )
    launch(_IMPL.value, shape)


if __name__ == '__main__':
    app.run(main)
