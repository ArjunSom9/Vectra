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

"""Runs Nsight Compute over a command and aggregates its counters.

  python -m benchmark.bandwidth.collect_nsight_metrics \
      --nvtx_range=vectra_region \
      -- python -m kernels.benchmarks.bandwidth --impl=fused --batch=1024

Every kernel inside the NVTX range is profiled with the fixed configuration
of kernels/benchmarks/nsight/README.md (L2 flushed before each kernel, base
clocks). The result sums bytes and time over the region's kernels and
averages DRAM throughput weighted by kernel time.
"""

from collections.abc import Sequence
import csv
import io
import json
import subprocess

from absl import app
from absl import flags

METRICS = (
    'dram__bytes_read.sum',
    'dram__bytes_write.sum',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'gpu__time_duration.sum',
)

_NCU = flags.DEFINE_string(
    'ncu', 'ncu', 'Nsight Compute CLI.', allow_override=True
)
_NVTX_RANGE = flags.DEFINE_string(
    'nvtx_range',
    'vectra_region',
    'Profiled NVTX push/pop range.',
    allow_override=True,
)


def ncu_command(
    command: Sequence[str], nvtx_range: str, ncu: str = 'ncu'
) -> list[str]:
    return [
        ncu,
        '--nvtx',
        '--nvtx-include',
        f'{nvtx_range}/',
        '--metrics',
        ','.join(METRICS),
        '--cache-control',
        'all',
        '--clock-control',
        'base',
        '--target-processes',
        'all',
        '--csv',
        '--print-units',
        'base',
        *command,
    ]


def _number(text: str) -> float:
    return float(text.replace(',', '')) if text not in ('', 'n/a') else 0.0


def parse_csv(text: str) -> dict[str, dict[str, float]]:
    """{kernel instance: {metric: value}} from `ncu --csv` output."""
    start = text.find('"ID"')
    if start < 0:
        raise ValueError('no Nsight Compute CSV table in the output')
    kernels = {}
    for row in csv.DictReader(io.StringIO(text[start:])):
        kernel_id, kernel_name = row['ID'], row['Kernel Name']
        key = f'{kernel_id}:{kernel_name}'
        kernels.setdefault(key, {})[row['Metric Name']] = _number(
            row['Metric Value']
        )
    return kernels


def aggregate(kernels: dict[str, dict[str, float]]) -> dict[str, float]:
    """Region totals: bytes and time summed, throughput time-weighted."""
    time_ns = sum(
        k.get('gpu__time_duration.sum', 0.0) for k in kernels.values()
    )
    weighted = sum(
        k.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', 0.0)
        * k.get('gpu__time_duration.sum', 0.0)
        for k in kernels.values()
    )
    read = sum(k.get('dram__bytes_read.sum', 0.0) for k in kernels.values())
    write = sum(k.get('dram__bytes_write.sum', 0.0) for k in kernels.values())
    return {
        'kernels': len(kernels),
        'dram_bytes_read': read,
        'dram_bytes_write': write,
        'dram_bytes': read + write,
        'gpu_time_ns': time_ns,
        'dram_throughput_pct': weighted / time_ns if time_ns else 0.0,
    }


def collect(
    command: Sequence[str], nvtx_range: str, ncu: str = 'ncu'
) -> dict[str, float]:
    output = subprocess.run(
        ncu_command(command, nvtx_range, ncu),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return aggregate(parse_csv(output))


def main(argv: Sequence[str]) -> None:
    if len(argv) < 2:
        raise app.UsageError('usage: collect_nsight_metrics -- COMMAND...')
    print(
        json.dumps(collect(argv[1:], _NVTX_RANGE.value, _NCU.value), indent=2)
    )


if __name__ == '__main__':
    app.run(main)
