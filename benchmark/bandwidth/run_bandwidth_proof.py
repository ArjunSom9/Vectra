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

"""Bytes per query and achieved DRAM bandwidth, eager versus fused.

  python -m benchmark.bandwidth.run_bandwidth_proof \
      --out=benchmark/reports/bandwidth.json

Profiles the attention region of kernels/benchmarks/bandwidth.py under
Nsight Compute for both implementations and two batches: the serving batch
(64) and a batch whose working set exceeds L2 (1024). It then checks
serving/configs/slo.yaml:

* B3 / O2: fused DRAM bytes per query <= 60% of eager, at both batches;
* C3 / O3: fused DRAM throughput >= 65% of peak at the L2-exceeding batch,
  where the measurement reflects streaming rather than launch latency.

The eager throughput at the same batch is the B1 baseline. The analytic
budgets of kernels/benchmarks/roofline.py are reported alongside, so a
counter that disagrees with the traffic model stands out.
"""

from collections.abc import Sequence
import json
import os
import sys

from absl import app
from absl import flags
import yaml

from benchmark.bandwidth import collect_nsight_metrics
from kernels.benchmarks import roofline

_NCU = flags.DEFINE_string(
    'ncu', 'ncu', 'Nsight Compute CLI.', allow_override=True
)
_PYTHON = flags.DEFINE_string(
    'python', sys.executable, 'Interpreter.', allow_override=True
)
_BATCHES = flags.DEFINE_list(
    'batches', ['64', '1024'], 'Batches profiled.', allow_override=True
)
_SLO = flags.DEFINE_string(
    'slo', 'serving/configs/slo.yaml', 'Targets.', allow_override=True
)
_OUT = flags.DEFINE_string(
    'out', 'benchmark/reports/bandwidth.json', 'Report.', allow_override=True
)

# The batch at which bandwidth utilization is judged (working set > L2).
UTILIZATION_BATCH = 1024


def evaluate(measured: dict, slo: dict) -> list[str]:
    """Failures of a {batch: {impl: counters}} measurement against the SLO."""
    max_ratio = slo['gpu']['dram_bytes_ratio_max']
    min_pct = slo['gpu']['dram_throughput_pct_min']
    failures = []
    for batch, runs in measured.items():
        ratio = runs['fused']['dram_bytes'] / runs['eager']['dram_bytes']
        if ratio > max_ratio:
            failures.append(
                f'batch {batch}: fused/eager bytes {ratio:.3f} > {max_ratio}'
            )
    utilization = measured.get(UTILIZATION_BATCH)
    if utilization is None:
        failures.append(f'batch {UTILIZATION_BATCH} was not profiled')
    elif utilization['fused']['dram_throughput_pct'] < min_pct:
        achieved = utilization['fused']['dram_throughput_pct']
        failures.append(f'fused DRAM throughput {achieved:.1f}% < {min_pct}%')
    return failures


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    with open(_SLO.value, encoding='utf-8') as f:
        slo = yaml.safe_load(f)
    measured, budgets = {}, {}
    for batch in (int(b) for b in _BATCHES.value):
        measured[batch] = {}
        for impl in ('eager', 'fused'):
            command = [
                _PYTHON.value,
                '-m',
                'kernels.benchmarks.bandwidth',
                f'--impl={impl}',
                f'--batch={batch}',
            ]
            counters = collect_nsight_metrics.collect(
                command, 'vectra_region', _NCU.value
            )
            counters['dram_bytes_per_query'] = counters['dram_bytes'] / batch
            measured[batch][impl] = counters
        shape = roofline.Shape(batch, 4, 65, 65, 32, 32, 2)
        budgets[batch] = roofline.analyze(shape)
    failures = evaluate(measured, slo)
    report = {
        'measured': measured,
        'analytic_budgets': budgets,
        'failures': failures,
        'passed': not failures,
    }
    os.makedirs(os.path.dirname(_OUT.value) or '.', exist_ok=True)
    with open(_OUT.value, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'failures': failures, 'passed': not failures}, indent=2))
    sys.exit(0 if not failures else 1)


if __name__ == '__main__':
    app.run(main)
