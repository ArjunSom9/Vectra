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

"""The O2 throughput gate: candidate >= 3.0x the eager baseline.

  python -m benchmark.gpu_throughput.compare_runs \
      --baseline=benchmark/reports/throughput_eager.json \
      --candidate=benchmark/reports/throughput_vectra.json

Numerics come first: a run whose output left the rtol 1e-3 band is
rejected before its speed is looked at. Both runs must use the same device,
batch and precision, or the ratio means nothing.
"""

from collections.abc import Sequence
import json
import sys

from absl import app
from absl import flags

_BASELINE = flags.DEFINE_string(
    'baseline', None, 'Eager run JSON.', allow_override=True
)
_CANDIDATE = flags.DEFINE_string(
    'candidate', None, 'Compiled run JSON.', allow_override=True
)
_MIN_RATIO = flags.DEFINE_float(
    'min_ratio', 3.0, 'Required speed-up.', allow_override=True
)

_MATCHED = ('device', 'batch_size', 'block_dtype', 'num_candidates')


def compare(
    baseline: dict, candidate: dict, min_ratio: float
) -> tuple[bool, list[str], float]:
    """(passed, problems, throughput ratio)."""
    problems = [
        f'{key} differs: {baseline.get(key)} vs {candidate.get(key)}'
        for key in _MATCHED
        if baseline.get(key) != candidate.get(key)
    ]
    tolerance = candidate.get('numerics_rtol', 1e-3)
    # The model's output is a log price near 13, so rtol 1e-3 allows ~0.013.
    if candidate['max_abs_error_vs_eager'] > 13.0 * tolerance + 1e-3:
        error = candidate['max_abs_error_vs_eager']
        problems.append(f'numerics: max |compiled - eager| = {error:.3g}')
    ratio = candidate['requests_per_s'] / baseline['requests_per_s']
    if ratio < min_ratio:
        problems.append(f'throughput ratio {ratio:.2f} < {min_ratio}')
    return not problems, problems, ratio


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    with open(_BASELINE.value, encoding='utf-8') as f:
        baseline = json.load(f)
    with open(_CANDIDATE.value, encoding='utf-8') as f:
        candidate = json.load(f)
    passed, problems, ratio = compare(baseline, candidate, _MIN_RATIO.value)
    mode, base, device = (
        candidate['mode'],
        baseline['mode'],
        candidate['device'],
    )
    print(f'{mode} / {base} throughput: {ratio:.2f}x on {device}')
    for problem in problems:
        print('FAIL', problem)
    print('PASS' if passed else 'FAIL')
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    flags.mark_flags_as_required(['baseline', 'candidate'])
    app.run(main)
