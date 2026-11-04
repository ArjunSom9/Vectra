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

"""Checks benchmark reports against the acceptance gates and a baseline.

  python -m tests.regression.compare_snapshots \
      --golden=tests/regression/golden_latency.json \
      --report=benchmark/reports/search.json \
      --report=benchmark/reports/ping.json

Each golden file holds `checks` (the design's acceptance thresholds, which
never move) and `regression.baseline` (a recorded run on the reference
environment, null until one is recorded). A report passes when it meets
every threshold of the check it belongs to and, if a baseline exists, does
not regress past the stated tolerance. `--update` records the reports as
the new baseline, stamped with the environment they came from.

Exit status: 0 when everything passes, 1 otherwise.
"""

from collections.abc import Sequence
import json
import platform
import sys

from absl import app
from absl import flags

_GOLDEN = flags.DEFINE_string(
    'golden', None, 'Golden JSON file.', allow_override=True
)
_REPORTS = flags.DEFINE_multi_string(
    'report', [], 'Benchmark report JSON.', allow_override=True
)
_UPDATE = flags.DEFINE_bool(
    'update',
    False,
    'Record the reports as the new baseline.',
    allow_override=True,
)
_ENVIRONMENT = flags.DEFINE_string(
    'environment',
    '',
    'Description of the machine the reports came from '
    '(required with --update).',
    allow_override=True,
)

# Metrics where larger is better; every other gated metric is a latency or
# a count where smaller is better.
_HIGHER_IS_BETTER = ('qps', 'recall')


def _higher_is_better(metric: str) -> bool:
    return any(token in metric for token in _HIGHER_IS_BETTER)


def check_for(golden: dict, report: dict) -> str | None:
    """Name of the check a report belongs to (matched on its `mode`)."""
    for name, check in golden['checks'].items():
        if check.get('mode', report.get('mode')) == report.get('mode'):
            return name
    return None


def gate(golden: dict, report: dict) -> list[str]:
    """Threshold violations of one report (empty when it passes)."""
    name = check_for(golden, report)
    if name is None:
        mode = report.get('mode')
        return [f'no check matches mode {mode!r}']
    check = golden['checks'][name]
    failures = []
    for metric, bound in check.get('min', {}).items():
        if metric not in report:
            failures.append(f'{name}: report lacks {metric}')
        elif report[metric] < bound:
            failures.append(f'{name}: {metric} {report[metric]} < {bound}')
    for metric, bound in check.get('max', {}).items():
        if metric not in report:
            failures.append(f'{name}: report lacks {metric}')
        elif report[metric] > bound:
            failures.append(f'{name}: {metric} {report[metric]} > {bound}')
    return failures


def regressions(golden: dict, report: dict) -> list[str]:
    """Regressions of one report against the recorded baseline, if any."""
    regression = golden.get('regression', {})
    baseline = (regression.get('baseline') or {}).get('reports', {})
    name = check_for(golden, report)
    if name not in baseline:
        return []
    previous = baseline[name]
    failures = []
    for kind in ('relative_tolerance', 'absolute_tolerance'):
        for metric, tolerance in regression.get(kind, {}).items():
            if metric not in report or metric not in previous:
                continue
            old, new = previous[metric], report[metric]
            allowed = tolerance * abs(old) if kind[0] == 'r' else tolerance
            worse = (old - new) if _higher_is_better(metric) else (new - old)
            if worse > allowed:
                failures.append(
                    f'{name}: {metric} regressed from {old} to '
                    f'{new} (tolerance {allowed:g})'
                )
    return failures


def update_baseline(
    golden: dict, reports: Sequence[dict], environment: str
) -> dict:
    """A copy of `golden` with `reports` recorded as the baseline."""
    if not environment:
        raise ValueError('--environment is required to record a baseline')
    recorded = {check_for(golden, r): r for r in reports}
    updated = json.loads(json.dumps(golden))
    updated.setdefault('regression', {})['baseline'] = {
        'environment': environment,
        'host': platform.node(),
        'reports': recorded,
    }
    return updated


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    with open(_GOLDEN.value, encoding='utf-8') as f:
        golden = json.load(f)
    reports = []
    for path in _REPORTS.value:
        with open(path, encoding='utf-8') as f:
            reports.append(json.load(f))
    failures = []
    for report in reports:
        failures += gate(golden, report) + regressions(golden, report)
    for failure in failures:
        print('FAIL', failure)
    if not failures and _UPDATE.value:
        with open(_GOLDEN.value, 'w', encoding='utf-8') as f:
            json.dump(
                update_baseline(golden, reports, _ENVIRONMENT.value),
                f,
                indent=2,
            )
            f.write('\n')
        print(f'recorded {len(reports)} report(s) as the baseline')
    print('PASS' if not failures else f'{len(failures)} failure(s)')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    flags.mark_flag_as_required('golden')
    app.run(main)
