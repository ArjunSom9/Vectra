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

"""Latency summaries and the maximum sustained rate of a load sweep.

  python -m benchmark.qps_latency.percentiles \
      --sweep=benchmark/reports/search_sweep.json --metric=p50_us --bound=1000

QPS is reported as the highest offered rate at which the latency bound
holds and the load was actually delivered, the definition of the design's
Section 9 and of ANN-Benchmarks [4].
"""

from collections.abc import Sequence
import json

from absl import app
from absl import flags
import numpy as np

_SWEEP = flags.DEFINE_string(
    'sweep', None, 'run_soak.py --sweep output.', allow_override=True
)
_METRIC = flags.DEFINE_string(
    'metric', 'p50_us', 'Latency percentile.', allow_override=True
)
_BOUND = flags.DEFINE_float(
    'bound', 1000.0, 'Latency bound, microseconds.', allow_override=True
)

# A run counts as sustained only if it delivered nearly all offered load.
MIN_DELIVERY = 0.99
_MISSING = float('nan')
_COLUMNS = (
    'offered_qps',
    'achieved_qps',
    'p50_us',
    'p90_us',
    'p99_us',
    'p999_us',
    'max_us',
    'errors',
)


def summarize(latencies_us: Sequence[float]) -> dict[str, float]:
    """The percentiles stress_client reports, from raw latencies."""
    values = np.asarray(latencies_us, dtype=np.float64)
    if not values.size:
        return {'count': 0}
    p50, p90, p99, p999 = np.percentile(values, [50, 90, 99, 99.9])
    return {
        'count': int(values.size),
        'p50_us': float(p50),
        'p90_us': float(p90),
        'p99_us': float(p99),
        'p999_us': float(p999),
        'max_us': float(values.max()),
        'mean_us': float(values.mean()),
    }


def sustained(report: dict, metric: str, bound: float) -> bool:
    """Whether one run delivered its load within the latency bound."""
    delivered = report['achieved_qps'] >= MIN_DELIVERY * report['offered_qps']
    return (
        delivered and report.get('errors', 0) == 0 and report[metric] <= bound
    )


def max_sustained_qps(
    reports: Sequence[dict], metric: str, bound: float
) -> float | None:
    """Highest offered rate that was sustained, or None if none was.

    Rates above the first failing one do not count: a sweep that passes
    again after failing is noise, not capacity.
    """
    best = None
    for report in sorted(reports, key=lambda r: r['offered_qps']):
        if not sustained(report, metric, bound):
            break
        best = report['offered_qps']
    return best


def table(reports: Sequence[dict]) -> str:
    """A Markdown table of a sweep."""
    lines = ['| ' + ' | '.join(_COLUMNS) + ' |', '|' + '---|' * len(_COLUMNS)]
    for report in sorted(reports, key=lambda r: r['offered_qps']):
        cells = [f'{report.get(column, _MISSING):,.0f}' for column in _COLUMNS]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    with open(_SWEEP.value, encoding='utf-8') as f:
        reports = json.load(f)['runs']
    print(table(reports))
    best = max_sustained_qps(reports, _METRIC.value, _BOUND.value)
    shown = 'none' if best is None else f'{best:,.0f}'
    print(
        f'\nmax sustained QPS with {_METRIC.value} <= {_BOUND.value:g}: {shown}'
    )


if __name__ == '__main__':
    flags.mark_flag_as_required('sweep')
    app.run(main)
