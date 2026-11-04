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

"""Turns a recall sweep into the report the regression gate reads.

  python -m benchmark.recall.analyze_recall \
      --sweep=benchmark/reports/recall_sweep.json \
      --out=benchmark/reports/recall.json

The report carries recall@10 at the configured ef_search (what production
runs), exact-mode boundary recall, the smallest ef_search on the curve that
meets the target, and the recall/latency curve itself for the
hnsw-parameter study.
"""

from collections.abc import Sequence
import json

from absl import app
from absl import flags

_SWEEP = flags.DEFINE_string(
    'sweep', None, 'run_recall_sweep.py output.', allow_override=True
)
_TARGET = flags.DEFINE_float(
    'target', 0.95, 'Recall@10 target.', allow_override=True
)
_OUT = flags.DEFINE_string(
    'out',
    'benchmark/reports/recall.json',
    'Report for compare_snapshots.py.',
    allow_override=True,
)


def summarize(sweep: dict, target: float) -> dict:
    """The gate report of one sweep.

    Raises:
      ValueError: If the configured ef_search is not on the curve.
    """
    curve = sorted(sweep['curve'], key=lambda p: p['ef_search'])
    configured = sweep['configured_ef_search']
    at_configured = [p for p in curve if p['ef_search'] == configured]
    if not at_configured:
        raise ValueError(
            f'the sweep does not include the configured ef_search {configured}'
        )
    meeting = [p['ef_search'] for p in curve if p['recall_at_10'] >= target]
    return {
        'recall_at_10': at_configured[0]['recall_at_10'],
        'ef_search': configured,
        'boundary_recall': sweep.get('boundary_recall'),
        'boundary_queries': sweep.get('boundary_queries', 0),
        'min_ef_search_for_target': meeting[0] if meeting else None,
        'target': target,
        'curve': curve,
        'hnswlib_curve': sweep.get('hnswlib_curve'),
    }


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    with open(_SWEEP.value, encoding='utf-8') as f:
        report = summarize(json.load(f), _TARGET.value)
    with open(_OUT.value, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ('curve', 'hnswlib_curve')
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    flags.mark_flag_as_required('sweep')
    app.run(main)
