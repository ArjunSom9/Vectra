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

"""The regression gate: thresholds, baselines and the golden files."""

import json

import pytest

from tests.regression import compare_snapshots


def _load(name):
    with open(f'tests/regression/{name}', encoding='utf-8') as f:
        return json.load(f)


def _search_report(**overrides):
    report = {
        'mode': 'search',
        'offered_qps': 50000.0,
        'achieved_qps': 49980.0,
        'p50_us': 640.0,
        'p99_us': 2100.0,
        'p999_us': 4000.0,
        'errors': 0,
    }
    report.update(overrides)
    return report


def test_golden_files_hold_the_design_gates_and_no_invented_baseline():
    latency = _load('golden_latency.json')
    assert latency['checks']['search']['max']['p50_us'] == 1000
    assert latency['checks']['dispatch']['max']['p99_us'] == 100
    recall = _load('golden_recall.json')
    assert recall['checks']['recall']['min']['recall_at_10'] == 0.95
    assert recall['checks']['recall']['min']['boundary_recall'] == 1.0
    assert latency['regression']['baseline'] is None
    assert recall['regression']['baseline'] is None


def test_thresholds():
    golden = _load('golden_latency.json')
    assert not compare_snapshots.gate(golden, _search_report())
    slow = compare_snapshots.gate(golden, _search_report(p50_us=1300.0))
    assert len(slow) == 1 and 'p50_us' in slow[0]
    short = compare_snapshots.gate(golden, _search_report(achieved_qps=41000.0))
    assert len(short) == 1 and 'achieved_qps' in short[0]
    ping = {
        'mode': 'ping',
        'offered_qps': 50000,
        'achieved_qps': 50000,
        'p99_us': 140.0,
        'errors': 0,
    }
    assert 'p99_us' in compare_snapshots.gate(golden, ping)[0]


def test_regressions_against_a_recorded_baseline():
    golden = compare_snapshots.update_baseline(
        _load('golden_latency.json'), [_search_report()], 'test machine'
    )
    assert golden['regression']['baseline']['environment'] == 'test machine'
    # 10% tolerance on p50: 700 passes, 720 regresses.
    assert not compare_snapshots.regressions(
        golden, _search_report(p50_us=700.0)
    )
    failures = compare_snapshots.regressions(
        golden, _search_report(p50_us=720.0)
    )
    assert len(failures) == 1 and 'p50_us' in failures[0]


def test_recall_regressions_use_absolute_tolerance():
    recorded = {'recall_at_10': 0.97, 'boundary_recall': 1.0}
    golden = compare_snapshots.update_baseline(
        _load('golden_recall.json'), [recorded], 'test machine'
    )
    assert not compare_snapshots.regressions(
        golden, dict(recorded, recall_at_10=0.966)
    )
    assert compare_snapshots.regressions(
        golden, dict(recorded, recall_at_10=0.96)
    )


def test_recording_a_baseline_requires_its_environment():
    with pytest.raises(ValueError):
        compare_snapshots.update_baseline(
            _load('golden_latency.json'), [_search_report()], ''
        )
