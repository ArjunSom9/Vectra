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

"""Sustained-load runs of the C++ load generator (deliverables A1, A3).

  # A3: 30 minutes of searches at 50,000 QPS through the geo router.
  python -m benchmark.qps_latency.run_soak --mode=search --qps=50000 \
      --duration=30m --config_dir=engine/configs \
      --queries=data/corpus/dev/queries --out=benchmark/reports/search.json

  # A1: no-op RPCs through the channel pool to one node.
  python -m benchmark.qps_latency.run_soak --mode=ping --qps=50000 \
      --target=10.0.1.11:7101 --out=benchmark/reports/ping.json

engine/apps/stress_client offers load open-loop: requests leave on a fixed
schedule regardless of the server's state, and each latency is measured from
its scheduled send time, so a stalled server cannot hide its own queueing
(coordinated omission). --sweep also runs a list of rates and records the
maximum sustained one (see percentiles.py). Gate the result with
tests/regression/compare_snapshots.py.
"""

from collections.abc import Sequence
import json
import os
import subprocess
import tempfile

from absl import app
from absl import flags
from absl import logging

from benchmark.qps_latency import percentiles

_STRESS_CLIENT = flags.DEFINE_string(
    'stress_client',
    'build/engine/apps/stress_client/stress_client',
    'Load generator binary.',
    allow_override=True,
)
_MODE = flags.DEFINE_enum(
    'mode',
    'search',
    ['search', 'ping'],
    'search: Search via the router; ping: no-op RPCs.',
    allow_override=True,
)
_QPS = flags.DEFINE_float('qps', 50000.0, 'Offered load.', allow_override=True)
_SWEEP = flags.DEFINE_list(
    'sweep', [], 'Also run these offered rates.', allow_override=True
)
_DURATION = flags.DEFINE_string(
    'duration', '60s', 'Measured duration.', allow_override=True
)
_WARMUP = flags.DEFINE_string(
    'warmup', '10s', 'Unmeasured warm-up.', allow_override=True
)
_TARGET = flags.DEFINE_string(
    'target', '', 'host:port (ping mode).', allow_override=True
)
_CONFIG_DIR = flags.DEFINE_string(
    'config_dir',
    'engine/configs',
    'Engine configuration (search mode).',
    allow_override=True,
)
_QUERIES = flags.DEFINE_string(
    'queries',
    '',
    'Query directory (query_embedding.npy, ...).',
    allow_override=True,
)
_SENDERS = flags.DEFINE_integer(
    'senders', 4, 'Pacing threads.', allow_override=True
)
_POLLERS = flags.DEFINE_integer(
    'pollers', 4, 'Completion-queue threads.', allow_override=True
)
_RADIUS_KM = flags.DEFINE_float(
    'radius_km', 5.0, 'Search radius.', allow_override=True
)
_TOP_K = flags.DEFINE_integer(
    'top_k', 64, 'Neighbors per search.', allow_override=True
)
_OUT = flags.DEFINE_string(
    'out',
    'benchmark/reports/soak.json',
    'Report for the --qps run.',
    allow_override=True,
)


def command(qps: float, report: str) -> list[str]:
    """The stress_client invocation for one offered rate."""
    args = [
        _STRESS_CLIENT.value,
        f'--mode={_MODE.value}',
        f'--qps={qps}',
        f'--duration={_DURATION.value}',
        f'--warmup={_WARMUP.value}',
        f'--senders={_SENDERS.value}',
        f'--pollers={_POLLERS.value}',
        f'--report={report}',
    ]
    if _MODE.value == 'ping' or _TARGET.value:
        args.append(f'--target={_TARGET.value}')
    else:
        args += [
            f'--config_dir={_CONFIG_DIR.value}',
            f'--radius_km={_RADIUS_KM.value}',
            f'--top_k={_TOP_K.value}',
        ]
        if _QUERIES.value:
            args.append(f'--queries={_QUERIES.value}')
    return args


def run_once(qps: float) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        report = os.path.join(directory, 'report.json')
        logging.info('offering %.0f QPS for %s', qps, _DURATION.value)
        subprocess.run(command(qps, report), check=True)
        with open(report, encoding='utf-8') as f:
            return json.load(f)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    if _MODE.value == 'ping' and not _TARGET.value:
        raise app.UsageError('--mode=ping needs --target')
    os.makedirs(os.path.dirname(_OUT.value) or '.', exist_ok=True)
    main_run = run_once(_QPS.value)
    with open(_OUT.value, 'w', encoding='utf-8') as f:
        json.dump(main_run, f, indent=2)
    print(json.dumps(main_run, indent=2))
    if _SWEEP.value:
        runs = [run_once(float(q)) for q in _SWEEP.value] + [main_run]
        metric = 'p99_us' if _MODE.value == 'ping' else 'p50_us'
        bound = 100.0 if _MODE.value == 'ping' else 1000.0
        best = percentiles.max_sustained_qps(runs, metric, bound)
        sweep_path = os.path.splitext(_OUT.value)[0] + '_sweep.json'
        with open(sweep_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'metric': metric,
                    'bound_us': bound,
                    'max_sustained_qps': best,
                    'runs': runs,
                },
                f,
                indent=2,
            )
        print(percentiles.table(runs))
        print(f'max sustained QPS ({metric} <= {bound:g} us): {best}')


if __name__ == '__main__':
    app.run(main)
