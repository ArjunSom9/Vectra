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

"""Prometheus exporter for the shard fleet's HealthService reports.

  python -m serving.engine_exporter --config_dir=engine/configs --port=9465

The C++ shards keep Prometheus and HTTP off their request path; they answer
HealthService.Report over gRPC instead. This exporter polls every shard
replica of the shard map and republishes the reports as metrics
(deploy/observability/prometheus.yml scrapes it):

  vectra_shard_up                        1 if the last Report succeeded
  vectra_shard_report_seconds            Report round-trip time
  vectra_shard_vectors / _capacity       index occupancy
  vectra_shard_index_bytes               resident index bytes
  vectra_shard_memory_locked             1 when the index is mlock()ed
  vectra_shard_rpcs_completed_total      per completion-queue worker
  vectra_shard_call_pool_high_water      peak in-flight calls per worker
  vectra_shard_call_pool_allocations     calls allocated after start-up
                                         (should stay 0)
"""

import asyncio
from collections.abc import Sequence
import dataclasses
import threading
import time

from absl import app
from absl import flags
from absl import logging
import grpc
from prometheus_client import core as prometheus_core
import prometheus_client

from data import routing
from vectra import health_pb2
from vectra import health_pb2_grpc

_CONFIG_DIR = flags.DEFINE_string(
    'config_dir',
    'engine/configs',
    'Engine configuration (shard map endpoints).',
    allow_override=True,
)
_PORT = flags.DEFINE_integer(
    'port', 9465, 'Metrics HTTP port.', allow_override=True
)
_INTERVAL_S = flags.DEFINE_float(
    'interval_s', 5.0, 'Seconds between polls.', allow_override=True
)
_TIMEOUT_S = flags.DEFINE_float(
    'timeout_s', 2.0, 'Report deadline.', allow_override=True
)


@dataclasses.dataclass
class Sample:
    """The latest Report from one replica."""

    shard_id: int
    endpoint: str
    up: bool
    seconds: float
    report: health_pb2.ReportResponse | None


class FleetCollector:
    """A Prometheus collector over the latest Report of every replica."""

    def __init__(self):
        self._lock = threading.Lock()
        self._samples: dict[str, Sample] = {}

    def update(self, sample: Sample) -> None:
        with self._lock:
            self._samples[sample.endpoint] = sample

    def collect(self):
        """Yields metric families (the prometheus_client collector API)."""
        with self._lock:
            samples = list(self._samples.values())
        labels = ['shard', 'endpoint']
        up = prometheus_core.GaugeMetricFamily(
            'vectra_shard_up', 'Last Report succeeded.', labels=labels
        )
        seconds = prometheus_core.GaugeMetricFamily(
            'vectra_shard_report_seconds',
            'Report round-trip time.',
            labels=labels,
        )
        gauges = {
            name: prometheus_core.GaugeMetricFamily(
                f'vectra_shard_{name}', text, labels=labels
            )
            for name, text in (
                ('vectors', 'Vectors in the index.'),
                ('capacity', 'Index capacity.'),
                ('index_bytes', 'Resident index bytes.'),
                ('memory_locked', '1 when the index is locked in RAM.'),
            )
        }
        worker_labels = labels + ['worker', 'core']
        rpcs = prometheus_core.CounterMetricFamily(
            'vectra_shard_rpcs_completed',
            'RPCs completed per completion-queue worker.',
            labels=worker_labels,
        )
        high_water = prometheus_core.GaugeMetricFamily(
            'vectra_shard_call_pool_high_water',
            'Peak in-flight calls per worker.',
            labels=worker_labels,
        )
        allocations = prometheus_core.GaugeMetricFamily(
            'vectra_shard_call_pool_allocations',
            'Calls allocated after start-up (should stay 0).',
            labels=worker_labels,
        )
        for sample in samples:
            key = [str(sample.shard_id), sample.endpoint]
            up.add_metric(key, 1.0 if sample.up else 0.0)
            seconds.add_metric(key, sample.seconds)
            report = sample.report
            if not sample.up or report is None:
                continue
            gauges['vectors'].add_metric(key, report.index.num_vectors)
            gauges['capacity'].add_metric(key, report.index.capacity)
            gauges['index_bytes'].add_metric(key, report.index.index_bytes)
            gauges['memory_locked'].add_metric(
                key, 1.0 if report.memory.locked else 0.0
            )
            for worker in report.workers:
                worker_key = key + [str(worker.worker), str(worker.core)]
                rpcs.add_metric(worker_key, worker.rpcs_completed)
                high_water.add_metric(worker_key, worker.call_pool_high_water)
                allocations.add_metric(worker_key, worker.call_pool_allocations)
        yield up
        yield seconds
        yield from gauges.values()
        yield rpcs
        yield high_water
        yield allocations


async def poll_once(
    channels: dict[str, tuple[int, grpc.aio.Channel]],
    collector: FleetCollector,
    timeout_s: float,
) -> None:
    """Asks every replica for its Report and records the results."""

    async def ask(endpoint: str, shard_id: int, channel: grpc.aio.Channel):
        stub = health_pb2_grpc.HealthServiceStub(channel)
        start = time.perf_counter()
        try:
            report = await stub.Report(
                health_pb2.ReportRequest(), timeout=timeout_s
            )
            up = True
        except grpc.aio.AioRpcError as e:
            logging.warning('shard %d at %s: %s', shard_id, endpoint, e.code())
            report, up = None, False
        collector.update(
            Sample(shard_id, endpoint, up, time.perf_counter() - start, report)
        )

    await asyncio.gather(*(ask(e, s, c) for e, (s, c) in channels.items()))


def open_channels(
    config: routing.EngineConfig,
) -> dict[str, tuple[int, grpc.aio.Channel]]:
    """One channel per replica endpoint (inside an event loop)."""
    return {
        endpoint: (shard.id, grpc.aio.insecure_channel(endpoint))
        for shard in config.shard_map.shards
        for endpoint in shard.endpoints
    }


async def _run(config: routing.EngineConfig, collector: FleetCollector):
    channels = open_channels(config)
    try:
        while True:
            await poll_once(channels, collector, _TIMEOUT_S.value)
            await asyncio.sleep(_INTERVAL_S.value)
    finally:
        await asyncio.gather(*(c.close() for _, c in channels.values()))


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    config = routing.load_engine_config(_CONFIG_DIR.value)
    collector = FleetCollector()
    registry = prometheus_client.CollectorRegistry()
    registry.register(collector)
    prometheus_client.start_http_server(_PORT.value, registry=registry)
    logging.info('serving shard metrics on :%d', _PORT.value)
    asyncio.run(_run(config, collector))


if __name__ == '__main__':
    app.run(main)
