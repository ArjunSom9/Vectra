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

"""The shard exporter turns HealthService reports into Prometheus metrics."""

import asyncio

import grpc
import prometheus_client

from serving import engine_exporter
from vectra import health_pb2
from vectra import health_pb2_grpc


class _Health(health_pb2_grpc.HealthServiceServicer):
    # grpc.aio servicers implement the generated methods as coroutines.
    async def Report(self, request, context):  # pylint: disable=invalid-name,invalid-overridden-method
        del request, context  # Unused.
        return health_pb2.ReportResponse(
            shard_id=2,
            index=health_pb2.IndexStats(
                num_vectors=1200, capacity=4000, index_bytes=987654
            ),
            memory=health_pb2.MemoryStats(locked=True),
            workers=[
                health_pb2.WorkerStats(
                    worker=0,
                    core=4,
                    rpcs_completed=50,
                    call_pool_high_water=12,
                    call_pool_allocations=0,
                )
            ],
        )


def test_reports_become_metrics_and_failures_mark_the_shard_down():
    async def scenario():
        server = grpc.aio.server()
        health_pb2_grpc.add_HealthServiceServicer_to_server(_Health(), server)
        port = server.add_insecure_port('127.0.0.1:0')
        await server.start()
        live = f'127.0.0.1:{port}'
        dead = '127.0.0.1:1'
        channels = {
            live: (2, grpc.aio.insecure_channel(live)),
            dead: (3, grpc.aio.insecure_channel(dead)),
        }
        collector = engine_exporter.FleetCollector()
        try:
            await engine_exporter.poll_once(channels, collector, 2.0)
        finally:
            await asyncio.gather(*(c.close() for _, c in channels.values()))
            await server.stop(grace=None)
        return collector, live, dead

    collector, live, dead = asyncio.run(scenario())
    registry = prometheus_client.CollectorRegistry()
    registry.register(collector)

    def value(name, **labels):
        return registry.get_sample_value(name, labels)

    assert value('vectra_shard_up', shard='2', endpoint=live) == 1.0
    assert value('vectra_shard_up', shard='3', endpoint=dead) == 0.0
    assert value('vectra_shard_vectors', shard='2', endpoint=live) == 1200
    assert value('vectra_shard_memory_locked', shard='2', endpoint=live) == 1
    worker = {'shard': '2', 'endpoint': live, 'worker': '0', 'core': '4'}
    assert value('vectra_shard_rpcs_completed_total', **worker) == 50
    assert value('vectra_shard_call_pool_allocations', **worker) == 0
    assert value('vectra_shard_vectors', shard='3', endpoint=dead) is None
