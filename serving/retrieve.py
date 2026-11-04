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

"""Retrieval: the gateway's asynchronous client for the shard fleet.

The Python counterpart of engine/client/geo_router.cc. Routing comes from
data.routing, which reads the same engine/configs files as the C++ engine,
so both agree on which shards a query must reach. Each shard is reached
through a pool of pre-warmed channels used round-robin. A single channel is
a single HTTP/2 connection, capped by the peer's
SETTINGS_MAX_CONCURRENT_STREAMS [20], so a pool keeps per-connection stream
counts far below the limit [27].
"""

import asyncio
import dataclasses
import itertools
import time

import grpc
import numpy as np

from data import routing
from vectra import common_pb2
from vectra import search_pb2
from vectra import search_pb2_grpc

# The same channel argument the C++ pool sets: a distinct value per channel,
# together with a channel-local subchannel pool, stops gRPC from collapsing
# the pool onto one shared connection [27].
CHANNEL_INDEX_ARG = 'vectra.channel_index'
_LOCAL_SUBCHANNEL_POOL_ARG = 'grpc.use_local_subchannel_pool'


def channel_options(
    index: int,
    keepalive_time_ms: int = 10_000,
    max_receive_bytes: int = 16 << 20,
) -> list[tuple]:
    """Channel arguments for the `index`-th channel of a pool."""
    return [
        (CHANNEL_INDEX_ARG, index),
        (_LOCAL_SUBCHANNEL_POOL_ARG, 1),
        ('grpc.keepalive_time_ms', keepalive_time_ms),
        ('grpc.max_receive_message_length', max_receive_bytes),
    ]


class ChannelPool:
    """Pre-warmed channels to one shard, chosen round-robin.

    Channel i connects to replica i mod len(targets). Create inside a
    running event loop.
    """

    def __init__(
        self, targets: list[str], size: int, keepalive_time_ms: int = 10_000
    ):
        """Opens the channels (connections are established lazily).

        Args:
          targets: host:port of every replica serving the shard.
          size: Channels in the pool.
          keepalive_time_ms: HTTP/2 keepalive ping interval.

        Raises:
          ValueError: Without targets or with a non-positive size.
        """
        if not targets or size < 1:
            raise ValueError('a pool needs targets and a positive size')
        self.targets = list(targets)
        # The replica each channel connects to.
        self.channel_targets = [
            self.targets[i % len(self.targets)] for i in range(size)
        ]
        self._channels = [
            grpc.aio.insecure_channel(
                target, options=channel_options(i, keepalive_time_ms)
            )
            for i, target in enumerate(self.channel_targets)
        ]
        self._stubs = [
            search_pb2_grpc.SearchServiceStub(channel)
            for channel in self._channels
        ]
        self._order = itertools.cycle(range(size))

    def __len__(self) -> int:
        return len(self._channels)

    def next_stub(self) -> search_pb2_grpc.SearchServiceStub:
        return self._stubs[next(self._order)]

    async def wait_ready(self, timeout_s: float) -> None:
        """Connects every channel (pre-warming).

        Raises:
          TimeoutError: If some channel is not ready within `timeout_s`.
        """
        await asyncio.wait_for(
            asyncio.gather(*(c.channel_ready() for c in self._channels)),
            timeout_s,
        )

    async def close(self) -> None:
        await asyncio.gather(*(c.close() for c in self._channels))


@dataclasses.dataclass
class Retrieval:
    """Merged results of one fanned-out search.

    Attributes:
      candidates: Nearest first by squared L2 embedding distance.
      shard_ids: Shards the query reached; the home shard first.
      radius_km: The radius searched (the policy default if 0 was asked).
      latency_us: Wall time of the fan-out.
    """

    candidates: list[common_pb2.Candidate]
    shard_ids: list[int]
    radius_km: float
    latency_us: int


class RetrievalError(RuntimeError):
    """A shard failed or the query cannot be routed.

    Attributes:
      code: The grpc.StatusCode to report to the caller.
    """

    def __init__(self, code: grpc.StatusCode, message: str):
        super().__init__(message)
        self.code = code


class Router:
    """Routes searches to the shards that own the cells a query touches."""

    def __init__(
        self,
        policy: routing.FanoutPolicy,
        pools: dict[int, ChannelPool],
        timeout_s: float,
    ):
        self._policy = policy
        self._pools = pools
        self._timeout_s = timeout_s

    @classmethod
    def create(
        cls,
        config: routing.EngineConfig,
        endpoints: dict[int, list[str]] | None = None,
    ) -> 'Router':
        """Builds a router from engine/configs (inside an event loop).

        Args:
          config: The engine configuration.
          endpoints: Overrides shard_map endpoints, keyed by shard id.

        Raises:
          ValueError: If a shard has no endpoint.
        """
        client = config.topology.client
        size = client.channel_pool_size or 64
        keepalive = client.keepalive_time_ms or 10_000
        timeout_s = (client.timeout_ms or 5) / 1000.0
        table = routing.RoutingTable.from_config(config)
        pools = {}
        for shard in config.shard_map.shards:
            targets = list((endpoints or {}).get(shard.id, shard.endpoints))
            if not targets:
                raise ValueError(f'shard {shard.id} has no endpoints')
            pools[shard.id] = ChannelPool(targets, size, keepalive)
        return cls(
            routing.FanoutPolicy.from_config(config, table), pools, timeout_s
        )

    @property
    def pools(self) -> dict[int, ChannelPool]:
        return self._pools

    async def wait_ready(self, timeout_s: float = 10.0) -> None:
        await asyncio.gather(
            *(pool.wait_ready(timeout_s) for pool in self._pools.values())
        )

    async def close(self) -> None:
        await asyncio.gather(*(pool.close() for pool in self._pools.values()))

    async def search(
        self,
        embedding: np.ndarray,
        latitude: float,
        longitude: float,
        radius_km: float = 0.0,
        top_k: int = 64,
        include_embeddings: bool = True,
        exact: bool = False,
        sold_before_epoch_day: int = 0,
        ef_search: int = 0,
    ) -> Retrieval:
        """Searches every shard the query's disk touches; merges by distance.

        With `sold_before_epoch_day` set, only earlier sales are returned
        (the shards filter during the search, so all top_k qualify).

        Raises:
          RetrievalError: INVALID_ARGUMENT for an unroutable query, or the
            status of the first shard that failed (the fan-out is all or
            nothing, as in the C++ router, so a boundary comparable is never
            silently missing).
        """
        start = time.perf_counter()
        try:
            plan = self._policy.plan(latitude, longitude, radius_km)
        except ValueError as e:
            raise RetrievalError(
                grpc.StatusCode.INVALID_ARGUMENT, str(e)
            ) from e
        if not plan.shard_ids:
            return Retrieval([], [], plan.radius_km, 0)
        request = search_pb2.SearchRequest(
            query_embedding=np.asarray(embedding, np.float32).tolist(),
            location=common_pb2.LatLng(latitude=latitude, longitude=longitude),
            radius_km=plan.radius_km,
            top_k=top_k,
            include_embeddings=include_embeddings,
            exact=exact,
            sold_before_epoch_day=sold_before_epoch_day,
            ef_search=ef_search,
        )
        calls = []
        for shard_id in plan.shard_ids:
            pool = self._pools.get(shard_id)
            if pool is None:
                raise RetrievalError(
                    grpc.StatusCode.UNAVAILABLE,
                    f'no channels for shard {shard_id}',
                )
            calls.append(
                pool.next_stub().Search(request, timeout=self._timeout_s)
            )
        try:
            responses = await asyncio.gather(*calls)
        except grpc.aio.AioRpcError as e:
            for call in calls:
                call.cancel()
            raise RetrievalError(e.code(), e.details() or str(e)) from e
        merged = [c for r in responses for c in r.candidates]
        merged.sort(key=lambda c: (c.squared_l2, c.property_id))
        return Retrieval(
            merged[:top_k],
            plan.shard_ids,
            plan.radius_km,
            int((time.perf_counter() - start) * 1e6),
        )
