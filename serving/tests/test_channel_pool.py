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

"""The channel pool opens one HTTP/2 connection per channel."""

import asyncio

import grpc
import numpy as np
import pytest

from data import routing
from serving import retrieve
from serving.tests import fake_engine
from vectra import common_pb2
from vectra import search_pb2

_CONFIG = routing.load_engine_config('engine/configs')


async def _one_shard():
    corpus = fake_engine.Corpus.generate(500, seed=1)
    cluster = await fake_engine.start_cluster(corpus, _CONFIG)
    shard_id = next(i for i, s in cluster.shards.items() if len(s._rows))  # pylint: disable=protected-access
    return corpus, cluster, shard_id


def _request(corpus) -> search_pb2.SearchRequest:
    r = corpus.records
    return search_pb2.SearchRequest(
        query_embedding=corpus.embeddings[0].tolist(),
        location=common_pb2.LatLng(
            latitude=r.latitude[0], longitude=r.longitude[0]
        ),
        radius_km=2.0,
        top_k=8,
    )


def test_each_channel_is_its_own_connection():
    async def scenario():
        corpus, cluster, shard_id = await _one_shard()
        pool = retrieve.ChannelPool(cluster.endpoints[shard_id], size=4)
        try:
            await pool.wait_ready(10.0)
            for _ in range(12):
                await pool.next_stub().Search(_request(corpus), timeout=5.0)
            return cluster.shards[shard_id].peers
        finally:
            await pool.close()
            await cluster.stop()

    peers = asyncio.run(scenario())
    # Four distinct client ports, used round-robin: three calls each.
    assert len(peers) == 4
    assert sorted(peers.values()) == [3, 3, 3, 3]


def test_identical_channels_would_share_a_connection():
    """Why the pool sets a distinct argument per channel [27]."""

    async def scenario():
        corpus, cluster, shard_id = await _one_shard()
        target = cluster.endpoints[shard_id][0]
        channels = [grpc.aio.insecure_channel(target) for _ in range(4)]
        try:
            for channel in channels:
                await channel.channel_ready()
                stub = retrieve.search_pb2_grpc.SearchServiceStub(channel)
                await stub.Search(_request(corpus), timeout=5.0)
            return cluster.shards[shard_id].peers
        finally:
            await asyncio.gather(*(c.close() for c in channels))
            await cluster.stop()

    assert len(asyncio.run(scenario())) == 1


def test_channels_spread_over_replicas():
    async def scenario():
        pool = retrieve.ChannelPool(['127.0.0.1:1', '127.0.0.1:2'], size=5)
        try:
            return pool.channel_targets
        finally:
            await pool.close()

    targets = asyncio.run(scenario())
    assert targets.count('127.0.0.1:1') == 3
    assert targets.count('127.0.0.1:2') == 2


def test_channel_options_match_the_engine_client():
    options = dict(retrieve.channel_options(7, keepalive_time_ms=1234))
    assert options[retrieve.CHANNEL_INDEX_ARG] == 7
    assert options['grpc.use_local_subchannel_pool'] == 1
    assert options['grpc.keepalive_time_ms'] == 1234


def test_pool_validates_its_arguments():
    with pytest.raises(ValueError):
        retrieve.ChannelPool([], size=4)
    with pytest.raises(ValueError):
        retrieve.ChannelPool(['127.0.0.1:1'], size=0)


def test_router_merges_shards_by_embedding_distance():
    async def scenario():
        corpus = fake_engine.Corpus.generate(4000, seed=2)
        cluster = await fake_engine.start_cluster(corpus, _CONFIG)
        router = retrieve.Router.create(_CONFIG, cluster.endpoints)
        try:
            await router.wait_ready()
            r = corpus.records
            row = int(np.argmax(r.metro_index == 0))
            found = await router.search(
                corpus.embeddings[row],
                float(r.latitude[row]),
                float(r.longitude[row]),
                3.0,
                16,
            )
            want, _ = corpus.exact_search(
                np.arange(len(r.property_id)),
                corpus.embeddings[row],
                float(r.latitude[row]),
                float(r.longitude[row]),
                3.0,
                16,
            )
            return found, r.property_id[want]
        finally:
            await router.close()
            await cluster.stop()

    found, want = asyncio.run(scenario())
    assert [c.property_id for c in found.candidates] == list(want)
    distances = [c.squared_l2 for c in found.candidates]
    assert distances == sorted(distances)
    assert found.radius_km == 3.0


def test_router_rejects_unroutable_queries():
    async def scenario():
        router = retrieve.Router.create(
            _CONFIG, {s.id: ['127.0.0.1:1'] for s in _CONFIG.shard_map.shards}
        )
        try:
            with pytest.raises(retrieve.RetrievalError) as error:
                await router.search(np.zeros(128), 37.77, -122.42, 400.0, 8)
            return error.value.code
        finally:
            await router.close()

    assert asyncio.run(scenario()) == grpc.StatusCode.INVALID_ARGUMENT
