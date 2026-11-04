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

"""The gateway end to end over exact fake shards: request in, price out."""

import asyncio
import math
import os
import shutil

from google.protobuf import text_format
import grpc
import numpy as np
import pytest
import torch

from data import routing
from models.price_transformer import model as model_lib
from serving import predict
from serving import service
from serving.tests import fake_engine
from vectra import common_pb2
from vectra import config_pb2
from vectra import pricing_pb2
from vectra import pricing_pb2_grpc

MODEL_CONFIG = model_lib.ModelConfig(
    d_model=32, num_heads=2, ffn_dim=64, num_candidates=16
)
# A point on San Francisco Bay, between San Francisco (cell 9q8) and Oakland
# (9q9): in the test topology the two sides belong to different shards.
BAY = (37.80, -122.36)
FUTURE_DAY = 30_000  # 2052: every synthetic sale is earlier.


def _edit(directory: str, name: str, message, edit) -> None:
    path = os.path.join(directory, f'{name}.textproto')
    with open(path, encoding='utf-8') as f:
        text_format.Parse(f.read(), message)
    edit(message)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text_format.MessageToString(message))


@pytest.fixture(scope='module', name='config_dir')
def _config_dir(tmp_path_factory) -> str:
    """engine/configs with Oakland's 3-character cell moved to shard 3.

    The per-shard deadline is also relaxed from the production 5 ms: the
    fake shards are brute-force Python, not the C++ engine.
    """
    directory = str(tmp_path_factory.mktemp('configs'))
    for name in os.listdir('engine/configs'):
        shutil.copy(os.path.join('engine/configs', name), directory)
    _edit(
        directory,
        'shard_map',
        config_pb2.ShardMap(),
        lambda m: m.assignments.add(prefix='9q9', shard_id=3),
    )
    _edit(
        directory,
        'cluster_topology',
        config_pb2.ClusterTopology(),
        lambda m: setattr(m.client, 'timeout_ms', 5000),
    )
    return directory


@pytest.fixture(scope='module', name='corpus')
def _corpus() -> fake_engine.Corpus:
    return fake_engine.Corpus.generate(8000, seed=3)


def _model() -> model_lib.PriceTransformer:
    torch.manual_seed(0)
    return model_lib.PriceTransformer(MODEL_CONFIG).eval()


def _request(lat=BAY[0], lng=BAY[1], **fields):
    attributes = common_pb2.PropertyAttributes(
        living_area_sqft=1850.0,
        lot_area_sqft=4200.0,
        bedrooms=3,
        bathrooms=2.0,
        year_built=1962,
    )
    fields.setdefault('valuation_epoch_day', FUTURE_DAY)
    fields.setdefault('radius_km', 5.0)
    return pricing_pb2.PredictPriceRequest(
        location=common_pb2.LatLng(latitude=lat, longitude=lng),
        attributes=attributes,
        **fields,
    )


def _run(config_dir, corpus, scenario):
    """Runs `scenario(stub, gateway, cluster, model)` against a gateway."""

    async def main():
        engine = routing.load_engine_config(config_dir)
        cluster = await fake_engine.start_cluster(corpus, engine)
        model = _model()
        config = service.ServiceConfig(
            engine_config_dir=config_dir,
            compiled=False,
            device='cpu',
            batch_window_ms=5.0,
            max_batch_size=8,
        )
        gateway = await service.Gateway.create(
            config,
            cluster.endpoints,
            model=(lambda inputs: model(*inputs), MODEL_CONFIG, 8),
        )
        server, port = await gateway.serve('127.0.0.1:0')
        channel = grpc.aio.insecure_channel(f'127.0.0.1:{port}')
        try:
            return await scenario(
                pricing_pb2_grpc.PricingServiceStub(channel),
                gateway,
                cluster,
                model,
            )
        finally:
            await channel.close()
            await server.stop(grace=None)
            await gateway.close()
            await cluster.stop()

    return asyncio.run(main())


def test_prices_a_property_from_comparables_on_both_shards(config_dir, corpus):
    async def scenario(stub, gateway, cluster, model):
        del gateway  # Unused.
        response = await stub.PredictPrice(_request())
        served = {
            shard_id: sum(s.peers.values())
            for shard_id, s in cluster.shards.items()
        }
        return response, served, model

    response, served, model = _run(config_dir, corpus, scenario)
    assert served[1] >= 1 and served[3] >= 1  # The fan-out crossed shards.
    # The comparables are the exact top 16 over the whole corpus.
    query = predict.FeatureBuilder(16).embed(*BAY, _request().attributes)
    rows, _ = corpus.exact_search(
        np.arange(len(corpus.cell6)), query, *BAY, 5.0, 16
    )
    assert list(response.comparable_ids) == list(
        corpus.records.property_id[rows]
    )
    # Both shards hold properties inside the radius, so the global answer
    # above needed the merge across shards.
    everything, _ = corpus.exact_search(
        np.arange(len(corpus.cell6)), query, *BAY, 5.0, len(corpus.cell6)
    )
    owners = fake_engine.shard_rows(
        corpus, routing.load_engine_config(config_dir)
    )
    assert {s for s, r in owners.items() if np.isin(everything, r).any()} == {
        1,
        3,
    }
    # And the price is what the model computes from them.
    candidates = [corpus.candidate(int(r), query, 0, 0.0, True) for r in rows]
    subject = predict.Subject(
        BAY[0], BAY[1], _request().attributes, FUTURE_DAY, 5.0, query
    )
    inputs = predict.FeatureBuilder(16).build([subject], [candidates])
    with torch.no_grad():
        want = float(model(*inputs)[0])
    assert response.log_price == pytest.approx(want, abs=1e-5)
    assert response.predicted_price_usd == pytest.approx(
        math.exp(want), rel=1e-6
    )
    assert 1 <= response.candidates_used <= 16
    latency = response.latency
    assert latency.total_us >= latency.retrieve_us > 0
    assert latency.inference_us > 0


def test_concurrent_requests_are_batched(config_dir, corpus):
    async def scenario(stub, gateway, cluster, model):
        del cluster, model  # Unused.
        offsets = np.linspace(-0.02, 0.02, 12)
        responses = await asyncio.gather(
            *(
                stub.PredictPrice(_request(BAY[0] + d, BAY[1] - d))
                for d in offsets
            )
        )
        registry = gateway.metrics.registry
        return (
            responses,
            registry.get_sample_value('vectra_gateway_batch_size_sum'),
            registry.get_sample_value('vectra_gateway_batch_size_count'),
        )

    responses, total, count = _run(config_dir, corpus, scenario)
    assert len(responses) == 12
    assert all(math.isfinite(r.log_price) for r in responses)
    # 12 requests, each recording the size of the batch it ran in.
    assert count == 12 and total > count


def test_rejects_invalid_requests(config_dir, corpus):
    async def scenario(stub, gateway, cluster, model):
        del gateway, cluster, model  # Unused.
        codes = {}
        bad = {
            'too_far': _request(radius_km=400.0),
            'too_many': _request(num_candidates=65),
            'no_location': pricing_pb2.PredictPriceRequest(
                attributes=_request().attributes
            ),
            'no_attributes': pricing_pb2.PredictPriceRequest(
                location=_request().location
            ),
            'nothing_sold_yet': _request(valuation_epoch_day=1),
        }
        for name, request in bad.items():
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.PredictPrice(request)
            codes[name] = error.value.code()
        return codes

    codes = _run(config_dir, corpus, scenario)
    assert codes == {
        'too_far': grpc.StatusCode.INVALID_ARGUMENT,
        'too_many': grpc.StatusCode.INVALID_ARGUMENT,
        'no_location': grpc.StatusCode.INVALID_ARGUMENT,
        'no_attributes': grpc.StatusCode.INVALID_ARGUMENT,
        'nothing_sold_yet': grpc.StatusCode.NOT_FOUND,
    }


def test_only_earlier_sales_are_comparables(config_dir, corpus):
    day = int(np.median(corpus.records.last_sale_epoch_day))

    async def scenario(stub, gateway, cluster, model):
        del gateway, cluster, model  # Unused.
        return await stub.PredictPrice(_request(valuation_epoch_day=day))

    response = _run(config_dir, corpus, scenario)
    sale_day = dict(
        zip(
            corpus.records.property_id,
            corpus.records.last_sale_epoch_day,
            strict=True,
        )
    )
    # The shards filter during the search: all 16 are earlier sales, and
    # they are the 16 nearest among the earlier sales.
    query = predict.FeatureBuilder(16).embed(*BAY, _request().attributes)
    rows, _ = corpus.exact_search(
        np.arange(len(corpus.cell6)),
        query,
        *BAY,
        5.0,
        16,
        sold_before_epoch_day=day,
    )
    assert list(response.comparable_ids) == list(
        corpus.records.property_id[rows]
    )
    assert len(response.comparable_ids) == 16
    assert all(sale_day[i] < day for i in response.comparable_ids)
