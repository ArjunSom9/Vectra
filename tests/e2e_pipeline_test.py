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

"""The whole system on a small corpus: coordinate in, price out.

corpus generation -> validation -> GeoHash assignment -> shard split ->
training -> deployment compile (bundle) -> shard servers -> gateway ->
PredictPrice.

The shard tier is the real C++ engine (indexer + vectra_shard) when it has
been built (`make build-engine`, or VECTRA_ENGINE_BUILD pointing at the CMake
build directory); otherwise it is the exact Python shard of
serving/tests/fake_engine.py, which implements the same SearchService
contract.
"""

import asyncio
import os
import shutil
import socket
import subprocess

from google.protobuf import text_format
import grpc
import numpy as np
import pytest
import torch

from data import embedding_model
from data import routing
from data import synthetic
from data.pipelines import assign_geohash
from data.pipelines import corpus_io
from data.pipelines import generate_embeddings
from data.pipelines import split_shards
from data.pipelines import validate_corpus
from models.export import compile_entry
from models.training import checkpoint
from models.training import dataset
from models.training import train
from serving import service
from serving.tests import fake_engine
from vectra import common_pb2
from vectra import config_pb2
from vectra import pricing_pb2
from vectra import pricing_pb2_grpc

_BUILD = os.environ.get('VECTRA_ENGINE_BUILD', 'build/engine')
_INDEXER = os.path.join(_BUILD, 'apps', 'indexer', 'indexer')
_SHARD = os.path.join(_BUILD, 'vectra_shard')
ENGINE_BUILT = os.access(_INDEXER, os.X_OK) and os.access(_SHARD, os.X_OK)

NUM_PROPERTIES = 8000
NUM_QUERIES = 60
NUM_CANDIDATES = 16

pytestmark = pytest.mark.slow


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope='module', name='workspace')
def _workspace(tmp_path_factory) -> dict[str, str]:
    """Corpus, shards, checkpoint and bundle, built by the real pipelines."""
    root = tmp_path_factory.mktemp('e2e')
    paths = {
        name: str(root / name)
        for name in ('corpus', 'shards', 'checkpoint', 'bundle', 'configs')
    }
    everything = synthetic.generate(NUM_PROPERTIES + NUM_QUERIES, seed=17)
    order = np.random.default_rng(18).permutation(len(everything))
    corpus = everything.take(np.sort(order[:NUM_PROPERTIES]))
    queries = everything.take(np.sort(order[NUM_PROPERTIES:]))
    embedder = embedding_model.PropertyEmbedder()
    columns = generate_embeddings.write_corpus(
        paths['corpus'], corpus, embedder, chunk=4096
    )
    corpus_io.write_column(
        paths['corpus'],
        corpus_io.GEOHASH_COLUMN,
        assign_geohash.assign(corpus.latitude, corpus.longitude),
    )
    corpus_io.write_column(paths['corpus'], 'metro_index', corpus.metro_index)
    corpus_io.write_manifest(
        paths['corpus'], columns + [corpus_io.GEOHASH_COLUMN, 'metro_index']
    )
    query_dir = os.path.join(paths['corpus'], 'queries')
    generate_embeddings.write_corpus(
        query_dir, queries, embedder, 4096, prefix='query_'
    )

    # Engine configuration sized for a test machine: a relaxed deadline and
    # shard capacity for this corpus instead of the production 4M rows.
    shutil.copytree('engine/configs', paths['configs'])
    topology_path = os.path.join(paths['configs'], 'cluster_topology.textproto')
    with open(topology_path, encoding='utf-8') as f:
        topology = text_format.Parse(f.read(), config_pb2.ClusterTopology())
    topology.client.timeout_ms = 5000
    with open(topology_path, 'w', encoding='utf-8') as f:
        f.write(text_format.MessageToString(topology))
    params_path = os.path.join(paths['configs'], 'hnsw_params.textproto')
    with open(params_path, encoding='utf-8') as f:
        params = text_format.Parse(f.read(), config_pb2.IndexParams())
    params.max_elements = 2 * NUM_PROPERTIES
    params.pq_training_samples = NUM_PROPERTIES
    params.build_threads = 2
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write(text_format.MessageToString(params))

    engine = routing.load_engine_config(paths['configs'])
    split_shards.write_shards(
        paths['corpus'],
        paths['shards'],
        routing.RoutingTable.from_config(engine),
    )

    data_config = dataset.DatasetConfig(
        corpus_dir=paths['corpus'],
        num_candidates=NUM_CANDIDATES,
        max_subjects=2000,
    )
    examples = dataset.build_examples(data_config)
    model_config, config = train.load_configs(
        'models/training/configs/smoke.yaml'
    )
    train_rows, validation_rows = dataset.split_by_day(
        examples, data_config.split_day
    )
    model, _ = train.train(
        model_config, config, examples, train_rows, validation_rows
    )
    checkpoint.save(paths['checkpoint'], model)
    compile_entry.export_bundle(
        paths['checkpoint'],
        paths['bundle'],
        compile_entry.ServingSpec('cpu', batch_size=8, block_dtype='float32'),
        numerics_batches=1,
    )
    return paths


def test_corpus_is_valid_and_fully_sharded(workspace):
    assert not validate_corpus.validate(
        workspace['corpus'], 'data/schemas', verify_checksums=True
    )
    total = 0
    for name in os.listdir(workspace['shards']):
        directory = os.path.join(workspace['shards'], name)
        total += len(corpus_io.read_column(directory, 'property_id'))
    assert total == NUM_PROPERTIES


async def _start_engine(workspace):
    """Shard servers: the C++ engine when built, else exact Python shards."""
    engine = routing.load_engine_config(workspace['configs'])
    if not ENGINE_BUILT:
        corpus = fake_engine.Corpus.from_dir(workspace['corpus'])
        cluster = await fake_engine.start_cluster(corpus, engine)
        return cluster.endpoints, cluster.stop
    processes, endpoints = [], {}
    configs = workspace['configs']
    for shard in engine.shard_map.shards:
        shard_dir = os.path.join(workspace['shards'], f'shard-{shard.id}')
        if not os.path.isdir(shard_dir):
            continue
        index = os.path.join(workspace['shards'], f'shard-{shard.id}.vidx')
        subprocess.run(
            [
                _INDEXER,
                f'--config_dir={configs}',
                f'--shard_dir={shard_dir}',
                f'--out={index}',
            ],
            check=True,
        )
        port = _free_port()
        processes.append(
            subprocess.Popen(
                [
                    _SHARD,
                    f'--config_dir={configs}',
                    f'--index={index}',
                    f'--shard_id={shard.id}',
                    f'--listen=127.0.0.1:{port}',
                    '--lock_memory=false',
                ]
            )
        )
        endpoints[shard.id] = [f'127.0.0.1:{port}']

    async def stop():
        for process in processes:
            process.terminate()
            process.wait(timeout=10)

    return endpoints, stop


def _queries(workspace):
    directory = os.path.join(workspace['corpus'], 'queries')
    return {
        name: corpus_io.read_column(directory, 'query_' + name)
        for name in corpus_io.RECORD_COLUMNS
    }


def test_coordinate_in_price_out(workspace):
    queries = _queries(workspace)
    corpus = fake_engine.Corpus.from_dir(workspace['corpus'])
    model = checkpoint.load(workspace['checkpoint'])

    async def scenario():
        endpoints, stop_engine = await _start_engine(workspace)
        config = service.ServiceConfig(
            engine_config_dir=workspace['configs'],
            compiled=True,
            bundle_dir=workspace['bundle'],
            device='cpu',
            batch_window_ms=2.0,
            connect_timeout_s=30.0,
        )
        gateway = await service.Gateway.create(config, endpoints)
        server, port = await gateway.serve('127.0.0.1:0')
        channel = grpc.aio.insecure_channel(f'127.0.0.1:{port}')
        stub = pricing_pb2_grpc.PricingServiceStub(channel)
        try:
            calls = []
            for i in range(NUM_QUERIES):
                calls.append(
                    stub.PredictPrice(
                        pricing_pb2.PredictPriceRequest(
                            location=common_pb2.LatLng(
                                latitude=float(queries['latitude'][i]),
                                longitude=float(queries['longitude'][i]),
                            ),
                            attributes=common_pb2.PropertyAttributes(
                                living_area_sqft=float(
                                    queries['living_area_sqft'][i]
                                ),
                                lot_area_sqft=float(
                                    queries['lot_area_sqft'][i]
                                ),
                                bedrooms=int(queries['bedrooms'][i]),
                                bathrooms=float(queries['bathrooms'][i]),
                                year_built=int(queries['year_built'][i]),
                            ),
                            radius_km=5.0,
                            valuation_epoch_day=int(
                                queries['last_sale_epoch_day'][i]
                            ),
                        )
                    )
                )
            return await asyncio.gather(*calls, return_exceptions=True)
        finally:
            await channel.close()
            await server.stop(grace=None)
            await gateway.close()
            await stop_engine()

    responses = asyncio.run(scenario())
    served = [
        (i, r)
        for i, r in enumerate(responses)
        if isinstance(r, pricing_pb2.PredictPriceResponse)
    ]
    failures = [
        r
        for r in responses
        if not isinstance(r, pricing_pb2.PredictPriceResponse)
    ]
    # Only rural queries with no earlier sale within 5 km may go unpriced.
    assert all(r.code() == grpc.StatusCode.NOT_FOUND for r in failures)
    assert len(served) >= 0.8 * NUM_QUERIES

    row_of = {int(p): row for row, p in enumerate(corpus.records.property_id)}
    errors = []
    for i, response in served:
        assert np.isfinite(response.log_price)
        assert response.latency.total_us > 0
        # Every comparable sold before the valuation day ...
        rows = [row_of[p] for p in response.comparable_ids]
        assert np.all(
            corpus.records.last_sale_epoch_day[rows]
            < queries['last_sale_epoch_day'][i]
        )
        # ... and the compiled model's price matches the eager model's on
        # those same comparables (the design's rtol 1e-3 numerics gate).
        builder = service.predict.FeatureBuilder(NUM_CANDIDATES)
        attributes = common_pb2.PropertyAttributes(
            living_area_sqft=float(queries['living_area_sqft'][i]),
            lot_area_sqft=float(queries['lot_area_sqft'][i]),
            bedrooms=int(queries['bedrooms'][i]),
            bathrooms=float(queries['bathrooms'][i]),
            year_built=int(queries['year_built'][i]),
        )
        lat = float(queries['latitude'][i])
        lng = float(queries['longitude'][i])
        query = builder.embed(lat, lng, attributes)
        subject = service.predict.Subject(
            lat,
            lng,
            attributes,
            int(queries['last_sale_epoch_day'][i]),
            5.0,
            query,
        )
        candidates = [
            corpus.candidate(row, query, 0, 0.0, True) for row in rows
        ]
        with torch.no_grad():
            eager = float(model(*builder.build([subject], [candidates]))[0])
        assert response.log_price == pytest.approx(eager, rel=1e-3)
        errors.append(
            abs(
                response.predicted_price_usd - queries['last_sale_price_usd'][i]
            )
            / queries['last_sale_price_usd'][i]
        )
    # A two-epoch smoke model is not accurate, but it must be sane.
    assert float(np.median(errors)) < 0.5
