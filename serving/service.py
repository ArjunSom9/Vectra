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

"""The inference gateway: coordinate and attributes in, price out.

  python -m serving.service --config=serving/configs/service.yaml

PricingService.PredictPrice embeds the subject property, retrieves its
comparables from the shards that own the surrounding GeoHash cells, batches
concurrent requests for the GPU, and runs the compiled price model. At
start-up the compiled bundle is recompiled and re-verified
(models/export/compile_entry.py) before the port opens, so the gateway
never serves the eager fallback by accident.
"""

import asyncio
from collections.abc import Sequence
import dataclasses
import math
import signal
import time

from absl import app
from absl import flags
from absl import logging
import grpc
import prometheus_client
import yaml

from data import routing
from models.export import compile_entry
from models.training import checkpoint
from serving import predict
from serving import retrieve
from vectra import pricing_pb2
from vectra import pricing_pb2_grpc

_CONFIG = flags.DEFINE_string(
    'config',
    'serving/configs/service.yaml',
    'Gateway configuration.',
    allow_override=True,
)
_LISTEN = flags.DEFINE_string(
    'listen', None, 'Overrides `listen`.', allow_override=True
)
_BUNDLE_DIR = flags.DEFINE_string(
    'bundle_dir', None, 'Overrides model.bundle_dir.', allow_override=True
)
_DEVICE = flags.DEFINE_string(
    'device', None, 'Overrides model.device.', allow_override=True
)


@dataclasses.dataclass(frozen=True)
class ServiceConfig:
    """serving/configs/service.yaml."""

    listen: str = '[::]:7000'
    metrics_port: int = 0
    engine_config_dir: str = 'engine/configs'
    connect_timeout_s: float = 10.0
    compiled: bool = True
    bundle_dir: str = 'dist/compiled_model'
    checkpoint_dir: str = 'checkpoints/price_transformer_v1'
    device: str = 'cuda'
    batch_window_ms: float = 2.0
    # Eager mode only; a compiled bundle fixes its own batch size.
    max_batch_size: int = 64

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> 'ServiceConfig':
        with open(path, encoding='utf-8') as f:
            values = yaml.safe_load(f)
        engine, model = values.get('engine', {}), values.get('model', {})
        fields = {
            'listen': values.get('listen'),
            'metrics_port': values.get('metrics_port'),
            'engine_config_dir': engine.get('config_dir'),
            'connect_timeout_s': engine.get('connect_timeout_s'),
            'compiled': model.get('compiled'),
            'bundle_dir': model.get('bundle_dir'),
            'checkpoint_dir': model.get('checkpoint_dir'),
            'device': model.get('device'),
            'batch_window_ms': model.get('batch_window_ms'),
            'max_batch_size': model.get('max_batch_size'),
        }
        fields.update(overrides)
        return cls(**{k: v for k, v in fields.items() if v is not None})


class _Metrics:
    """Prometheus instruments, on a registry owned by one gateway."""

    def __init__(self):
        self.registry = prometheus_client.CollectorRegistry()
        self.stage_seconds = prometheus_client.Histogram(
            'vectra_gateway_stage_seconds',
            'Wall time per request stage.',
            ['stage'],
            registry=self.registry,
            buckets=(
                1e-4,
                2.5e-4,
                5e-4,
                1e-3,
                2.5e-3,
                5e-3,
                1e-2,
                2.5e-2,
                5e-2,
                0.1,
                0.25,
            ),
        )
        self.requests = prometheus_client.Counter(
            'vectra_gateway_requests',
            'PredictPrice calls by status.',
            ['code'],
            registry=self.registry,
        )
        self.batch_size = prometheus_client.Histogram(
            'vectra_gateway_batch_size',
            'Requests per model call.',
            registry=self.registry,
            buckets=(1, 2, 4, 8, 16, 32, 64, 128),
        )


def load_model(config: ServiceConfig):
    """(predict_fn, ModelConfig, batch size) for the configured model."""
    if config.compiled:
        serving = compile_entry.load_bundle(config.bundle_dir, config.device)
        return serving.predict, serving.config, serving.spec.batch_size
    logging.warning(
        'serving the EAGER model from %s (model.compiled is false)',
        config.checkpoint_dir,
    )
    model = checkpoint.load(config.checkpoint_dir, config.device)

    def eager(inputs):
        return model(*inputs.to(config.device))

    return eager, model.config, config.max_batch_size


class PricingServicer(pricing_pb2_grpc.PricingServiceServicer):
    """PricingService.PredictPrice."""

    def __init__(
        self,
        router: retrieve.Router,
        builder: predict.FeatureBuilder,
        batcher: predict.Batcher,
        metrics: _Metrics,
    ):
        self._router = router
        self._builder = builder
        self._batcher = batcher
        self._metrics = metrics

    async def _fail(self, context, code: grpc.StatusCode, message: str):
        self._metrics.requests.labels(code.name).inc()
        await context.abort(code, message)

    # grpc.aio servicers implement the generated (synchronous-looking)
    # methods as coroutines.
    async def PredictPrice(self, request, context):  # pylint: disable=invalid-name,invalid-overridden-method
        start = time.perf_counter()
        k = self._builder.num_candidates
        top_k = request.num_candidates or k
        if not request.HasField('location'):
            await self._fail(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                'location is required',
            )
        if (
            not request.HasField('attributes')
            or request.attributes.living_area_sqft <= 0
        ):
            await self._fail(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                'attributes.living_area_sqft must be positive',
            )
        if top_k > k:
            await self._fail(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                f'num_candidates must be at most {k}',
            )
        if request.valuation_epoch_day < 0:
            await self._fail(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                'valuation_epoch_day must be non-negative',
            )
        valuation_day = request.valuation_epoch_day or predict.today_epoch_day()
        lat, lng = request.location.latitude, request.location.longitude
        embedding = self._builder.embed(lat, lng, request.attributes)
        try:
            found = await self._router.search(
                embedding,
                lat,
                lng,
                request.radius_km,
                top_k=top_k,
                include_embeddings=True,
                sold_before_epoch_day=valuation_day,
            )
        except retrieve.RetrievalError as e:
            await self._fail(context, e.code, str(e))
        eligible = [
            c
            for c in found.candidates
            if c.attributes.last_sale_price_usd > 0
            and c.attributes.last_sale_epoch_day < valuation_day
        ]
        if not eligible:
            await self._fail(
                context,
                grpc.StatusCode.NOT_FOUND,
                'no comparable sales before the valuation day '
                f'within {found.radius_km} km',
            )
        subject = predict.Subject(
            lat,
            lng,
            request.attributes,
            valuation_day,
            found.radius_km,
            embedding,
        )
        batched = await self._batcher.submit((subject, eligible))
        total_us = int((time.perf_counter() - start) * 1e6)
        latency = pricing_pb2.StageLatency(
            retrieve_us=found.latency_us,
            batch_wait_us=batched.wait_us,
            inference_us=batched.inference_us,
            total_us=total_us,
        )
        for stage, micros in (
            ('retrieve', found.latency_us),
            ('batch_wait', batched.wait_us),
            ('inference', batched.inference_us),
            ('total', total_us),
        ):
            self._metrics.stage_seconds.labels(stage).observe(micros / 1e6)
        self._metrics.batch_size.observe(batched.batch_size)
        self._metrics.requests.labels(grpc.StatusCode.OK.name).inc()
        return pricing_pb2.PredictPriceResponse(
            predicted_price_usd=math.exp(batched.value.log_price),
            log_price=batched.value.log_price,
            candidates_used=batched.value.candidates_used,
            comparable_ids=[c.property_id for c in eligible],
            latency=latency,
        )


class Gateway:
    """The router, model, batcher and servicer of one gateway process."""

    def __init__(self, router, batcher, servicer, metrics):
        self.router = router
        self.batcher = batcher
        self.servicer = servicer
        self.metrics = metrics

    @classmethod
    async def create(
        cls,
        config: ServiceConfig,
        endpoints: dict[int, list[str]] | None = None,
        model=None,
    ) -> 'Gateway':
        """Builds a gateway inside the running event loop.

        Args:
          config: Gateway configuration.
          endpoints: Overrides the shard endpoints in engine/configs.
          model: (predict_fn, ModelConfig, batch size); loaded from
            `config` when None.
        """
        predict_fn, model_config, batch_size = model or load_model(config)
        engine = routing.load_engine_config(config.engine_config_dir)
        router = retrieve.Router.create(engine, endpoints)
        await router.wait_ready(config.connect_timeout_s)
        builder = predict.FeatureBuilder(model_config.num_candidates)
        predictor = predict.Predictor(
            predict_fn, builder, model_config.min_candidates
        )
        batcher = predict.Batcher(
            predictor, batch_size, config.batch_window_ms / 1000.0
        )
        batcher.start()
        metrics = _Metrics()
        return cls(
            router,
            batcher,
            PricingServicer(router, builder, batcher, metrics),
            metrics,
        )

    async def serve(self, listen: str) -> tuple[grpc.aio.Server, int]:
        """Starts the gRPC server; returns it and its bound port."""
        server = grpc.aio.server()
        pricing_pb2_grpc.add_PricingServiceServicer_to_server(
            self.servicer, server
        )
        port = server.add_insecure_port(listen)
        await server.start()
        return server, port

    async def close(self) -> None:
        await self.batcher.close()
        await self.router.close()


async def _run(config: ServiceConfig) -> None:
    gateway = await Gateway.create(config)
    if config.metrics_port:
        prometheus_client.start_http_server(
            config.metrics_port, registry=gateway.metrics.registry
        )
    server, port = await gateway.serve(config.listen)
    logging.info('PricingService listening on port %d', port)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    await stop.wait()
    await server.stop(grace=5.0)
    await gateway.close()


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    overrides = {
        'listen': _LISTEN.value,
        'bundle_dir': _BUNDLE_DIR.value,
        'device': _DEVICE.value,
    }
    config = ServiceConfig.from_yaml(
        _CONFIG.value, **{k: v for k, v in overrides.items() if v is not None}
    )
    asyncio.run(_run(config))


if __name__ == '__main__':
    app.run(main)
