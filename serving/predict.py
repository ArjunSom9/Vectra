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

"""Prediction: model inputs from retrieved comparables, and batching.

The features are assembled exactly as models/training/dataset.py assembles
them for training: the subject's embedding and standardized attributes, and
for each comparable its embedding, attributes, log sale price, sale day,
location and 6-character GeoHash cell. Only sales before the valuation day
are valid comparables; the gateway asks the shards for exactly those
(SearchRequest.sold_before_epoch_day), as the training set was drawn.
"""

import asyncio
from collections.abc import Callable, Sequence
import concurrent.futures
import dataclasses
import time
import typing

import numpy as np
import torch

from data import embedding_model
from data import geo
from models.price_transformer import microhood
from models.price_transformer import model as model_lib
from vectra import common_pb2

_SECONDS_PER_DAY = 86_400


def today_epoch_day() -> int:
    return int(time.time() // _SECONDS_PER_DAY)


def _attribute_columns(
    attributes: Sequence[common_pb2.PropertyAttributes],
) -> dict[str, np.ndarray]:
    return {
        name: np.array([getattr(a, name) for a in attributes])
        for name in embedding_model.ATTRIBUTE_COLUMNS
    }


@dataclasses.dataclass
class Subject:
    """The property being valued.

    Attributes:
      latitude: Degrees.
      longitude: Degrees.
      attributes: Its physical attributes (sale fields are ignored).
      valuation_day: Days since the epoch; only earlier sales count.
      radius_km: Comparable search radius.
      embedding: [E] float32, as embedded for the index.
    """

    latitude: float
    longitude: float
    attributes: common_pb2.PropertyAttributes
    valuation_day: int
    radius_km: float
    embedding: np.ndarray


class FeatureBuilder:
    """Turns subjects and retrieved candidates into PricingInputs."""

    def __init__(
        self,
        num_candidates: int,
        embedder: embedding_model.PropertyEmbedder | None = None,
    ):
        self.num_candidates = num_candidates
        self.embedder = embedder or embedding_model.PropertyEmbedder()

    def embed(
        self,
        latitude: float,
        longitude: float,
        attributes: common_pb2.PropertyAttributes,
    ) -> np.ndarray:
        """The subject's embedding, from the same network as the corpus."""
        return self.embedder.embed(
            _attribute_columns([attributes]),
            np.array([latitude]),
            np.array([longitude]),
        )[0]

    def build(
        self,
        subjects: Sequence[Subject],
        candidates: Sequence[Sequence[common_pb2.Candidate]],
    ) -> model_lib.PricingInputs:
        """Model inputs for a batch; comparables beyond K are dropped.

        Raises:
          ValueError: If a candidate lacks its embedding (the search must
            request include_embeddings).
        """
        n, k = len(subjects), self.num_candidates
        dim = self.embedder.dimension
        cand_embedding = np.zeros((n, k, dim), np.float32)
        cand_attributes = np.zeros(
            (n, k, len(embedding_model.ATTRIBUTE_COLUMNS)), np.float32
        )
        cand_log_price = np.zeros((n, k), np.float32)
        cand_sale_day = np.zeros((n, k), np.float32)
        cand_location = np.zeros((n, k, 2), np.float32)
        cand_cell6 = np.zeros((n, k), np.int64)
        cand_valid = np.zeros((n, k), bool)
        for i, (subject, rows) in enumerate(
            zip(subjects, candidates, strict=True)
        ):
            rows = [
                c
                for c in rows
                if c.attributes.last_sale_price_usd > 0
                and c.attributes.last_sale_epoch_day < subject.valuation_day
            ]
            rows = rows[:k]
            if not rows:
                continue
            m = len(rows)
            if any(len(c.embedding) != dim for c in rows):
                raise ValueError('candidates must carry their embeddings')
            cand_embedding[i, :m] = [list(c.embedding) for c in rows]
            cand_attributes[i, :m] = embedding_model.attribute_features(
                _attribute_columns([c.attributes for c in rows])
            )
            cand_log_price[i, :m] = np.log(
                [c.attributes.last_sale_price_usd for c in rows]
            )
            cand_sale_day[i, :m] = [
                c.attributes.last_sale_epoch_day for c in rows
            ]
            cand_location[i, :m] = [
                (c.location.latitude, c.location.longitude) for c in rows
            ]
            cand_cell6[i, :m] = [c.geohash6 for c in rows]
            cand_valid[i, :m] = True
        arrays = model_lib.PricingInputs(
            query_embedding=np.stack([s.embedding for s in subjects]).astype(
                np.float32
            ),
            query_attributes=embedding_model.attribute_features(
                _attribute_columns([s.attributes for s in subjects])
            ),
            query_location=np.array(
                [(s.latitude, s.longitude) for s in subjects], np.float32
            ),
            query_cell6=np.array(
                [geo.encode_cell(s.latitude, s.longitude, 6) for s in subjects],
                np.int64,
            ),
            valuation_day=np.array(
                [s.valuation_day for s in subjects], np.float32
            ),
            radius_km=np.array([s.radius_km for s in subjects], np.float32),
            cand_embedding=cand_embedding,
            cand_attributes=cand_attributes,
            cand_log_price=cand_log_price,
            cand_sale_day=cand_sale_day,
            cand_location=cand_location,
            cand_cell6=cand_cell6,
            cand_valid=cand_valid,
        )
        return model_lib.PricingInputs(*(torch.from_numpy(a) for a in arrays))


def comparables_used(
    inputs: model_lib.PricingInputs, min_candidates: int
) -> torch.Tensor:
    """[B] comparables the model attends to after radius trimming."""
    distance = microhood.haversine_km(
        inputs.query_location, inputs.cand_location
    )
    mask = model_lib.trim_to_radius(
        distance,
        inputs.radius_km,
        inputs.cand_valid,
        min(min_candidates, inputs.cand_valid.shape[-1]),
    )
    return mask.sum(-1)


@dataclasses.dataclass
class Prediction:
    log_price: float
    candidates_used: int


class Predictor:
    """Runs one batch of subjects through the model."""

    def __init__(
        self,
        predict_fn: Callable[[model_lib.PricingInputs], torch.Tensor],
        builder: FeatureBuilder,
        min_candidates: int,
    ):
        """Initializes the predictor.

        Args:
          predict_fn: Maps a batch of inputs to [B] log prices, for example
            compile_entry.ServingModel.predict.
          builder: Feature assembly.
          min_candidates: The model's radius-trimming fallback size.
        """
        self._predict_fn = predict_fn
        self._builder = builder
        self._min_candidates = min_candidates

    def __call__(
        self, batch: Sequence[tuple[Subject, Sequence[common_pb2.Candidate]]]
    ) -> list[Prediction]:
        inputs = self._builder.build(
            [s for s, _ in batch], [c for _, c in batch]
        )
        with torch.no_grad():
            log_price = self._predict_fn(inputs).float().cpu()
        used = comparables_used(inputs, self._min_candidates)
        return [
            Prediction(float(p), int(u))
            for p, u in zip(log_price, used, strict=True)
        ]


_Item = typing.TypeVar('_Item')
_Result = typing.TypeVar('_Result')


@dataclasses.dataclass
class Batched(typing.Generic[_Result]):
    """One request's result and where its time went (microseconds)."""

    value: _Result
    batch_size: int
    wait_us: int
    inference_us: int


class Batcher(typing.Generic[_Item, _Result]):
    """Groups concurrent requests into one model call (dynamic batching).

    The first request of a batch opens a window; the batch runs when the
    window closes or `max_batch_size` requests are waiting, whichever comes
    first. The batch function runs on one dedicated thread, so the event
    loop keeps accepting requests while the GPU works, and batches never
    overlap on the device.
    """

    def __init__(
        self,
        run_batch: Callable[[list[_Item]], list[_Result]],
        max_batch_size: int,
        window_s: float,
    ):
        if max_batch_size < 1 or window_s < 0:
            raise ValueError('need max_batch_size >= 1 and window_s >= 0')
        self._run_batch = run_batch
        self._max_batch_size = max_batch_size
        self._window_s = window_s
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='inference'
        )

    def start(self) -> None:
        """Starts the batching loop on the running event loop."""
        self._queue = asyncio.Queue()
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=True)

    async def submit(self, item: _Item) -> Batched[_Result]:
        """Queues `item`; resolves when its batch has run.

        Raises:
          Exception: Whatever the batch function raised for its batch.
        """
        if self._queue is None:
            raise RuntimeError('Batcher.start() was not called')
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((item, future, time.perf_counter()))
        return await future

    async def _collect(self) -> list[tuple]:
        batch = [await self._queue.get()]
        deadline = time.perf_counter() + self._window_s
        while len(batch) < self._max_batch_size:
            # Requests that queued behind the previous batch join at once.
            try:
                batch.append(self._queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(
                    await asyncio.wait_for(self._queue.get(), remaining)
                )
            except TimeoutError:
                break
        return batch

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._collect()
            batch = [entry for entry in batch if not entry[1].cancelled()]
            if not batch:
                continue
            started = time.perf_counter()
            try:
                results = await loop.run_in_executor(
                    self._executor, self._run_batch, [b[0] for b in batch]
                )
                if len(results) != len(batch):
                    raise RuntimeError(
                        f'the batch function returned {len(results)} '
                        f'results for {len(batch)} requests'
                    )
            except Exception as e:  # pylint: disable=broad-except
                # Isolation point: the error belongs to this batch's callers,
                # and the loop must keep serving later batches.
                for _, future, _ in batch:
                    if not future.done():
                        future.set_exception(e)
                continue
            inference_us = int((time.perf_counter() - started) * 1e6)
            for (_, future, queued), result in zip(batch, results, strict=True):
                if not future.done():
                    future.set_result(
                        Batched(
                            result,
                            len(batch),
                            int((started - queued) * 1e6),
                            inference_us,
                        )
                    )
