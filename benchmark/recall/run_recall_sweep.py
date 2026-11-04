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

"""Recall@10 versus ef_search against exact ground truth (A2, A3).

  python -m benchmark.recall.run_recall_sweep \
      --corpus_dir=data/corpus/dev --config_dir=engine/configs \
      --ef=16,32,64,128,256,512 --out=benchmark/reports/recall_sweep.json \
      [--oracle]

Follows the ANN-Benchmarks protocol [4]: ground truth is exact brute force
(computed here in NumPy, independently of the engine), recall@k is the
fraction of the true k nearest returned, and every point of the curve is one
ef_search value. Queries are the held-out properties of the corpus, searched
within --radius_km of their own location through the running shards (the
same geo router as production). Boundary queries, whose disk reaches more
than one shard, are also searched in exact mode, where recall must be 1.0:
routing may never lose a comparable. --oracle adds hnswlib's curve at the
same M and ef_construction on the unfiltered corpus, as a reference for the
graph's quality.
"""

import asyncio
from collections.abc import Sequence
import json
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np

from data import geo
from data import routing
from data.pipelines import corpus_io
from serving import retrieve

_CORPUS_DIR = flags.DEFINE_string(
    'corpus_dir',
    'data/corpus/dev',
    'Indexed corpus (with queries/).',
    allow_override=True,
)
_CONFIG_DIR = flags.DEFINE_string(
    'config_dir', 'engine/configs', 'Engine configuration.', allow_override=True
)
_ENDPOINTS = flags.DEFINE_list(
    'endpoints',
    [],
    'Overrides as shard=host:port, e.g. 1=127.0.0.1:7101.',
    allow_override=True,
)
_EF = flags.DEFINE_list(
    'ef',
    ['16', '32', '64', '128', '256', '512'],
    'ef_search values.',
    allow_override=True,
)
_K = flags.DEFINE_integer('k', 10, 'Recall depth.', allow_override=True)
_RADIUS_KM = flags.DEFINE_float(
    'radius_km', 5.0, 'Search radius.', allow_override=True
)
_NUM_QUERIES = flags.DEFINE_integer(
    'num_queries', 1000, 'Queries to use.', allow_override=True
)
_ORACLE = flags.DEFINE_bool(
    'oracle', False, 'Also sweep hnswlib.', allow_override=True
)
_OUT = flags.DEFINE_string(
    'out',
    'benchmark/reports/recall_sweep.json',
    'Output JSON.',
    allow_override=True,
)


class GroundTruth:
    """Exact k nearest within a radius, by brute force over prefilter cells."""

    def __init__(self, corpus_dir: str):
        self.ids = corpus_io.read_column(corpus_dir, 'property_id')
        self.lat = corpus_io.read_column(corpus_dir, 'latitude')
        self.lng = corpus_io.read_column(corpus_dir, 'longitude')
        self.embeddings = corpus_io.read_column(
            corpus_dir, corpus_io.EMBEDDING_COLUMN, mmap=True
        )
        cell5 = (
            corpus_io.read_column(corpus_dir, corpus_io.GEOHASH_COLUMN).astype(
                np.int64
            )
            >> 5
        )
        order = np.argsort(cell5, kind='stable')
        keys, starts = np.unique(cell5[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        self._buckets = {
            int(c): order[s:e]
            for c, s, e in zip(keys, starts, ends, strict=True)
        }

    def search(
        self,
        query: np.ndarray,
        lat: float,
        lng: float,
        radius_km: float,
        k: int,
    ) -> list[int]:
        cells = geo.cover_disk(lat, lng, radius_km, 5, max_rings=64) or []
        pools = [self._buckets[c] for c in cells if c in self._buckets]
        if not pools:
            return []
        rows = np.concatenate(pools)
        rows = rows[
            geo.haversine_km_array(lat, lng, self.lat[rows], self.lng[rows])
            <= radius_km
        ]
        squared = np.sum((np.asarray(self.embeddings[rows]) - query) ** 2, 1)
        order = np.lexsort((self.ids[rows], squared))[:k]
        return self.ids[rows[order]].tolist()


def recall(truth: Sequence[int], found: Sequence[int]) -> float:
    if not truth:
        return 1.0
    return len(set(truth) & set(found)) / len(truth)


def parse_endpoints(entries: Sequence[str]) -> dict[int, list[str]] | None:
    """{shard: [host:port, ...]} from shard=host:port entries."""
    if not entries:
        return None
    result = {}
    for entry in entries:
        shard, target = entry.split('=', 1)
        result.setdefault(int(shard), []).append(target)
    return result


async def sweep_engine(
    queries,
    truths,
    boundary,
    config: routing.EngineConfig,
    endpoints: dict[int, list[str]] | None,
    efs: Sequence[int],
    radius_km: float,
    k: int,
) -> dict:
    """Recall and latency per ef_search, and exact boundary recall."""
    router = retrieve.Router.create(config, endpoints)
    try:
        await router.wait_ready(30.0)
        curve = []
        for ef in efs:
            scores, latencies = [], []
            for (embedding, lat, lng), truth in zip(
                queries, truths, strict=True
            ):
                start = time.perf_counter()
                found = await router.search(
                    embedding,
                    lat,
                    lng,
                    radius_km,
                    k,
                    include_embeddings=False,
                    ef_search=ef,
                )
                latencies.append((time.perf_counter() - start) * 1e6)
                scores.append(
                    recall(truth, [c.property_id for c in found.candidates])
                )
            curve.append(
                {
                    'ef_search': ef,
                    'recall_at_10': float(np.mean(scores)),
                    'p50_us': float(np.percentile(latencies, 50)),
                    'p99_us': float(np.percentile(latencies, 99)),
                }
            )
            logging.info('ef %d: recall %.4f', ef, curve[-1]['recall_at_10'])
        exact_scores = []
        for i in boundary:
            embedding, lat, lng = queries[i]
            found = await router.search(
                embedding,
                lat,
                lng,
                radius_km,
                k,
                include_embeddings=False,
                exact=True,
            )
            exact_scores.append(
                recall(truths[i], [c.property_id for c in found.candidates])
            )
        return {
            'curve': curve,
            'boundary_queries': len(boundary),
            'boundary_recall': (
                float(np.mean(exact_scores)) if exact_scores else None
            ),
        }
    finally:
        await router.close()


def global_knn(
    data: np.ndarray, vectors: np.ndarray, k: int, block: int = 65_536
) -> np.ndarray:
    """Exact k nearest rows of `data` for each vector, in blocks of rows."""
    norms = np.sum(vectors.astype(np.float64) ** 2, axis=1)
    best_d = np.full((len(vectors), k), np.inf)
    best_i = np.zeros((len(vectors), k), np.int64)
    for start in range(0, len(data), block):
        chunk = np.asarray(data[start : start + block], np.float64)
        d = (
            norms[:, None]
            - 2.0 * vectors @ chunk.T
            + np.sum(chunk**2, axis=1)[None, :]
        )
        ids = np.broadcast_to(np.arange(start, start + len(chunk)), d.shape)
        all_d = np.concatenate([best_d, d], axis=1)
        all_i = np.concatenate([best_i, ids], axis=1)
        top = np.argpartition(all_d, k - 1, axis=1)[:, :k]
        best_d = np.take_along_axis(all_d, top, axis=1)
        best_i = np.take_along_axis(all_i, top, axis=1)
    order = np.argsort(best_d, axis=1, kind='stable')
    return np.take_along_axis(best_i, order, axis=1)


def sweep_hnswlib(
    truth_index: GroundTruth,
    queries,
    config: routing.EngineConfig,
    efs: Sequence[int],
    k: int,
) -> list[dict]:
    """hnswlib on the whole corpus, unfiltered, against global brute force."""
    import hnswlib  # pylint: disable=import-outside-toplevel

    data = np.asarray(truth_index.embeddings)
    index = hnswlib.Index(space='l2', dim=data.shape[1])
    index.init_index(
        max_elements=len(data),
        M=config.index.m,
        ef_construction=config.index.ef_construction,
        random_seed=config.index.seed,
    )
    index.add_items(data, np.arange(len(data)))
    vectors = np.stack([q[0] for q in queries]).astype(np.float64)
    truth = global_knn(data, vectors, k)
    curve = []
    for ef in efs:
        index.set_ef(max(ef, k))
        labels, _ = index.knn_query(vectors.astype(np.float32), k=k)
        curve.append(
            {
                'ef_search': ef,
                'recall_at_10': float(
                    np.mean(
                        [
                            recall(t.tolist(), found.tolist())
                            for t, found in zip(truth, labels, strict=True)
                        ]
                    )
                ),
            }
        )
    return curve


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    config = routing.load_engine_config(_CONFIG_DIR.value)
    policy = routing.FanoutPolicy.from_config(
        config, routing.RoutingTable.from_config(config)
    )
    query_dir = f'{_CORPUS_DIR.value}/queries'
    embeddings = corpus_io.read_column(query_dir, 'query_embedding')
    lats = corpus_io.read_column(query_dir, 'query_latitude')
    lngs = corpus_io.read_column(query_dir, 'query_longitude')
    count = min(_NUM_QUERIES.value, len(lats))
    queries = [
        (embeddings[i], float(lats[i]), float(lngs[i])) for i in range(count)
    ]
    truth_index = GroundTruth(_CORPUS_DIR.value)
    truths = [
        truth_index.search(e, lat, lng, _RADIUS_KM.value, _K.value)
        for e, lat, lng in queries
    ]
    boundary = [
        i
        for i, (_, lat, lng) in enumerate(queries)
        if len(policy.plan(lat, lng, _RADIUS_KM.value).shard_ids) > 1
    ]
    efs = [int(e) for e in _EF.value]
    result = asyncio.run(
        sweep_engine(
            queries,
            truths,
            boundary,
            config,
            parse_endpoints(_ENDPOINTS.value),
            efs,
            _RADIUS_KM.value,
            _K.value,
        )
    )
    result.update(
        {
            'k': _K.value,
            'radius_km': _RADIUS_KM.value,
            'queries': count,
            'configured_ef_search': config.index.ef_search,
        }
    )
    if _ORACLE.value:
        result['hnswlib_curve'] = sweep_hnswlib(
            truth_index, queries, config, efs, _K.value
        )
    with open(_OUT.value, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    app.run(main)
