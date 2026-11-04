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

"""An exact, in-process stand-in for the C++ shard servers.

Each fake shard owns the corpus rows whose shard-key cell the shard map
assigns to it, and answers Search by brute force: every row within the
radius, nearest first by squared L2 embedding distance. It implements the
same SearchService contract as engine/src/server, so the gateway can be
tested without the C++ build, and it is the exact oracle for recall checks.
"""

import collections
import dataclasses
import os

import grpc
import numpy as np

from data import embedding_model
from data import geo
from data import routing
from data import synthetic
from data.pipelines import corpus_io
from vectra import common_pb2
from vectra import search_pb2
from vectra import search_pb2_grpc


@dataclasses.dataclass
class Corpus:
    """Records, embeddings and 6-character cells, as the indexer sees them."""

    records: synthetic.Corpus
    embeddings: np.ndarray
    cell6: np.ndarray

    @classmethod
    def generate(cls, count: int, seed: int = 0) -> 'Corpus':
        records = synthetic.generate(count, seed)
        embeddings = embedding_model.PropertyEmbedder().embed(
            records.attributes(), records.latitude, records.longitude
        )
        cells = geo.encode_cells(records.latitude, records.longitude, 6)
        return cls(records, embeddings, cells)

    @classmethod
    def from_dir(cls, directory: str) -> 'Corpus':
        """Loads a corpus written by data/pipelines/generate_embeddings.py."""
        columns = {
            name: corpus_io.read_column(directory, name)
            for name in corpus_io.RECORD_COLUMNS
        }
        count = len(columns['property_id'])
        metro = (
            corpus_io.read_column(directory, 'metro_index')
            if os.path.exists(corpus_io.column_path(directory, 'metro_index'))
            else np.zeros(count, np.int16)
        )
        return cls(
            synthetic.Corpus(**columns, metro_index=metro),
            corpus_io.read_column(directory, corpus_io.EMBEDDING_COLUMN),
            corpus_io.read_column(directory, corpus_io.GEOHASH_COLUMN).astype(
                np.uint64
            ),
        )

    def candidate(
        self,
        row: int,
        query: np.ndarray,
        shard_id: int,
        distance_km: float,
        include_embedding: bool,
    ) -> common_pb2.Candidate:
        r = self.records
        embedding = self.embeddings[row]
        return common_pb2.Candidate(
            property_id=int(r.property_id[row]),
            squared_l2=float(np.sum((embedding - query) ** 2)),
            geo_distance_km=float(distance_km),
            location=common_pb2.LatLng(
                latitude=float(r.latitude[row]),
                longitude=float(r.longitude[row]),
            ),
            attributes=common_pb2.PropertyAttributes(
                living_area_sqft=float(r.living_area_sqft[row]),
                lot_area_sqft=float(r.lot_area_sqft[row]),
                bedrooms=int(r.bedrooms[row]),
                bathrooms=float(r.bathrooms[row]),
                year_built=int(r.year_built[row]),
                last_sale_price_usd=float(r.last_sale_price_usd[row]),
                last_sale_epoch_day=int(r.last_sale_epoch_day[row]),
            ),
            embedding=embedding.tolist() if include_embedding else [],
            squared_norm=float(np.sum(embedding**2)),
            geohash6=int(self.cell6[row]),
            shard_id=shard_id,
        )

    def exact_search(
        self,
        rows: np.ndarray,
        query: np.ndarray,
        lat: float,
        lng: float,
        radius_km: float,
        top_k: int,
        sold_before_epoch_day: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(rows, distances in km) within the radius, nearest in embedding.

        Only rows sold before `sold_before_epoch_day` qualify when positive.
        """
        r = self.records
        distance = geo.haversine_km_array(
            lat, lng, r.latitude[rows], r.longitude[rows]
        )
        inside = distance <= radius_km
        if sold_before_epoch_day > 0:
            inside &= r.last_sale_epoch_day[rows] < sold_before_epoch_day
        rows, distance = rows[inside], distance[inside]
        squared = np.sum((self.embeddings[rows] - query) ** 2, axis=1)
        order = np.lexsort((r.property_id[rows], squared))[:top_k]
        return rows[order], distance[order]


class ExactShard(search_pb2_grpc.SearchServiceServicer):
    """SearchService over one shard's rows, by brute force.

    Attributes:
      shard_id: The shard served.
      peers: Calls received per client connection (context.peer()).
    """

    def __init__(
        self,
        shard_id: int,
        corpus: Corpus,
        rows: np.ndarray,
        policy: routing.FanoutPolicy,
    ):
        self.shard_id = shard_id
        self.peers = collections.Counter()
        self._corpus = corpus
        self._rows = rows
        self._policy = policy

    # grpc.aio servicers implement the generated methods as coroutines.
    async def Search(self, request, context):  # pylint: disable=invalid-name,invalid-overridden-method
        self.peers[context.peer()] += 1
        dimension = self._corpus.embeddings.shape[1]
        if len(request.query_embedding) != dimension:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f'query_embedding must have {dimension} values',
            )
        radius = request.radius_km or self._policy.default_radius_km
        if not 0 < radius <= self._policy.max_radius_km:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f'radius_km out of range: {radius}',
            )
        if not 1 <= request.top_k <= 1024:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, 'top_k must be in [1, 1024]'
            )
        if request.sold_before_epoch_day < 0:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                'sold_before_epoch_day must be non-negative',
            )
        query = np.asarray(request.query_embedding, np.float32)
        rows, distance = self._corpus.exact_search(
            self._rows,
            query,
            request.location.latitude,
            request.location.longitude,
            radius,
            request.top_k,
            request.sold_before_epoch_day,
        )
        return search_pb2.SearchResponse(
            candidates=[
                self._corpus.candidate(
                    int(row),
                    query,
                    self.shard_id,
                    km,
                    request.include_embeddings,
                )
                for row, km in zip(rows, distance, strict=True)
            ],
            shard_id=self.shard_id,
            stats=search_pb2.SearchStats(
                distance_evaluations=len(rows),
                prefilter_population=len(self._rows),
                used_exhaustive_scan=True,
            ),
        )


@dataclasses.dataclass
class Cluster:
    """Running fake shards.

    Attributes:
      servers: The gRPC servers, one per shard.
      shards: Servicers by shard id.
      endpoints: host:port by shard id (for Router.create overrides).
    """

    servers: list[grpc.aio.Server]
    shards: dict[int, ExactShard]
    endpoints: dict[int, list[str]]

    async def stop(self) -> None:
        for server in self.servers:
            await server.stop(grace=None)


def shard_rows(
    corpus: Corpus, config: routing.EngineConfig
) -> dict[int, np.ndarray]:
    """Corpus rows owned by each shard (by shard-key cell)."""
    table = routing.RoutingTable.from_config(config)
    owners = table.shards_for_cells(corpus.cell6, 6)
    return {
        shard.id: np.flatnonzero(owners == shard.id)
        for shard in config.shard_map.shards
    }


async def start_cluster(
    corpus: Corpus, config: routing.EngineConfig
) -> Cluster:
    """Starts one ExactShard per configured shard on 127.0.0.1."""
    policy = routing.FanoutPolicy.from_config(
        config, routing.RoutingTable.from_config(config)
    )
    servers, shards, endpoints = [], {}, {}
    for shard_id, rows in shard_rows(corpus, config).items():
        server = grpc.aio.server()
        servicer = ExactShard(shard_id, corpus, rows, policy)
        search_pb2_grpc.add_SearchServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port('127.0.0.1:0')
        await server.start()
        servers.append(server)
        shards[shard_id] = servicer
        endpoints[shard_id] = [f'127.0.0.1:{port}']
    return Cluster(servers, shards, endpoints)
