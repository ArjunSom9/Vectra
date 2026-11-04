// Copyright 2025 Texas A&M University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "vectra/service/search_handler.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "grpcpp/support/status.h"
#include "vectra/index/search_scratch.h"
#include "vectra/routing/geo_math.h"
#include "vectra/service/proto_convert.h"

namespace vectra::service {
namespace {

grpc::Status InvalidArgument(std::string message) {
  return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, std::move(message));
}

}  // namespace

SearchHandler::SearchHandler(const index::ShardIndex* index, uint32_t shard_id,
                             SearchLimits limits)
    : index_(index), shard_id_(shard_id), limits_(limits) {}

grpc::Status SearchHandler::Search(const search::v1::SearchRequest& request,
                                   search::v1::SearchResponse* response) const {
  const auto start = std::chrono::steady_clock::now();
  if (request.query_embedding_size() != index_->dimension()) {
    return InvalidArgument(
        absl::StrCat("query_embedding has ", request.query_embedding_size(),
                     " values; the shard expects ", index_->dimension()));
  }
  if (request.top_k() == 0 || request.top_k() > limits_.max_top_k) {
    return InvalidArgument(
        absl::StrCat("top_k must be in [1, ", limits_.max_top_k, "]"));
  }
  if (request.ef_search() > limits_.max_ef_search) {
    return InvalidArgument(
        absl::StrCat("ef_search must not exceed ", limits_.max_ef_search));
  }
  const double radius = request.radius_km() == 0.0 ? limits_.default_radius_km
                                                   : request.radius_km();
  const routing::LatLng center = FromProto(request.location());
  if (!std::isfinite(radius) || radius < 0.0 ||
      radius > limits_.max_radius_km) {
    return InvalidArgument(
        absl::StrCat("radius_km must be in [0, ", limits_.max_radius_km, "]"));
  }
  if (radius > 0.0 && !routing::IsValidLatLng(center)) {
    return InvalidArgument("location is not a valid coordinate");
  }
  if (request.sold_before_epoch_day() < 0) {
    return InvalidArgument("sold_before_epoch_day must be non-negative");
  }

  index::IndexQuery query;
  query.embedding = request.query_embedding().data();
  query.center = center;
  query.radius_km = radius;
  query.k = static_cast<int>(request.top_k());
  query.ef = static_cast<int>(request.ef_search());
  query.exact = request.exact();
  query.sold_before_epoch_day = request.sold_before_epoch_day();

  // One scratch per worker thread for the life of the thread.
  thread_local index::SearchScratch scratch;
  thread_local std::vector<index::IndexHit> hits;
  index::IndexSearchStats stats;
  index_->Search(query, &scratch, &hits, &stats);

  response->set_shard_id(shard_id_);
  response->mutable_candidates()->Reserve(static_cast<int>(hits.size()));
  for (const index::IndexHit& hit : hits) {
    ToCandidate(*index_, hit, shard_id_, request.include_embeddings(),
                response->add_candidates());
  }
  search::v1::SearchStats* out = response->mutable_stats();
  out->set_distance_evaluations(stats.distance_evaluations);
  out->set_prefilter_population(stats.prefilter_population);
  out->set_used_exhaustive_scan(stats.exhaustive);
  out->set_server_latency_ns(static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count()));
  return grpc::Status::OK;
}

}  // namespace vectra::service
