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

#ifndef VECTRA_SERVICE_SEARCH_HANDLER_H_
#define VECTRA_SERVICE_SEARCH_HANDLER_H_

#include <cstdint>

#include "grpcpp/support/status.h"
#include "vectra/index/shard_index.h"
#include "vectra/search.pb.h"

namespace vectra::service {

// Limits applied to every search request.
struct SearchLimits {
  uint32_t max_top_k = 1024;
  uint32_t max_ef_search = 4096;
  double default_radius_km = 0.0;
  double max_radius_km = 15.0;
};

// Serves vectra.search.v1.SearchService/Search against one shard. Runs on the
// completion-queue worker that received the request; per-thread scratch makes
// steady-state searches allocation-free inside the index. Thread-safe.
class SearchHandler {
 public:
  // `index` must outlive the handler.
  SearchHandler(const index::ShardIndex* index, uint32_t shard_id,
                SearchLimits limits);

  grpc::Status Search(const search::v1::SearchRequest& request,
                      search::v1::SearchResponse* response) const;

 private:
  const index::ShardIndex* index_;
  uint32_t shard_id_;
  SearchLimits limits_;
};

}  // namespace vectra::service

#endif  // VECTRA_SERVICE_SEARCH_HANDLER_H_
