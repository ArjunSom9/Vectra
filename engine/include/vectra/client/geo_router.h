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

#ifndef VECTRA_CLIENT_GEO_ROUTER_H_
#define VECTRA_CLIENT_GEO_ROUTER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "vectra/client/async_search.h"
#include "vectra/client/channel_pool.h"
#include "vectra/config/engine_config.h"
#include "vectra/routing/fanout_policy.h"
#include "vectra/routing/routing_table.h"
#include "vectra/search.pb.h"

namespace vectra::client {

// The stateless routing layer: computes the query's GeoHash cell, sends the
// search to the shard that owns it (and, only when the radius crosses a cell
// boundary, to the owners of the intersected neighbor cells, at most nine
// cells in all), and merges the shard top-k lists. Thread-safe.
class GeoRouter {
 public:
  using Callback =
      absl::AnyInvocable<void(absl::StatusOr<search::v1::SearchResponse>) &&>;

  // Builds channel pools for every shard endpoint in the shard map.
  static absl::StatusOr<std::unique_ptr<GeoRouter>> Create(
      const config::EngineConfig& config, int polling_threads);

  GeoRouter(const GeoRouter&) = delete;
  GeoRouter& operator=(const GeoRouter&) = delete;

  // Routes `request` and calls `done` exactly once with the merged result.
  void Search(const search::v1::SearchRequest& request, Callback done);

  // Blocking convenience wrapper for tools and tests.
  absl::StatusOr<search::v1::SearchResponse> SearchBlocking(
      const search::v1::SearchRequest& request);

  // Connects every pooled channel.
  bool WaitForConnected(absl::Time deadline);

  const routing::FanoutPolicy& policy() const { return *policy_; }

 private:
  GeoRouter(std::unique_ptr<routing::RoutingTable> table,
            double default_radius_km, double max_radius_km,
            absl::Duration timeout, int polling_threads);

  std::unique_ptr<routing::RoutingTable> table_;
  std::unique_ptr<routing::FanoutPolicy> policy_;
  absl::Duration timeout_;
  // One pool per shard (first endpoint; replicas are an operator concern).
  std::map<uint32_t, std::unique_ptr<ChannelPool>> pools_;
  AsyncClient client_;
};

}  // namespace vectra::client

#endif  // VECTRA_CLIENT_GEO_ROUTER_H_
