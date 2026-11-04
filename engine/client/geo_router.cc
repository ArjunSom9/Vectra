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

#include "vectra/client/geo_router.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "vectra/service/proto_convert.h"

namespace vectra::client {
namespace {

absl::Status FromGrpc(const grpc::Status& status) {
  return absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                      status.error_message());
}

// Collects the shard responses of one routed query.
class FanoutState {
 public:
  FanoutState(int expected, uint32_t top_k, GeoRouter::Callback done)
      : remaining_(expected), top_k_(top_k), done_(std::move(done)) {}

  void Add(const grpc::Status& status, search::v1::SearchResponse* response) {
    {
      absl::MutexLock lock(&mutex_);
      if (!status.ok()) {
        if (error_.ok()) error_ = FromGrpc(status);
      } else {
        for (common::v1::Candidate& candidate :
             *response->mutable_candidates()) {
          candidates_.push_back(std::move(candidate));
        }
      }
    }
    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) Finish();
  }

 private:
  void Finish() {
    absl::StatusOr<search::v1::SearchResponse> result;
    {
      absl::MutexLock lock(&mutex_);
      if (!error_.ok()) {
        result = error_;
      } else {
        // Shards own disjoint cells, so merging is a sort by exact distance.
        std::sort(
            candidates_.begin(), candidates_.end(),
            [](const common::v1::Candidate& a, const common::v1::Candidate& b) {
              return a.squared_l2() < b.squared_l2() ||
                     (a.squared_l2() == b.squared_l2() &&
                      a.property_id() < b.property_id());
            });
        search::v1::SearchResponse merged;
        const size_t keep = std::min<size_t>(candidates_.size(), top_k_);
        for (size_t i = 0; i < keep; ++i) {
          *merged.add_candidates() = std::move(candidates_[i]);
        }
        result = std::move(merged);
      }
    }
    std::move(done_)(std::move(result));
    delete this;
  }

  std::atomic<int> remaining_;
  uint32_t top_k_;
  GeoRouter::Callback done_;
  absl::Mutex mutex_;
  absl::Status error_ ABSL_GUARDED_BY(mutex_);
  std::vector<common::v1::Candidate> candidates_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace

GeoRouter::GeoRouter(std::unique_ptr<routing::RoutingTable> table,
                     double default_radius_km, double max_radius_km,
                     absl::Duration timeout, int polling_threads)
    : table_(std::move(table)),
      policy_(std::make_unique<routing::FanoutPolicy>(
          table_.get(), default_radius_km, max_radius_km)),
      timeout_(timeout),
      client_(polling_threads) {}

absl::StatusOr<std::unique_ptr<GeoRouter>> GeoRouter::Create(
    const config::EngineConfig& config, int polling_threads) {
  absl::StatusOr<std::unique_ptr<routing::RoutingTable>> table =
      config::BuildRoutingTable(config);
  if (!table.ok()) return table.status();
  const config::v1::ClientOptions& client = config.topology.client();
  const absl::Duration timeout =
      absl::Milliseconds(client.timeout_ms() == 0 ? 2 : client.timeout_ms());
  std::unique_ptr<GeoRouter> router(
      new GeoRouter(*std::move(table), config.geohash.default_radius_km(),
                    config.geohash.max_radius_km(), timeout, polling_threads));
  ChannelPoolOptions options;
  if (client.channel_pool_size() > 0) {
    options.size = static_cast<int>(client.channel_pool_size());
  }
  if (client.keepalive_time_ms() > 0) {
    options.keepalive_time_ms = static_cast<int>(client.keepalive_time_ms());
  }
  for (const config::v1::ShardEndpoint& shard : config.shard_map.shards()) {
    if (shard.endpoints().empty()) {
      return absl::InvalidArgumentError("shard without endpoints");
    }
    router->pools_[shard.id()] = std::make_unique<ChannelPool>(
        std::vector<std::string>(shard.endpoints().begin(),
                                 shard.endpoints().end()),
        options);
  }
  return router;
}

void GeoRouter::Search(const search::v1::SearchRequest& request,
                       Callback done) {
  const routing::RoutePlan plan = policy_->Plan(
      service::FromProto(request.location()), request.radius_km());
  if (!plan.ok()) {
    std::move(done)(absl::InvalidArgumentError(plan.message));
    return;
  }
  std::vector<ChannelPool*> targets;
  for (const routing::ShardTarget& target : plan.targets) {
    auto it = pools_.find(target.shard_id);
    if (it != pools_.end()) targets.push_back(it->second.get());
  }
  if (targets.empty()) {
    std::move(done)(search::v1::SearchResponse());  // No coverage there.
    return;
  }
  search::v1::SearchRequest routed = request;
  routed.set_radius_km(plan.radius_km);
  auto* state = new FanoutState(static_cast<int>(targets.size()),
                                request.top_k(), std::move(done));
  for (ChannelPool* pool : targets) {
    client_.Search(pool->NextSearchStub(), routed, timeout_,
                   [state](const grpc::Status& status,
                           search::v1::SearchResponse* response) {
                     state->Add(status, response);
                   });
  }
}

absl::StatusOr<search::v1::SearchResponse> GeoRouter::SearchBlocking(
    const search::v1::SearchRequest& request) {
  absl::Notification done;
  absl::StatusOr<search::v1::SearchResponse> result;
  Search(request, [&](absl::StatusOr<search::v1::SearchResponse> response) {
    result = std::move(response);
    done.Notify();
  });
  done.WaitForNotification();
  return result;
}

bool GeoRouter::WaitForConnected(absl::Time deadline) {
  bool ready = true;
  for (auto& [shard, pool] : pools_) {
    ready = pool->WaitForConnected(deadline) && ready;
  }
  return ready;
}

}  // namespace vectra::client
