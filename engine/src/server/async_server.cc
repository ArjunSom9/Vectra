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

#include "vectra/server/async_server.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server_builder.h"
#include "vectra/server/core_pin.h"
#include "vectra/server/cq_worker.h"

namespace vectra::server {
namespace {

// Method indices within each queue's CallPool.
enum Method : int { kSearch = 0, kIngest = 1, kReport = 2, kPing = 3 };

}  // namespace

AsyncServer::AsyncServer(ServerOptions options, ShardHandlers handlers)
    : options_(std::move(options)), handlers_(handlers) {
  search_fn_ = [this](const search::v1::SearchRequest& request,
                      search::v1::SearchResponse* response) {
    return handlers_.search->Search(request, response);
  };
  ingest_fn_ = [this](const ingest::v1::IngestRequest& request,
                      ingest::v1::IngestResponse* response) {
    return handlers_.ingest->Ingest(request, response);
  };
  report_fn_ = [this](const health::v1::ReportRequest& request,
                      health::v1::ReportResponse* response) {
    const grpc::Status status = handlers_.health->Report(request, response);
    FillWorkerStats(response);
    return status;
  };
  ping_fn_ = [this](const health::v1::PingRequest& request,
                    health::v1::PingResponse* response) {
    return handlers_.health->Ping(request, response);
  };
}

AsyncServer::~AsyncServer() { Shutdown(absl::ZeroDuration()); }

std::vector<std::unique_ptr<MethodBinding>> AsyncServer::BindingsFor(
    grpc::ServerCompletionQueue* queue) {
  std::vector<std::unique_ptr<MethodBinding>> bindings;
  bindings.push_back(
      std::make_unique<UnaryBinding<SearchService, search::v1::SearchRequest,
                                    search::v1::SearchResponse>>(
          &search_service_, &SearchService::RequestSearch, &search_fn_, queue));
  bindings.push_back(
      std::make_unique<UnaryBinding<IngestService, ingest::v1::IngestRequest,
                                    ingest::v1::IngestResponse>>(
          &ingest_service_, &IngestService::RequestIngest, &ingest_fn_, queue));
  bindings.push_back(
      std::make_unique<UnaryBinding<HealthService, health::v1::ReportRequest,
                                    health::v1::ReportResponse>>(
          &health_service_, &HealthService::RequestReport, &report_fn_, queue));
  bindings.push_back(
      std::make_unique<UnaryBinding<HealthService, health::v1::PingRequest,
                                    health::v1::PingResponse>>(
          &health_service_, &HealthService::RequestPing, &ping_fn_, queue));
  return bindings;
}

absl::Status AsyncServer::Start() {
  if (running_) return absl::FailedPreconditionError("server already started");
  grpc::ServerBuilder builder;
  builder.AddListeningPort(options_.listen_address,
                           grpc::InsecureServerCredentials(), &port_);
  builder.SetMaxReceiveMessageSize(options_.max_receive_message_bytes);
  builder.RegisterService(&search_service_);
  builder.RegisterService(&ingest_service_);
  builder.RegisterService(&health_service_);

  queue_cores_ = options_.worker_cores;
  const size_t queue_count =
      queue_cores_.empty() ? static_cast<size_t>(AvailableHardwareThreads())
                           : queue_cores_.size();
  for (size_t i = 0; i < queue_count; ++i) {
    queues_.push_back(builder.AddCompletionQueue());
  }
  server_ = builder.BuildAndStart();
  if (server_ == nullptr || port_ == 0) {
    return absl::UnavailableError(
        absl::StrCat("cannot listen on ", options_.listen_address));
  }
  running_ = true;
  const int threads_per_queue = std::max(1, options_.threads_per_queue);
  for (size_t i = 0; i < queues_.size(); ++i) {
    counters_.push_back(std::make_unique<QueueCounters>());
    pools_.push_back(std::make_unique<CallPool>(BindingsFor(queues_[i].get()),
                                                counters_.back().get()));
    const int core = queue_cores_.empty() ? -1 : queue_cores_[i];
    for (int t = 0; t < threads_per_queue; ++t) {
      workers_.emplace_back(RunCompletionQueue, queues_[i].get(), core);
    }
  }
  for (const std::unique_ptr<CallPool>& pool : pools_) {
    pool->Prime(options_.requests_per_queue);
  }
  return absl::OkStatus();
}

void AsyncServer::Shutdown(absl::Duration grace) {
  if (!running_) return;
  running_ = false;
  for (const std::unique_ptr<CallPool>& pool : pools_) pool->StopArming();
  // Server first, queues second: the order the tutorial requires [19].
  server_->Shutdown(absl::ToChronoTime(absl::Now() + grace));
  for (const std::unique_ptr<grpc::ServerCompletionQueue>& queue : queues_) {
    queue->Shutdown();
  }
  for (std::thread& worker : workers_) worker.join();
  workers_.clear();
  // Every call has returned to its pool; the pools now free them.
  pools_.clear();
}

void AsyncServer::FillWorkerStats(health::v1::ReportResponse* response) const {
  for (size_t i = 0; i < counters_.size(); ++i) {
    health::v1::WorkerStats* stats = response->add_workers();
    stats->set_worker(static_cast<uint32_t>(i));
    stats->set_core(queue_cores_.empty() ? -1 : queue_cores_[i]);
    stats->set_rpcs_completed(
        counters_[i]->rpcs_completed.load(std::memory_order_relaxed));
    stats->set_call_pool_high_water(
        counters_[i]->high_water.load(std::memory_order_relaxed));
    stats->set_call_pool_allocations(
        counters_[i]->allocations.load(std::memory_order_relaxed));
  }
}

}  // namespace vectra::server
