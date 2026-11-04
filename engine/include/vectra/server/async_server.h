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

#ifndef VECTRA_SERVER_ASYNC_SERVER_H_
#define VECTRA_SERVER_ASYNC_SERVER_H_

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "grpcpp/completion_queue.h"
#include "grpcpp/server.h"
#include "vectra/health.grpc.pb.h"
#include "vectra/ingest.grpc.pb.h"
#include "vectra/search.grpc.pb.h"
#include "vectra/server/call_data.h"
#include "vectra/service/health_handler.h"
#include "vectra/service/ingest_handler.h"
#include "vectra/service/search_handler.h"

namespace vectra::server {

struct ServerOptions {
  std::string listen_address = "0.0.0.0:7101";
  // One completion queue per listed core, its pollers pinned to that core.
  // Empty: one queue per hardware thread, unpinned.
  std::vector<int> worker_cores;
  // Pollers per queue: 1 gives the thread-per-core model of the design;
  // gRPC's performance guide reports 2 as the best trade-off [27].
  int threads_per_queue = 1;
  // Listeners pre-registered per method per queue [27].
  int requests_per_queue = 64;
  int max_receive_message_bytes = 16 << 20;
};

// The handlers a shard serves; not owned, must outlive the server.
struct ShardHandlers {
  service::SearchHandler* search = nullptr;
  service::IngestHandler* ingest = nullptr;
  service::HealthHandler* health = nullptr;
};

// The shard's gRPC server on the asynchronous completion-queue API [19]: no
// thread blocks inside the runtime, and every request is parsed, searched and
// answered on the worker thread whose queue received it.
class AsyncServer {
 public:
  AsyncServer(ServerOptions options, ShardHandlers handlers);
  AsyncServer(const AsyncServer&) = delete;
  AsyncServer& operator=(const AsyncServer&) = delete;
  ~AsyncServer();

  // Binds, starts the workers, and primes every queue with listeners.
  absl::Status Start();

  // The bound port (useful with a listen address ending in ":0").
  int port() const { return port_; }

  // Stops accepting work, lets in-flight RPCs finish for up to `grace`,
  // shuts down and drains every queue, and joins the workers. Idempotent.
  void Shutdown(absl::Duration grace);

 private:
  using SearchService = search::v1::SearchService::AsyncService;
  using IngestService = ingest::v1::IngestService::AsyncService;
  using HealthService = health::v1::HealthService::AsyncService;

  std::vector<std::unique_ptr<MethodBinding>> BindingsFor(
      grpc::ServerCompletionQueue* queue);
  void FillWorkerStats(health::v1::ReportResponse* response) const;

  ServerOptions options_;
  ShardHandlers handlers_;
  SearchService search_service_;
  IngestService ingest_service_;
  HealthService health_service_;
  // Type-erased handlers shared by every queue's calls.
  UnaryCall<SearchService, search::v1::SearchRequest,
            search::v1::SearchResponse>::Handler search_fn_;
  UnaryCall<IngestService, ingest::v1::IngestRequest,
            ingest::v1::IngestResponse>::Handler ingest_fn_;
  UnaryCall<HealthService, health::v1::ReportRequest,
            health::v1::ReportResponse>::Handler report_fn_;
  UnaryCall<HealthService, health::v1::PingRequest,
            health::v1::PingResponse>::Handler ping_fn_;
  std::unique_ptr<grpc::Server> server_;
  std::vector<std::unique_ptr<grpc::ServerCompletionQueue>> queues_;
  std::vector<int> queue_cores_;
  std::vector<std::unique_ptr<QueueCounters>> counters_;
  std::vector<std::unique_ptr<CallPool>> pools_;
  std::vector<std::thread> workers_;
  int port_ = 0;
  bool running_ = false;
};

}  // namespace vectra::server

#endif  // VECTRA_SERVER_ASYNC_SERVER_H_
