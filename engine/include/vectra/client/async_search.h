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

#ifndef VECTRA_CLIENT_ASYNC_SEARCH_H_
#define VECTRA_CLIENT_ASYNC_SEARCH_H_

#include <memory>
#include <thread>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/time/time.h"
#include "grpcpp/completion_queue.h"
#include "grpcpp/support/status.h"
#include "vectra/health.grpc.pb.h"
#include "vectra/search.grpc.pb.h"

namespace vectra::client {

// Issues unary RPCs on the completion-queue async API and runs each
// completion callback on one of the client's polling threads. Callbacks must
// not block. Thread-safe; destruction waits for outstanding calls.
class AsyncClient {
 public:
  using SearchCallback =
      absl::AnyInvocable<void(const grpc::Status&,
                              search::v1::SearchResponse*) &&>;
  using PingCallback = absl::AnyInvocable<void(const grpc::Status&) &&>;

  explicit AsyncClient(int polling_threads = 1);
  AsyncClient(const AsyncClient&) = delete;
  AsyncClient& operator=(const AsyncClient&) = delete;
  ~AsyncClient();

  void Search(search::v1::SearchService::StubInterface* stub,
              const search::v1::SearchRequest& request, absl::Duration timeout,
              SearchCallback done);

  void Ping(health::v1::HealthService::StubInterface* stub,
            const health::v1::PingRequest& request, absl::Duration timeout,
            PingCallback done);

 private:
  // One in-flight call; its address is the completion-queue tag.
  class PendingCall {
   public:
    virtual ~PendingCall() = default;
    virtual void Complete() = 0;
  };
  class PendingSearch;
  class PendingPing;

  void Poll();

  grpc::CompletionQueue queue_;
  std::vector<std::thread> pollers_;
};

}  // namespace vectra::client

#endif  // VECTRA_CLIENT_ASYNC_SEARCH_H_
