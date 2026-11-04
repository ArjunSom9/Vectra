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

#include "vectra/client/async_search.h"

#include <algorithm>
#include <memory>
#include <thread>
#include <utility>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"
#include "grpcpp/support/async_unary_call.h"

namespace vectra::client {

class AsyncClient::PendingSearch final : public PendingCall {
 public:
  explicit PendingSearch(SearchCallback done) : done_(std::move(done)) {}

  void Complete() override { std::move(done_)(status_, &response_); }

  grpc::ClientContext context_;
  search::v1::SearchResponse response_;
  grpc::Status status_;
  std::unique_ptr<
      grpc::ClientAsyncResponseReaderInterface<search::v1::SearchResponse>>
      reader_;

 private:
  SearchCallback done_;
};

class AsyncClient::PendingPing final : public PendingCall {
 public:
  explicit PendingPing(PingCallback done) : done_(std::move(done)) {}

  void Complete() override { std::move(done_)(status_); }

  grpc::ClientContext context_;
  health::v1::PingResponse response_;
  grpc::Status status_;
  std::unique_ptr<
      grpc::ClientAsyncResponseReaderInterface<health::v1::PingResponse>>
      reader_;

 private:
  PingCallback done_;
};

AsyncClient::AsyncClient(int polling_threads) {
  for (int i = 0; i < std::max(1, polling_threads); ++i) {
    pollers_.emplace_back(&AsyncClient::Poll, this);
  }
}

AsyncClient::~AsyncClient() {
  queue_.Shutdown();
  for (std::thread& poller : pollers_) poller.join();
}

void AsyncClient::Search(search::v1::SearchService::StubInterface* stub,
                         const search::v1::SearchRequest& request,
                         absl::Duration timeout, SearchCallback done) {
  auto* call = new PendingSearch(std::move(done));
  call->context_.set_deadline(absl::ToChronoTime(absl::Now() + timeout));
  call->reader_ = stub->PrepareAsyncSearch(&call->context_, request, &queue_);
  call->reader_->StartCall();
  call->reader_->Finish(&call->response_, &call->status_, call);
}

void AsyncClient::Ping(health::v1::HealthService::StubInterface* stub,
                       const health::v1::PingRequest& request,
                       absl::Duration timeout, PingCallback done) {
  auto* call = new PendingPing(std::move(done));
  call->context_.set_deadline(absl::ToChronoTime(absl::Now() + timeout));
  call->reader_ = stub->PrepareAsyncPing(&call->context_, request, &queue_);
  call->reader_->StartCall();
  call->reader_->Finish(&call->response_, &call->status_, call);
}

void AsyncClient::Poll() {
  void* tag = nullptr;
  bool ok = false;
  while (queue_.Next(&tag, &ok)) {
    // Client-side Finish always reports ok; failures arrive in the status.
    std::unique_ptr<PendingCall> call(static_cast<PendingCall*>(tag));
    call->Complete();
  }
}

}  // namespace vectra::client
