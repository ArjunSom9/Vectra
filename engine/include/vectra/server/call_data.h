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

// The per-RPC state machine of the asynchronous server.
//
// This is the CallData pattern of the gRPC asynchronous tutorial [19]: every
// in-flight RPC is an object whose own address is the void* tag of each
// asynchronous operation it starts, so a completion-queue event dispatches
// with one cast, and a fresh listener is armed before the current request is
// processed. Two production changes: finished calls are recycled through a
// per-queue pool instead of `delete this`, and each queue is primed with many
// listeners per method, as gRPC's performance guide advises [27].

#ifndef VECTRA_SERVER_CALL_DATA_H_
#define VECTRA_SERVER_CALL_DATA_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/completion_queue.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/async_unary_call.h"
#include "grpcpp/support/status.h"

namespace vectra::server {

// Telemetry for one completion queue, readable from any thread.
struct QueueCounters {
  std::atomic<uint64_t> rpcs_completed{0};
  std::atomic<uint64_t> in_flight{0};
  std::atomic<uint64_t> high_water{0};
  std::atomic<uint64_t> allocations{0};
};

// One in-flight RPC.
class CallData {
 public:
  virtual ~CallData() = default;

  // Advances the state machine after the operation tagged with `this`
  // completes. `ok` is false when the operation was cancelled: server
  // shutdown, an expired deadline, or a vanished client.
  virtual void Proceed(bool ok) = 0;
};

// A CallData that is re-armed for another request instead of destroyed.
class RecyclableCall : public CallData {
 public:
  explicit RecyclableCall(int method) : method_(method) {}

  // Resets per-call state and asks gRPC for the next request of the method.
  virtual void Arm() = 0;

  int method() const { return method_; }

 private:
  const int method_;
};

class CallPool;

// Creates calls for one RPC method on one completion queue.
class MethodBinding {
 public:
  virtual ~MethodBinding() = default;
  virtual std::unique_ptr<RecyclableCall> NewCall(int method,
                                                  CallPool* pool) = 0;
};

// Owns every CallData of one completion queue and recycles finished calls.
// Shared by the threads polling that queue; the lock is uncontended with one
// thread per queue.
class CallPool {
 public:
  CallPool(std::vector<std::unique_ptr<MethodBinding>> bindings,
           QueueCounters* counters);
  CallPool(const CallPool&) = delete;
  CallPool& operator=(const CallPool&) = delete;
  ~CallPool();

  // Arms `per_method` listeners for every method.
  void Prime(int per_method);

  // Arms one more listener for `method`, reusing an idle call if possible.
  // Does nothing once StopArming has been called.
  void ArmAnother(int method);

  // Takes back a call whose last operation has completed.
  void Release(RecyclableCall* call);

  // Called before server shutdown so no new listeners race the drain.
  void StopArming() { stopping_.store(true, std::memory_order_release); }

  QueueCounters* counters() { return counters_; }

 private:
  RecyclableCall* Acquire(int method);

  std::vector<std::unique_ptr<MethodBinding>> bindings_;
  QueueCounters* counters_;
  std::atomic<bool> stopping_{false};
  absl::Mutex mutex_;
  std::vector<std::vector<RecyclableCall*>> idle_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::unique_ptr<RecyclableCall>> owned_ ABSL_GUARDED_BY(mutex_);
};

// Serves one unary method (`Request` -> `Response`) of an async `Service`.
template <typename Service, typename Request, typename Response>
class UnaryCall final : public RecyclableCall {
 public:
  using RequestMethod = void (Service::*)(
      grpc::ServerContext*, Request*,
      grpc::ServerAsyncResponseWriter<Response>*, grpc::CompletionQueue*,
      grpc::ServerCompletionQueue*, void*);
  using Handler = std::function<grpc::Status(const Request&, Response*)>;

  // None of the pointers are owned; all must outlive the call.
  UnaryCall(int method, Service* service, RequestMethod request_method,
            const Handler* handler, grpc::ServerCompletionQueue* queue,
            CallPool* pool)
      : RecyclableCall(method),
        service_(service),
        request_method_(request_method),
        handler_(handler),
        queue_(queue),
        pool_(pool) {}

  void Arm() override {
    state_ = State::kListening;
    request_.Clear();
    response_.Clear();
    // A ServerContext serves exactly one RPC, so it is rebuilt in place.
    responder_.reset();
    context_.emplace();
    responder_.emplace(&*context_);
    (service_->*request_method_)(&*context_, &request_, &*responder_, queue_,
                                 queue_, this);
  }

  void Proceed(bool ok) override {
    if (state_ == State::kListening) {
      if (!ok) {  // The server is shutting down; nothing was received.
        pool_->Release(this);
        return;
      }
      // Re-arm before doing any work, so this queue never goes deaf.
      pool_->ArmAnother(method());
      const grpc::Status status = (*handler_)(request_, &response_);
      state_ = State::kFinishing;
      if (status.ok()) {
        responder_->Finish(response_, status, this);
      } else {
        responder_->FinishWithError(status, this);
      }
      return;
    }
    pool_->counters()->rpcs_completed.fetch_add(1, std::memory_order_relaxed);
    pool_->Release(this);
  }

 private:
  enum class State { kListening, kFinishing };

  Service* service_;
  RequestMethod request_method_;
  const Handler* handler_;
  grpc::ServerCompletionQueue* queue_;
  CallPool* pool_;
  State state_ = State::kListening;
  std::optional<grpc::ServerContext> context_;
  std::optional<grpc::ServerAsyncResponseWriter<Response>> responder_;
  Request request_;
  Response response_;
};

// MethodBinding for UnaryCall.
template <typename Service, typename Request, typename Response>
class UnaryBinding final : public MethodBinding {
 public:
  using Call = UnaryCall<Service, Request, Response>;

  UnaryBinding(Service* service, typename Call::RequestMethod request_method,
               const typename Call::Handler* handler,
               grpc::ServerCompletionQueue* queue)
      : service_(service),
        request_method_(request_method),
        handler_(handler),
        queue_(queue) {}

  std::unique_ptr<RecyclableCall> NewCall(int method, CallPool* pool) override {
    return std::make_unique<Call>(method, service_, request_method_, handler_,
                                  queue_, pool);
  }

 private:
  Service* service_;
  typename Call::RequestMethod request_method_;
  const typename Call::Handler* handler_;
  grpc::ServerCompletionQueue* queue_;
};

}  // namespace vectra::server

#endif  // VECTRA_SERVER_CALL_DATA_H_
