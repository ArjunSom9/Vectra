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

#include "vectra/server/call_data.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"

namespace vectra::server {

CallPool::CallPool(std::vector<std::unique_ptr<MethodBinding>> bindings,
                   QueueCounters* counters)
    : bindings_(std::move(bindings)),
      counters_(counters),
      idle_(bindings_.size()) {}

CallPool::~CallPool() = default;

void CallPool::Prime(int per_method) {
  for (int method = 0; method < static_cast<int>(bindings_.size()); ++method) {
    for (int i = 0; i < per_method; ++i) ArmAnother(method);
  }
  // Start-up allocations are expected; only later ones signal pool pressure.
  counters_->allocations.store(0, std::memory_order_relaxed);
}

RecyclableCall* CallPool::Acquire(int method) {
  RecyclableCall* call = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    std::vector<RecyclableCall*>& idle = idle_[static_cast<size_t>(method)];
    if (!idle.empty()) {
      call = idle.back();
      idle.pop_back();
    } else {
      owned_.push_back(
          bindings_[static_cast<size_t>(method)]->NewCall(method, this));
      call = owned_.back().get();
      counters_->allocations.fetch_add(1, std::memory_order_relaxed);
    }
  }
  const uint64_t in_flight =
      counters_->in_flight.fetch_add(1, std::memory_order_relaxed) + 1;
  uint64_t high = counters_->high_water.load(std::memory_order_relaxed);
  while (in_flight > high && !counters_->high_water.compare_exchange_weak(
                                 high, in_flight, std::memory_order_relaxed)) {
  }
  return call;
}

void CallPool::ArmAnother(int method) {
  if (stopping_.load(std::memory_order_acquire)) return;
  Acquire(method)->Arm();
}

void CallPool::Release(RecyclableCall* call) {
  counters_->in_flight.fetch_sub(1, std::memory_order_relaxed);
  absl::MutexLock lock(&mutex_);
  idle_[static_cast<size_t>(call->method())].push_back(call);
}

}  // namespace vectra::server
