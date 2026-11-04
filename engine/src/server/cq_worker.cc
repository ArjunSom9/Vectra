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

#include "vectra/server/cq_worker.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "grpcpp/completion_queue.h"
#include "vectra/server/call_data.h"
#include "vectra/server/core_pin.h"

namespace vectra::server {

void RunCompletionQueue(grpc::ServerCompletionQueue* queue, int core) {
  if (core >= 0) {
    const absl::Status pinned = PinCurrentThreadToCore(core);
    if (!pinned.ok()) LOG(WARNING) << "worker unpinned: " << pinned;
  }
  void* tag = nullptr;
  bool ok = false;
  while (queue->Next(&tag, &ok)) {
    static_cast<CallData*>(tag)->Proceed(ok);
  }
}

}  // namespace vectra::server
