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

#ifndef VECTRA_SERVICE_HEALTH_HANDLER_H_
#define VECTRA_SERVICE_HEALTH_HANDLER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "grpcpp/support/status.h"
#include "vectra/health.pb.h"
#include "vectra/index/shard_index.h"

namespace vectra::service {

// Outcome of pinning the index in RAM at start-up.
struct MemoryStatus {
  bool locked = false;
  uint64_t locked_bytes = 0;
  std::string error;
};

// Serves vectra.health.v1.HealthService. Worker statistics are appended by
// the server, which owns the completion queues. Thread-safe.
class HealthHandler {
 public:
  // `index` must outlive the handler.
  HealthHandler(const index::ShardIndex* index, uint32_t shard_id,
                std::vector<std::string> owned_prefixes, MemoryStatus memory,
                std::string build_version);

  grpc::Status Report(const health::v1::ReportRequest& request,
                      health::v1::ReportResponse* response) const;

  // Echoes the payload: the no-op RPC of the A1 transport test.
  grpc::Status Ping(const health::v1::PingRequest& request,
                    health::v1::PingResponse* response) const;

 private:
  const index::ShardIndex* index_;
  uint32_t shard_id_;
  std::vector<std::string> owned_prefixes_;
  MemoryStatus memory_;
  std::string build_version_;
};

}  // namespace vectra::service

#endif  // VECTRA_SERVICE_HEALTH_HANDLER_H_
