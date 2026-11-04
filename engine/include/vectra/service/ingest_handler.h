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

#ifndef VECTRA_SERVICE_INGEST_HANDLER_H_
#define VECTRA_SERVICE_INGEST_HANDLER_H_

#include <cstdint>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/support/status.h"
#include "vectra/index/shard_index.h"
#include "vectra/ingest.pb.h"
#include "vectra/routing/routing_table.h"

namespace vectra::service {

// Serves vectra.ingest.v1.IngestService/Ingest. Batches are validated in full
// (dimension, coordinate, and cell ownership against the routing table)
// before anything is written, then applied by a single writer while searches
// continue. Thread-safe.
class IngestHandler {
 public:
  // Neither pointer is owned; both must outlive the handler.
  IngestHandler(index::ShardIndex* index, const routing::RoutingTable* table,
                uint32_t shard_id);

  grpc::Status Ingest(const ingest::v1::IngestRequest& request,
                      ingest::v1::IngestResponse* response);

 private:
  index::ShardIndex* index_;
  const routing::RoutingTable* table_;
  uint32_t shard_id_;
  // ShardIndex::Insert requires a single writer.
  absl::Mutex writer_mutex_;
};

}  // namespace vectra::service

#endif  // VECTRA_SERVICE_INGEST_HANDLER_H_
