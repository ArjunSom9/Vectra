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

#include "vectra/service/health_handler.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "grpcpp/support/status.h"

namespace vectra::service {

HealthHandler::HealthHandler(const index::ShardIndex* index, uint32_t shard_id,
                             std::vector<std::string> owned_prefixes,
                             MemoryStatus memory, std::string build_version)
    : index_(index),
      shard_id_(shard_id),
      owned_prefixes_(std::move(owned_prefixes)),
      memory_(std::move(memory)),
      build_version_(std::move(build_version)) {}

grpc::Status HealthHandler::Report(const health::v1::ReportRequest& request,
                                   health::v1::ReportResponse* response) const {
  static_cast<void>(request);  // The request carries no fields.
  response->set_shard_id(shard_id_);
  for (const std::string& prefix : owned_prefixes_) {
    response->add_owned_prefixes(prefix);
  }
  health::v1::IndexStats* stats = response->mutable_index();
  stats->set_num_vectors(index_->size());
  stats->set_capacity(index_->capacity());
  stats->set_dimension(static_cast<uint32_t>(index_->dimension()));
  stats->set_pq_subspaces(
      static_cast<uint32_t>(index_->options().pq_subspaces));
  stats->set_hnsw_m(static_cast<uint32_t>(index_->options().m));
  stats->set_max_level(
      static_cast<uint32_t>(std::max(0, index_->graph().entry().level)));
  stats->set_index_bytes(index_->ResidentBytes());
  health::v1::MemoryStats* memory = response->mutable_memory();
  memory->set_locked(memory_.locked);
  memory->set_locked_bytes(memory_.locked_bytes);
  memory->set_lock_error(memory_.error);
  response->set_build_version(build_version_);
  return grpc::Status::OK;
}

grpc::Status HealthHandler::Ping(const health::v1::PingRequest& request,
                                 health::v1::PingResponse* response) const {
  response->set_payload(request.payload());
  return grpc::Status::OK;
}

}  // namespace vectra::service
