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

#include "vectra/service/ingest_handler.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/support/status.h"
#include "vectra/index/property_store.h"
#include "vectra/routing/geo_math.h"
#include "vectra/service/proto_convert.h"

namespace vectra::service {

IngestHandler::IngestHandler(index::ShardIndex* index,
                             const routing::RoutingTable* table,
                             uint32_t shard_id)
    : index_(index), table_(table), shard_id_(shard_id) {}

grpc::Status IngestHandler::Ingest(const ingest::v1::IngestRequest& request,
                                   ingest::v1::IngestResponse* response) {
  const int dimension = index_->dimension();
  std::vector<float> embeddings;
  std::vector<index::PropertyRecord> records;
  embeddings.reserve(static_cast<size_t>(request.records_size()) *
                     static_cast<size_t>(dimension));
  records.reserve(static_cast<size_t>(request.records_size()));
  for (const common::v1::PropertyRecord& record : request.records()) {
    if (record.embedding_size() != dimension) {
      return grpc::Status(
          grpc::StatusCode::INVALID_ARGUMENT,
          absl::StrCat("record ", record.property_id(), " has ",
                       record.embedding_size(), " embedding values, expected ",
                       dimension));
    }
    const routing::LatLng location = FromProto(record.location());
    if (!routing::IsValidLatLng(location)) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          absl::StrCat("record ", record.property_id(),
                                       " has no valid location"));
    }
    const std::optional<uint32_t> owner = table_->ShardForPoint(location);
    if (!owner.has_value() || *owner != shard_id_) {
      return grpc::Status(
          grpc::StatusCode::FAILED_PRECONDITION,
          absl::StrCat("record ", record.property_id(),
                       " lies in a cell this shard does not own"));
    }
    embeddings.insert(embeddings.end(), record.embedding().begin(),
                      record.embedding().end());
    index::PropertyRecord row;
    row.id = record.property_id();
    row.location = location;
    row.attributes = FromProto(record.attributes());
    records.push_back(row);
  }

  absl::MutexLock lock(&writer_mutex_);
  std::string error;
  if (!index_->Insert(embeddings.data(), records.data(), records.size(),
                      &error)) {
    return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, error);
  }
  response->set_accepted(static_cast<uint32_t>(records.size()));
  response->set_index_size(index_->size());
  return grpc::Status::OK;
}

}  // namespace vectra::service
