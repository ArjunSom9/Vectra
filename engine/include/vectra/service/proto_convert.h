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

// Conversions between the wire contract (contracts/proto) and core types.
// This is the engine's single serialization boundary.

#ifndef VECTRA_SERVICE_PROTO_CONVERT_H_
#define VECTRA_SERVICE_PROTO_CONVERT_H_

#include <cstdint>

#include "grpcpp/support/status.h"
#include "vectra/common.pb.h"
#include "vectra/index/property_store.h"
#include "vectra/index/shard_index.h"
#include "vectra/routing/fanout_policy.h"
#include "vectra/routing/geo_math.h"

namespace vectra::service {

routing::LatLng FromProto(const common::v1::LatLng& location);
void ToProto(const routing::LatLng& location, common::v1::LatLng* out);

index::PropertyAttributes FromProto(const common::v1::PropertyAttributes& in);
void ToProto(const index::PropertyAttributes& in,
             common::v1::PropertyAttributes* out);

// Fills a candidate from a shard hit; copies the embedding only when
// `include_embedding` is set.
void ToCandidate(const index::ShardIndex& index, const index::IndexHit& hit,
                 uint32_t shard_id, bool include_embedding,
                 common::v1::Candidate* out);

// Maps a routing failure to the gRPC status a client should see.
grpc::Status ToGrpcStatus(const routing::RoutePlan& plan);

}  // namespace vectra::service

#endif  // VECTRA_SERVICE_PROTO_CONVERT_H_
