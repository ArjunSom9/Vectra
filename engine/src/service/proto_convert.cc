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

#include "vectra/service/proto_convert.h"

#include <cstdint>

#include "grpcpp/support/status.h"

namespace vectra::service {

routing::LatLng FromProto(const common::v1::LatLng& location) {
  return {location.latitude(), location.longitude()};
}

void ToProto(const routing::LatLng& location, common::v1::LatLng* out) {
  out->set_latitude(location.lat);
  out->set_longitude(location.lng);
}

index::PropertyAttributes FromProto(const common::v1::PropertyAttributes& in) {
  index::PropertyAttributes out;
  out.living_area_sqft = in.living_area_sqft();
  out.lot_area_sqft = in.lot_area_sqft();
  out.bedrooms = in.bedrooms();
  out.bathrooms = in.bathrooms();
  out.year_built = in.year_built();
  out.last_sale_price_usd = in.last_sale_price_usd();
  out.last_sale_epoch_day = in.last_sale_epoch_day();
  return out;
}

void ToProto(const index::PropertyAttributes& in,
             common::v1::PropertyAttributes* out) {
  out->set_living_area_sqft(in.living_area_sqft);
  out->set_lot_area_sqft(in.lot_area_sqft);
  out->set_bedrooms(in.bedrooms);
  out->set_bathrooms(in.bathrooms);
  out->set_year_built(in.year_built);
  out->set_last_sale_price_usd(in.last_sale_price_usd);
  out->set_last_sale_epoch_day(in.last_sale_epoch_day);
}

void ToCandidate(const index::ShardIndex& index, const index::IndexHit& hit,
                 uint32_t shard_id, bool include_embedding,
                 common::v1::Candidate* out) {
  const index::PropertyStore& properties = index.properties();
  out->set_property_id(properties.id(hit.node));
  out->set_squared_l2(hit.squared_l2);
  out->set_geo_distance_km(hit.geo_distance_km);
  ToProto(properties.location(hit.node), out->mutable_location());
  ToProto(properties.attributes(hit.node), out->mutable_attributes());
  out->set_squared_norm(index.vectors().SquaredNorm(hit.node));
  out->set_geohash6(properties.cell6(hit.node));
  out->set_shard_id(shard_id);
  if (include_embedding) {
    const float* row = index.vectors().Row(hit.node);
    out->mutable_embedding()->Assign(row, row + index.dimension());
  }
}

grpc::Status ToGrpcStatus(const routing::RoutePlan& plan) {
  switch (plan.status) {
    case routing::PlanStatus::kOk:
      return grpc::Status::OK;
    case routing::PlanStatus::kInvalidCoordinate:
    case routing::PlanStatus::kInvalidRadius:
    case routing::PlanStatus::kRadiusExceedsBound:
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, plan.message);
  }
  return grpc::Status(grpc::StatusCode::INTERNAL, "unknown routing status");
}

}  // namespace vectra::service
