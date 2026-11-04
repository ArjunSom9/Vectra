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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "gtest/gtest.h"
#include "testing/synthetic_data.h"
#include "vectra/index/shard_index.h"
#include "vectra/ingest.grpc.pb.h"
#include "vectra/search.grpc.pb.h"
#include "vectra/server/async_server.h"
#include "vectra/service/health_handler.h"
#include "vectra/service/ingest_handler.h"
#include "vectra/service/search_handler.h"

namespace vectra {
namespace {

constexpr int kDimension = 16;
constexpr routing::LatLng kCenter{37.7749, -122.4194};

class IngestRoundTripTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<routing::LatLng> locations =
        testing::PointsAround(1000, kCenter, 5.0, 3);
    const std::vector<float> embeddings =
        testing::GeoAwareEmbeddings(locations, kDimension, 4);
    const std::vector<index::PropertyRecord> records =
        testing::RecordsAt(locations, 1);
    index::IndexOptions options;
    options.dimension = kDimension;
    options.pq_subspaces = 4;
    options.ef_construction = 64;
    options.max_elements = 1500;
    std::string error;
    index_ = index::ShardIndex::Build(options, embeddings.data(),
                                      records.data(), 1000, &error);
    ASSERT_NE(index_, nullptr) << error;
    // This shard owns the 9q8 cells; San Francisco is in 9q8y.
    table_ = std::make_unique<routing::RoutingTable>(
        *routing::RoutingTable::Build(4, {{"9q8", 7}, {"dr5", 8}}, nullptr));
    search_ = std::make_unique<service::SearchHandler>(index_.get(), 7,
                                                       service::SearchLimits());
    ingest_ =
        std::make_unique<service::IngestHandler>(index_.get(), table_.get(), 7);
    health_ = std::make_unique<service::HealthHandler>(
        index_.get(), 7, std::vector<std::string>{"9q8"},
        service::MemoryStatus(), "test");
    server::ServerOptions server_options;
    server_options.listen_address = "127.0.0.1:0";
    server_options.worker_cores = {-1};
    server_options.requests_per_queue = 2;
    server_ = std::make_unique<server::AsyncServer>(
        server_options,
        server::ShardHandlers{search_.get(), ingest_.get(), health_.get()});
    ASSERT_TRUE(server_->Start().ok());
    channel_ = grpc::CreateChannel(absl::StrCat("127.0.0.1:", server_->port()),
                                   grpc::InsecureChannelCredentials());
  }

  void TearDown() override { server_->Shutdown(absl::Seconds(1)); }

  static common::v1::PropertyRecord Record(int64_t id,
                                           const routing::LatLng& where,
                                           float fill) {
    common::v1::PropertyRecord record;
    record.set_property_id(id);
    record.mutable_location()->set_latitude(where.lat);
    record.mutable_location()->set_longitude(where.lng);
    record.mutable_attributes()->set_bedrooms(3);
    for (int d = 0; d < kDimension; ++d) record.add_embedding(fill);
    return record;
  }

  std::unique_ptr<index::ShardIndex> index_;
  std::unique_ptr<routing::RoutingTable> table_;
  std::unique_ptr<service::SearchHandler> search_;
  std::unique_ptr<service::IngestHandler> ingest_;
  std::unique_ptr<service::HealthHandler> health_;
  std::unique_ptr<server::AsyncServer> server_;
  std::shared_ptr<grpc::Channel> channel_;
};

TEST_F(IngestRoundTripTest, IngestedRecordsAreServed) {
  auto ingest = ingest::v1::IngestService::NewStub(channel_);
  ingest::v1::IngestRequest request;
  *request.add_records() = Record(424242, kCenter, 9.0f);
  ingest::v1::IngestResponse response;
  grpc::ClientContext context;
  const grpc::Status status = ingest->Ingest(&context, request, &response);
  ASSERT_TRUE(status.ok()) << status.error_message();
  EXPECT_EQ(response.accepted(), 1u);
  EXPECT_EQ(response.index_size(), 1001u);

  auto search = search::v1::SearchService::NewStub(channel_);
  search::v1::SearchRequest query;
  for (int d = 0; d < kDimension; ++d) query.add_query_embedding(9.0f);
  query.mutable_location()->set_latitude(kCenter.lat);
  query.mutable_location()->set_longitude(kCenter.lng);
  query.set_radius_km(1.0);
  query.set_top_k(1);
  query.set_include_embeddings(true);
  search::v1::SearchResponse found;
  grpc::ClientContext search_context;
  ASSERT_TRUE(search->Search(&search_context, query, &found).ok());
  ASSERT_EQ(found.candidates_size(), 1);
  EXPECT_EQ(found.candidates(0).property_id(), 424242);
  EXPECT_EQ(found.candidates(0).embedding_size(), kDimension);
  EXPECT_FLOAT_EQ(found.candidates(0).squared_norm(), 81.0f * kDimension);
}

TEST_F(IngestRoundTripTest, RejectsRecordsOutsideOwnedCells) {
  auto ingest = ingest::v1::IngestService::NewStub(channel_);
  ingest::v1::IngestRequest request;
  *request.add_records() = Record(1, kCenter, 1.0f);
  *request.add_records() = Record(2, {40.7128, -74.0060}, 1.0f);  // New York.
  ingest::v1::IngestResponse response;
  grpc::ClientContext context;
  EXPECT_EQ(ingest->Ingest(&context, request, &response).error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_EQ(index_->size(), 1000u);  // All-or-nothing.
}

}  // namespace
}  // namespace vectra
