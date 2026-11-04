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
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "gtest/gtest.h"
#include "testing/synthetic_data.h"
#include "vectra/client/async_search.h"
#include "vectra/client/channel_pool.h"
#include "vectra/health.grpc.pb.h"
#include "vectra/index/search_scratch.h"
#include "vectra/index/shard_index.h"
#include "vectra/search.grpc.pb.h"
#include "vectra/server/async_server.h"
#include "vectra/service/health_handler.h"
#include "vectra/service/ingest_handler.h"
#include "vectra/service/search_handler.h"

namespace vectra {
namespace {

constexpr int kDimension = 32;
constexpr size_t kCount = 3000;
constexpr routing::LatLng kCenter{37.7749, -122.4194};

class SearchRoundTripTest : public ::testing::Test {
 protected:
  void SetUp() override {
    locations_ = testing::PointsAround(kCount, kCenter, 10.0, 1);
    embeddings_ = testing::GeoAwareEmbeddings(locations_, kDimension, 2);
    const std::vector<index::PropertyRecord> records =
        testing::RecordsAt(locations_, 500);
    index::IndexOptions options;
    options.dimension = kDimension;
    options.pq_subspaces = 8;
    options.ef_construction = 64;
    std::string error;
    index_ = index::ShardIndex::Build(options, embeddings_.data(),
                                      records.data(), kCount, &error);
    ASSERT_NE(index_, nullptr) << error;
    table_ = std::make_unique<routing::RoutingTable>(
        *routing::RoutingTable::Build(4, {{"9q", 1}}, nullptr));
    search_ = std::make_unique<service::SearchHandler>(index_.get(), 1,
                                                       service::SearchLimits());
    ingest_ =
        std::make_unique<service::IngestHandler>(index_.get(), table_.get(), 1);
    health_ = std::make_unique<service::HealthHandler>(
        index_.get(), 1, std::vector<std::string>{"9q"},
        service::MemoryStatus(), "test");
    server::ServerOptions options_server;
    options_server.listen_address = "127.0.0.1:0";
    options_server.worker_cores = {-1, -1};  // Two unpinned queues.
    options_server.requests_per_queue = 4;
    server_ = std::make_unique<server::AsyncServer>(
        options_server,
        server::ShardHandlers{search_.get(), ingest_.get(), health_.get()});
    ASSERT_TRUE(server_->Start().ok());
    target_ = absl::StrCat("127.0.0.1:", server_->port());
  }

  void TearDown() override { server_->Shutdown(absl::Seconds(1)); }

  search::v1::SearchRequest Request(size_t row, double radius_km) const {
    search::v1::SearchRequest request;
    const float* embedding = embeddings_.data() + row * kDimension;
    request.mutable_query_embedding()->Assign(embedding,
                                              embedding + kDimension);
    request.mutable_location()->set_latitude(locations_[row].lat);
    request.mutable_location()->set_longitude(locations_[row].lng);
    request.set_radius_km(radius_km);
    request.set_top_k(10);
    return request;
  }

  std::vector<routing::LatLng> locations_;
  std::vector<float> embeddings_;
  std::unique_ptr<index::ShardIndex> index_;
  std::unique_ptr<routing::RoutingTable> table_;
  std::unique_ptr<service::SearchHandler> search_;
  std::unique_ptr<service::IngestHandler> ingest_;
  std::unique_ptr<service::HealthHandler> health_;
  std::unique_ptr<server::AsyncServer> server_;
  std::string target_;
};

TEST_F(SearchRoundTripTest, AsyncSearchMatchesDirectIndexSearch) {
  client::ChannelPool pool(target_,
                           client::ChannelPoolOptions{4, 10000, 16 << 20});
  ASSERT_TRUE(pool.WaitForConnected(absl::Now() + absl::Seconds(5)));
  client::AsyncClient client(2);
  index::SearchScratch scratch;
  for (size_t row = 0; row < 50; ++row) {
    const search::v1::SearchRequest request = Request(row * 37, 2.0);
    absl::Notification done;
    grpc::Status status;
    search::v1::SearchResponse response;
    client.Search(pool.NextSearchStub(), request, absl::Seconds(5),
                  [&](const grpc::Status& s, search::v1::SearchResponse* r) {
                    status = s;
                    response = *r;
                    done.Notify();
                  });
    done.WaitForNotification();
    ASSERT_TRUE(status.ok()) << status.error_message();

    index::IndexQuery query;
    query.embedding = request.query_embedding().data();
    query.center = locations_[row * 37];
    query.radius_km = 2.0;
    query.k = 10;
    std::vector<index::IndexHit> hits;
    index::IndexSearchStats stats;
    index_->Search(query, &scratch, &hits, &stats);
    ASSERT_EQ(response.candidates_size(), static_cast<int>(hits.size()));
    for (int i = 0; i < response.candidates_size(); ++i) {
      EXPECT_EQ(response.candidates(i).property_id(),
                index_->properties().id(hits[static_cast<size_t>(i)].node));
      EXPECT_LE(response.candidates(i).geo_distance_km(), 2.0 + 1e-6);
      EXPECT_EQ(response.candidates(i).shard_id(), 1u);
    }
  }
}

TEST_F(SearchRoundTripTest, RejectsMalformedRequests) {
  auto stub = search::v1::SearchService::NewStub(
      grpc::CreateChannel(target_, grpc::InsecureChannelCredentials()));
  search::v1::SearchRequest request = Request(0, 2.0);
  request.mutable_query_embedding()->RemoveLast();
  grpc::ClientContext context;
  search::v1::SearchResponse response;
  EXPECT_EQ(stub->Search(&context, request, &response).error_code(),
            grpc::StatusCode::INVALID_ARGUMENT);

  request = Request(0, 500.0);  // Beyond the policy radius.
  grpc::ClientContext context2;
  EXPECT_EQ(stub->Search(&context2, request, &response).error_code(),
            grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(SearchRoundTripTest, PingEchoesAndReportCountsWork) {
  auto stub = health::v1::HealthService::NewStub(
      grpc::CreateChannel(target_, grpc::InsecureChannelCredentials()));
  for (int i = 0; i < 20; ++i) {
    health::v1::PingRequest ping;
    ping.set_payload(absl::StrCat("probe-", i));
    health::v1::PingResponse pong;
    grpc::ClientContext context;
    ASSERT_TRUE(stub->Ping(&context, ping, &pong).ok());
    EXPECT_EQ(pong.payload(), ping.payload());
  }
  grpc::ClientContext context;
  health::v1::ReportResponse report;
  ASSERT_TRUE(
      stub->Report(&context, health::v1::ReportRequest(), &report).ok());
  EXPECT_EQ(report.shard_id(), 1u);
  EXPECT_EQ(report.index().num_vectors(), kCount);
  EXPECT_EQ(report.workers_size(), 2);
  uint64_t completed = 0;
  for (const health::v1::WorkerStats& worker : report.workers()) {
    completed += worker.rpcs_completed();
  }
  EXPECT_GE(completed, 20u);
}

}  // namespace
}  // namespace vectra
