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

#include "vectra/index/shard_index.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "testing/synthetic_data.h"
#include "vectra/index/property_store.h"
#include "vectra/index/search_scratch.h"
#include "vectra/routing/geo_math.h"

namespace vectra::index {
namespace {

constexpr int kDimension = 32;
constexpr size_t kCount = 8000;
constexpr routing::LatLng kSanFrancisco{37.7749, -122.4194};

IndexOptions SmallOptions() {
  IndexOptions options;
  options.dimension = kDimension;
  options.m = 16;
  options.m0 = 32;
  options.ef_construction = 100;
  options.ef_search = 64;
  options.rerank_depth = 64;
  options.pq_subspaces = 8;
  options.pq_kmeans_iterations = 10;
  options.exhaustive_scan_threshold = 256;
  options.build_threads = 2;
  options.max_elements = kCount + 200;
  return options;
}

class ShardIndexTest : public ::testing::Test {
 protected:
  void SetUp() override {
    locations_ = testing::PointsAround(kCount, kSanFrancisco, 20.0, 1);
    embeddings_ = testing::GeoAwareEmbeddings(locations_, kDimension, 2);
    records_ = testing::RecordsAt(locations_, 1000);
    std::string error;
    index_ = ShardIndex::Build(SmallOptions(), embeddings_.data(),
                               records_.data(), kCount, &error);
    ASSERT_NE(index_, nullptr) << error;
  }

  std::vector<uint32_t> Nodes(const std::vector<IndexHit>& hits) const {
    std::vector<uint32_t> nodes;
    for (const IndexHit& hit : hits) nodes.push_back(hit.node);
    return nodes;
  }

  // Ground truth: the k nearest embeddings among nodes within the radius
  // (0: anywhere) that sold before `sold_before` (0: any time).
  std::vector<uint32_t> Truth(const float* query, const routing::LatLng& center,
                              double radius_km, size_t k,
                              int32_t sold_before = 0) const {
    std::vector<bool> allowed(kCount, true);
    for (size_t i = 0; i < kCount; ++i) {
      if (radius_km > 0.0) {
        allowed[i] = routing::HaversineKm(center, locations_[i]) <= radius_km;
      }
      if (sold_before > 0 &&
          records_[i].attributes.last_sale_epoch_day >= sold_before) {
        allowed[i] = false;
      }
    }
    return testing::BruteForceKnn(embeddings_, kDimension, query, k, allowed);
  }

  std::vector<routing::LatLng> locations_;
  std::vector<float> embeddings_;
  std::vector<PropertyRecord> records_;
  std::unique_ptr<ShardIndex> index_;
};

TEST_F(ShardIndexTest, UnfilteredSearchMeetsRecallTarget) {
  SearchScratch scratch;
  double recall = 0.0;
  for (size_t q = 0; q < 100; ++q) {
    const float* query = embeddings_.data() + (q * 79) * kDimension;
    IndexQuery request;
    request.embedding = query;
    request.k = 10;
    std::vector<IndexHit> hits;
    IndexSearchStats stats;
    index_->Search(request, &scratch, &hits, &stats);
    recall += testing::Recall(Truth(query, {}, 0.0, 10), Nodes(hits));
  }
  EXPECT_GE(recall / 100.0, 0.95);
}

TEST_F(ShardIndexTest, RadiusSearchStaysInsideTheDiskAndMeetsRecall) {
  SearchScratch scratch;
  double recall = 0.0;
  int graph_searches = 0;
  for (size_t q = 0; q < 100; ++q) {
    const routing::LatLng center = locations_[q * 61];
    const float* query = embeddings_.data() + (q * 61) * kDimension;
    const double radius = 1.0 + static_cast<double>(q % 5);
    IndexQuery request;
    request.embedding = query;
    request.center = center;
    request.radius_km = radius;
    request.k = 10;
    std::vector<IndexHit> hits;
    IndexSearchStats stats;
    index_->Search(request, &scratch, &hits, &stats);
    graph_searches += stats.exhaustive ? 0 : 1;
    for (const IndexHit& hit : hits) {
      EXPECT_LE(hit.geo_distance_km, radius + 1e-6);
    }
    recall += testing::Recall(Truth(query, center, radius, 10), Nodes(hits));
  }
  EXPECT_GE(recall / 100.0, 0.95);
  EXPECT_GT(graph_searches, 0);  // Larger radii exceed the scan threshold.
}

TEST_F(ShardIndexTest, ExactModeEqualsBruteForce) {
  SearchScratch scratch;
  for (size_t q = 0; q < 20; ++q) {
    const routing::LatLng center = locations_[q * 97];
    const float* query = embeddings_.data() + (q * 13) * kDimension;
    IndexQuery request;
    request.embedding = query;
    request.center = center;
    request.radius_km = 4.0;
    request.k = 10;
    request.exact = true;
    std::vector<IndexHit> hits;
    IndexSearchStats stats;
    index_->Search(request, &scratch, &hits, &stats);
    EXPECT_TRUE(stats.exhaustive);
    EXPECT_EQ(Nodes(hits), Truth(query, center, 4.0, 10));
  }
}

TEST_F(ShardIndexTest, SoldBeforeBoundSeesOnlyEarlierSales) {
  SearchScratch scratch;
  double recall = 0.0;
  int queries = 0;
  for (size_t q = 0; q < 60; ++q) {
    const routing::LatLng center = locations_[q * 113];
    const float* query = embeddings_.data() + (q * 113) * kDimension;
    const int32_t bound = 17000 + static_cast<int32_t>(q * 40);
    for (const bool exact : {true, false}) {
      for (const double radius : {0.0, 3.0, 8.0}) {
        IndexQuery request;
        request.embedding = query;
        request.center = center;
        request.radius_km = radius;
        request.k = 10;
        request.exact = exact;
        request.sold_before_epoch_day = bound;
        std::vector<IndexHit> hits;
        IndexSearchStats stats;
        index_->Search(request, &scratch, &hits, &stats);
        for (const IndexHit& hit : hits) {
          EXPECT_LT(records_[hit.node].attributes.last_sale_epoch_day, bound);
          if (radius > 0.0) EXPECT_LE(hit.geo_distance_km, radius + 1e-6);
        }
        const std::vector<uint32_t> truth =
            Truth(query, center, radius, 10, bound);
        if (exact) {
          EXPECT_EQ(Nodes(hits), truth);
        } else {
          recall += testing::Recall(truth, Nodes(hits));
          ++queries;
        }
      }
    }
  }
  EXPECT_GE(recall / queries, 0.95);
}

TEST_F(ShardIndexTest, SaveAndLoadPreserveResults) {
  const std::string path = ::testing::TempDir() + "/shard.vidx";
  std::string error;
  ASSERT_TRUE(index_->Save(path, &error)) << error;
  std::unique_ptr<ShardIndex> loaded = ShardIndex::Load(path, &error);
  ASSERT_NE(loaded, nullptr) << error;
  EXPECT_EQ(loaded->size(), index_->size());
  EXPECT_EQ(loaded->capacity(), index_->capacity());
  SearchScratch scratch;
  for (size_t q = 0; q < 25; ++q) {
    IndexQuery request;
    request.embedding = embeddings_.data() + (q * 31) * kDimension;
    request.center = locations_[q * 31];
    request.radius_km = q % 2 == 0 ? 0.0 : 3.0;
    request.k = 10;
    std::vector<IndexHit> before;
    std::vector<IndexHit> after;
    IndexSearchStats stats;
    index_->Search(request, &scratch, &before, &stats);
    loaded->Search(request, &scratch, &after, &stats);
    EXPECT_EQ(Nodes(before), Nodes(after));
  }
  EXPECT_EQ(loaded->properties().id(5), index_->properties().id(5));
}

TEST_F(ShardIndexTest, InsertedRecordsBecomeSearchable) {
  const routing::LatLng spot{37.80, -122.27};  // Oakland.
  const std::vector<routing::LatLng> places =
      testing::PointsAround(100, spot, 0.5, 9);
  const std::vector<float> vectors =
      testing::GeoAwareEmbeddings(places, kDimension, 10);
  const std::vector<PropertyRecord> records =
      testing::RecordsAt(places, 900000);
  std::string error;
  ASSERT_TRUE(index_->Insert(vectors.data(), records.data(), 100, &error))
      << error;
  EXPECT_EQ(index_->size(), kCount + 100);

  SearchScratch scratch;
  IndexQuery request;
  request.embedding = vectors.data() + 42 * kDimension;
  request.center = places[42];
  request.radius_km = 1.0;
  request.k = 1;
  std::vector<IndexHit> hits;
  IndexSearchStats stats;
  index_->Search(request, &scratch, &hits, &stats);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(index_->properties().id(hits[0].node), 900042);

  // Capacity is kCount + 200: another 150 cannot fit, and nothing is added.
  const std::vector<PropertyRecord> more =
      testing::RecordsAt(testing::PointsAround(150, spot, 0.5, 11), 1);
  const std::vector<float> more_vectors(150 * kDimension, 0.5f);
  EXPECT_FALSE(index_->Insert(more_vectors.data(), more.data(), 150, &error));
  EXPECT_EQ(index_->size(), kCount + 100);
}

TEST_F(ShardIndexTest, RejectsInvalidLocations) {
  std::vector<PropertyRecord> records = testing::RecordsAt({{95.0, 0.0}}, 1);
  const std::vector<float> vector(kDimension, 0.0f);
  std::string error;
  EXPECT_FALSE(index_->Insert(vector.data(), records.data(), 1, &error));
  EXPECT_NE(error.find("invalid location"), std::string::npos);
}

}  // namespace
}  // namespace vectra::index
