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

// Deliverable A2's correctness suite: queries sampled near shard-cell
// boundaries, with random radii, checked against brute-force ground truth
// over the whole corpus.

#include <cstdint>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "testing/in_process_cluster.h"
#include "testing/synthetic_data.h"
#include "vectra/index/property_store.h"
#include "vectra/index/shard_index.h"
#include "vectra/routing/geo_math.h"
#include "vectra/routing/geohash.h"
#include "vectra/routing/routing_table.h"

namespace vectra {
namespace {

constexpr int kDimension = 32;
constexpr size_t kCorpus = 30000;
constexpr int kQueries = 250;
constexpr routing::LatLng kSanFrancisco{37.7749, -122.4194};

class AdjacentRecallTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    places_ = new std::vector<routing::LatLng>(
        testing::PointsAround(kCorpus, kSanFrancisco, 30.0, 2025));
    embeddings_ = new std::vector<float>(
        testing::GeoAwareEmbeddings(*places_, kDimension, 7));
    records_ =
        new std::vector<index::PropertyRecord>(testing::RecordsAt(*places_, 0));
    // Checkerboard of three shards over 4-character cells, so every cell
    // edge is also a shard edge.
    std::set<std::string> cells;
    for (const routing::LatLng& place : *places_) {
      cells.insert(routing::EncodeGeohash(place, 4));
    }
    std::vector<routing::PrefixAssignment> assignments;
    for (const std::string& cell : cells) {
      const routing::CellIndex index =
          routing::Deinterleave(*routing::ParseGeohash(cell));
      assignments.push_back({cell, (index.column + index.row) % 3});
    }
    table_ = new routing::RoutingTable(
        *routing::RoutingTable::Build(4, assignments, nullptr));
    index::IndexOptions options;
    options.dimension = kDimension;
    options.pq_subspaces = 8;
    options.ef_construction = 100;
    options.exhaustive_scan_threshold = 512;
    options.build_threads = 4;
    cluster_ = new testing::InProcessCluster(table_, options, *embeddings_,
                                             *records_, 15.0);
  }

  // Ground truth over the whole corpus: the k nearest embeddings within
  // the radius.
  static std::vector<int64_t> Truth(const float* query,
                                    const routing::LatLng& center,
                                    double radius_km, size_t k) {
    std::vector<bool> allowed(kCorpus);
    for (size_t i = 0; i < kCorpus; ++i) {
      allowed[i] = routing::HaversineKm(center, (*places_)[i]) <= radius_km;
    }
    std::vector<int64_t> ids;
    for (uint32_t row :
         testing::BruteForceKnn(*embeddings_, kDimension, query, k, allowed)) {
      ids.push_back((*records_)[row].id);
    }
    return ids;
  }

  // A query center within 500 m of an edge of a random populated cell.
  static routing::LatLng NearBoundary(std::mt19937_64& rng) {
    const routing::LatLng anchor = (*places_)[rng() % kCorpus];
    const routing::GeoBox box =
        routing::CellBox(routing::EncodeCell(anchor, 4));
    const double offset_deg =
        (0.5 / 111.195) * static_cast<double>(rng() % 1000) / 1000.0;
    switch (rng() % 4) {
      case 0:
        return {box.max_lat - offset_deg, anchor.lng};
      case 1:
        return {box.min_lat + offset_deg, anchor.lng};
      case 2:
        return {anchor.lat, box.max_lng - offset_deg};
      default:
        return {anchor.lat, box.min_lng + offset_deg};
    }
  }

  static std::vector<routing::LatLng>* places_;
  static std::vector<float>* embeddings_;
  static std::vector<index::PropertyRecord>* records_;
  static routing::RoutingTable* table_;
  static testing::InProcessCluster* cluster_;
};

std::vector<routing::LatLng>* AdjacentRecallTest::places_ = nullptr;
std::vector<float>* AdjacentRecallTest::embeddings_ = nullptr;
std::vector<index::PropertyRecord>* AdjacentRecallTest::records_ = nullptr;
routing::RoutingTable* AdjacentRecallTest::table_ = nullptr;
testing::InProcessCluster* AdjacentRecallTest::cluster_ = nullptr;

double RecallOf(const std::vector<int64_t>& truth,
                const std::vector<int64_t>& found) {
  if (truth.empty()) return 1.0;
  const std::set<int64_t> found_set(found.begin(), found.end());
  size_t hits = 0;
  for (int64_t id : truth) hits += found_set.count(id);
  return static_cast<double>(hits) / static_cast<double>(truth.size());
}

TEST_F(AdjacentRecallTest, ExactRoutingFindsEveryBoundaryComparable) {
  std::mt19937_64 rng(1);
  int multi_shard = 0;
  for (int q = 0; q < kQueries; ++q) {
    const routing::LatLng center = NearBoundary(rng);
    const double radius = 0.25 + 9.0 * static_cast<double>(rng() % 1000) / 1e3;
    const float* query = embeddings_->data() + (rng() % kCorpus) * kDimension;
    const testing::InProcessCluster::Result result =
        cluster_->Search(query, center, radius, 10, /*exact=*/true);
    EXPECT_LE(result.cells_intersected, 9);
    multi_shard += result.shards_touched > 1;
    EXPECT_EQ(result.property_ids, Truth(query, center, radius, 10))
        << "center " << center.lat << "," << center.lng << " r=" << radius;
  }
  EXPECT_GT(multi_shard, kQueries / 2);  // The suite really crosses edges.
}

TEST_F(AdjacentRecallTest, ApproximateRoutingMeetsA2RecallTarget) {
  std::mt19937_64 rng(2);
  double recall = 0.0;
  for (int q = 0; q < kQueries; ++q) {
    const routing::LatLng center = NearBoundary(rng);
    const double radius = 0.25 + 9.0 * static_cast<double>(rng() % 1000) / 1e3;
    const float* query = embeddings_->data() + (rng() % kCorpus) * kDimension;
    const testing::InProcessCluster::Result result =
        cluster_->Search(query, center, radius, 10, /*exact=*/false,
                         /*ef=*/128);
    recall += RecallOf(Truth(query, center, radius, 10), result.property_ids);
  }
  // Deliverable A2: recall@10 >= 0.99 over randomly sampled radii.
  EXPECT_GE(recall / kQueries, 0.99);
}

}  // namespace
}  // namespace vectra
