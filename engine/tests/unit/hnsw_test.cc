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

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "testing/synthetic_data.h"
#include "vectra/index/flat_store.h"
#include "vectra/index/hnsw_builder.h"
#include "vectra/index/hnsw_graph.h"
#include "vectra/index/hnsw_searcher.h"
#include "vectra/index/pq_codec.h"
#include "vectra/index/search_scratch.h"
#include "vectra/index/simd_distance.h"

namespace vectra::index {
namespace {

constexpr int kDimension = 64;
constexpr size_t kCount = 20000;
constexpr size_t kQueries = 200;
constexpr int kSubspaces = 16;

// A small but realistic index: a Gaussian mixture, PQ codes, and the graph,
// with queries drawn from the same distribution.
class HnswFixture {
 public:
  explicit HnswFixture(int build_threads, size_t capacity = kCount) {
    const std::vector<float> all =
        testing::ClusteredVectors(kCount + kQueries, kDimension, 64, 0.12f, 42);
    data_.assign(all.begin(), all.begin() + kCount * kDimension);
    queries_.assign(all.begin() + kCount * kDimension, all.end());
    store_.emplace(*FlatStore::Create(kDimension, capacity));
    for (size_t i = 0; i < kCount; ++i) {
      store_->Append(data_.data() + i * kDimension);
    }
    PqTrainOptions options;
    options.num_subspaces = kSubspaces;
    options.kmeans_iterations = 10;
    options.threads = 4;
    pq_.emplace(*ProductQuantizer::Train(data_.data(), kCount, kDimension,
                                         kDimension, options, nullptr));
    std::string error;
    graph_.emplace(*HnswGraph::Create(capacity, 16, 32, kSubspaces, 7, &error));
    for (uint32_t i = 0; i < kCount; ++i) {
      pq_->Encode(data_.data() + i * kDimension, graph_->MutableCode(i));
    }
    builder_ = std::make_unique<HnswBuilder>(&*graph_, &*store_, 100);
    builder_->InsertRange(0, kCount, build_threads);
  }

  // Mean recall@10 over the query set.
  double MeanRecall(int ef, int rerank_depth) const {
    HnswSearcher searcher(&*graph_, &*store_, &*pq_);
    SearchScratch scratch;
    SearchParams params;
    params.k = 10;
    params.ef = ef;
    params.rerank_depth = rerank_depth;
    double total = 0.0;
    for (size_t q = 0; q < kQueries; ++q) {
      const float* query = queries_.data() + q * kDimension;
      std::vector<Neighbor> results;
      SearchCounters counters;
      searcher.Search(query, params, nullptr, kCount, &scratch, &results,
                      &counters);
      std::vector<uint32_t> found;
      for (const Neighbor& n : results) found.push_back(n.node);
      total += testing::Recall(
          testing::BruteForceKnn(data_, kDimension, query, 10, {}), found);
    }
    return total / kQueries;
  }

  HnswGraph& graph() { return *graph_; }
  FlatStore& store() { return *store_; }
  const ProductQuantizer& pq() const { return *pq_; }
  HnswBuilder& builder() { return *builder_; }
  const std::vector<float>& queries() const { return queries_; }

 private:
  std::vector<float> data_;
  std::vector<float> queries_;
  std::optional<FlatStore> store_;
  std::optional<ProductQuantizer> pq_;
  std::optional<HnswGraph> graph_;
  std::unique_ptr<HnswBuilder> builder_;
};

TEST(HnswTest, RecallAtTenMeetsTargetWithExactRerank) {
  const HnswFixture fixture(/*build_threads=*/1);
  // The O1 acceptance threshold is recall@10 >= 0.95 [4]. Isotropic 64-d
  // Gaussian clusters are a hard case for 16-byte PQ codes, so the operating
  // point re-ranks a 256-wide beam (see docs/research/hnsw-parameter-study.md).
  EXPECT_GE(fixture.MeanRecall(/*ef=*/256, /*rerank_depth=*/256), 0.95);
}

TEST(HnswTest, NavigationFindsTheQuantizedNearestNeighbors) {
  // Graph quality, isolated from PQ quality: the beam search must recover
  // nearly everything an exhaustive scan ranked by the same PQ codes finds.
  HnswFixture fixture(/*build_threads=*/1);
  const ProductQuantizer& pq = fixture.pq();
  HnswSearcher searcher(&fixture.graph(), &fixture.store(), &pq);
  std::vector<float> table(kSubspaces * kPqCentroids);
  SearchScratch scratch;
  SearchParams params;
  params.k = 64;
  params.ef = 64;
  params.rerank_depth = 64;
  double overlap = 0.0;
  for (size_t q = 0; q < kQueries; ++q) {
    const float* query = fixture.queries().data() + q * kDimension;
    pq.ComputeDistanceTable(query, table.data());
    std::vector<std::pair<float, uint32_t>> scored;
    for (uint32_t i = 0; i < kCount; ++i) {
      scored.emplace_back(
          PqAdcDistance(table.data(), fixture.graph().Code(i), kSubspaces), i);
    }
    std::partial_sort(scored.begin(), scored.begin() + 64, scored.end());
    std::vector<uint32_t> exhaustive;
    for (size_t i = 0; i < 64; ++i) exhaustive.push_back(scored[i].second);
    std::vector<Neighbor> results;
    SearchCounters counters;
    searcher.Search(query, params, nullptr, kCount, &scratch, &results,
                    &counters);
    std::vector<uint32_t> found;
    for (const Neighbor& n : results) found.push_back(n.node);
    overlap += testing::Recall(exhaustive, found);
  }
  EXPECT_GE(overlap / kQueries, 0.97);
}

TEST(HnswTest, ExactRerankRecoversRecallLostToQuantization) {
  const HnswFixture fixture(/*build_threads=*/1);
  // Ranking by 16-byte PQ codes alone (re-ranking only the final ten) is
  // measurably worse than re-scoring the whole beam exactly; this is why the
  // engine keeps the FP32 vectors resident.
  const double pq_only = fixture.MeanRecall(64, 10);
  const double reranked = fixture.MeanRecall(64, 64);
  EXPECT_GT(reranked, pq_only);
}

TEST(HnswTest, MultithreadedBuildKeepsRecall) {
  const HnswFixture fixture(/*build_threads=*/4);
  EXPECT_GE(fixture.MeanRecall(256, 256), 0.95);
}

TEST(HnswTest, LevelsFollowTheGeometricDistributionDeterministically) {
  std::string error;
  std::optional<HnswGraph> a = HnswGraph::Create(100000, 16, 32, 8, 3, &error);
  std::optional<HnswGraph> b = HnswGraph::Create(100000, 16, 32, 8, 3, &error);
  ASSERT_TRUE(a.has_value() && b.has_value()) << error;
  EXPECT_EQ(a->levels(), b->levels());
  size_t upper = 0;
  for (uint8_t level : a->levels()) upper += level >= 1;
  // P(level >= 1) = 1/M = 6.25% at M = 16.
  EXPECT_NEAR(static_cast<double>(upper) / 100000.0, 1.0 / 16.0, 0.005);
}

TEST(HnswTest, SearchesProceedDuringSingleWriterInsertion) {
  // Reserve room for 2,000 more nodes than the initial build.
  HnswFixture fixture(/*build_threads=*/2, kCount + 2000);
  const std::vector<float> extra =
      testing::ClusteredVectors(2000, kDimension, 64, 0.12f, 43);
  std::atomic<uint32_t> published{static_cast<uint32_t>(kCount)};
  std::atomic<bool> done{false};
  std::atomic<size_t> searches{0};
  std::vector<std::thread> readers;
  for (int t = 0; t < 3; ++t) {
    readers.emplace_back([&, t] {
      HnswSearcher searcher(&fixture.graph(), &fixture.store(), &fixture.pq());
      SearchScratch scratch;
      SearchParams params;
      size_t q = static_cast<size_t>(t);
      while (!done.load()) {
        std::vector<Neighbor> results;
        SearchCounters counters;
        const uint32_t visible = published.load(std::memory_order_acquire);
        searcher.Search(fixture.queries().data() + (q % kQueries) * kDimension,
                        params, nullptr, visible, &scratch, &results,
                        &counters);
        EXPECT_EQ(results.size(), 10u);
        ++q;
        searches.fetch_add(1);
      }
    });
  }
  for (uint32_t i = 0; i < 2000; ++i) {
    const uint32_t node =
        *fixture.store().Append(extra.data() + i * kDimension);
    fixture.pq().Encode(extra.data() + i * kDimension,
                        fixture.graph().MutableCode(node));
    fixture.builder().Insert(node);
    published.store(node + 1, std::memory_order_release);
  }
  done.store(true);
  for (std::thread& reader : readers) reader.join();
  EXPECT_GT(searches.load(), 0u);
  // A freshly inserted vector is its own nearest neighbor.
  HnswSearcher searcher(&fixture.graph(), &fixture.store(), &fixture.pq());
  SearchScratch scratch;
  std::vector<Neighbor> results;
  SearchCounters counters;
  searcher.Search(extra.data() + 1999 * kDimension, SearchParams(), nullptr,
                  published.load(), &scratch, &results, &counters);
  ASSERT_FALSE(results.empty());
  EXPECT_EQ(results.front().node, kCount + 1999);
  EXPECT_FLOAT_EQ(results.front().distance, 0.0f);
}

}  // namespace
}  // namespace vectra::index
