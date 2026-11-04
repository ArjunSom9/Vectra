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

// Recall/latency sweep of the shard index
// (docs/research/hnsw-parameter-study.md).
//
//   index_study EMBEDDINGS.npy [queries]       # a corpus's embeddings
//   index_study [count] [dimension] [clusters] [queries]   # Gaussian mixture
//
// For each re-rank depth it builds an index, then sweeps ef_search and
// prints CSV rows: rerank_depth, ef_search, recall@10, mean microseconds
// per query, mean distance evaluations. A final row reports the recall of
// an exhaustive search ranked by PQ distances alone, the ceiling of any
// search that never looks at the float32 vectors.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "testing/synthetic_data.h"
#include "vectra/index/pq_codec.h"
#include "vectra/index/search_scratch.h"
#include "vectra/index/shard_index.h"
#include "vectra/index/simd_distance.h"
#include "vectra/io/npy.h"

namespace vectra {
namespace {

constexpr size_t kK = 10;

size_t Argument(int argc, char** argv, int position, size_t fallback) {
  return argc > position ? std::strtoull(argv[position], nullptr, 10)
                         : fallback;
}

// Recall@k of an exhaustive search that ranks every vector by its PQ
// distance (asymmetric distance computation).
double PqCeiling(const std::vector<float>& data, int dimension, size_t count,
                 const std::vector<size_t>& query_rows,
                 const std::vector<std::vector<uint32_t>>& truth) {
  index::PqTrainOptions options;
  options.threads = 8;
  std::string error;
  std::optional<index::ProductQuantizer> pq = index::ProductQuantizer::Train(
      data.data(), count, static_cast<size_t>(dimension), dimension, options,
      &error);
  if (!pq.has_value()) {
    std::fprintf(stderr, "pq: %s\n", error.c_str());
    return 0.0;
  }
  const int m = pq->num_subspaces();
  std::vector<uint8_t> codes(count * static_cast<size_t>(m));
  for (size_t i = 0; i < count; ++i) {
    pq->Encode(data.data() + i * dimension, codes.data() + i * m);
  }
  std::vector<float> table(static_cast<size_t>(m) * index::kPqCentroids);
  std::vector<uint32_t> order(count);
  std::vector<float> distance(count);
  double recall = 0.0;
  for (size_t q = 0; q < query_rows.size(); ++q) {
    pq->ComputeDistanceTable(data.data() + query_rows[q] * dimension,
                             table.data());
    for (size_t i = 0; i < count; ++i) {
      distance[i] = index::PqAdcDistance(table.data(), codes.data() + i * m, m);
    }
    std::iota(order.begin(), order.end(), 0u);
    std::partial_sort(
        order.begin(), order.begin() + kK, order.end(),
        [&](uint32_t a, uint32_t b) { return distance[a] < distance[b]; });
    recall += testing::Recall(
        truth[q], std::vector<uint32_t>(order.begin(), order.begin() + kK));
  }
  return recall / static_cast<double>(query_rows.size());
}

int Main(int argc, char** argv) {
  std::vector<float> data;
  size_t count = 0;
  int dimension = 0;
  size_t num_queries = 0;
  const std::string first = argc > 1 ? argv[1] : "";
  if (first.size() > 4 && first.ends_with(".npy")) {
    std::string error;
    std::optional<io::NpyArray> array = io::ReadNpy(first, &error);
    if (!array.has_value() || array->descr != "<f4" ||
        array->shape.size() != 2) {
      std::fprintf(stderr, "%s: need a [n, d] float32 array %s\n",
                   first.c_str(), error.c_str());
      return 1;
    }
    count = array->shape[0];
    dimension = static_cast<int>(array->shape[1]);
    data.assign(array->As<float>(),
                array->As<float>() + array->element_count());
    num_queries = Argument(argc, argv, 2, 500);
  } else {
    count = Argument(argc, argv, 1, 50000);
    dimension = static_cast<int>(Argument(argc, argv, 2, 128));
    const int clusters = static_cast<int>(Argument(argc, argv, 3, 64));
    num_queries = Argument(argc, argv, 4, 500);
    data = testing::ClusteredVectors(count, dimension, clusters, 0.35f, 7);
  }
  const std::vector<index::PropertyRecord> records = testing::RecordsAt(
      testing::PointsAround(count, {37.7749, -122.4194}, 30.0, 8), 1);
  std::vector<size_t> query_rows(num_queries);
  for (size_t q = 0; q < num_queries; ++q) {
    query_rows[q] = (q * 7919) % count;
  }
  // Each query is a corpus vector; its own row is excluded from the truth
  // and from the results by asking for k + 1 and dropping it.
  std::vector<std::vector<uint32_t>> truth(num_queries);
  for (size_t q = 0; q < num_queries; ++q) {
    std::vector<uint32_t> nearest = testing::BruteForceKnn(
        data, dimension, data.data() + query_rows[q] * dimension, kK + 1, {});
    std::erase(nearest, static_cast<uint32_t>(query_rows[q]));
    nearest.resize(kK);
    truth[q] = nearest;
  }

  std::printf(
      "rerank_depth,ef_search,recall_at_10,us_per_query,"
      "distance_evaluations\n");
  for (const int rerank : {static_cast<int>(kK) + 1, 64, 128, 256}) {
    index::IndexOptions options;
    options.dimension = dimension;
    options.max_elements = count;
    options.rerank_depth = rerank;
    options.build_threads = 8;
    std::string error;
    std::unique_ptr<index::ShardIndex> shard = index::ShardIndex::Build(
        options, data.data(), records.data(), count, &error);
    if (shard == nullptr) {
      std::fprintf(stderr, "build: %s\n", error.c_str());
      return 1;
    }
    index::SearchScratch scratch;
    std::vector<index::IndexHit> hits;
    for (const int ef : {16, 32, 64, 128, 256, 512}) {
      double recall = 0.0;
      double evaluations = 0.0;
      const auto start = std::chrono::steady_clock::now();
      for (size_t q = 0; q < num_queries; ++q) {
        index::IndexQuery query;
        query.embedding = data.data() + query_rows[q] * dimension;
        query.k = static_cast<int>(kK) + 1;
        query.ef = ef;
        index::IndexSearchStats stats;
        shard->Search(query, &scratch, &hits, &stats);
        std::vector<uint32_t> found;
        for (const index::IndexHit& hit : hits) {
          if (hit.node != query_rows[q]) found.push_back(hit.node);
        }
        found.resize(std::min(found.size(), kK));
        recall += testing::Recall(truth[q], found);
        evaluations += stats.distance_evaluations;
      }
      const double seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
      const double n = static_cast<double>(num_queries);
      std::printf("%d,%d,%.4f,%.1f,%.0f\n", rerank, ef, recall / n,
                  seconds / n * 1e6, evaluations / n);
    }
  }
  std::printf("pq_exhaustive,-,%.4f,-,-\n",
              PqCeiling(data, dimension, count, query_rows, truth));
  return 0;
}

}  // namespace
}  // namespace vectra

int main(int argc, char** argv) { return vectra::Main(argc, argv); }
