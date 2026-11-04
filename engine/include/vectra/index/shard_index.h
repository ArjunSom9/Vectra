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

#ifndef VECTRA_INDEX_SHARD_INDEX_H_
#define VECTRA_INDEX_SHARD_INDEX_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vectra/index/flat_store.h"
#include "vectra/index/geofilter.h"
#include "vectra/index/hnsw_builder.h"
#include "vectra/index/hnsw_graph.h"
#include "vectra/index/hnsw_searcher.h"
#include "vectra/index/pq_codec.h"
#include "vectra/index/property_store.h"
#include "vectra/index/search_scratch.h"
#include "vectra/routing/geo_math.h"

namespace vectra::index {

// Construction and query defaults; mirrors vectra.config.v1.IndexParams.
struct IndexOptions {
  int dimension = 128;
  int m = 16;
  int m0 = 32;
  int ef_construction = 200;
  int ef_search = 64;
  int rerank_depth = 64;
  size_t max_elements = 0;
  int pq_subspaces = 16;
  size_t pq_training_samples = 65536;
  int pq_kmeans_iterations = 25;
  // Radius populations at or below this size are scanned exhaustively.
  size_t exhaustive_scan_threshold = 2048;
  int build_threads = 1;
  uint64_t seed = 100;
};

struct IndexQuery {
  // dimension() floats.
  const float* embedding = nullptr;
  routing::LatLng center;
  // Geodesic radius; 0 disables geographic filtering.
  double radius_km = 0.0;
  int k = 10;
  // Beam width override; 0 uses IndexOptions::ef_search.
  int ef = 0;
  // Exhaustive search over the radius population (or the whole shard).
  bool exact = false;
  // When positive, only properties whose last sale precedes this day (days
  // since 1970-01-01) qualify: a valuation as of a past date sees only the
  // sales that had happened by then.
  int32_t sold_before_epoch_day = 0;
};

struct IndexHit {
  uint32_t node = 0;
  float squared_l2 = 0.0f;
  float geo_distance_km = 0.0f;
};

struct IndexSearchStats {
  uint32_t distance_evaluations = 0;
  uint32_t prefilter_population = 0;
  bool exhaustive = false;
};

// A memory region to pin with memory::LockRegion.
struct MemoryRegion {
  const void* address = nullptr;
  size_t bytes = 0;
};

// Everything one shard serves: FP32 vectors, PQ codes, the flat HNSW graph,
// property metadata, and the GeoHash prefilter.
//
// Search is thread-safe. Insert must be serialized by the caller (one
// writer); it may run concurrently with searches, which observe a consistent
// prefix of the inserted nodes.
class ShardIndex {
 public:
  // Builds an index over `count` embeddings (row-major, `options.dimension`
  // floats each) and their records, reserving
  // max(count, options.max_elements) slots.
  static std::unique_ptr<ShardIndex> Build(const IndexOptions& options,
                                           const float* embeddings,
                                           const PropertyRecord* records,
                                           size_t count, std::string* error);

  static std::unique_ptr<ShardIndex> Load(const std::string& path,
                                          std::string* error);

  ShardIndex(const ShardIndex&) = delete;
  ShardIndex& operator=(const ShardIndex&) = delete;
  ~ShardIndex();

  bool Save(const std::string& path, std::string* error) const;

  // Writes up to query.k hits, nearest first by exact squared L2, to `hits`.
  void Search(const IndexQuery& query, SearchScratch* scratch,
              std::vector<IndexHit>* hits, IndexSearchStats* stats) const;

  // Appends `count` records. Returns false (inserting nothing) when capacity
  // would be exceeded or a location is invalid.
  bool Insert(const float* embeddings, const PropertyRecord* records,
              size_t count, std::string* error);

  // Number of searchable nodes.
  size_t size() const { return published_.load(std::memory_order_acquire); }
  size_t capacity() const { return vectors_.capacity(); }
  int dimension() const { return options_.dimension; }
  const IndexOptions& options() const { return options_; }
  const PropertyStore& properties() const { return properties_; }
  const FlatStore& vectors() const { return vectors_; }
  const HnswGraph& graph() const { return graph_; }
  size_t ResidentBytes() const;
  std::vector<MemoryRegion> MemoryRegions() const;

 private:
  ShardIndex(IndexOptions options, FlatStore vectors, PropertyStore properties,
             ProductQuantizer pq, HnswGraph graph);

  // Appends vectors, metadata and PQ codes for `count` records without
  // linking them into the graph.
  bool AppendRows(const float* embeddings, const PropertyRecord* records,
                  size_t count, std::string* error);

  // Rebuilds the prefilter over the first `count` nodes and publishes it.
  void RebuildFilter(size_t count);

  void ExhaustiveSearch(const float* embedding,
                        const SearchPredicate* predicate,
                        const GeoFilter* filter,
                        const std::vector<routing::GeohashCell>* cells,
                        uint32_t published, int k, SearchScratch* scratch,
                        std::vector<Neighbor>* results,
                        uint32_t* evaluations) const;

  IndexOptions options_;
  FlatStore vectors_;
  PropertyStore properties_;
  ProductQuantizer pq_;
  HnswGraph graph_;
  HnswSearcher searcher_;
  std::unique_ptr<HnswBuilder> builder_;
  std::atomic<size_t> published_{0};
  // The current prefilter. Superseded filters are retired and freed only
  // after kFilterGracePeriod, far longer than any search can run.
  std::atomic<const GeoFilter*> filter_{nullptr};
  std::mutex retired_mutex_;
  std::vector<std::pair<std::chrono::steady_clock::time_point,
                        std::unique_ptr<const GeoFilter>>>
      retired_;
  std::unique_ptr<const GeoFilter> current_filter_;
};

}  // namespace vectra::index

#endif  // VECTRA_INDEX_SHARD_INDEX_H_
