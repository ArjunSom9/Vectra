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

#ifndef VECTRA_INDEX_HNSW_BUILDER_H_
#define VECTRA_INDEX_HNSW_BUILDER_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "vectra/index/flat_store.h"
#include "vectra/index/hnsw_graph.h"
#include "vectra/index/search_scratch.h"

namespace vectra::index {

// Inserts nodes into an HnswGraph with Algorithm 1 of [21] and the neighbor
// selection heuristic of Algorithm 4 (the variant hnswlib [23] uses: no
// candidate extension, no kept pruned connections). Construction scores with
// exact FP32 distances from the FlatStore so graph quality does not depend on
// quantization error.
//
// InsertRange may run several threads; neighbor-list updates take a striped
// lock and readers use atomic word access, so searches may proceed
// concurrently with a single inserting thread.
class HnswBuilder {
 public:
  // `graph` and `vectors` must outlive the builder.
  HnswBuilder(HnswGraph* graph, const FlatStore* vectors, int ef_construction);

  HnswBuilder(const HnswBuilder&) = delete;
  HnswBuilder& operator=(const HnswBuilder&) = delete;

  // Inserts nodes [begin, end) on `threads` threads. Every node's vector must
  // already be in the FlatStore (node id == row id).
  void InsertRange(uint32_t begin, uint32_t end, int threads);

  // Inserts one node on the calling thread.
  void Insert(uint32_t node);

 private:
  void InsertWithScratch(uint32_t node, SearchScratch* scratch);

  float Distance(uint32_t a, uint32_t b) const;

  // Beam search of one layer from `entry_slot` for the `ef` slots nearest to
  // `node`, written to scratch->results sorted nearest first.
  void SearchLayer(uint32_t node, uint32_t entry_slot, int level, int ef,
                   SearchScratch* scratch) const;

  // Algorithm 4: keeps up to `max_count` of the (sorted) candidates that are
  // closer to the base than to any already selected neighbor.
  void SelectNeighbors(std::vector<Neighbor>* candidates, int level,
                       int max_count) const;

  // Adds bidirectional edges between `slot` and `neighbors` at `level`.
  void Connect(uint32_t node, int level,
               const std::vector<Neighbor>& neighbors);

  std::mutex& StripeFor(uint32_t node) {
    return stripes_[node & (kStripes - 1)];
  }

  static constexpr uint32_t kStripes = 1 << 16;

  HnswGraph* graph_;
  const FlatStore* vectors_;
  int ef_construction_;
  std::unique_ptr<std::mutex[]> stripes_;
  // Held for a whole insertion whose level exceeds the current top level.
  std::mutex entry_mutex_;
  // Scratch for the single-node Insert path.
  SearchScratch scratch_;
};

}  // namespace vectra::index

#endif  // VECTRA_INDEX_HNSW_BUILDER_H_
