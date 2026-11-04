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

#ifndef VECTRA_INDEX_HNSW_SEARCHER_H_
#define VECTRA_INDEX_HNSW_SEARCHER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vectra/index/flat_store.h"
#include "vectra/index/geofilter.h"
#include "vectra/index/hnsw_graph.h"
#include "vectra/index/pq_codec.h"
#include "vectra/index/search_scratch.h"

namespace vectra::index {

struct SearchParams {
  // Results to return.
  int k = 10;
  // Beam width at layer 0 (ef_search in [21]).
  int ef = 64;
  // PQ-ranked candidates re-scored with exact distances (at least k).
  int rerank_depth = 64;
  // Upper bound on nodes scored at layer 0, or 0 for none. Caps the work of a
  // filtered search whose predicate admits few nodes near the query.
  size_t visit_budget = 0;
};

struct SearchCounters {
  uint32_t distance_evaluations = 0;
  bool budget_exhausted = false;
};

// Query-time HNSW search. Navigation scores PQ codes by asymmetric distance
// (m table lookups per node, from L1-resident tables); the best rerank_depth
// admitted candidates are then re-scored against the FP32 vectors, because
// PQ-only distances cannot reach recall@10 >= 0.95 at 8-16 byte codes (PQ
// systems report 1-recall@1 near 0.5 [12]).
//
// The filtered beam search follows hnswlib [23]: every visited node may extend
// the frontier, but only admitted nodes enter the result set, and the search
// stops early only once the result set is full.
//
// Immutable; Search is safe to call concurrently with one inserting writer.
class HnswSearcher {
 public:
  // None of the arguments are owned; all must outlive the searcher.
  HnswSearcher(const HnswGraph* graph, const FlatStore* vectors,
               const ProductQuantizer* pq);

  // Writes up to params.k neighbors of `query`, nearest first with exact
  // squared L2 distances, to `results` (replacing its contents). Only nodes
  // below `published` are considered; `predicate` may be null to admit all.
  void Search(const float* query, const SearchParams& params,
              const SearchPredicate* predicate, uint32_t published,
              SearchScratch* scratch, std::vector<Neighbor>* results,
              SearchCounters* counters) const;

 private:
  float CodeDistance(const float* table, uint32_t node) const {
    return PqAdcDistance(table, graph_->Code(node), pq_->code_bytes());
  }

  const HnswGraph* graph_;
  const FlatStore* vectors_;
  const ProductQuantizer* pq_;
};

}  // namespace vectra::index

#endif  // VECTRA_INDEX_HNSW_SEARCHER_H_
