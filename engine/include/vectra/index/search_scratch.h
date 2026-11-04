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

// Reusable per-thread working memory for graph search. Each worker owns one
// scratch object for its lifetime, so steady-state queries allocate nothing.

#ifndef VECTRA_INDEX_SEARCH_SCRATCH_H_
#define VECTRA_INDEX_SEARCH_SCRATCH_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vectra::index {

// A node and its distance to the query (squared L2, exact or PQ-estimated
// depending on the phase).
struct Neighbor {
  float distance = 0.0f;
  uint32_t node = 0;
};

// Orders by distance, ties broken by node id for determinism.
struct CloserFirst {
  bool operator()(const Neighbor& a, const Neighbor& b) const {
    return a.distance < b.distance ||
           (a.distance == b.distance && a.node < b.node);
  }
};

// The set of nodes visited by one search. An open-addressing table tagged with
// a per-search epoch: Clear() is O(1), and a search touching a few thousand
// nodes stays L2-resident, unlike a bitmap over the whole shard.
class VisitedSet {
 public:
  // Enough for a search that touches a few thousand nodes without growing.
  static constexpr size_t kDefaultSlots = 16384;

  // `initial_slots` is rounded up to a power of two.
  explicit VisitedSet(size_t initial_slots = kDefaultSlots);

  // Starts a new search.
  void Clear();

  // Marks `key` visited. Returns true if it was not visited before.
  bool Insert(uint32_t key);

  size_t size() const { return size_; }

 private:
  void Grow();

  std::vector<uint32_t> keys_;
  std::vector<uint32_t> epochs_;
  uint32_t epoch_ = 1;
  size_t size_ = 0;
  size_t mask_ = 0;
};

// Working memory for HnswSearcher and HnswBuilder.
struct SearchScratch {
  VisitedSet visited;
  // PQ distance table: subspaces x kPqCentroids floats.
  std::vector<float> distance_table;
  // Frontier (min-heap) and the best results so far (max-heap).
  std::vector<Neighbor> candidates;
  std::vector<Neighbor> results;
  // Exact re-rank buffer.
  std::vector<Neighbor> rerank;
};

}  // namespace vectra::index

#endif  // VECTRA_INDEX_SEARCH_SCRATCH_H_
