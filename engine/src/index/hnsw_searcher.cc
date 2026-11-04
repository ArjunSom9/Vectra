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

#include "vectra/index/hnsw_searcher.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "vectra/common/port.h"
#include "vectra/index/simd_distance.h"

namespace vectra::index {
namespace {

struct FartherFirst {
  bool operator()(const Neighbor& a, const Neighbor& b) const {
    return CloserFirst()(b, a);
  }
};

}  // namespace

HnswSearcher::HnswSearcher(const HnswGraph* graph, const FlatStore* vectors,
                           const ProductQuantizer* pq)
    : graph_(graph), vectors_(vectors), pq_(pq) {}

void HnswSearcher::Search(const float* query, const SearchParams& params,
                          const SearchPredicate* predicate, uint32_t published,
                          SearchScratch* scratch,
                          std::vector<Neighbor>* results,
                          SearchCounters* counters) const {
  results->clear();
  // The entry point may be a node an in-flight insertion has not published
  // yet; its records were written before the entry word was released, so it
  // is safe to start from it.
  const HnswGraph::EntryPoint entry = graph_->entry();
  if (entry.node == HnswGraph::kNoNode) return;

  std::vector<float>& table = scratch->distance_table;
  table.resize(static_cast<size_t>(pq_->num_subspaces()) * kPqCentroids);
  pq_->ComputeDistanceTable(query, table.data());
  uint32_t evaluations = 0;

  // Greedy descent through the upper layers.
  uint32_t slot = graph_->SlotOf(entry.node, entry.level);
  float distance = CodeDistance(table.data(), entry.node);
  ++evaluations;
  for (int level = entry.level; level > 0; --level) {
    const uint32_t max_degree = static_cast<uint32_t>(graph_->MaxDegree(level));
    for (bool improved = true; improved;) {
      improved = false;
      const uint32_t* links = graph_->Links(level, slot);
      const uint32_t count = std::min(
          LoadGraphWord(links[0], std::memory_order_acquire), max_degree);
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t candidate = LoadGraphWord(links[1 + i]);
        const uint32_t node = graph_->NodeAt(level, candidate);
        if (node >= published) continue;
        const float d = CodeDistance(table.data(), node);
        ++evaluations;
        if (d < distance) {
          distance = d;
          slot = candidate;
          improved = true;
        }
      }
    }
    slot = graph_->Down(level, slot);
  }

  // Beam search of layer 0, where slots are node ids.
  VisitedSet& visited = scratch->visited;
  std::vector<Neighbor>& frontier = scratch->candidates;
  std::vector<Neighbor>& best = scratch->results;
  visited.Clear();
  frontier.clear();
  best.clear();
  const size_t width = static_cast<size_t>(std::max(params.ef, params.k));
  const uint32_t max_degree = static_cast<uint32_t>(graph_->m0());
  auto admitted = [predicate](uint32_t node) {
    return predicate == nullptr || predicate->Admits(node);
  };

  float lower_bound = std::numeric_limits<float>::max();
  visited.Insert(slot);
  frontier.push_back({distance, slot});
  if (admitted(slot)) {
    best.push_back({distance, slot});
    lower_bound = distance;
  }
  size_t scored = 0;
  while (!frontier.empty()) {
    const Neighbor current = frontier.front();
    if (current.distance > lower_bound && best.size() >= width) break;
    if (params.visit_budget != 0 && scored >= params.visit_budget) {
      counters->budget_exhausted = true;
      break;
    }
    std::pop_heap(frontier.begin(), frontier.end(), FartherFirst());
    frontier.pop_back();

    const uint32_t* links = graph_->Links(0, current.node);
    const uint32_t count = std::min(
        LoadGraphWord(links[0], std::memory_order_acquire), max_degree);
    for (uint32_t i = 0; i < count; ++i) {
      PrefetchForRead(graph_->Code(LoadGraphWord(links[1 + i])));
    }
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t node = LoadGraphWord(links[1 + i]);
      if (node >= published || !visited.Insert(node)) continue;
      const float d = CodeDistance(table.data(), node);
      ++scored;
      if (best.size() < width || d < lower_bound) {
        frontier.push_back({d, node});
        std::push_heap(frontier.begin(), frontier.end(), FartherFirst());
        if (admitted(node)) {
          best.push_back({d, node});
          std::push_heap(best.begin(), best.end(), CloserFirst());
          if (best.size() > width) {
            std::pop_heap(best.begin(), best.end(), CloserFirst());
            best.pop_back();
          }
        }
        if (!best.empty()) lower_bound = best.front().distance;
      }
    }
  }
  evaluations += static_cast<uint32_t>(scored);

  // Exact re-rank of the best PQ-ranked candidates.
  std::sort(best.begin(), best.end(), CloserFirst());
  const size_t depth =
      std::min(best.size(),
               static_cast<size_t>(std::max(params.rerank_depth, params.k)));
  std::vector<Neighbor>& rerank = scratch->rerank;
  rerank.clear();
  for (size_t i = 0; i < depth; ++i) {
    const uint32_t node = best[i].node;
    rerank.push_back(
        {L2SquaredDistance(query, vectors_->Row(node), vectors_->dimension()),
         node});
  }
  evaluations += static_cast<uint32_t>(depth);
  const size_t k = std::min(rerank.size(), static_cast<size_t>(params.k));
  std::partial_sort(rerank.begin(),
                    rerank.begin() + static_cast<std::ptrdiff_t>(k),
                    rerank.end(), CloserFirst());
  results->assign(rerank.begin(),
                  rerank.begin() + static_cast<std::ptrdiff_t>(k));
  counters->distance_evaluations += evaluations;
}

}  // namespace vectra::index
