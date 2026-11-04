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

#include "vectra/index/hnsw_builder.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "vectra/index/simd_distance.h"

namespace vectra::index {
namespace {

// Orders a std:: heap so that the nearest element is on top.
struct FartherFirst {
  bool operator()(const Neighbor& a, const Neighbor& b) const {
    return CloserFirst()(b, a);
  }
};

}  // namespace

HnswBuilder::HnswBuilder(HnswGraph* graph, const FlatStore* vectors,
                         int ef_construction)
    : graph_(graph),
      vectors_(vectors),
      ef_construction_(std::max(ef_construction, graph->m())),
      stripes_(std::make_unique<std::mutex[]>(kStripes)) {}

float HnswBuilder::Distance(uint32_t a, uint32_t b) const {
  return L2SquaredDistance(vectors_->Row(a), vectors_->Row(b),
                           vectors_->dimension());
}

void HnswBuilder::InsertRange(uint32_t begin, uint32_t end, int threads) {
  if (begin >= end) return;
  std::atomic<uint32_t> next{begin};
  auto work = [this, &next, end] {
    SearchScratch scratch;
    for (;;) {
      const uint32_t node = next.fetch_add(1, std::memory_order_relaxed);
      if (node >= end) return;
      InsertWithScratch(node, &scratch);
    }
  };
  const int workers = std::max(1, threads);
  if (workers == 1) {
    work();
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(workers));
  for (int t = 0; t < workers; ++t) pool.emplace_back(work);
  for (std::thread& thread : pool) thread.join();
}

void HnswBuilder::Insert(uint32_t node) { InsertWithScratch(node, &scratch_); }

void HnswBuilder::InsertWithScratch(uint32_t node, SearchScratch* scratch) {
  const int level = graph_->Level(node);
  std::unique_lock<std::mutex> entry_lock(entry_mutex_);
  const HnswGraph::EntryPoint entry = graph_->entry();
  if (entry.node == HnswGraph::kNoNode) {
    graph_->SetEntry(node, level);
    return;
  }
  // Only an insertion that raises the top level keeps the lock throughout.
  if (level <= entry.level) entry_lock.unlock();

  // Greedy descent (ef = 1) through the layers above the node's level.
  uint32_t slot = graph_->SlotOf(entry.node, entry.level);
  float distance = Distance(node, entry.node);
  for (int l = entry.level; l > level; --l) {
    for (bool improved = true; improved;) {
      improved = false;
      const uint32_t* links = graph_->Links(l, slot);
      const uint32_t count =
          std::min<uint32_t>(LoadGraphWord(links[0], std::memory_order_acquire),
                             static_cast<uint32_t>(graph_->MaxDegree(l)));
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t candidate = LoadGraphWord(links[1 + i]);
        const float d = Distance(node, graph_->NodeAt(l, candidate));
        if (d < distance) {
          distance = d;
          slot = candidate;
          improved = true;
        }
      }
    }
    slot = graph_->Down(l, slot);
  }

  for (int l = std::min(level, entry.level); l >= 0; --l) {
    SearchLayer(node, slot, l, ef_construction_, scratch);
    std::vector<Neighbor> neighbors = scratch->results;
    const uint32_t nearest = neighbors.front().node;
    SelectNeighbors(&neighbors, l, graph_->m());
    Connect(node, l, neighbors);
    if (l > 0) slot = graph_->Down(l, nearest);
  }
  if (level > entry.level) graph_->SetEntry(node, level);
}

void HnswBuilder::SearchLayer(uint32_t node, uint32_t entry_slot, int level,
                              int ef, SearchScratch* scratch) const {
  VisitedSet& visited = scratch->visited;
  std::vector<Neighbor>& frontier = scratch->candidates;
  std::vector<Neighbor>& best = scratch->results;
  visited.Clear();
  frontier.clear();
  best.clear();
  const size_t width = static_cast<size_t>(ef);
  const uint32_t max_degree = static_cast<uint32_t>(graph_->MaxDegree(level));

  const Neighbor start{Distance(node, graph_->NodeAt(level, entry_slot)),
                       entry_slot};
  visited.Insert(entry_slot);
  frontier.push_back(start);
  best.push_back(start);
  while (!frontier.empty()) {
    std::pop_heap(frontier.begin(), frontier.end(), FartherFirst());
    const Neighbor current = frontier.back();
    frontier.pop_back();
    if (best.size() >= width && current.distance > best.front().distance) {
      break;
    }
    const uint32_t* links = graph_->Links(level, current.node);
    const uint32_t count = std::min(
        LoadGraphWord(links[0], std::memory_order_acquire), max_degree);
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t slot = LoadGraphWord(links[1 + i]);
      if (!visited.Insert(slot)) continue;
      const float d = Distance(node, graph_->NodeAt(level, slot));
      if (best.size() < width || d < best.front().distance) {
        frontier.push_back({d, slot});
        std::push_heap(frontier.begin(), frontier.end(), FartherFirst());
        best.push_back({d, slot});
        std::push_heap(best.begin(), best.end(), CloserFirst());
        if (best.size() > width) {
          std::pop_heap(best.begin(), best.end(), CloserFirst());
          best.pop_back();
        }
      }
    }
  }
  std::sort(best.begin(), best.end(), CloserFirst());
}

void HnswBuilder::SelectNeighbors(std::vector<Neighbor>* candidates, int level,
                                  int max_count) const {
  const size_t limit = static_cast<size_t>(max_count);
  if (candidates->size() <= limit) return;
  std::vector<Neighbor> selected;
  selected.reserve(limit);
  for (const Neighbor& candidate : *candidates) {
    if (selected.size() >= limit) break;
    const uint32_t candidate_node = graph_->NodeAt(level, candidate.node);
    bool diverse = true;
    for (const Neighbor& kept : selected) {
      if (Distance(candidate_node, graph_->NodeAt(level, kept.node)) <
          candidate.distance) {
        diverse = false;
        break;
      }
    }
    if (diverse) selected.push_back(candidate);
  }
  *candidates = std::move(selected);
}

void HnswBuilder::Connect(uint32_t node, int level,
                          const std::vector<Neighbor>& neighbors) {
  const uint32_t own_slot = graph_->SlotOf(node, level);
  const uint32_t max_degree = static_cast<uint32_t>(graph_->MaxDegree(level));
  {
    std::lock_guard<std::mutex> lock(StripeFor(node));
    uint32_t* links = graph_->MutableLinks(level, own_slot);
    uint32_t count = 0;
    for (const Neighbor& neighbor : neighbors) {
      if (neighbor.node == own_slot || count == max_degree) continue;
      StoreGraphWord(links[1 + count], neighbor.node);
      ++count;
    }
    StoreGraphWord(links[0], count, std::memory_order_release);
  }
  for (const Neighbor& neighbor : neighbors) {
    if (neighbor.node == own_slot) continue;
    const uint32_t other = graph_->NodeAt(level, neighbor.node);
    std::lock_guard<std::mutex> lock(StripeFor(other));
    uint32_t* links = graph_->MutableLinks(level, neighbor.node);
    const uint32_t count = LoadGraphWord(links[0]);
    bool present = false;
    for (uint32_t i = 0; i < count && !present; ++i) {
      present = LoadGraphWord(links[1 + i]) == own_slot;
    }
    if (present) continue;
    if (count < max_degree) {
      StoreGraphWord(links[1 + count], own_slot);
      StoreGraphWord(links[0], count + 1, std::memory_order_release);
      continue;
    }
    // Full: re-select among the existing links plus the new node.
    std::vector<Neighbor> pool;
    pool.reserve(count + 1);
    pool.push_back({Distance(other, node), own_slot});
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t slot = LoadGraphWord(links[1 + i]);
      pool.push_back({Distance(other, graph_->NodeAt(level, slot)), slot});
    }
    std::sort(pool.begin(), pool.end(), CloserFirst());
    SelectNeighbors(&pool, level, static_cast<int>(max_degree));
    for (size_t i = 0; i < pool.size(); ++i) {
      StoreGraphWord(links[1 + i], pool[i].node);
    }
    StoreGraphWord(links[0], static_cast<uint32_t>(pool.size()),
                   std::memory_order_release);
  }
}

}  // namespace vectra::index
