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

#ifndef VECTRA_INDEX_HNSW_GRAPH_H_
#define VECTRA_INDEX_HNSW_GRAPH_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vectra/memory/arena.h"

namespace vectra::index {

// Reads a graph word that a concurrent inserter may be writing. The graph
// lives in a mutable arena, so casting away const for an atomic load is safe.
inline uint32_t LoadGraphWord(
    const uint32_t& word, std::memory_order order = std::memory_order_relaxed) {
  return std::atomic_ref<uint32_t>(const_cast<uint32_t&>(word)).load(order);
}

// Writes a graph word that concurrent searches may be reading.
inline void StoreGraphWord(
    uint32_t& word, uint32_t value,
    std::memory_order order = std::memory_order_relaxed) {
  std::atomic_ref<uint32_t>(word).store(value, order);
}

// A fixed-capacity Hierarchical Navigable Small World graph [21] stored in
// flat memory.
//
// Layout, all inside one arena with one contiguous block per layer:
//   * Layer 0 holds one record per node (slot == node id):
//       [PQ code, padded to 4 bytes][count:u32][neighbor slot:u32 x m0]
//     The code leads the record so that scoring a neighbor pulls the start of
//     that neighbor's adjacency list into the same cache line.
//   * Layer l >= 1 holds one record per node of level >= l, in id order:
//       [node:u32][down:u32][count:u32][neighbor slot:u32 x m]
//     Neighbors are slots of layer l, and `down` is the node's slot in layer
//     l - 1, so descending a layer is one array read.
// Every edge is a 32-bit offset; the structure contains no pointers and is
// saved and loaded verbatim.
//
// Node levels are drawn when the graph is created, from the geometric
// distribution of [21] with normalization m_L = 1 / ln(M). That fixes the
// size of every layer's block before the first insertion.
//
// Thread safety: accessors are const and may run concurrently with one
// inserter provided neighbor lists are read with LoadGraphWord.
class HnswGraph {
 public:
  static constexpr uint32_t kNoNode = UINT32_MAX;
  // Levels are capped well above anything the distribution produces at
  // ten-million scale (P(level >= 16) is about 16^-16 at M = 16).
  static constexpr int kMaxLevel = 16;

  // Creates an empty graph for `capacity` nodes with degree caps `m` (upper
  // layers) and `m0` (layer 0) and `code_bytes` of PQ code per node.
  static std::optional<HnswGraph> Create(size_t capacity, int m, int m0,
                                         int code_bytes, uint64_t seed,
                                         std::string* error);

  // Recreates a graph from saved levels and arena contents (see Save).
  static std::optional<HnswGraph> FromSaved(size_t capacity, int m, int m0,
                                            int code_bytes,
                                            std::vector<uint8_t> levels,
                                            const void* arena_bytes,
                                            size_t arena_size,
                                            std::string* error);

  HnswGraph(HnswGraph&&) noexcept = default;
  HnswGraph& operator=(HnswGraph&&) noexcept = default;
  HnswGraph(const HnswGraph&) = delete;
  HnswGraph& operator=(const HnswGraph&) = delete;

  int m() const { return m_; }
  int m0() const { return m0_; }
  int code_bytes() const { return code_bytes_; }
  size_t capacity() const { return levels_.size(); }

  int Level(uint32_t node) const { return levels_[node]; }
  int top_level() const { return top_level_; }
  int MaxDegree(int level) const { return level == 0 ? m0_ : m_; }

  // Layer-0 access. Lists are [count, slot 0, slot 1, ...].
  const uint8_t* Code(uint32_t node) const {
    return arena_.At<uint8_t>(level0_offset_ + node * level0_stride_);
  }
  uint8_t* MutableCode(uint32_t node) {
    return arena_.At<uint8_t>(level0_offset_ + node * level0_stride_);
  }

  // Neighbor list of the node at `slot` in `level`: [count, slot...].
  const uint32_t* Links(int level, uint32_t slot) const;
  uint32_t* MutableLinks(int level, uint32_t slot);

  // Node id of the record at `slot` in `level`.
  uint32_t NodeAt(int level, uint32_t slot) const;

  // Slot of `node` in `level` (which must not exceed Level(node)).
  uint32_t SlotOf(uint32_t node, int level) const;

  // Slot in `level` - 1 of the record at `slot` in `level` >= 1.
  uint32_t Down(int level, uint32_t slot) const {
    return UpperRecord(level, slot)[1];
  }

  // The entry point, read and published as one atomic word.
  struct EntryPoint {
    uint32_t node = kNoNode;  // kNoNode while the graph is empty.
    int level = -1;
  };
  EntryPoint entry() const;
  void SetEntry(uint32_t node, int level);

  // Raw state for persistence and residency locking.
  const std::vector<uint8_t>& levels() const { return levels_; }
  const void* arena_data() const { return arena_.base(); }
  size_t arena_bytes() const { return arena_.used(); }
  size_t ResidentBytes() const {
    return arena_.capacity() + levels_.size() + upper_slots_.size() * 4 +
           upper_base_.size() * 4;
  }

 private:
  HnswGraph() = default;

  // Computes the layout for the current levels_ and allocates the arena.
  bool AllocateLayout(std::string* error);

  const uint32_t* UpperRecord(int level, uint32_t slot) const {
    return arena_.At<uint32_t>(layer_offsets_[level] + slot * upper_stride_);
  }
  uint32_t* MutableUpperRecord(int level, uint32_t slot) {
    return arena_.At<uint32_t>(layer_offsets_[level] + slot * upper_stride_);
  }

  int m_ = 0;
  int m0_ = 0;
  int code_bytes_ = 0;
  // Code bytes rounded up so the count that follows stays 4-byte aligned.
  size_t padded_code_bytes_ = 0;
  size_t level0_stride_ = 0;
  size_t upper_stride_ = 0;
  int top_level_ = 0;
  std::vector<uint8_t> levels_;
  // Arena offset of each layer's block (index 0 is layer 0).
  std::vector<size_t> layer_offsets_;
  // upper_slots_[upper_base_[node] + level - 1] is the node's slot in that
  // level; upper_base_ is kNoNode for nodes that live only in layer 0.
  std::vector<uint32_t> upper_base_;
  std::vector<uint32_t> upper_slots_;
  size_t level0_offset_ = 0;
  memory::Arena arena_;
  // (level << 32) | node, or all ones when empty. Accessed through
  // std::atomic_ref so that the graph itself stays movable.
  alignas(8) uint64_t entry_ = ~uint64_t{0};
};

}  // namespace vectra::index

#endif  // VECTRA_INDEX_HNSW_GRAPH_H_
