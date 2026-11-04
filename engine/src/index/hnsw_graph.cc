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

#include "vectra/index/hnsw_graph.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace vectra::index {
namespace {

size_t RoundUp(size_t value, size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

// Uniform double in (0, 1] from the top 53 bits of a 64-bit draw. Avoids
// std::uniform_real_distribution, whose output is implementation-defined, so
// level assignment is identical across standard libraries.
double UniformOpenClosed(std::mt19937_64& rng) {
  return (static_cast<double>(rng() >> 11) + 1.0) * 0x1.0p-53;
}

}  // namespace

std::optional<HnswGraph> HnswGraph::Create(size_t capacity, int m, int m0,
                                           int code_bytes, uint64_t seed,
                                           std::string* error) {
  if (capacity == 0 || capacity >= kNoNode || m < 2 || m0 < m ||
      code_bytes < 0) {
    if (error != nullptr) *error = "invalid HNSW geometry";
    return std::nullopt;
  }
  HnswGraph graph;
  graph.m_ = m;
  graph.m0_ = m0;
  graph.code_bytes_ = code_bytes;
  graph.levels_.resize(capacity);
  // Level l is reached with probability M^-l [21].
  const double normalization = 1.0 / std::log(static_cast<double>(m));
  std::mt19937_64 rng(seed);
  for (uint8_t& level : graph.levels_) {
    const double drawn = -std::log(UniformOpenClosed(rng)) * normalization;
    level = static_cast<uint8_t>(std::min(static_cast<int>(drawn), kMaxLevel));
  }
  if (!graph.AllocateLayout(error)) return std::nullopt;
  return graph;
}

std::optional<HnswGraph> HnswGraph::FromSaved(
    size_t capacity, int m, int m0, int code_bytes, std::vector<uint8_t> levels,
    const void* arena_bytes, size_t arena_size, std::string* error) {
  if (levels.size() != capacity) {
    if (error != nullptr) *error = "saved HNSW levels do not match capacity";
    return std::nullopt;
  }
  HnswGraph graph;
  graph.m_ = m;
  graph.m0_ = m0;
  graph.code_bytes_ = code_bytes;
  graph.levels_ = std::move(levels);
  if (!graph.AllocateLayout(error)) return std::nullopt;
  if (arena_size != graph.arena_.used()) {
    if (error != nullptr) *error = "saved HNSW arena has the wrong size";
    return std::nullopt;
  }
  std::memcpy(graph.arena_.At<std::byte>(0), arena_bytes, arena_size);
  return graph;
}

bool HnswGraph::AllocateLayout(std::string* error) {
  padded_code_bytes_ = RoundUp(static_cast<size_t>(code_bytes_), 4);
  // 32-byte multiples keep a code of up to 32 bytes inside one cache line.
  level0_stride_ =
      RoundUp(padded_code_bytes_ + 4 + 4 * static_cast<size_t>(m0_), 32);
  upper_stride_ = RoundUp(12 + 4 * static_cast<size_t>(m_), 16);

  top_level_ = 0;
  for (uint8_t level : levels_) top_level_ = std::max<int>(top_level_, level);
  std::vector<size_t> layer_sizes(static_cast<size_t>(top_level_) + 1, 0);
  layer_sizes[0] = levels_.size();
  upper_base_.assign(levels_.size(), kNoNode);
  upper_slots_.clear();
  for (size_t node = 0; node < levels_.size(); ++node) {
    if (levels_[node] == 0) continue;
    upper_base_[node] = static_cast<uint32_t>(upper_slots_.size());
    for (int level = 1; level <= levels_[node]; ++level) {
      upper_slots_.push_back(static_cast<uint32_t>(layer_sizes[level]++));
    }
  }

  size_t total = layer_sizes[0] * level0_stride_ + memory::kCacheLineBytes;
  for (int level = 1; level <= top_level_; ++level) {
    total += layer_sizes[level] * upper_stride_ + memory::kCacheLineBytes;
  }
  std::optional<memory::Arena> arena = memory::Arena::Create(total);
  if (!arena.has_value()) {
    if (error != nullptr) *error = "cannot allocate the HNSW arena";
    return false;
  }
  arena_ = std::move(*arena);
  layer_offsets_.assign(static_cast<size_t>(top_level_) + 1, 0);
  for (int level = 0; level <= top_level_; ++level) {
    const size_t stride = level == 0 ? level0_stride_ : upper_stride_;
    std::optional<size_t> offset = arena_.Allocate(layer_sizes[level] * stride);
    if (!offset.has_value()) {
      if (error != nullptr) *error = "HNSW arena layout overflow";
      return false;
    }
    layer_offsets_[level] = *offset;
  }
  level0_offset_ = layer_offsets_[0];

  // Upper records know their node and their slot one layer down.
  for (size_t node = 0; node < levels_.size(); ++node) {
    for (int level = 1; level <= levels_[node]; ++level) {
      const uint32_t id = static_cast<uint32_t>(node);
      uint32_t* record = MutableUpperRecord(level, SlotOf(id, level));
      record[0] = id;
      record[1] = SlotOf(id, level - 1);
    }
  }
  return true;
}

const uint32_t* HnswGraph::Links(int level, uint32_t slot) const {
  if (level == 0) {
    return arena_.At<uint32_t>(level0_offset_ + slot * level0_stride_ +
                               padded_code_bytes_);
  }
  return UpperRecord(level, slot) + 2;
}

uint32_t* HnswGraph::MutableLinks(int level, uint32_t slot) {
  if (level == 0) {
    return arena_.At<uint32_t>(level0_offset_ + slot * level0_stride_ +
                               padded_code_bytes_);
  }
  return MutableUpperRecord(level, slot) + 2;
}

uint32_t HnswGraph::NodeAt(int level, uint32_t slot) const {
  return level == 0 ? slot : UpperRecord(level, slot)[0];
}

uint32_t HnswGraph::SlotOf(uint32_t node, int level) const {
  if (level == 0) return node;
  return upper_slots_[upper_base_[node] + static_cast<uint32_t>(level) - 1];
}

HnswGraph::EntryPoint HnswGraph::entry() const {
  const uint64_t packed =
      std::atomic_ref<uint64_t>(const_cast<uint64_t&>(entry_))
          .load(std::memory_order_acquire);
  if (packed == ~uint64_t{0}) return EntryPoint{kNoNode, -1};
  return EntryPoint{static_cast<uint32_t>(packed),
                    static_cast<int>(packed >> 32)};
}

void HnswGraph::SetEntry(uint32_t node, int level) {
  const uint64_t entry = (static_cast<uint64_t>(level) << 32) | node;
  std::atomic_ref<uint64_t>(entry_).store(entry, std::memory_order_release);
}

}  // namespace vectra::index
