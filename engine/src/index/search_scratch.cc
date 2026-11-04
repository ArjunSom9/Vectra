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

#include "vectra/index/search_scratch.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vectra::index {
namespace {

size_t Slot(uint32_t key, size_t mask) {
  // Fibonacci hashing spreads consecutive node ids across the table.
  return static_cast<size_t>((uint64_t{key} * 0x9E3779B97F4A7C15ULL) >> 32) &
         mask;
}

}  // namespace

VisitedSet::VisitedSet(size_t initial_slots) {
  size_t slots = 16;
  while (slots < initial_slots) slots <<= 1;
  keys_.assign(slots, 0);
  epochs_.assign(slots, 0);
  mask_ = slots - 1;
}

void VisitedSet::Clear() {
  size_ = 0;
  if (++epoch_ == 0) {
    // The epoch wrapped: forget every stale tag once.
    std::fill(epochs_.begin(), epochs_.end(), 0);
    epoch_ = 1;
  }
}

bool VisitedSet::Insert(uint32_t key) {
  if (2 * (size_ + 1) > keys_.size()) Grow();
  for (size_t slot = Slot(key, mask_);; slot = (slot + 1) & mask_) {
    if (epochs_[slot] != epoch_) {
      epochs_[slot] = epoch_;
      keys_[slot] = key;
      ++size_;
      return true;
    }
    if (keys_[slot] == key) return false;
  }
}

void VisitedSet::Grow() {
  std::vector<uint32_t> live;
  live.reserve(size_);
  for (size_t slot = 0; slot < keys_.size(); ++slot) {
    if (epochs_[slot] == epoch_) live.push_back(keys_[slot]);
  }
  const size_t slots = keys_.size() * 2;
  keys_.assign(slots, 0);
  epochs_.assign(slots, 0);
  mask_ = slots - 1;
  epoch_ = 1;
  size_ = 0;
  for (uint32_t key : live) {
    size_t slot = Slot(key, mask_);
    while (epochs_[slot] == epoch_) slot = (slot + 1) & mask_;
    epochs_[slot] = epoch_;
    keys_[slot] = key;
    ++size_;
  }
}

}  // namespace vectra::index
