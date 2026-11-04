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

#ifndef VECTRA_MEMORY_ARENA_H_
#define VECTRA_MEMORY_ARENA_H_

#include <cstddef>
#include <optional>
#include <utility>

#include "vectra/memory/aligned_buffer.h"

namespace vectra::memory {

// A fixed-capacity region that hands out blocks by byte offset.
//
// Index structures keep integer offsets into an arena instead of pointers:
// the layout is position independent (it is saved and loaded verbatim), and
// related records sit next to each other so that one traversal step touches
// contiguous memory.
//
// Example:
//   std::optional<Arena> arena = Arena::Create(1 << 20);
//   std::optional<size_t> offset = arena->Allocate(4096);
//   uint32_t* block = arena->At<uint32_t>(*offset);
class Arena {
 public:
  // Creates an empty, zero-capacity arena.
  Arena() = default;

  // Reserves `capacity_bytes` of zeroed memory. Returns nullopt on failure.
  static std::optional<Arena> Create(size_t capacity_bytes);

  Arena(Arena&& other) noexcept = default;
  Arena& operator=(Arena&& other) noexcept = default;
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  // Reserves `bytes` aligned to `alignment`, which must be a power of two no
  // larger than kCacheLineBytes. Returns the block's offset, or nullopt when
  // the arena cannot fit it.
  std::optional<size_t> Allocate(size_t bytes,
                                 size_t alignment = kCacheLineBytes);

  // Returns the object at `offset`. The caller is responsible for having
  // allocated a block of a suitable size and alignment there.
  template <typename T>
  T* At(size_t offset) {
    return reinterpret_cast<T*>(buffer_.data() + offset);
  }
  template <typename T>
  const T* At(size_t offset) const {
    return reinterpret_cast<const T*>(buffer_.data() + offset);
  }

  const std::byte* base() const { return buffer_.data(); }
  size_t capacity() const { return buffer_.size(); }
  size_t used() const { return used_; }

 private:
  explicit Arena(AlignedBuffer buffer) : buffer_(std::move(buffer)) {}

  AlignedBuffer buffer_;
  size_t used_ = 0;
};

}  // namespace vectra::memory

#endif  // VECTRA_MEMORY_ARENA_H_
