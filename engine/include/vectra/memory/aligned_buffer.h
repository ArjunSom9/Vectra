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

#ifndef VECTRA_MEMORY_ALIGNED_BUFFER_H_
#define VECTRA_MEMORY_ALIGNED_BUFFER_H_

#include <cstddef>
#include <optional>

namespace vectra::memory {

// Alignment of every buffer: one cache line on the x86-64 production nodes.
inline constexpr size_t kCacheLineBytes = 64;

// Requests at least this large are served by an anonymous mapping that, on
// Linux, is advised to use transparent huge pages so that random traversal of
// multi-gigabyte index structures does not thrash the TLB.
inline constexpr size_t kHugePageBytes = size_t{2} << 20;

// An owning, move-only block of zero-initialized memory aligned to a cache
// line. The address never changes while the buffer is alive, so structures
// stored inside it can refer to each other by integer offset.
class AlignedBuffer {
 public:
  // Creates an empty buffer.
  AlignedBuffer() = default;

  // Allocates at least `size_bytes` of zeroed memory, rounded up to the
  // backing granularity. Returns nullopt when the allocation fails. A zero
  // size yields an empty buffer.
  static std::optional<AlignedBuffer> Allocate(size_t size_bytes);

  AlignedBuffer(AlignedBuffer&& other) noexcept;
  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  ~AlignedBuffer();

  std::byte* data() { return data_; }
  const std::byte* data() const { return data_; }

  // Usable size in bytes (the rounded-up allocation size).
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

 private:
  enum class Backing { kNone, kHeap, kMapping };

  AlignedBuffer(std::byte* data, size_t size, Backing backing)
      : data_(data), size_(size), backing_(backing) {}

  void Release();

  std::byte* data_ = nullptr;
  size_t size_ = 0;
  Backing backing_ = Backing::kNone;
};

}  // namespace vectra::memory

#endif  // VECTRA_MEMORY_ALIGNED_BUFFER_H_
