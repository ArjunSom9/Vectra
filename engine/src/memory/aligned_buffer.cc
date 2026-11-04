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

#include "vectra/memory/aligned_buffer.h"

#include <sys/mman.h>

#include <cstdlib>
#include <cstring>
#include <utility>

namespace vectra::memory {
namespace {

size_t RoundUp(size_t value, size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

}  // namespace

std::optional<AlignedBuffer> AlignedBuffer::Allocate(size_t size_bytes) {
  if (size_bytes == 0) return AlignedBuffer();
  if (size_bytes >= kHugePageBytes) {
    const size_t length = RoundUp(size_bytes, kHugePageBytes);
    void* mapping = mmap(nullptr, length, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, /*fd=*/-1, /*offset=*/0);
    if (mapping == MAP_FAILED) return std::nullopt;
#if defined(__linux__)
    // Best effort: the kernel may decline, which only costs TLB reach.
    madvise(mapping, length, MADV_HUGEPAGE);
#endif
    // Anonymous mappings are zero-filled by the kernel.
    return AlignedBuffer(static_cast<std::byte*>(mapping), length,
                         Backing::kMapping);
  }
  const size_t length = RoundUp(size_bytes, kCacheLineBytes);
  void* memory = std::aligned_alloc(kCacheLineBytes, length);
  if (memory == nullptr) return std::nullopt;
  std::memset(memory, 0, length);
  return AlignedBuffer(static_cast<std::byte*>(memory), length, Backing::kHeap);
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      backing_(std::exchange(other.backing_, Backing::kNone)) {}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
  if (this != &other) {
    Release();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    backing_ = std::exchange(other.backing_, Backing::kNone);
  }
  return *this;
}

AlignedBuffer::~AlignedBuffer() { Release(); }

void AlignedBuffer::Release() {
  switch (backing_) {
    case Backing::kNone:
      break;
    case Backing::kHeap:
      std::free(data_);
      break;
    case Backing::kMapping:
      munmap(data_, size_);
      break;
  }
  data_ = nullptr;
  size_ = 0;
  backing_ = Backing::kNone;
}

}  // namespace vectra::memory
