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

#include "vectra/memory/arena.h"

#include <utility>

namespace vectra::memory {

std::optional<Arena> Arena::Create(size_t capacity_bytes) {
  std::optional<AlignedBuffer> buffer = AlignedBuffer::Allocate(capacity_bytes);
  if (!buffer.has_value()) return std::nullopt;
  return Arena(std::move(*buffer));
}

std::optional<size_t> Arena::Allocate(size_t bytes, size_t alignment) {
  const bool power_of_two =
      alignment != 0 && (alignment & (alignment - 1)) == 0;
  if (!power_of_two || alignment > kCacheLineBytes) return std::nullopt;
  const size_t offset = (used_ + alignment - 1) & ~(alignment - 1);
  if (offset > capacity() || bytes > capacity() - offset) return std::nullopt;
  used_ = offset + bytes;
  return offset;
}

}  // namespace vectra::memory
