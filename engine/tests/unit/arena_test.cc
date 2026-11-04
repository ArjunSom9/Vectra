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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "vectra/memory/aligned_buffer.h"
#include "vectra/memory/ram_lock.h"

namespace vectra::memory {
namespace {

TEST(AlignedBufferTest, SmallAndHugeAllocationsAreAlignedAndZeroed) {
  for (size_t size : {size_t{1}, size_t{100}, kHugePageBytes + 1}) {
    std::optional<AlignedBuffer> buffer = AlignedBuffer::Allocate(size);
    ASSERT_TRUE(buffer.has_value());
    EXPECT_GE(buffer->size(), size);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer->data()) % kCacheLineBytes,
              0u);
    for (size_t i = 0; i < size; i += 4093) {
      EXPECT_EQ(buffer->data()[i], std::byte{0});
    }
  }
  std::optional<AlignedBuffer> empty = AlignedBuffer::Allocate(0);
  ASSERT_TRUE(empty.has_value());
  EXPECT_TRUE(empty->empty());
}

TEST(AlignedBufferTest, MoveTransfersOwnership) {
  std::optional<AlignedBuffer> a = AlignedBuffer::Allocate(256);
  ASSERT_TRUE(a.has_value());
  std::byte* data = a->data();
  AlignedBuffer b = std::move(*a);
  EXPECT_EQ(b.data(), data);
  EXPECT_TRUE(a->empty());  // NOLINT(bugprone-use-after-move)
}

TEST(ArenaTest, HandsOutAlignedNonOverlappingBlocks) {
  std::optional<Arena> arena = Arena::Create(1024);
  ASSERT_TRUE(arena.has_value());
  const std::optional<size_t> a = arena->Allocate(10, 1);
  const std::optional<size_t> b = arena->Allocate(100, 64);
  const std::optional<size_t> c = arena->Allocate(4, 4);
  ASSERT_TRUE(a && b && c);
  EXPECT_EQ(*a, 0u);
  EXPECT_EQ(*b % 64, 0u);
  EXPECT_GE(*b, *a + 10);
  EXPECT_GE(*c, *b + 100);
  EXPECT_FALSE(arena->Allocate(4096).has_value());    // Too large.
  EXPECT_FALSE(arena->Allocate(8, 3).has_value());    // Not a power of two.
  EXPECT_FALSE(arena->Allocate(8, 128).has_value());  // Beyond a cache line.
  *arena->At<uint32_t>(*c) = 42;
  EXPECT_EQ(*arena->At<uint32_t>(*c), 42u);
}

TEST(RamLockTest, LocksAndUnlocksSmallRegionWithinLimit) {
  std::optional<AlignedBuffer> buffer = AlignedBuffer::Allocate(4096);
  ASSERT_TRUE(buffer.has_value());
  const std::optional<uint64_t> limit = MemlockLimitBytes();
  if (limit.has_value() && *limit < 4096) {
    GTEST_SKIP() << "RLIMIT_MEMLOCK is below one page";
  }
  std::string error;
  ASSERT_TRUE(LockRegion(buffer->data(), buffer->size(), &error)) << error;
  EXPECT_TRUE(UnlockRegion(buffer->data(), buffer->size(), &error)) << error;
}

}  // namespace
}  // namespace vectra::memory
