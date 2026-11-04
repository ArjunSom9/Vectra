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

// RAM residency for the index. A locked page can never be evicted, so graph
// traversal can never degrade into disk I/O (the failure mode that DiskANN
// [12] and the billion-scale challenge [7] document for non-resident indexes).

#ifndef VECTRA_MEMORY_RAM_LOCK_H_
#define VECTRA_MEMORY_RAM_LOCK_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace vectra::memory {

// Pins [address, address + bytes) in physical memory with mlock(2). Returns
// false and describes the failure in `error` (when non-null) if the kernel
// refuses, typically because RLIMIT_MEMLOCK is too small.
bool LockRegion(const void* address, size_t bytes, std::string* error);

// Reverses LockRegion.
bool UnlockRegion(const void* address, size_t bytes, std::string* error);

// Locks every current and future mapping of the process (mlockall). Only
// supported on Linux; elsewhere it fails with a descriptive error.
bool LockAllMemory(std::string* error);

// Returns the RLIMIT_MEMLOCK soft limit in bytes, or nullopt if unlimited.
std::optional<uint64_t> MemlockLimitBytes();

}  // namespace vectra::memory

#endif  // VECTRA_MEMORY_RAM_LOCK_H_
