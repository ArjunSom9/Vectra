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

#include "vectra/memory/ram_lock.h"

#include <sys/mman.h>
#include <sys/resource.h>

#include <cerrno>
#include <cstring>
#include <string>

namespace vectra::memory {
namespace {

bool Fail(const char* operation, std::string* error) {
  if (error != nullptr) {
    *error = std::string(operation) + " failed: " + std::strerror(errno);
  }
  return false;
}

}  // namespace

bool LockRegion(const void* address, size_t bytes, std::string* error) {
  if (bytes == 0) return true;
  if (mlock(address, bytes) != 0) return Fail("mlock", error);
  return true;
}

bool UnlockRegion(const void* address, size_t bytes, std::string* error) {
  if (bytes == 0) return true;
  if (munlock(address, bytes) != 0) return Fail("munlock", error);
  return true;
}

bool LockAllMemory(std::string* error) {
#if defined(__linux__)
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) return Fail("mlockall", error);
  return true;
#else
  if (error != nullptr) *error = "mlockall is only supported on Linux";
  return false;
#endif
}

std::optional<uint64_t> MemlockLimitBytes() {
  rlimit limit{};
  if (getrlimit(RLIMIT_MEMLOCK, &limit) != 0) return 0;
  if (limit.rlim_cur == RLIM_INFINITY) return std::nullopt;
  return static_cast<uint64_t>(limit.rlim_cur);
}

}  // namespace vectra::memory
