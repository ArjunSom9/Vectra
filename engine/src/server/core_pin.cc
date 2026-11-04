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

#include "vectra/server/core_pin.h"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include <algorithm>
#include <cstring>
#include <thread>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace vectra::server {

absl::Status PinCurrentThreadToCore(int core) {
#if defined(__linux__)
  if (core < 0 || core >= CPU_SETSIZE) {
    return absl::InvalidArgumentError(
        absl::StrCat("core ", core, " is out of range"));
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  const int result = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (result != 0) {
    return absl::InternalError(absl::StrCat("pthread_setaffinity_np(", core,
                                            "): ", std::strerror(result)));
  }
  return absl::OkStatus();
#else
  return absl::UnimplementedError(
      absl::StrCat("core pinning is Linux-only; core ", core, " not pinned"));
#endif
}

int AvailableHardwareThreads() {
  return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}

}  // namespace vectra::server
