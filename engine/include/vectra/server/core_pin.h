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

#ifndef VECTRA_SERVER_CORE_PIN_H_
#define VECTRA_SERVER_CORE_PIN_H_

#include "absl/status/status.h"

namespace vectra::server {

// Restricts the calling thread to one CPU core, so a completion-queue worker
// keeps its L1/L2 working set and never migrates. Linux only; elsewhere
// returns UNIMPLEMENTED and the worker runs unpinned.
absl::Status PinCurrentThreadToCore(int core);

// Number of hardware threads available to the process.
int AvailableHardwareThreads();

}  // namespace vectra::server

#endif  // VECTRA_SERVER_CORE_PIN_H_
