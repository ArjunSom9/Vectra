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

// The project's designated portability header. The Google C++ style guide
// permits compiler-specific extensions only behind a wrapper of this kind, so
// every such extension the engine uses lives here and nowhere else.

#ifndef VECTRA_COMMON_PORT_H_
#define VECTRA_COMMON_PORT_H_

namespace vectra {

// Hints the hardware to fetch the cache line holding `address` for a later
// read. Compiles to nothing on toolchains without the builtin.
inline void PrefetchForRead(const void* address) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(address, /*rw=*/0, /*locality=*/3);
#else
  static_cast<void>(address);
#endif
}

}  // namespace vectra

#endif  // VECTRA_COMMON_PORT_H_
