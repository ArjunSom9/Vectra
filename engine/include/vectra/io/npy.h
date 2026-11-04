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

// Minimal reader and writer for NumPy .npy files (format versions 1.0-3.0,
// C order, little-endian), the interchange format between the Python corpus
// pipelines and the C++ indexer.

#ifndef VECTRA_IO_NPY_H_
#define VECTRA_IO_NPY_H_

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace vectra::io {

struct NpyArray {
  // NumPy dtype descriptor, for example "<f4", "<f8", "<i8", "<u4".
  std::string descr;
  std::vector<size_t> shape;
  std::vector<std::byte> data;

  size_t element_count() const;

  template <typename T>
  const T* As() const {
    return reinterpret_cast<const T*>(data.data());
  }
};

// Reads a C-order, little-endian array. Fails on Fortran order, object
// arrays, or big-endian data.
std::optional<NpyArray> ReadNpy(const std::string& path, std::string* error);

// Writes `array` as a version 1.0 file.
bool WriteNpy(const std::string& path, const NpyArray& array,
              std::string* error);

// Bytes per element of a supported descriptor, or 0 if unsupported.
size_t NpyItemSize(const std::string& descr);

}  // namespace vectra::io

#endif  // VECTRA_IO_NPY_H_
