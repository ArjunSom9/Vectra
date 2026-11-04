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

// A sectioned binary container with per-section checksums, used for index
// files. Each section is [tag:u32][reserved:u32][length:u64][bytes][fnv1a:u64]
// in little-endian order (the only byte order the engine targets).

#ifndef VECTRA_IO_BINARY_FILE_H_
#define VECTRA_IO_BINARY_FILE_H_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vectra::io {

// 64-bit FNV-1a over `bytes`, continuing from `seed`.
uint64_t Fnv1a64(const void* bytes, size_t length,
                 uint64_t seed = 0xcbf29ce484222325ULL);

// Writes sections to `<path>.tmp` and atomically renames it to `path` on
// Close, so a crash never leaves a truncated index in place.
class SectionWriter {
 public:
  static std::optional<SectionWriter> Open(const std::string& path,
                                           std::string* error);

  SectionWriter(SectionWriter&&) noexcept = default;
  SectionWriter& operator=(SectionWriter&&) noexcept = default;
  SectionWriter(const SectionWriter&) = delete;
  SectionWriter& operator=(const SectionWriter&) = delete;

  // Appends one section. Returns false on an I/O error.
  bool Write(uint32_t tag, const void* data, size_t length);

  // Flushes, closes, and publishes the file.
  bool Close(std::string* error);

 private:
  struct FileCloser {
    void operator()(std::FILE* file) const { std::fclose(file); }
  };

  SectionWriter(std::string path, std::FILE* file)
      : path_(std::move(path)), file_(file) {}

  std::string path_;
  std::unique_ptr<std::FILE, FileCloser> file_;
  bool failed_ = false;
};

// Reads sections written by SectionWriter, verifying tags and checksums.
class SectionReader {
 public:
  static std::optional<SectionReader> Open(const std::string& path,
                                           std::string* error);

  SectionReader(SectionReader&&) noexcept = default;
  SectionReader& operator=(SectionReader&&) noexcept = default;
  SectionReader(const SectionReader&) = delete;
  SectionReader& operator=(const SectionReader&) = delete;

  // Reads the next section, which must carry `tag` and be exactly `length`
  // bytes, into `out`.
  bool ReadInto(uint32_t tag, void* out, size_t length, std::string* error);

  // Reads the next section, which must carry `tag`, into `out`.
  bool ReadBytes(uint32_t tag, std::vector<std::byte>* out, std::string* error);

 private:
  struct FileCloser {
    void operator()(std::FILE* file) const { std::fclose(file); }
  };

  explicit SectionReader(std::FILE* file) : file_(file) {}

  bool ReadHeader(uint32_t tag, uint64_t* length, std::string* error);
  bool ReadPayload(void* out, uint64_t length, std::string* error);

  std::unique_ptr<std::FILE, FileCloser> file_;
};

}  // namespace vectra::io

#endif  // VECTRA_IO_BINARY_FILE_H_
