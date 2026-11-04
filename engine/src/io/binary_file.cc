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

#include "vectra/io/binary_file.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vectra::io {
namespace {

constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

struct SectionHeader {
  uint32_t tag;
  uint32_t reserved;
  uint64_t length;
};

std::string ErrnoMessage(const std::string& what) {
  return what + ": " + std::strerror(errno);
}

}  // namespace

uint64_t Fnv1a64(const void* bytes, size_t length, uint64_t seed) {
  const auto* data = static_cast<const unsigned char*>(bytes);
  uint64_t hash = seed;
  for (size_t i = 0; i < length; ++i) {
    hash ^= data[i];
    hash *= kFnvPrime;
  }
  return hash;
}

std::optional<SectionWriter> SectionWriter::Open(const std::string& path,
                                                 std::string* error) {
  std::FILE* file = std::fopen((path + ".tmp").c_str(), "wb");
  if (file == nullptr) {
    if (error != nullptr) *error = ErrnoMessage("cannot create " + path);
    return std::nullopt;
  }
  return SectionWriter(path, file);
}

bool SectionWriter::Write(uint32_t tag, const void* data, size_t length) {
  if (failed_) return false;
  const SectionHeader header{tag, 0, length};
  const uint64_t checksum = Fnv1a64(data, length);
  failed_ = std::fwrite(&header, sizeof(header), 1, file_.get()) != 1 ||
            (length != 0 && std::fwrite(data, length, 1, file_.get()) != 1) ||
            std::fwrite(&checksum, sizeof(checksum), 1, file_.get()) != 1;
  return !failed_;
}

bool SectionWriter::Close(std::string* error) {
  std::FILE* file = file_.release();
  const bool closed = file != nullptr && std::fclose(file) == 0;
  if (failed_ || !closed) {
    if (error != nullptr) *error = ErrnoMessage("cannot write " + path_);
    std::remove((path_ + ".tmp").c_str());
    return false;
  }
  if (std::rename((path_ + ".tmp").c_str(), path_.c_str()) != 0) {
    if (error != nullptr) *error = ErrnoMessage("cannot publish " + path_);
    return false;
  }
  return true;
}

std::optional<SectionReader> SectionReader::Open(const std::string& path,
                                                 std::string* error) {
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    if (error != nullptr) *error = ErrnoMessage("cannot open " + path);
    return std::nullopt;
  }
  return SectionReader(file);
}

bool SectionReader::ReadHeader(uint32_t tag, uint64_t* length,
                               std::string* error) {
  SectionHeader header{};
  if (std::fread(&header, sizeof(header), 1, file_.get()) != 1) {
    if (error != nullptr) *error = "truncated index file (section header)";
    return false;
  }
  if (header.tag != tag) {
    if (error != nullptr) {
      *error = "unexpected section tag " + std::to_string(header.tag) +
               " (wanted " + std::to_string(tag) + ")";
    }
    return false;
  }
  *length = header.length;
  return true;
}

bool SectionReader::ReadPayload(void* out, uint64_t length,
                                std::string* error) {
  if (length != 0 && std::fread(out, length, 1, file_.get()) != 1) {
    if (error != nullptr) *error = "truncated index file (section payload)";
    return false;
  }
  uint64_t checksum = 0;
  if (std::fread(&checksum, sizeof(checksum), 1, file_.get()) != 1 ||
      checksum != Fnv1a64(out, length)) {
    if (error != nullptr) *error = "index section checksum mismatch";
    return false;
  }
  return true;
}

bool SectionReader::ReadInto(uint32_t tag, void* out, size_t length,
                             std::string* error) {
  uint64_t stored = 0;
  if (!ReadHeader(tag, &stored, error)) return false;
  if (stored != length) {
    if (error != nullptr) {
      *error = "section " + std::to_string(tag) + " has " +
               std::to_string(stored) + " bytes, expected " +
               std::to_string(length);
    }
    return false;
  }
  return ReadPayload(out, stored, error);
}

bool SectionReader::ReadBytes(uint32_t tag, std::vector<std::byte>* out,
                              std::string* error) {
  uint64_t stored = 0;
  if (!ReadHeader(tag, &stored, error)) return false;
  out->resize(stored);
  return ReadPayload(out->data(), stored, error);
}

}  // namespace vectra::io
