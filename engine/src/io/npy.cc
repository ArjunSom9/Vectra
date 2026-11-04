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

#include "vectra/io/npy.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vectra::io {
namespace {

constexpr char kMagic[] = "\x93NUMPY";
constexpr size_t kMagicLength = 6;

struct FileCloser {
  void operator()(std::FILE* file) const { std::fclose(file); }
};
using File = std::unique_ptr<std::FILE, FileCloser>;

bool Fail(const std::string& message, std::string* error) {
  if (error != nullptr) *error = message;
  return false;
}

// Returns the text after `'key':` in a header dict, trimmed of spaces.
std::string_view ValueAfterKey(std::string_view header, std::string_view key) {
  const std::string quoted = "'" + std::string(key) + "'";
  size_t position = header.find(quoted);
  if (position == std::string_view::npos) return {};
  position = header.find(':', position + quoted.size());
  if (position == std::string_view::npos) return {};
  ++position;
  while (position < header.size() && header[position] == ' ') ++position;
  return header.substr(position);
}

bool ParseHeader(std::string_view header, NpyArray* array, std::string* error) {
  std::string_view descr = ValueAfterKey(header, "descr");
  if (descr.empty() || descr.front() != '\'') {
    return Fail("npy header lacks a descr", error);
  }
  const size_t descr_end = descr.find('\'', 1);
  if (descr_end == std::string_view::npos) {
    return Fail("malformed npy descr", error);
  }
  array->descr = std::string(descr.substr(1, descr_end - 1));
  if (NpyItemSize(array->descr) == 0) {
    return Fail("unsupported npy dtype " + array->descr, error);
  }
  std::string_view fortran = ValueAfterKey(header, "fortran_order");
  if (fortran.substr(0, 5) != "False") {
    return Fail("Fortran-ordered npy arrays are not supported", error);
  }
  std::string_view shape = ValueAfterKey(header, "shape");
  if (shape.empty() || shape.front() != '(') {
    return Fail("npy header lacks a shape", error);
  }
  array->shape.clear();
  size_t value = 0;
  bool in_number = false;
  for (size_t i = 1; i < shape.size() && shape[i] != ')'; ++i) {
    const char c = shape[i];
    if (c >= '0' && c <= '9') {
      value = value * 10 + static_cast<size_t>(c - '0');
      in_number = true;
    } else if (in_number) {
      array->shape.push_back(value);
      value = 0;
      in_number = false;
    }
  }
  if (in_number) array->shape.push_back(value);
  return true;
}

}  // namespace

size_t NpyArray::element_count() const {
  size_t count = 1;
  for (size_t dim : shape) count *= dim;
  return count;
}

size_t NpyItemSize(const std::string& descr) {
  if (descr == "<f4" || descr == "<i4" || descr == "<u4") return 4;
  if (descr == "<f8" || descr == "<i8" || descr == "<u8") return 8;
  if (descr == "|u1" || descr == "|i1" || descr == "|b1") return 1;
  if (descr == "<i2" || descr == "<u2" || descr == "<f2") return 2;
  return 0;
}

std::optional<NpyArray> ReadNpy(const std::string& path, std::string* error) {
  File file(std::fopen(path.c_str(), "rb"));
  if (file == nullptr) {
    Fail("cannot open " + path, error);
    return std::nullopt;
  }
  char magic[kMagicLength] = {};
  unsigned char version[2] = {};
  if (std::fread(magic, 1, kMagicLength, file.get()) != kMagicLength ||
      std::memcmp(magic, kMagic, kMagicLength) != 0 ||
      std::fread(version, 1, 2, file.get()) != 2) {
    Fail(path + " is not an npy file", error);
    return std::nullopt;
  }
  uint32_t header_length = 0;
  if (version[0] == 1) {
    unsigned char bytes[2] = {};
    if (std::fread(bytes, 1, 2, file.get()) != 2) {
      Fail("truncated npy header in " + path, error);
      return std::nullopt;
    }
    header_length = bytes[0] | (uint32_t{bytes[1]} << 8);
  } else if (version[0] == 2 || version[0] == 3) {
    unsigned char bytes[4] = {};
    if (std::fread(bytes, 1, 4, file.get()) != 4) {
      Fail("truncated npy header in " + path, error);
      return std::nullopt;
    }
    header_length = bytes[0] | (uint32_t{bytes[1]} << 8) |
                    (uint32_t{bytes[2]} << 16) | (uint32_t{bytes[3]} << 24);
  } else {
    Fail("unsupported npy version in " + path, error);
    return std::nullopt;
  }
  std::string header(header_length, '\0');
  if (std::fread(header.data(), 1, header_length, file.get()) !=
      header_length) {
    Fail("truncated npy header in " + path, error);
    return std::nullopt;
  }
  NpyArray array;
  if (!ParseHeader(header, &array, error)) return std::nullopt;
  const size_t bytes = array.element_count() * NpyItemSize(array.descr);
  array.data.resize(bytes);
  if (bytes != 0 &&
      std::fread(array.data.data(), 1, bytes, file.get()) != bytes) {
    Fail("truncated npy payload in " + path, error);
    return std::nullopt;
  }
  return array;
}

bool WriteNpy(const std::string& path, const NpyArray& array,
              std::string* error) {
  std::string shape = "(";
  for (size_t i = 0; i < array.shape.size(); ++i) {
    shape += std::to_string(array.shape[i]);
    shape += array.shape.size() == 1 ? ","
                                     : (i + 1 < array.shape.size() ? ", " : "");
  }
  shape += ")";
  std::string header = "{'descr': '" + array.descr +
                       "', 'fortran_order': False, 'shape': " + shape + ", }";
  // Pad so that the payload starts on a 64-byte boundary.
  const size_t prefix = kMagicLength + 2 + 2;
  const size_t total = (prefix + header.size() + 1 + 63) / 64 * 64;
  header.append(total - prefix - header.size() - 1, ' ');
  header.push_back('\n');
  File file(std::fopen(path.c_str(), "wb"));
  if (file == nullptr) return Fail("cannot create " + path, error);
  const unsigned char version[2] = {1, 0};
  const unsigned char length[2] = {
      static_cast<unsigned char>(header.size() & 0xff),
      static_cast<unsigned char>((header.size() >> 8) & 0xff)};
  const bool ok =
      std::fwrite(kMagic, 1, kMagicLength, file.get()) == kMagicLength &&
      std::fwrite(version, 1, 2, file.get()) == 2 &&
      std::fwrite(length, 1, 2, file.get()) == 2 &&
      std::fwrite(header.data(), 1, header.size(), file.get()) ==
          header.size() &&
      (array.data.empty() ||
       std::fwrite(array.data.data(), 1, array.data.size(), file.get()) ==
           array.data.size());
  if (!ok) return Fail("cannot write " + path, error);
  return true;
}

}  // namespace vectra::io
