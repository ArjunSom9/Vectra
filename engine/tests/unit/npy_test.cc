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
#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace vectra::io {
namespace {

std::string TempPath(const std::string& name) {
  return ::testing::TempDir() + "/" + name;
}

TEST(NpyTest, RoundTripsTwoDimensionalFloats) {
  NpyArray array;
  array.descr = "<f4";
  array.shape = {3, 2};
  const float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  array.data.resize(sizeof(values));
  std::memcpy(array.data.data(), values, sizeof(values));
  const std::string path = TempPath("matrix.npy");
  std::string error;
  ASSERT_TRUE(WriteNpy(path, array, &error)) << error;
  std::optional<NpyArray> read = ReadNpy(path, &error);
  ASSERT_TRUE(read.has_value()) << error;
  EXPECT_EQ(read->descr, "<f4");
  EXPECT_EQ(read->shape, (std::vector<size_t>{3, 2}));
  EXPECT_EQ(read->element_count(), 6u);
  EXPECT_EQ(read->As<float>()[5], 6.0f);
}

TEST(NpyTest, ParsesHeaderWrittenByNumpy) {
  // Byte-for-byte header of np.save(path, np.arange(3, dtype=np.int64)) with
  // NumPy 2.2 (format 1.0, padded to 64 bytes).
  const std::string header =
      "{'descr': '<i8', 'fortran_order': False, 'shape': (3,), }";
  // NumPy pads magic + version + length + header to a multiple of 64.
  const size_t total = (10 + header.size() + 1 + 63) / 64 * 64;
  std::string padded = header;
  padded.append(total - 10 - header.size() - 1, ' ');
  padded.push_back('\n');
  const std::string path = TempPath("numpy.npy");
  std::FILE* file = std::fopen(path.c_str(), "wb");
  ASSERT_NE(file, nullptr);
  std::fwrite("\x93NUMPY\x01\x00", 1, 8, file);
  const unsigned char length[2] = {static_cast<unsigned char>(padded.size()),
                                   0};
  std::fwrite(length, 1, 2, file);
  std::fwrite(padded.data(), 1, padded.size(), file);
  const int64_t values[3] = {0, 1, 2};
  std::fwrite(values, sizeof(values), 1, file);
  std::fclose(file);

  std::string error;
  std::optional<NpyArray> read = ReadNpy(path, &error);
  ASSERT_TRUE(read.has_value()) << error;
  EXPECT_EQ(read->shape, (std::vector<size_t>{3}));
  EXPECT_EQ(read->As<int64_t>()[2], 2);
}

TEST(NpyTest, RejectsFortranOrderAndUnknownTypes) {
  for (const std::string& header :
       {std::string("{'descr': '<f4', 'fortran_order': True, 'shape': (2,), }"),
        std::string("{'descr': '<c8', 'fortran_order': False, 'shape': (2,), "
                    "}")}) {
    const std::string path = TempPath("bad.npy");
    std::FILE* file = std::fopen(path.c_str(), "wb");
    ASSERT_NE(file, nullptr);
    std::fwrite("\x93NUMPY\x01\x00", 1, 8, file);
    const unsigned char length[2] = {static_cast<unsigned char>(header.size()),
                                     0};
    std::fwrite(length, 1, 2, file);
    std::fwrite(header.data(), 1, header.size(), file);
    std::fclose(file);
    std::string error;
    EXPECT_FALSE(ReadNpy(path, &error).has_value());
    EXPECT_FALSE(error.empty());
  }
}

}  // namespace
}  // namespace vectra::io
