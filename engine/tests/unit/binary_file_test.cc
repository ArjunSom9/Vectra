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

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace vectra::io {
namespace {

std::string TempPath(const std::string& name) {
  return ::testing::TempDir() + "/" + name;
}

TEST(BinaryFileTest, SectionsRoundTrip) {
  const std::string path = TempPath("sections.bin");
  std::optional<SectionWriter> writer = SectionWriter::Open(path, nullptr);
  ASSERT_TRUE(writer.has_value());
  const uint64_t header = 0x1122334455667788ULL;
  const std::vector<float> payload = {1.0f, 2.5f, -3.0f};
  ASSERT_TRUE(writer->Write(1, &header, sizeof(header)));
  ASSERT_TRUE(writer->Write(2, payload.data(), payload.size() * sizeof(float)));
  ASSERT_TRUE(writer->Write(3, nullptr, 0));
  std::string error;
  ASSERT_TRUE(writer->Close(&error)) << error;

  std::optional<SectionReader> reader = SectionReader::Open(path, &error);
  ASSERT_TRUE(reader.has_value()) << error;
  uint64_t read_header = 0;
  ASSERT_TRUE(reader->ReadInto(1, &read_header, sizeof(read_header), &error));
  EXPECT_EQ(read_header, header);
  std::vector<std::byte> bytes;
  ASSERT_TRUE(reader->ReadBytes(2, &bytes, &error)) << error;
  ASSERT_EQ(bytes.size(), payload.size() * sizeof(float));
  EXPECT_EQ(reinterpret_cast<const float*>(bytes.data())[1], 2.5f);
  ASSERT_TRUE(reader->ReadBytes(3, &bytes, &error)) << error;
  EXPECT_TRUE(bytes.empty());
}

TEST(BinaryFileTest, DetectsWrongTagsAndCorruption) {
  const std::string path = TempPath("corrupt.bin");
  std::optional<SectionWriter> writer = SectionWriter::Open(path, nullptr);
  ASSERT_TRUE(writer.has_value());
  const uint32_t value = 7;
  ASSERT_TRUE(writer->Write(5, &value, sizeof(value)));
  ASSERT_TRUE(writer->Close(nullptr));

  std::string error;
  std::optional<SectionReader> reader = SectionReader::Open(path, &error);
  ASSERT_TRUE(reader.has_value());
  uint32_t out = 0;
  EXPECT_FALSE(reader->ReadInto(6, &out, sizeof(out), &error));
  EXPECT_NE(error.find("tag"), std::string::npos);

  // Flip one payload byte (after the 16-byte section header).
  std::FILE* file = std::fopen(path.c_str(), "r+b");
  ASSERT_NE(file, nullptr);
  std::fseek(file, 16, SEEK_SET);
  std::fputc(0xFF, file);
  std::fclose(file);
  reader = SectionReader::Open(path, &error);
  ASSERT_TRUE(reader.has_value());
  EXPECT_FALSE(reader->ReadInto(5, &out, sizeof(out), &error));
  EXPECT_NE(error.find("checksum"), std::string::npos);
}

TEST(BinaryFileTest, Fnv1aMatchesPublishedVectors) {
  // Test vectors from the FNV reference ("" and "a").
  EXPECT_EQ(Fnv1a64("", 0), 0xcbf29ce484222325ULL);
  EXPECT_EQ(Fnv1a64("a", 1), 0xaf63dc4c8601ec8cULL);
}

}  // namespace
}  // namespace vectra::io
