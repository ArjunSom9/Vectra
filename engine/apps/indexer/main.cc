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

// indexer: builds one shard's index from the columnar .npy files written by
// data/pipelines/split_shards.py.
//
//   indexer --config_dir=engine/configs \
//       --shard_dir=data/corpus/shards/shard-1 \
//       --out=/var/lib/vectra/index/shard-1.vidx

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "vectra/config/engine_config.h"
#include "vectra/index/property_store.h"
#include "vectra/index/shard_index.h"
#include "vectra/io/npy.h"

ABSL_FLAG(std::string, config_dir, "engine/configs",
          "Directory holding hnsw_params.textproto and friends.");
ABSL_FLAG(std::string, shard_dir, "",
          "Directory of one shard's .npy columns (split_shards.py output).");
ABSL_FLAG(std::string, out, "", "Output .vidx path.");
ABSL_FLAG(int32_t, build_threads, 0,
          "Overrides hnsw_params build_threads when positive.");

namespace {

using ::vectra::io::NpyArray;

absl::StatusOr<NpyArray> ReadColumn(const std::string& dir,
                                    const std::string& name,
                                    const std::string& descr, size_t rows) {
  std::string error;
  std::optional<NpyArray> array =
      vectra::io::ReadNpy(absl::StrCat(dir, "/", name, ".npy"), &error);
  if (!array.has_value()) return absl::NotFoundError(error);
  if (array->descr != descr || array->shape.empty() ||
      (rows != 0 && array->shape[0] != rows)) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, ".npy must be ", descr, " with ", rows, " rows"));
  }
  return *std::move(array);
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  const std::string dir = absl::GetFlag(FLAGS_shard_dir);
  QCHECK(!dir.empty() && !absl::GetFlag(FLAGS_out).empty())
      << "--shard_dir and --out are required";

  absl::StatusOr<vectra::config::EngineConfig> config =
      vectra::config::LoadEngineConfig(absl::GetFlag(FLAGS_config_dir));
  CHECK_OK(config.status());
  vectra::index::IndexOptions options =
      vectra::config::ToIndexOptions(config->index);
  if (absl::GetFlag(FLAGS_build_threads) > 0) {
    options.build_threads = absl::GetFlag(FLAGS_build_threads);
  }

  absl::StatusOr<NpyArray> embeddings = ReadColumn(dir, "embeddings", "<f4", 0);
  CHECK_OK(embeddings.status());
  QCHECK_EQ(embeddings->shape.size(), 2u);
  QCHECK_EQ(embeddings->shape[1], static_cast<size_t>(options.dimension))
      << "embedding dimension disagrees with hnsw_params";
  const size_t rows = embeddings->shape[0];

  absl::StatusOr<NpyArray> ids = ReadColumn(dir, "property_id", "<i8", rows);
  absl::StatusOr<NpyArray> lat = ReadColumn(dir, "latitude", "<f8", rows);
  absl::StatusOr<NpyArray> lng = ReadColumn(dir, "longitude", "<f8", rows);
  absl::StatusOr<NpyArray> living =
      ReadColumn(dir, "living_area_sqft", "<f4", rows);
  absl::StatusOr<NpyArray> lot = ReadColumn(dir, "lot_area_sqft", "<f4", rows);
  absl::StatusOr<NpyArray> beds = ReadColumn(dir, "bedrooms", "<i4", rows);
  absl::StatusOr<NpyArray> baths = ReadColumn(dir, "bathrooms", "<f4", rows);
  absl::StatusOr<NpyArray> built = ReadColumn(dir, "year_built", "<i4", rows);
  absl::StatusOr<NpyArray> price =
      ReadColumn(dir, "last_sale_price_usd", "<f8", rows);
  absl::StatusOr<NpyArray> sale_day =
      ReadColumn(dir, "last_sale_epoch_day", "<i4", rows);
  for (const absl::StatusOr<NpyArray>* column :
       {&ids, &lat, &lng, &living, &lot, &beds, &baths, &built, &price,
        &sale_day}) {
    CHECK_OK(column->status());
  }

  std::vector<vectra::index::PropertyRecord> records(rows);
  for (size_t i = 0; i < rows; ++i) {
    vectra::index::PropertyRecord& record = records[i];
    record.id = ids->As<int64_t>()[i];
    record.location = {lat->As<double>()[i], lng->As<double>()[i]};
    record.attributes.living_area_sqft = living->As<float>()[i];
    record.attributes.lot_area_sqft = lot->As<float>()[i];
    record.attributes.bedrooms = beds->As<int32_t>()[i];
    record.attributes.bathrooms = baths->As<float>()[i];
    record.attributes.year_built = built->As<int32_t>()[i];
    record.attributes.last_sale_price_usd = price->As<double>()[i];
    record.attributes.last_sale_epoch_day = sale_day->As<int32_t>()[i];
  }

  LOG(INFO) << "building shard index over " << rows
            << " vectors (M=" << options.m
            << ", ef_construction=" << options.ef_construction << ", PQ "
            << options.pq_subspaces << " bytes/vector, "
            << options.build_threads << " threads)";
  const auto start = std::chrono::steady_clock::now();
  std::string error;
  std::unique_ptr<vectra::index::ShardIndex> index =
      vectra::index::ShardIndex::Build(options, embeddings->As<float>(),
                                       records.data(), rows, &error);
  if (index == nullptr) LOG(FATAL) << "build failed: " << error;
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  if (!index->Save(absl::GetFlag(FLAGS_out), &error)) {
    LOG(FATAL) << "save failed: " << error;
  }
  LOG(INFO) << "wrote " << absl::GetFlag(FLAGS_out) << " in " << seconds
            << " s; resident " << index->ResidentBytes() / (1 << 20) << " MiB";
  return 0;
}
