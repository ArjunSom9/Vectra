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

// shard_inspector: reports shard ownership, index composition and RAM
// residency, either from live shards (HealthService/Report over the shard
// map) or offline from an index file.
//
//   shard_inspector --config_dir=engine/configs
//   shard_inspector --index=/var/lib/vectra/index/shard-1.vidx

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "vectra/config/engine_config.h"
#include "vectra/health.grpc.pb.h"
#include "vectra/index/shard_index.h"

ABSL_FLAG(std::string, config_dir, "engine/configs",
          "Engine config directory (live mode).");
ABSL_FLAG(std::string, index, "", "Inspect this index file offline instead.");

namespace {

constexpr double kMiB = 1 << 20;

void InspectFile(const std::string& path) {
  std::string error;
  const auto start = std::chrono::steady_clock::now();
  std::unique_ptr<vectra::index::ShardIndex> index =
      vectra::index::ShardIndex::Load(path, &error);
  if (index == nullptr) LOG(FATAL) << error;
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  const vectra::index::IndexOptions& options = index->options();
  absl::PrintF("index      %s (loaded in %.2f s)\n", path, seconds);
  absl::PrintF("vectors    %d of %d reserved\n", index->size(),
               index->capacity());
  absl::PrintF("dimension  %d\n", index->dimension());
  absl::PrintF("hnsw       M=%d M0=%d ef_search=%d rerank=%d top_level=%d\n",
               options.m, options.m0, options.ef_search, options.rerank_depth,
               index->graph().entry().level);
  absl::PrintF("pq         %d bytes/vector (%.0fx smaller than FP32)\n",
               options.pq_subspaces,
               4.0 * index->dimension() / options.pq_subspaces);
  absl::PrintF("resident   %.1f MiB\n",
               static_cast<double>(index->ResidentBytes()) / kMiB);
}

void InspectCluster(const std::string& config_dir) {
  absl::StatusOr<vectra::config::EngineConfig> config =
      vectra::config::LoadEngineConfig(config_dir);
  CHECK_OK(config.status());
  absl::PrintF("%-6s %-22s %-12s %10s %10s %7s %s\n", "shard", "endpoint",
               "region", "vectors", "MiB", "locked", "prefixes");
  for (const auto& shard : config->shard_map.shards()) {
    for (const std::string& endpoint : shard.endpoints()) {
      auto stub = vectra::health::v1::HealthService::NewStub(
          grpc::CreateChannel(endpoint, grpc::InsecureChannelCredentials()));
      grpc::ClientContext context;
      context.set_deadline(absl::ToChronoTime(absl::Now() + absl::Seconds(2)));
      vectra::health::v1::ReportResponse report;
      const grpc::Status status =
          stub->Report(&context, vectra::health::v1::ReportRequest(), &report);
      if (!status.ok()) {
        absl::PrintF("%-6d %-22s %-12s unreachable: %s\n", shard.id(), endpoint,
                     shard.macro_region(), status.error_message());
        continue;
      }
      std::string prefixes;
      for (const std::string& prefix : report.owned_prefixes()) {
        prefixes += prefix + " ";
      }
      absl::PrintF("%-6d %-22s %-12s %10d %10.1f %7s %s\n", shard.id(),
                   endpoint, shard.macro_region(), report.index().num_vectors(),
                   static_cast<double>(report.index().index_bytes()) / kMiB,
                   report.memory().locked() ? "yes" : "NO", prefixes);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  if (!absl::GetFlag(FLAGS_index).empty()) {
    InspectFile(absl::GetFlag(FLAGS_index));
  } else {
    InspectCluster(absl::GetFlag(FLAGS_config_dir));
  }
  return 0;
}
