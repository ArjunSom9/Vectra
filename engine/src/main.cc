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

// vectra_shard: serves one shard's index over the asynchronous gRPC API.
//
//   vectra_shard --config_dir=engine/configs --shard_id=1 \
//       --index=/var/lib/vectra/index/shard-1.vidx

#include <pthread.h>
#include <signal.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "vectra/config/engine_config.h"
#include "vectra/index/shard_index.h"
#include "vectra/memory/ram_lock.h"
#include "vectra/server/async_server.h"
#include "vectra/service/health_handler.h"
#include "vectra/service/ingest_handler.h"
#include "vectra/service/search_handler.h"

ABSL_FLAG(std::string, config_dir, "engine/configs",
          "Directory holding the engine .textproto configuration.");
ABSL_FLAG(std::string, index, "", "Path of this shard's .vidx index file.");
ABSL_FLAG(uint32_t, shard_id, 0, "Shard id from the shard map.");
ABSL_FLAG(std::string, listen, "",
          "host:port to bind; defaults to the shard map endpoint's port.");
ABSL_FLAG(std::string, worker_cores, "",
          "cpulist override for completion-queue workers, e.g. 4-95.");
ABSL_FLAG(bool, lock_memory, true, "Pin the index in RAM with mlock.");

namespace {

constexpr char kBuildVersion[] = "vectra-engine 1.0.0";

// Pins every index region; returns what was achieved.
vectra::service::MemoryStatus LockIndex(const vectra::index::ShardIndex& index,
                                        bool lock_all) {
  vectra::service::MemoryStatus status;
  status.locked = true;
  for (const vectra::index::MemoryRegion& region : index.MemoryRegions()) {
    std::string error;
    if (vectra::memory::LockRegion(region.address, region.bytes, &error)) {
      status.locked_bytes += region.bytes;
    } else {
      status.locked = false;
      status.error = error;
    }
  }
  if (lock_all) {
    std::string error;
    if (!vectra::memory::LockAllMemory(&error)) status.error = error;
  }
  return status;
}

std::string ListenAddress(const vectra::config::EngineConfig& config,
                          uint32_t shard_id) {
  if (!absl::GetFlag(FLAGS_listen).empty()) return absl::GetFlag(FLAGS_listen);
  for (const auto& shard : config.shard_map.shards()) {
    if (shard.id() != shard_id || shard.endpoints().empty()) continue;
    std::vector<std::string> parts = absl::StrSplit(shard.endpoints(0), ':');
    return absl::StrCat("0.0.0.0:", parts.back());
  }
  return "0.0.0.0:7101";
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  // Termination signals are handled synchronously by the main thread; block
  // them before any worker thread exists so that all threads inherit the mask.
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGINT);
  sigaddset(&signals, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);

  absl::StatusOr<vectra::config::EngineConfig> config =
      vectra::config::LoadEngineConfig(absl::GetFlag(FLAGS_config_dir));
  CHECK_OK(config.status());
  absl::StatusOr<std::unique_ptr<vectra::routing::RoutingTable>> table =
      vectra::config::BuildRoutingTable(*config);
  CHECK_OK(table.status());

  const uint32_t shard_id = absl::GetFlag(FLAGS_shard_id);
  std::string error;
  std::unique_ptr<vectra::index::ShardIndex> index =
      vectra::index::ShardIndex::Load(absl::GetFlag(FLAGS_index), &error);
  if (index == nullptr) LOG(FATAL) << "cannot load index: " << error;

  const bool lock = absl::GetFlag(FLAGS_lock_memory) &&
                    config->topology().server().lock_memory();
  const vectra::service::MemoryStatus memory =
      lock ? LockIndex(*index, /*lock_all=*/true)
           : vectra::service::MemoryStatus();
  if (lock && !memory.error.empty()) {
    LOG(WARNING) << "index not fully locked in RAM: " << memory.error;
  }

  vectra::service::SearchLimits limits;
  limits.max_radius_km = config->geohash().max_radius_km();
  limits.default_radius_km = config->geohash().default_radius_km();
  vectra::service::SearchHandler search(index.get(), shard_id, limits);
  vectra::service::IngestHandler ingest(index.get(), table->get(), shard_id);
  vectra::service::HealthHandler health(index.get(), shard_id,
                                        (*table)->PrefixesForShard(shard_id),
                                        memory, kBuildVersion);

  const vectra::config::v1::ServerOptions& server_config =
      config->topology().server();
  std::string cores = absl::GetFlag(FLAGS_worker_cores);
  if (cores.empty()) cores = server_config.worker_cores();
  absl::StatusOr<std::vector<int>> worker_cores =
      vectra::config::ParseCpuList(cores);
  CHECK_OK(worker_cores.status());

  vectra::server::ServerOptions options;
  options.listen_address = ListenAddress(*config, shard_id);
  options.worker_cores = *std::move(worker_cores);
  options.threads_per_queue =
      static_cast<int>(std::max(1u, server_config.threads_per_queue()));
  options.requests_per_queue =
      static_cast<int>(std::max(1u, server_config.requests_per_queue()));
  if (server_config.max_receive_message_bytes() > 0) {
    options.max_receive_message_bytes =
        static_cast<int>(server_config.max_receive_message_bytes());
  }
  vectra::server::AsyncServer server(
      options, vectra::server::ShardHandlers{&search, &ingest, &health});
  CHECK_OK(server.Start());
  LOG(INFO) << "shard " << shard_id << " serving " << index->size()
            << " vectors (" << index->ResidentBytes() / (1 << 20)
            << " MiB resident, " << memory.locked_bytes / (1 << 20)
            << " MiB locked) on port " << server.port();

  int received = 0;
  sigwait(&signals, &received);
  LOG(INFO) << "signal " << received << ": draining";
  server.Shutdown(absl::Seconds(5));
  return 0;
}
