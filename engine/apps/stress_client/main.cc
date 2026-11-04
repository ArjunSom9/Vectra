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

// stress_client: an open-loop load generator for the A1 and O1 acceptance
// runs.
//
// Requests are sent on a fixed schedule (offered load), independent of how
// fast responses return, and each latency is measured from the request's
// scheduled send time. A closed loop would slow down when the server does
// and hide exactly the queueing it is meant to expose (coordinated omission).
//
//   stress_client --mode=ping --target=localhost:7101 --qps=50000 \
//       --duration=60s                                  # A1 transport test
//   stress_client --mode=search --config_dir=engine/configs --qps=50000 \
//       --duration=30m --queries=data/corpus/queries --report=soak.json

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "vectra/client/async_search.h"
#include "vectra/client/channel_pool.h"
#include "vectra/client/geo_router.h"
#include "vectra/common/latency_histogram.h"
#include "vectra/config/engine_config.h"
#include "vectra/health.grpc.pb.h"
#include "vectra/io/npy.h"
#include "vectra/search.pb.h"

ABSL_FLAG(std::string, mode, "search",
          "ping (no-op RPCs, deliverable A1) or search (routed searches).");
ABSL_FLAG(std::string, target, "",
          "host:port of one shard; ping mode and direct search mode.");
ABSL_FLAG(std::string, config_dir, "",
          "Engine config directory; search mode routes over its shard map.");
ABSL_FLAG(double, qps, 50000.0, "Offered load in requests per second.");
ABSL_FLAG(absl::Duration, duration, absl::Seconds(60), "Measured duration.");
ABSL_FLAG(absl::Duration, warmup, absl::Seconds(5),
          "Initial period excluded from the statistics.");
ABSL_FLAG(int32_t, senders, 4, "Pacing threads issuing requests.");
ABSL_FLAG(int32_t, pollers, 4, "Completion-queue polling threads.");
ABSL_FLAG(int32_t, pool_size, 64, "Channels per endpoint (direct mode).");
ABSL_FLAG(absl::Duration, timeout, absl::Milliseconds(50), "RPC deadline.");
ABSL_FLAG(int32_t, top_k, 32, "Neighbors per search.");
ABSL_FLAG(double, radius_km, 2.0, "Search radius.");
ABSL_FLAG(std::string, queries, "",
          "Directory with query_embedding.npy, query_latitude.npy and "
          "query_longitude.npy; synthetic queries around --center if empty.");
ABSL_FLAG(int32_t, dimension, 128, "Embedding dimension (synthetic mode).");
ABSL_FLAG(double, center_lat, 37.7749, "Synthetic query center latitude.");
ABSL_FLAG(double, center_lng, -122.4194, "Synthetic query center longitude.");
ABSL_FLAG(std::string, report, "", "Write a JSON summary to this path.");

namespace {

using Clock = std::chrono::steady_clock;

// Per-thread histograms, merged at the end.
class Recorder {
 public:
  void Record(Clock::time_point scheduled, bool ok, bool measured) {
    thread_local LatencyHistogramSlot* slot = nullptr;
    if (slot == nullptr) slot = Register();
    if (!measured) return;
    if (!ok) {
      slot->errors.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    const auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - scheduled);
    absl::MutexLock lock(&slot->mutex);
    slot->histogram.Record(static_cast<uint64_t>(latency.count()));
  }

  vectra::LatencyHistogram Merged(uint64_t* errors) {
    vectra::LatencyHistogram merged;
    absl::MutexLock lock(&mutex_);
    for (const std::unique_ptr<LatencyHistogramSlot>& slot : slots_) {
      absl::MutexLock slot_lock(&slot->mutex);
      merged.Merge(slot->histogram);
      *errors += slot->errors.load();
    }
    return merged;
  }

 private:
  struct LatencyHistogramSlot {
    absl::Mutex mutex;
    vectra::LatencyHistogram histogram;
    std::atomic<uint64_t> errors{0};
  };

  LatencyHistogramSlot* Register() {
    absl::MutexLock lock(&mutex_);
    slots_.push_back(std::make_unique<LatencyHistogramSlot>());
    return slots_.back().get();
  }

  absl::Mutex mutex_;
  std::vector<std::unique_ptr<LatencyHistogramSlot>> slots_;
};

struct QuerySet {
  int dimension = 0;
  std::vector<float> embeddings;
  std::vector<double> latitudes;
  std::vector<double> longitudes;

  size_t size() const { return latitudes.size(); }
};

QuerySet LoadQueries() {
  QuerySet set;
  const std::string dir = absl::GetFlag(FLAGS_queries);
  if (!dir.empty()) {
    std::string error;
    std::optional<vectra::io::NpyArray> embeddings =
        vectra::io::ReadNpy(dir + "/query_embedding.npy", &error);
    std::optional<vectra::io::NpyArray> lat =
        vectra::io::ReadNpy(dir + "/query_latitude.npy", &error);
    std::optional<vectra::io::NpyArray> lng =
        vectra::io::ReadNpy(dir + "/query_longitude.npy", &error);
    QCHECK(embeddings && lat && lng) << error;
    set.dimension = static_cast<int>(embeddings->shape[1]);
    set.embeddings.assign(
        embeddings->As<float>(),
        embeddings->As<float>() + embeddings->element_count());
    set.latitudes.assign(lat->As<double>(),
                         lat->As<double>() + lat->element_count());
    set.longitudes.assign(lng->As<double>(),
                          lng->As<double>() + lng->element_count());
    return set;
  }
  // Synthetic: Gaussian embeddings, locations within ~5 km of the center.
  set.dimension = absl::GetFlag(FLAGS_dimension);
  std::mt19937_64 rng(1);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  for (int i = 0; i < 4096; ++i) {
    for (int d = 0; d < set.dimension; ++d)
      set.embeddings.push_back(normal(rng));
    set.latitudes.push_back(absl::GetFlag(FLAGS_center_lat) +
                            0.045 * normal(rng));
    set.longitudes.push_back(absl::GetFlag(FLAGS_center_lng) +
                             0.045 * normal(rng));
  }
  return set;
}

vectra::search::v1::SearchRequest MakeRequest(const QuerySet& queries,
                                              size_t index) {
  vectra::search::v1::SearchRequest request;
  const size_t i = index % queries.size();
  const float* embedding =
      queries.embeddings.data() + i * static_cast<size_t>(queries.dimension);
  request.mutable_query_embedding()->Assign(embedding,
                                            embedding + queries.dimension);
  request.mutable_location()->set_latitude(queries.latitudes[i]);
  request.mutable_location()->set_longitude(queries.longitudes[i]);
  request.set_radius_km(absl::GetFlag(FLAGS_radius_km));
  request.set_top_k(static_cast<uint32_t>(absl::GetFlag(FLAGS_top_k)));
  return request;
}

// Sleeps until `when`, spinning for the last stretch for sub-100 us accuracy.
void WaitUntil(Clock::time_point when) {
  constexpr auto kSpin = std::chrono::microseconds(100);
  if (when - Clock::now() > kSpin) std::this_thread::sleep_until(when - kSpin);
  while (Clock::now() < when) {
  }
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  const std::string mode = absl::GetFlag(FLAGS_mode);
  QCHECK(mode == "ping" || mode == "search") << "--mode must be ping|search";
  const double qps = absl::GetFlag(FLAGS_qps);
  QCHECK_GT(qps, 0.0);

  vectra::client::AsyncClient client(absl::GetFlag(FLAGS_pollers));
  std::unique_ptr<vectra::client::ChannelPool> pool;
  std::vector<std::unique_ptr<vectra::health::v1::HealthService::Stub>> pings;
  std::unique_ptr<vectra::client::GeoRouter> router;
  if (!absl::GetFlag(FLAGS_target).empty()) {
    vectra::client::ChannelPoolOptions options;
    options.size = absl::GetFlag(FLAGS_pool_size);
    pool = std::make_unique<vectra::client::ChannelPool>(
        std::vector<std::string>{absl::GetFlag(FLAGS_target)}, options);
    QCHECK(pool->WaitForConnected(absl::Now() + absl::Seconds(10)))
        << "cannot connect to " << absl::GetFlag(FLAGS_target);
    for (int i = 0; i < pool->size(); ++i) {
      pings.push_back(
          vectra::health::v1::HealthService::NewStub(pool->NextChannel()));
    }
  } else {
    QCHECK(mode == "search") << "--mode=ping needs --target";
    absl::StatusOr<vectra::config::EngineConfig> config =
        vectra::config::LoadEngineConfig(absl::GetFlag(FLAGS_config_dir));
    CHECK_OK(config.status());
    absl::StatusOr<std::unique_ptr<vectra::client::GeoRouter>> created =
        vectra::client::GeoRouter::Create(*config,
                                          absl::GetFlag(FLAGS_pollers));
    CHECK_OK(created.status());
    router = *std::move(created);
    QCHECK(router->WaitForConnected(absl::Now() + absl::Seconds(10)));
  }
  const QuerySet queries = LoadQueries();

  Recorder recorder;
  std::atomic<uint64_t> issued{0};
  std::atomic<uint64_t> completed{0};
  const int senders = std::max(1, absl::GetFlag(FLAGS_senders));
  const auto interval = std::chrono::duration<double>(senders / qps);
  const auto start = Clock::now() + std::chrono::milliseconds(100);
  const auto measure_from =
      start + absl::ToChronoNanoseconds(absl::GetFlag(FLAGS_warmup));
  const auto end =
      measure_from + absl::ToChronoNanoseconds(absl::GetFlag(FLAGS_duration));
  const absl::Duration timeout = absl::GetFlag(FLAGS_timeout);

  std::vector<std::thread> threads;
  for (int s = 0; s < senders; ++s) {
    threads.emplace_back([&, s] {
      // Senders are phase-shifted so their combined schedule is uniform.
      const auto phase = interval * (static_cast<double>(s) / senders);
      for (uint64_t i = 0;; ++i) {
        const auto scheduled =
            start + std::chrono::duration_cast<Clock::duration>(
                        phase + interval * static_cast<double>(i));
        if (scheduled >= end) return;
        WaitUntil(scheduled);
        const bool measured = scheduled >= measure_from;
        const uint64_t n = issued.fetch_add(1, std::memory_order_relaxed);
        if (mode == "ping") {
          client.Ping(pings[n % pings.size()].get(),
                      vectra::health::v1::PingRequest(), timeout,
                      [&, scheduled, measured](const grpc::Status& status) {
                        recorder.Record(scheduled, status.ok(), measured);
                        completed.fetch_add(1, std::memory_order_relaxed);
                      });
        } else if (router != nullptr) {
          router->Search(
              MakeRequest(queries, n),
              [&, scheduled, measured](
                  absl::StatusOr<vectra::search::v1::SearchResponse> result) {
                recorder.Record(scheduled, result.ok(), measured);
                completed.fetch_add(1, std::memory_order_relaxed);
              });
        } else {
          client.Search(
              pool->NextSearchStub(), MakeRequest(queries, n), timeout,
              [&, scheduled, measured](const grpc::Status& status,
                                       vectra::search::v1::SearchResponse*) {
                recorder.Record(scheduled, status.ok(), measured);
                completed.fetch_add(1, std::memory_order_relaxed);
              });
        }
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  // Allow in-flight calls to finish.
  const auto drain_deadline = Clock::now() + std::chrono::seconds(5);
  while (completed.load() < issued.load() && Clock::now() < drain_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  uint64_t errors = 0;
  const vectra::LatencyHistogram histogram = recorder.Merged(&errors);
  const double seconds = absl::ToDoubleSeconds(absl::GetFlag(FLAGS_duration));
  const auto us = [&](double q) {
    return static_cast<double>(histogram.Quantile(q)) / 1000.0;
  };
  const std::string summary = absl::StrFormat(
      "{\"mode\": \"%s\", \"offered_qps\": %.1f, \"achieved_qps\": %.1f, "
      "\"duration_s\": %.1f, \"completed\": %d, \"errors\": %d, "
      "\"p50_us\": %.1f, \"p90_us\": %.1f, \"p99_us\": %.1f, "
      "\"p999_us\": %.1f, \"max_us\": %.1f, \"mean_us\": %.1f}",
      mode, qps, static_cast<double>(histogram.count()) / seconds, seconds,
      histogram.count(), errors, us(0.5), us(0.9), us(0.99), us(0.999),
      static_cast<double>(histogram.max()) / 1000.0, histogram.Mean() / 1000.0);
  absl::PrintF("%s\n", summary);
  if (!absl::GetFlag(FLAGS_report).empty()) {
    std::ofstream(absl::GetFlag(FLAGS_report)) << summary << "\n";
  }
  return errors == 0 ? 0 : 1;
}
