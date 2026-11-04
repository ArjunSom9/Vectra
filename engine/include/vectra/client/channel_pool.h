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

#ifndef VECTRA_CLIENT_CHANNEL_POOL_H_
#define VECTRA_CLIENT_CHANNEL_POOL_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "vectra/search.grpc.pb.h"

namespace vectra::client {

struct ChannelPoolOptions {
  int size = 64;
  int keepalive_time_ms = 10000;
  int max_receive_message_bytes = 16 << 20;
};

// A pre-warmed pool of channels to one shard, used round-robin.
//
// One channel is one HTTP/2 connection whose concurrency is capped by the
// peer's SETTINGS_MAX_CONCURRENT_STREAMS [20]; beyond it, gRPC queues calls
// inside the client [27], which would erase the server's asynchronous
// advantage at 50,000 QPS. Each channel gets a distinct channel argument and
// a local subchannel pool so that gRPC cannot collapse them onto one shared
// connection, as the performance guide prescribes [27]. Channel i connects
// to replica i mod targets.size(), so load is spread over every replica.
// Thread-safe.
class ChannelPool {
 public:
  // `targets` (host:port of each replica) must not be empty.
  ChannelPool(const std::vector<std::string>& targets,
              const ChannelPoolOptions& options);
  ChannelPool(const ChannelPool&) = delete;
  ChannelPool& operator=(const ChannelPool&) = delete;

  // The next stub in round-robin order.
  search::v1::SearchService::Stub* NextSearchStub();
  const std::shared_ptr<grpc::Channel>& NextChannel();

  // Connects every channel; returns false if any is not ready by `deadline`.
  bool WaitForConnected(absl::Time deadline);

  int size() const { return static_cast<int>(channels_.size()); }
  const std::vector<std::string>& targets() const { return targets_; }

 private:
  size_t NextIndex() {
    return static_cast<size_t>(next_.fetch_add(1, std::memory_order_relaxed) %
                               channels_.size());
  }

  std::vector<std::string> targets_;
  std::vector<std::shared_ptr<grpc::Channel>> channels_;
  std::vector<std::unique_ptr<search::v1::SearchService::Stub>> stubs_;
  std::atomic<uint64_t> next_{0};
};

}  // namespace vectra::client

#endif  // VECTRA_CLIENT_CHANNEL_POOL_H_
