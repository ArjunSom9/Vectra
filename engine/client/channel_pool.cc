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

#include "vectra/client/channel_pool.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/time/time.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"

namespace vectra::client {

ChannelPool::ChannelPool(const std::vector<std::string>& targets,
                         const ChannelPoolOptions& options)
    : targets_(targets) {
  CHECK(!targets_.empty()) << "ChannelPool needs at least one target";
  const int size = std::max(1, options.size);
  channels_.reserve(static_cast<size_t>(size));
  stubs_.reserve(static_cast<size_t>(size));
  for (int i = 0; i < size; ++i) {
    grpc::ChannelArguments args;
    // Distinct arguments plus a private subchannel pool force a separate
    // HTTP/2 connection per channel.
    args.SetInt("vectra.channel_index", i);
    args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, 1);
    args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, options.keepalive_time_ms);
    args.SetMaxReceiveMessageSize(options.max_receive_message_bytes);
    channels_.push_back(grpc::CreateCustomChannel(
        targets_[static_cast<size_t>(i) % targets_.size()],
        grpc::InsecureChannelCredentials(), args));
    stubs_.push_back(search::v1::SearchService::NewStub(channels_.back()));
  }
}

search::v1::SearchService::Stub* ChannelPool::NextSearchStub() {
  return stubs_[NextIndex()].get();
}

const std::shared_ptr<grpc::Channel>& ChannelPool::NextChannel() {
  return channels_[NextIndex()];
}

bool ChannelPool::WaitForConnected(absl::Time deadline) {
  bool all_ready = true;
  for (const std::shared_ptr<grpc::Channel>& channel : channels_) {
    all_ready =
        channel->WaitForConnected(absl::ToChronoTime(deadline)) && all_ready;
  }
  return all_ready;
}

}  // namespace vectra::client
