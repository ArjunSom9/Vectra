#!/usr/bin/env bash
# Copyright 2025 Texas A&M University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Runs the three-shard development cluster of engine/configs on this host:
# indexes the dev corpus (creating it if needed) and starts one vectra_shard
# per shard on the ports of engine/configs/shard_map.textproto (7101-7103).
# Ctrl-C stops every shard. Needs `make build-engine`.
#
#   scripts/run_local_cluster.sh                  # shards only
#   scripts/run_local_cluster.sh --with-gateway   # plus the gateway on :7000

set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly BUILD="${VECTRA_ENGINE_BUILD:-${ROOT}/build/engine}"
readonly CORPUS="${CORPUS:-${ROOT}/data/corpus/dev}"
readonly CONFIGS="${ROOT}/engine/configs"
readonly PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/kernels:${ROOT}/compiler:${ROOT}/build/gen/python"

pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ ! -x "${BUILD}/vectra_shard" || ! -x "${BUILD}/apps/indexer/indexer" ]]; then
  echo "engine binaries not found under ${BUILD}; run 'make build-engine'" >&2
  exit 1
fi

if [[ ! -f "${CORPUS}/manifest.json" ]]; then
  (cd "${ROOT}" && make corpus CORPUS="${CORPUS}")
fi

for shard_dir in "${CORPUS}"/shards/shard-*; do
  shard_id="${shard_dir##*-}"
  index="${shard_dir}.vidx"
  if [[ ! -f "${index}" ]]; then
    "${BUILD}/apps/indexer/indexer" --config_dir="${CONFIGS}" \
      --shard_dir="${shard_dir}" --out="${index}"
  fi
  port=$((7100 + shard_id))
  "${BUILD}/vectra_shard" --config_dir="${CONFIGS}" --index="${index}" \
    --shard_id="${shard_id}" --listen="127.0.0.1:${port}" \
    --lock_memory=false &
  pids+=("$!")
  echo "shard ${shard_id} on 127.0.0.1:${port}"
done

if [[ "${1:-}" == "--with-gateway" ]]; then
  "${PYTHON}" -m serving.service --config="${ROOT}/serving/configs/service.yaml" &
  pids+=("$!")
  echo "gateway on :7000"
fi

wait
