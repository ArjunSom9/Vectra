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
# Sets up a development host (Ubuntu 24.04, x86-64): GCC 14, the Python 3.12
# environment from requirements-lock.txt, and gRPC 1.71.0 (with its vendored
# protobuf 29.0 and Abseil 20240722.0) installed under GRPC_PREFIX.
#
#   scripts/bootstrap.sh              # everything
#   scripts/bootstrap.sh --python     # the Python environment only
#   scripts/bootstrap.sh --grpc       # gRPC only

set -euo pipefail

readonly GRPC_VERSION="1.71.0"
readonly GRPC_PREFIX="${GRPC_PREFIX:-/opt/vectra/grpc-${GRPC_VERSION}}"
readonly UV_VERSION="0.7.8"
readonly PYTHON_VERSION="3.12"
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly JOBS="$(nproc)"

log() { printf '\033[1m[bootstrap]\033[0m %s\n' "$*"; }

install_system_packages() {
  log "system packages (GCC 14, Python ${PYTHON_VERSION})"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    build-essential gcc-14 g++-14 git curl ca-certificates pkg-config \
    "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" \
    "python${PYTHON_VERSION}-dev"
}

install_python_environment() {
  log "uv ${UV_VERSION} and the locked Python environment in .venv"
  if ! command -v uv >/dev/null || [[ "$(uv --version)" != *"${UV_VERSION}"* ]]; then
    "python${PYTHON_VERSION}" -m pip install --user "uv==${UV_VERSION}"
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
  cd "${ROOT}"
  uv venv --python "${PYTHON_VERSION}" .venv
  # --exclude-newer also bounds the build dependencies of source-only
  # packages (hnswlib) to the project window.
  uv pip sync --python .venv/bin/python \
    --exclude-newer 2025-05-30T23:59:59Z requirements-lock.txt
  uv pip install --python .venv/bin/python --no-deps -e .
  scripts/gen_proto.sh
}

build_grpc() {
  if [[ -f "${GRPC_PREFIX}/lib/cmake/grpc/gRPCConfig.cmake" ]]; then
    log "gRPC ${GRPC_VERSION} already installed in ${GRPC_PREFIX}"
    return
  fi
  log "building gRPC ${GRPC_VERSION} into ${GRPC_PREFIX}"
  local src
  src="$(mktemp -d)"
  git clone --depth 1 --branch "v${GRPC_VERSION}" --recurse-submodules \
    --shallow-submodules https://github.com/grpc/grpc "${src}/grpc"
  sudo mkdir -p "${GRPC_PREFIX}"
  sudo chown "$(id -u):$(id -g)" "${GRPC_PREFIX}"
  CC=gcc-14 CXX=g++-14 "${ROOT}/.venv/bin/cmake" -S "${src}/grpc" \
    -B "${src}/build" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="${ROOT}/.venv/bin/ninja" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_INSTALL_PREFIX="${GRPC_PREFIX}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=OFF \
    -DgRPC_BUILD_CSHARP_EXT=OFF \
    -DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_PHP_PLUGIN=OFF \
    -DgRPC_BUILD_GRPC_RUBY_PLUGIN=OFF \
    -DABSL_PROPAGATE_CXX_STD=ON
  "${ROOT}/.venv/bin/cmake" --build "${src}/build" -j "${JOBS}"
  "${ROOT}/.venv/bin/cmake" --install "${src}/build"
  log "gRPC installed; build the engine with GRPC_PREFIX=${GRPC_PREFIX}"
}

main() {
  case "${1:-all}" in
    --python) install_python_environment ;;
    --grpc) build_grpc ;;
    all)
      install_system_packages
      install_python_environment
      build_grpc
      ;;
    *)
      echo "usage: $0 [--python|--grpc]" >&2
      exit 2
      ;;
  esac
}

main "$@"
