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
# Prepares an Ubuntu 24.04 H100 host for the inference container: the R570
# driver, the NVIDIA container toolkit, persistence mode, MIG off, and
# profiler access for Nsight Compute (deploy/gpu/driver_matrix.md).
#
#   sudo deploy/gpu/h100_setup.sh

set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "run as root" >&2
  exit 1
fi

apt-get update
apt-get install -y --no-install-recommends nvidia-driver-570-server \
  nvidia-utils-570-server curl gnupg

# NVIDIA container toolkit repository.
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Keep the driver loaded between jobs and give the serving process the
# whole GPU.
nvidia-smi -pm 1
nvidia-smi -mig 0 || true

# Nsight Compute needs performance counters for non-root users.
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
  > /etc/modprobe.d/nvidia-profiling.conf
update-initramfs -u

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "Reboot to apply the profiling permission."
