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
# Run-time latency tuning of an engine node, applied at every boot by the
# vectra-tune.service unit (provision.yml). Kernel boot parameters
# (isolcpus, nohz_full, rcu_nocbs) are set separately in GRUB, because they
# need a reboot.
#
#   tune_latency.sh WORKER_CORES HOUSEKEEPING_CORES NIC
#   tune_latency.sh 4-95 0-3 enp65s0f0np0

set -euo pipefail

readonly WORKER_CORES="${1:?worker cores, e.g. 4-95}"
readonly HOUSEKEEPING_CORES="${2:?housekeeping cores, e.g. 0-3}"
readonly NIC="${3:?network interface}"

# Fixed, maximum clocks: no frequency ramps in the middle of a burst.
for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance > "${governor}"
done

# Transparent huge pages only where requested: the index arena asks for them
# with madvise(MADV_HUGEPAGE); background compaction of other memory would
# stall the request path.
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag

# NIC interrupts belong to the housekeeping cores, never to the cores that
# run the completion-queue workers.
systemctl stop irqbalance 2>/dev/null || true
for irq in $(grep "${NIC}" /proc/interrupts | cut -d: -f1 | tr -d ' '); do
  echo "${HOUSEKEEPING_CORES}" > "/proc/irq/${irq}/smp_affinity_list"
done

# Socket backlogs and buffers sized for tens of thousands of RPCs a second.
sysctl -q -w net.core.somaxconn=65535
sysctl -q -w net.core.netdev_max_backlog=250000
sysctl -q -w net.ipv4.tcp_max_syn_backlog=65535
sysctl -q -w net.core.rmem_max=134217728
sysctl -q -w net.core.wmem_max=134217728
sysctl -q -w net.ipv4.tcp_slow_start_after_idle=0
# Never trade a page of the index for page cache.
sysctl -q -w vm.swappiness=0

echo "tuned: workers ${WORKER_CORES}, housekeeping ${HOUSEKEEPING_CORES}, nic ${NIC}"
