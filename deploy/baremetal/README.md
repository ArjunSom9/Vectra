# Engine nodes: bare-metal provisioning

Engine nodes run directly on bare metal (see
[ADR-007](../../docs/decisions/adr-007-bare-metal-over-k8s.md)). One shard
process owns each node. It pins one completion-queue worker per isolated
core and locks its whole index in RAM.

## Files

| File | Purpose |
|---|---|
| `provision.yml` | Ansible playbook (ansible-core 2.18.6): binaries, configuration, limits, CPU isolation, systemd units |
| `tune_latency.sh` | Run-time tuning at every boot: performance governor, THP `madvise`, NIC IRQs off the worker cores, socket sysctls |
| `mlock_limits.conf` | Unlimited `RLIMIT_MEMLOCK` for the `vectra` account |

## Bring-up

```bash
make dist                                    # binaries into dist/engine
ansible-playbook deploy/baremetal/provision.yml -i deploy/envs/prod.yaml
# Copy each shard's index to /var/lib/vectra/index/shard-<id>.vidx, then
# reboot once so the isolcpus/nohz_full boot parameters apply.
ansible engine -i deploy/envs/prod.yaml -b -m reboot
```

Each host's inventory entry sets `shard_id`, `shard_port`, `numa_node`,
`worker_cores` (the isolated cores the workers are pinned to),
`housekeeping_cores` (everything else, including NIC interrupts), and `nic`.

## Rules

1. **No CPU quotas.** CFS throttling breaks thread-per-core determinism. Use
   bare metal, or a `cpuset` without a quota.
2. **One NUMA node.** The workers, the index arena (`numactl --membind`) and
   the NIC interrupts all stay on the same node. Losing cross-socket
   locality erases the flat-memory cache gains.
3. **Lock the index.** `--lock_memory=true` with `LimitMEMLOCK=infinity`. A
   shard whose index is not locked reports `vectra_shard_memory_locked 0`,
   and the `IndexNotLocked` alert fires.
4. **Verify after every change** with
   `shard_inspector --config_dir=/opt/vectra/configs` (shard ownership, index
   composition, per-worker counters) and the runbook
   [node-bringup](../../docs/runbooks/node-bringup.md).
