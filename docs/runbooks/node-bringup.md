# Runbook: engine node bring-up and drain

**When:** a new or replaced shard host, or after maintenance.
**Owner:** engine on-call. **Time:** about 30 minutes plus the index copy.

## Bring up

1. Add the host to the inventory (`deploy/envs/prod.yaml`) with its
   `shard_id`, `shard_port`, `numa_node`, `worker_cores`,
   `housekeeping_cores` and `nic`. Add its `host:port` to the shard's
   `endpoints` in `engine/configs/shard_map.textproto` (one PR, reviewed).
2. Provision:
   ```bash
   make dist
   ansible-playbook deploy/baremetal/provision.yml -i deploy/envs/prod.yaml --limit <host>
   ```
3. Copy the shard's index to `/var/lib/vectra/index/shard-<id>.vidx`
   ([index-rebuild](index-rebuild.md) builds it), then reboot once so the
   CPU-isolation boot parameters apply.
4. Verify the host before it takes traffic:
   ```bash
   cat /sys/devices/system/cpu/isolated              # = worker_cores
   systemctl status vectra-tune vectra-shard
   /opt/vectra/bin/shard_inspector --config_dir=/opt/vectra/configs
   ```
   The inspector's report for the shard must show the owned prefixes of the
   shard map, `num_vectors` equal to the shard's row count, `locked: true`,
   and one worker per worker core with its pinned `core`.
5. Check the NIC: interrupts only on housekeeping cores
   (`grep <nic> /proc/interrupts`), and the coalescing settings of the
   hardware profile (`ethtool -c <nic>`).
6. Smoke-test from the monitoring host:
   ```bash
   stress_client --mode=ping --target=<host>:<port> --qps=5000 --duration=30s
   ```
   p99 should be within the A1 budget (< 100 µs dispatch on the reference
   fabric), with no errors.
7. Roll the gateways (they read endpoints at start-up) and watch
   `vectra_shard_up` and the retrieval latency panel in Grafana.

## Drain

1. Remove the endpoint from `shard_map.textproto` and roll the gateways. A
   shard with one remaining replica stays in service; never drain the last
   replica of a shard.
2. `systemctl stop vectra-shard`. The server stops arming new calls,
   finishes in-flight ones within the shutdown deadline, and drains its
   completion queues.
