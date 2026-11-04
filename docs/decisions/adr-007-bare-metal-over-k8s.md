# ADR-007: Engine nodes on bare metal, not Kubernetes

* **Status:** Accepted (January 2025)
* **Deciders:** Workstream A, operations
* **Runbook:** [node-bringup](../runbooks/node-bringup.md), [`deploy/baremetal/`](../../deploy/baremetal/README.md)

## Context

The engine's latency model is thread-per-core: isolated cores, pinned
completion-queue workers, NIC interrupts on housekeeping cores, the index
locked in RAM on one NUMA node. Container platforms commonly impose CFS
quotas (throttling a process at period boundaries), share cores with
neighbors, and restrict `mlock`.

## Decision

Run shards directly on bare-metal Ubuntu 24.04 hosts, provisioned by Ansible
(`deploy/baremetal/provision.yml`): `isolcpus`/`nohz_full` on the worker
cores, a systemd unit with `numactl --cpunodebind/--membind`,
`LimitMEMLOCK=infinity`, and no CPU quota. The GPU inference tier, which
has no such constraints, runs containerized
(`deploy/gpu/Dockerfile.inference`).

## Consequences

* No CPU quotas, ever: a quota is a scheduler decision injected into a loop
  designed to have none.
* Scheduling and failover are ours (systemd restarts, N+1 replicas per
  region).
* A future containerized engine is acceptable only with exclusive `cpuset`
  cores, no quota, and `mlock` allowed.
