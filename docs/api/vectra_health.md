# `vectra.health.v1.HealthService`

Source: [`contracts/proto/vectra/health.proto`](../../contracts/proto/vectra/health.proto).
Served by every shard (`engine/src/service/health_handler.cc`). Read by
`shard_inspector` and the Prometheus exporter (`serving/engine_exporter.py`).

## `Report(ReportRequest) → ReportResponse`

| Field | Meaning |
|---|---|
| `shard_id`, `owned_prefixes` | Ownership according to the loaded shard map |
| `index` (`IndexStats`) | `num_vectors`, `capacity`, `dimension`, `pq_subspaces`, `hnsw_m`, `max_level`, `index_bytes` (resident) |
| `workers` (`WorkerStats`, one per completion-queue worker) | `worker`, pinned `core` (−1 if unpinned), `rpcs_completed`, `call_pool_high_water`, `call_pool_allocations` (allocations after start-up; should stay 0) |
| `memory` (`MemoryStats`) | `locked`, `locked_bytes`, `lock_error` |
| `build_version` | The binary's version |

The exporter republishes these as `vectra_shard_*` metrics
(`deploy/observability/`).

## `Ping(PingRequest) → PingResponse`

Echoes `payload`. Deliverable A1 drives this no-op RPC at 50,000 QPS
through the channel pool to measure dispatch overhead without search work
(`stress_client --mode=ping`).
