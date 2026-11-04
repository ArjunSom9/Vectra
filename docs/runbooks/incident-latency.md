# Runbook: latency incident

**When:** `RetrievalP50AboveOneMillisecond`, `GatewayP99High` or
`SearchQpsShortfall` fires, or clients report slow prices.

## 1. Locate the stage

The gateway's `vectra_gateway_stage_seconds` histogram splits every request
into `retrieve`, `batch_wait`, `inference` and `total` (Grafana: *Vectra:
retrieval and pricing latency*). `PredictPriceResponse.latency` carries the
same split per request.

| Slow stage | Go to |
|---|---|
| `retrieve` | §2 |
| `batch_wait` | §3 |
| `inference` | §4 |

## 2. Retrieval

1. Is one shard slow, or all? Compare `vectra_shard_report_seconds` and the
   per-shard RPC rate. One slow shard bounds every query it serves.
2. On the slow host:
   * `vectra_shard_memory_locked` must be 1. An unlocked index pages.
   * `vectra_shard_call_pool_allocations` must stay flat. Growth means the
     call pool is undersized for the offered load: raise
     `requests_per_queue` and restart.
   * Worker cores must be isolated and the governor at `performance`
     (`deploy/baremetal/tune_latency.sh` ran?). Check for CPU quotas (there
     must be none) and for processes on worker cores (`ps -eLo psr,comm`).
   * NIC interrupts only on housekeeping cores.
3. Is it the query mix? Large radii raise the prefilter population and push
   searches from the exhaustive scan to the graph. Compare with a
   `stress_client` run of known queries.
4. Is the channel pool saturated? More in-flight RPCs per connection than
   `MAX_CONCURRENT_STREAMS` allows queue inside the client. Raise
   `channel_pool_size`.

## 3. Batch wait

`batch_wait` is bounded by the batching window (2 ms) plus the time a batch
spends behind the previous one. If it grows, the GPU is the bottleneck:
check `inference`, and add a GPU host if the offered load has grown.

## 4. Inference

1. `GpuDramActivityDropped`, or a sudden step in `inference`: confirm that
   the gateway started from a compiled bundle. Its log line
   `[vectra] serving model ready: N site(s)` must show one site per layer,
   and `model.compiled` must be true.
2. Thermal or clock throttling: `nvidia-smi -q -d PERFORMANCE`.
3. XID errors: see [gpu-node-recovery](gpu-node-recovery.md).

## 5. Afterwards

Record the incident, and the benchmark report if a fix changed
performance (`tests/regression/compare_snapshots.py --update`, reviewed).
