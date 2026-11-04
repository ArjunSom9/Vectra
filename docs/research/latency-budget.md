# Retrieval latency budget

The budget that objective O1 (p50 < 1 ms at 50,000 QPS) is designed
against. Each line is an allocation, not a measurement. Measurements
replace allocations only through recorded `make bench-qps` reports on the
reference environment ([01 §4](../architecture/01-overview.md)).

## The p50 ledger

| # | Stage | Code | Allocation |
|---|---|---|---|
| 1 | Validate, embed the subject (25 → 256 → 128 MLP) | `serving/service.py` | 20 µs |
| 2 | Disk cover at precision 4, owner lookup | `serving/retrieve.py`, `data/routing.py` | 10 µs |
| 3 | Gateway → shard hop, intra-DC, one pooled channel | gRPC | 50 µs |
| 4 | Completion-queue wake-up and dispatch | `engine/src/server/` | 5 µs |
| 5 | Prefilter: cover at precision 5, range lookups | `geofilter.cc` | 15 µs |
| 6 | Search: exhaustive scan or filtered beam + re-rank | `shard_index.cc` | 150 µs |
| 7 | Encode and send up to 64 candidates with embeddings | `search_handler.cc` | 30 µs |
| 8 | Shard → gateway hop | gRPC | 50 µs |
| 9 | Merge by distance across shards | `retrieve.py` | 10 µs |
| 10 | Reserve for interrupt coalescing, cold caches, boundary fan-out | — | 160 µs |
| | **Retrieval p50 allocation** | | **500 µs** |

The allocation leaves half of the 1 ms objective as headroom.

## Where the numbers come from

* **Search (6).** `index_study` measures about 60 µs per unfiltered query at
  `ef_search` 64 and about 104 µs at 128 (re-rank 128), single-threaded
  over 200,000 vectors on an M1 Pro
  ([`hnsw-parameter-study.md`](hnsw-parameter-study.md)). Radius-filtered
  searches with small populations are exhaustive scans of at most 2,048
  rows. A production shard holds about 3.3M vectors; graph search grows
  roughly logarithmically, so the allocation is 150 µs.
* **Hops (3, 8).** Intra-data-center RPC round trips on 100 GbE are tens
  of microseconds. Deliverable A1 measures dispatch overhead directly: p99 <
  100 µs for no-op RPCs at 50,000 QPS (`stress_client --mode=ping`).
* **Fan-out (10).** A boundary query waits for the slowest of two or three
  shards. The reserve covers that tail at p50.

## Throughput

At 50,000 QPS over three shards, each shard serves about 16,700 QPS. With
92 worker cores per shard host and about 200 µs of worker time per search
(stages 4 to 7), each core is about 4% busy. The engine is sized for its
tail, not its mean.

## Inference tier

End-to-end `PredictPrice` latency adds the batching window (2 ms) and the
compiled forward pass. It is a separate, looser objective, watched by
`GatewayP99High` (25 ms) in `deploy/observability/alerts.yml`.
