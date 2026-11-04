# 01 — System Overview

| | |
|---|---|
| **Status** | Accepted (v1.1, corrected against the implementation) |
| **Synthesizes** | Design document §2 (objectives) and §4 (architecture overview) |
| **Deep dives** | [02](02-why-not-managed.md) · [03](03-async-grpc.md) · [04](04-geohash-sharding.md) · [05](05-hnsw-flat-memory.md) · [06](06-memory-bottleneck.md) · [07](07-triton-kernels.md) · [08](08-fx-compilation.md) · [09](09-subgraph-rewrite.md) |
| **Change control** | ADR-backed PRs; performance figures change only with benchmark reports recorded in `tests/regression/` |

This document is the map. It describes the system as implemented in this
repository. Two kinds of statements appear and are kept apart:

* **Implemented and tested.** Behavior that a test in this repository
  checks. The test is named next to the claim.
* **Target.** An acceptance number of the design (§7) that needs the
  reference hardware to measure. A target is never reported as a result
  until its harness has run there and its report has been recorded with
  `tests/regression/compare_snapshots.py --update`.

---

## 1. The problem

Real-time property pricing needs two things on every request: the subject's
comparable sales (a nearest-neighbor search over property embeddings,
constrained to a geographic radius), and a Transformer forward pass that
scores the subject against those comparables. The project replaces a managed
vector backend and an eager PyTorch serving path with:

* a bespoke C++ vector search engine, sharded by GeoHash and served over
  asynchronous gRPC (Workstream A);
* a fused Triton kernel for the model's radial-basis (RBF) spatial attention
  (Workstream B);
* a compiler pass that puts that kernel into ordinary PyTorch models at
  deployment time (Workstream C).

## 2. Objectives

| # | Objective | Pillar | Gate |
|---|---|---|---|
| O1 | 50,000 searches/s against 10M property embeddings with p50 < 1 ms and recall@10 ≥ 0.95 | `engine/` | `make bench-qps`, `make bench-recall` |
| O2 | Fused kernel moves ≤ 60% of the eager scorer's DRAM bytes; serving throughput ≥ 3.0× eager at matched numerics (rtol 1e-3) | `kernels/` | `make bench-bandwidth`, `make bench-gpu` |
| O3 | Deployment compile rewrites every attention site with zero graph breaks; the substituted kernel reaches ≥ 65% of peak DRAM throughput (40% eager baseline) | `compiler/` | `make test-compiler`, `make bench-bandwidth` |

The objectives depend on each other. O3 exists so that O2's kernel reaches
production without anyone calling it by hand, and O1 exists because the
managed backend cannot meet it ([02](02-why-not-managed.md)).

## 3. Architecture

```
  client ── PricingService.PredictPrice ──▶ inference gateway (serving/)
                                              │  embed subject (data/embedding_model.py)
                                              │  route: 4-char GeoHash cells the
                                              │  search disk touches → their shards
                 ┌────────────────────────────┤
                 ▼ async gRPC, 64 pooled      │
   ┌──────────────────────────┐ channels      │  batch requests (2 ms window)
   │ C++ shards (engine/)     │ per shard     ▼
   │ · CompletionQueue per    │        compiled price model (models/)
   │   core, CallData tags    │        torch.compile(backend="vectra")
   │ · 5-char prefilter       │         └ RBF attention → vectra::rbf_attention
   │ · HNSW + PQ, exact       │            (fused Triton kernel, kernels/)
   │   re-rank, sale-date     │
   │   bound                  │        price out, per-stage latency
   └──────────────────────────┘
```

### Pillar I: `engine/` (C++20, gRPC 1.71.0)

Each shard owns the properties whose 4-character GeoHash cell its shard map
assigns to it. The asynchronous server runs one `CompletionQueue` per worker
core with `CallData` objects whose addresses are the completion tags
([03](03-async-grpc.md)). A search is answered by the shard index: a
5-character GeoHash prefilter, then either an exhaustive scan (small
populations) or a filtered HNSW traversal on product-quantized codes,
followed by an exact float32 re-rank ([05](05-hnsw-flat-memory.md)).

### Pillar II: `kernels/` (Triton 3.3.0)

One fused kernel computes `softmax(-scale·‖q−k‖² + bias) @ v`. It uses the
expansion `‖q−k‖² = ‖q‖² − 2q·k + ‖k‖²`, so the distance becomes a
tensor-core matrix product, and an online softmax, so no Nq×Nk matrix ever
reaches HBM ([07](07-triton-kernels.md)). It is registered as the PyTorch
operator `vectra::rbf_attention`.

### Pillar III: `compiler/` (PyTorch 2.7.0)

`torch.compile(backend="vectra")` is TorchInductor plus a graph pass
installed at Inductor's joint-graph pre-pass. The pass matches every
recognized formulation of RBF attention by use-def topology, checks the
match semantically, and replaces it with the operator
([08](08-fx-compilation.md), [09](09-subgraph-rewrite.md)).

## 4. Topology

| Tier | Reference deployment |
|---|---|
| Search shards | 3 macro-regions × (N+1) nodes; single-socket AMD EPYC, 512 GB DDR5, 100 GbE, Ubuntu 24.04, bare metal |
| Inference | NVIDIA H100 SXM5 80 GB (HBM3 about 3.35 TB/s, 50 MB L2), containerized |
| Gateway | Co-located with the inference GPU |

External clients talk only to the gateway (`:7000`). Shard assignments,
endpoints and client options live in `engine/configs/*.textproto` (schema
`contracts/proto/vectra/config.proto`). The gateway and the C++ engine read
the same files, so they always agree on ownership. `scripts/run_local_cluster.sh`
runs the three-shard topology on one machine.

**GeoHash precision ladder** ([04](04-geohash-sharding.md)):

| Precision | Cell, width × height at 45° N | Role |
|---|---|---|
| 3 | ≈ 110 km × 156 km | macro-region (data center) |
| 4 | ≈ 27.6 km × 19.5 km | shard key |
| 5 | ≈ 3.5 km × 4.9 km | in-shard prefilter |
| 6 | ≈ 0.86 km × 0.61 km | micro-neighborhood feature in the model |

## 5. Request lifecycle

| # | Stage | Code |
|---|---|---|
| 1 | Validate the request; embed the subject from its attributes and location | `serving/service.py`, `data/embedding_model.py` |
| 2 | Cover the search disk with precision-4 cells (at most the 3×3 block around the home cell; larger disks are rejected) and collect their owners | `serving/retrieve.py`, `data/routing.py` (mirror of `engine/src/routing/`) |
| 3 | Send one `SearchRequest` per owning shard through that shard's channel pool, with `sold_before_epoch_day` set to the valuation day | `serving/retrieve.py` |
| 4 | Shard ingress: the worker's `CompletionQueue` yields the `CallData` tag; a fresh call is armed before the search runs | `engine/src/server/call_data.cc` |
| 5 | Prefilter: precision-5 cells covering the disk select candidate rows | `engine/src/index/geofilter.cc` |
| 6 | Exhaustive scan if at most 2,048 rows qualify; otherwise filtered HNSW on PQ codes, then exact re-rank of the best 128 | `engine/src/index/shard_index.cc`, `hnsw_searcher.cc` |
| 7 | Merge shard results by squared L2 distance; any shard failure fails the request | `serving/retrieve.py` |
| 8 | Batch with concurrent requests (2 ms window, up to the compiled batch size) | `serving/predict.py` |
| 9 | Compiled forward pass; the price is `exp` of the predicted log price | `models/export/compile_entry.py` |
| 10 | Reply with comparables and per-stage microseconds; record Prometheus histograms | `serving/service.py` |

The sub-millisecond objective is the retrieval tier's (stages 2 to 7). The
inference tier spends a bounded batching window to keep the GPU busy, so
end-to-end latency is a separate, looser objective (`GatewayP99High` in
`deploy/observability/alerts.yml`).

## 6. Integration contracts

| Contract | Location | Consumer → producer |
|---|---|---|
| gRPC services | `contracts/proto/vectra/{search,ingest,health,pricing}.proto` | gateway → shards, clients → gateway |
| Engine configuration | `contracts/proto/vectra/config.proto`, `engine/configs/` | engine, gateway, data pipelines |
| Custom operator | `torch.ops.vectra.rbf_attention` (`kernels/vectra_kernels/custom_ops.py`) | compiler replacement → kernel |
| Deployment compile | `models/export/compile_entry.py` → `dist/compiled_model/` | gateway → model |

`models/` contains no kernel calls and no GPU-specific imports. The rewrite
is the only path by which the kernel enters the model.

## 7. Verification status

| Objective / property | Status | Evidence |
|---|---|---|
| Exact routing loses no comparable across shard boundaries | Implemented and tested | `engine/tests/boundary/` (exact mode: recall 1.0), `serving/tests/test_gateway_flow.py` |
| Approximate recall@10 on synthetic data | Implemented and tested | `engine/tests/unit/shard_index_test.cc` (≥ 0.95), `engine/tests/boundary/adjacent_recall_test.cc` (≥ 0.99) |
| Rewrite matches every authoring style and no look-alike | Implemented and tested | `compiler/tests/` |
| Compiled model equals eager (rtol 1e-3) | Implemented and tested (CPU) | `compiler/tests/test_e2e_compile.py`, `models/tests/test_train_and_export.py` |
| Kernel numerics vs float64 oracle | Implemented and tested (CPU emulation; Triton on GPU in CI) | `kernels/tests/` |
| O1: 50,000 QPS, p50 < 1 ms, 10M vectors | Target | `make bench-qps` on the reference cluster |
| O1: recall@10 ≥ 0.95 on the production corpus | Target | `make bench-recall` |
| O2: ≤ 60% DRAM bytes, ≥ 3.0× throughput | Target (analytic budget: about 1.4% of eager bytes at the serving shape, `python -m kernels.benchmarks.roofline`) | `make bench-bandwidth`, `make bench-gpu` |
| O3: ≥ 65% of peak DRAM throughput | Target, judged at batch 1024 (see [06](06-memory-bottleneck.md)) | `make bench-bandwidth` |

## 8. Capacity model

Per-vector resident bytes, from the data structures in `engine/src/index/`
at the configured parameters (dimension 128, 16 PQ subspaces, M = 16,
M0 = 32):

| Structure | Bytes per vector |
|---|---|
| Float32 vectors for the exact re-rank (64-byte-aligned rows) plus norm | 516 |
| Layer-0 graph record: PQ code (16 B), count, 32 neighbors (padded to 32 B) | 160 |
| Upper layers (about 1/16 of nodes, 80 B each, geometric) | about 5 |
| Property columns (id, location, unit vector, cell, sale day, attributes) | 96 |
| Prefilter (cell, id) | 8 |
| **Total** | **about 785** |

10M vectors therefore occupy about 7.9 GB across the fleet, about 2.6 GB per
shard at three shards. That leaves more than two orders of magnitude of
headroom on 512 GB nodes, all of it locked in RAM (`mlockall`). The full
float32 vectors are kept on purpose: ranking by PQ distances alone caps
recall below the target (see
[`hnsw-parameter-study.md`](../research/hnsw-parameter-study.md)).

## 9. Failure behavior

| Failure | Behavior | Runbook |
|---|---|---|
| A shard does not answer | The request fails with that shard's status (`UNAVAILABLE` or `DEADLINE_EXCEEDED`); a partial comparable set is never priced | [`node-bringup.md`](../runbooks/node-bringup.md) |
| Radius beyond the adjacent cells | `INVALID_ARGUMENT` before any RPC | — |
| No earlier sale within the radius | `NOT_FOUND` | — |
| Compiled bundle does not rewrite every site, or breaks the graph | The gateway refuses to start | [`gpu-node-recovery.md`](../runbooks/gpu-node-recovery.md) |
| Latency regression | Alerts in `deploy/observability/alerts.yml`; the regression gate blocks recorded regressions | [`incident-latency.md`](../runbooks/incident-latency.md) |

Degrade loudly, never silently.

## 10. Principles

1. **Tail latency over mean throughput.** Thread-per-core, no garbage
   collector, no consensus on the read path.
2. **Locality is the algorithm.** GeoHash prefixes, flat arrays with integer
   offsets, tiles that stay on chip.
3. **Algebra before silicon.** The distance expansion turns a memory-bound
   elementwise chain into a tensor-core product.
4. **Oracle before optimization.** hnswlib and brute force check recall, the
   float64 reference checks the kernel, the eager model checks the rewrite.
5. **The modeling surface stays eager.** Performance enters at the compile
   boundary only.
6. **Every number has a harness.** Targets are labelled as targets until
   their harness has run on the reference hardware.

## 11. Repository map

| Concept | Code |
|---|---|
| Async server, `CallData`, completion queues | `engine/src/server/` |
| Search, ingest and health handlers | `engine/src/service/` |
| GeoHash, disk cover, routing table, fan-out | `engine/src/routing/` |
| HNSW, PQ, flat store, prefilter, shard index | `engine/src/index/` |
| Aligned buffers, arena, RAM locking | `engine/src/memory/` |
| Channel pool, async client, geo router | `engine/client/` |
| Fused kernel, references, custom operator | `kernels/vectra_kernels/` |
| Pattern catalog, filters, rewrite pass, backend | `compiler/vectra_compiler/` |
| Price model, training, evaluation, deployment compile | `models/` |
| Gateway, router, batching, shard exporter | `serving/` |
| Synthetic corpus, embeddings, pipelines | `data/` |
| Benchmarks and regression gates | `benchmark/`, `tests/regression/` |

## 12. Non-goals

* Write-heavy workloads: ingestion is batch-oriented and single-writer per
  shard.
* Cross-shard consistency: per-shard state is local; there is no consensus
  on the read path.
* Elastic re-sharding: a runbook operation
  ([`resharding.md`](../runbooks/resharding.md)).
* A CPU inference tier: the operator runs on CPU for tests, but performance
  targets assume H100 tensor cores.
* Multi-tenant isolation.

## 13. Document set

| Doc | Scope |
|---|---|
| 01 (this) | Synthesis, lifecycle, status, capacity |
| [02](02-why-not-managed.md) | Why the managed backend cannot meet O1 |
| [03](03-async-grpc.md) | Asynchronous gRPC, `CallData`, channel pools |
| [04](04-geohash-sharding.md) | GeoHash encoding, disk cover, shard routing |
| [05](05-hnsw-flat-memory.md) | HNSW, PQ, flat layout, filtered search |
| [06](06-memory-bottleneck.md) | The eager scorer's HBM traffic and the roofline |
| [07](07-triton-kernels.md) | The fused kernel |
| [08](08-fx-compilation.md) | TorchDynamo capture, graph breaks, `torch.cond` |
| [09](09-subgraph-rewrite.md) | Pattern matching, filters, replacement |

Also: [decisions](../decisions/) (ADR-001 to ADR-008),
[research notes](../research/), [runbooks](../runbooks/),
[API reference](../api/), and the
[annotated bibliography](../research/bibliography.md). Diagrams:
[request-lifecycle](diagrams/request-lifecycle.svg),
[shard-routing](diagrams/shard-routing.svg),
[memory-hierarchy](diagrams/memory-hierarchy.svg).
