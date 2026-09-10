# 01 — System Overview

| | |
|---|---|
| **Status** | Accepted — Frozen Spec (Phase 0) |
| **Version** | 1.0.0 |
| **Synthesizes** | Design document §1 (Paradigm Shift) and §10 (System Synthesis) |
| **Deep dives** | [02](02-why-not-managed.md) · [03](03-async-grpc.md) · [04](04-geohash-sharding.md) · [05](05-hnsw-flat-memory.md) · [06](06-memory-bottleneck.md) · [07](07-triton-kernels.md) · [08](08-fx-compilation.md) · [09](09-subgraph-rewrite.md) |
| **Change control** | ADR-backed PRs only; performance figures move only with benchmark PRs that update the regression goldens |

This document is the map. It orients every reader — engineer, operator, or
reviewer — before the pillar-specific deep dives. It describes the system
**as built and verified**, not as proposed: every performance figure in §8
is reproduced by a `make bench-*` target and enforced by the regression
goldens in `tests/regression/`.

---

## 1. The Problem

Transformer-based real-time property price prediction has outgrown its
hosting infrastructure. Modern pricing models consume dense geographical
context — the historical properties *around* a subject property — alongside
traditional econometric indicators. Every prediction therefore begins with
a nearest-neighbor retrieval over a massive embedding corpus, and concludes
with a deep-learning forward pass whose hottest operation is a spatial
attention score between the query and every retrieved candidate.

This composite workload — large-scale information retrieval fused with
high-performance inference — sits in the gap between two infrastructure
families, and neither fills it:

- **Managed vector databases** are built for durable, general-purpose
  workloads. Their consensus protocols, generalized sharding, and
  fan-out query patterns make single-query latency a function of the
  *slowest* shard in a cluster. (Full analysis: [02](02-why-not-managed.md).)
- **Eager-mode deep learning runtimes** are built for developer velocity.
  Their sequential op-by-op execution materializes O(N²) intermediate
  tensors through GPU global memory, starving Tensor Cores at 10–20%
  utilization. (Full analysis: [06](06-memory-bottleneck.md).)

## 2. The Three Requirements

The system is engineered against three aggressive, measurable requirements.
Each maps to one architectural pillar, one directory, and one CI gate:

| # | Requirement | Pillar | Verified by |
|---|---|---|---|
| R1 | **50,000 QPS against 10M property embeddings at sub-millisecond latency**, via a bespoke geo-distributed C++ engine with asynchronous gRPC and spatial sharding | I — `engine/` | `make bench-qps`, `make bench-recall` |
| R2 | **≥ 40% GPU memory-bandwidth reduction and ≥ 3× inference throughput** via custom fused Triton kernels with explicitly managed global-memory access | II — `kernels/` | `make bench-bandwidth`, `make bench-gpu` |
| R3 | **Zero developer-facing friction** — a custom PyTorch compiler pass that detects unoptimized spatial scoring in `torch.fx` and swaps in the Triton kernel automatically | III — `compiler/` | `make test-compiler` |

The requirements are not independent: R3 exists so that R2's performance
reaches production *without* demanding that data scientists hand-invoke
kernels; R1 exists because no managed system satisfies it (§4 of the design
document's analysis, elaborated in [02](02-why-not-managed.md)).

## 3. The Three-Pillar Architecture

```
                          ┌─────────────────────────────────┐
                          │         Clients / API           │
                          └────────────────┬────────────────┘
                                           │ PredictPrice(lat, lon, …, top_k)
                                           ▼
                          ┌─────────────────────────────────┐
                          │     INFERENCE GATEWAY           │
                          │     serving/ (asyncio + gRPC)   │
                          │     :7000                       │
                          └────────┬───────────────┬────────┘
                 3-char GeoHash    │               │ top-k candidates
              (data-center tier)  ─┘               ▼
        ┌───────────────────────────┐   ┌──────────────────────────────────┐
        │  ROUTING LAYER            │   │   GPU INFERENCE TIER (H100 SXM5) │
        │  engine/client            │   │   models/ · compiler/ · kernels/ │
        │  · 4-char prefix → shard  │   │                                  │
        │  · 8-adjacent-cell fanout │   │   torch.compile(                 │
        │    (radius-intersecting   │   │     backend="custom_backend")    │
        │    shards only)           │   │    └─ FX subgraph rewrite →      │
        │  · pooled channels (64),  │   │       fused Triton kernel:       │
        │    round-robin            │   │       2(Q·K) − ‖Q‖² − ‖K‖²       │
        └──────────┬────────────────┘   │       on Tensor Cores, all       │
                   │ async gRPC         │       intermediates in SRAM/     │
                   │ 1 CQ per core,     │       registers, one load and    │
                   │ thread-per-core,   │       one store per invocation   │
                   │ CallData tags      │                                  │
        ┌──────────▼─────────────────┐  │   Price prediction out           │
        │  BARE-METAL C++ SHARDS     │  └──────────────────────────────────┘
        │  engine/src                │
        │ · HNSW, flat contiguous    │
        │   arena, integer offsets   │
        │ · PQ-compressed embeddings │
        │   (64 B/vector)            │
        │ · AVX-512 SIMD distances   │
        │ · 5-char GeoHash prefilter │
        │ · 10M vectors, ~2.1 GB,    │
        │   mlock'd, zero disk spill │
        └────────────────────────────┘
```

### Pillar I — `engine/` (C++20, gRPC 1.83 Asynchronous API)

A bespoke, geo-distributed vector search engine. Deterministic memory
management; one polling thread per `CompletionQueue` per physical core with
`CallData` objects tagged by address; GeoHash spatial sharding that routes
each query to *exactly one* shard except when the search radius crosses a
cell boundary; an HNSW graph stored as one contiguous arena with integer
offsets instead of pointers, so a node's topological neighborhood enters
L1/L2 with the node itself. Details: [03](03-async-grpc.md),
[04](04-geohash-sharding.md), [05](05-hnsw-flat-memory.md).

### Pillar II — `kernels/` (Triton on H100)

A single fused kernel resolves the spatial attention scores. The algebraic
identity −‖Q−K‖² = 2(Q·K) − ‖Q‖² − ‖K‖² converts a memory-bound elementwise
operation into a Tensor-Core matrix multiplication; pre-computed norms are
subtracted inside the inner loop; running maxima and sums live in thread
registers; inputs are read from HBM exactly once and the output written
exactly once. Details: [07](07-triton-kernels.md).

### Pillar III — `compiler/` (torch.fx subgraph rewriting)

TorchDynamo captures the model's forward pass as an FX graph;
a subgraph-isomorphism pass matches the eager spatial-attention pattern on
use-def dependencies — not variable names — and splices in the registered
Triton operator through TorchInductor's post-grad hook. Model authors write
plain PyTorch; the deployment boundary injects the performance.
Details: [08](08-fx-compilation.md), [09](09-subgraph-rewrite.md).

## 4. System Topology

**Reference deployment** (the environment against which every figure in §8
was measured):

| Tier | Nodes | Specification |
|---|---|---|
| Search shards | 3 | Single-socket AMD EPYC 9005 (96c, Zen 5), 512 GB DDR5, 100 GbE, Ubuntu Server 24.04, bare metal |
| GPU inference | 1 | NVIDIA H100 SXM5 (80 GB HBM3, 3.35 TB/s, 50 MB L2), containerized |
| Gateway | 1 | Co-located with the GPU tier |

**Boundaries.** External clients speak only to the gateway (`:7000`).
Internal boundaries: gateway → shard nodes (async gRPC, channel pool),
gateway → compiled model (in-process). The shards' raw search port serves
the model tier, never external clients. Shard ports and assignments are
config-driven (`engine/configs/shard_map.yaml`,
`cluster_topology.yaml`); the local development cluster
(`scripts/run_local_cluster.sh`) runs the identical topology on one box.

**GeoHash precision ladder** — one encoding, four architectural roles
([04](04-geohash-sharding.md)):

| Precision | Cell size | Role |
|---|---|---|
| 3 chars | ~156 × 156 km | Data-center / macro-region assignment |
| 4 chars | ~39.1 × 19.5 km | **Primary shard key** — deterministic node routing |
| 5 chars | ~4.89 × 4.89 km | In-memory inverted index, applied before vector traversal |
| 6 chars | ~1.22 × 0.61 km | Micro-neighborhood isolation in pricing features |

## 5. The Request Lifecycle

The full path of a real-time prediction, with the measured p50 ledger from
[`docs/research/latency-budget.md`](../research/latency-budget.md):

| # | Stage | Owner | p50 |
|---|---|---|---|
| 1 | **Ingest.** `PredictPrice` arrives at the gateway; the asyncio handler validates and computes the 4-char GeoHash of the target plus its 8 adjacent cells | `serving/service.py` | 7 µs |
| 2 | **Route.** The routing table maps the 4-char prefix to exactly one shard; a radius-vs-cell-bounds check decides whether any adjacent cells intersect — the query is duplicated *only* to those shards | `engine/client/geo_router.cpp` | 2 µs |
| 3 | **Transport.** The pre-warmed channel pool (64 channels, round-robin) issues the RPC asynchronously; HTTP/2 stream limits are never encountered | `engine/client/channel_pool.cpp` | 52 µs |
| 4 | **Shard ingress.** A dedicated polling thread retrieves the event tag from its `CompletionQueue`, casts it back to the `CallData` object, transitions the state machine, immediately spawns a fresh `CallData` to keep listening, and executes the search on the same thread — no context switch, no scheduler yield | `engine/src/server/` | 3 µs |
| 5 | **Prefilter.** The 5-char inverted index narrows the candidate space before any vector is touched | `engine/src/index/geofilter.cpp` | 12 µs |
| 6 | **Traversal.** HNSW descends from the top-layer entry to Layer 0. Because the graph is one contiguous arena with integer offsets, each node fetch pulls its neighborhood into L1/L2; AVX-512 computes PQ-decoded distances | `engine/src/index/hnsw_searcher.cpp` | 118 µs |
| 7 | **Merge & reply.** Per-shard top-k; the single-shard common case is pass-through; serialization and return | `engine/src/service/search_handler.cpp` | 21 µs |
| 8 | **Reply hop & assembly.** | gateway | 46 µs |
| 9 | **Jitter reserve** (measured tail contributor: interrupt coalescing, cold-cache paths) | — | 29 µs |
| | **Search path total** | | **290 µs** |
| 10 | **Handoff.** Candidates stream into the model's candidate encoder | `serving/retrieve.py` | pipelined |
| 11 | **Inference.** The compiled FX graph executes; spatial attention runs as the fused Triton kernel — `tl.dot` on Tensor Cores, norms subtracted in-loop, online max/sum in registers, one HBM load and one store | `kernels/`, `compiler/` | see §8 |
| 12 | **Price out.** Prediction serialized; per-stage µs recorded to the latency histogram | `serving/predict.py` | — |

**Tier discipline:** the sub-millisecond envelope is the *retrieval tier*
SLA. The *inference tier* deliberately trades a bounded 2 ms batching
window (`serving/configs/service.yaml`, `batch_window_ms`) for Tensor-Core
utilization — coordinate-in to price-out p50 is ~3.2 ms, of which the
batching window is the largest single component. This split is a design
decision, not a limitation: the retrieval tier is the latency-critical
lookup; the inference tier is the throughput-critical scorer.

## 6. Pillar Integration Contracts

The pillars are strictly isolated and integrate through three narrow,
versioned contracts — the reason they can be built, benchmarked, and
deployed independently (the parallel lanes of the implementation plan):

| Contract | Location | Consumer → Producer |
|---|---|---|
| **gRPC/protobuf services** | `contracts/proto/` (`vectra_search`, `vectra_ingest`, `vectra_health`) | `serving/` → `engine/` |
| **Registered custom operator** | `kernels/vectra_kernels/custom_ops.py` (`torch.library`) | `compiler/` replacement → `kernels/` kernel |
| **Compile entry point** | `models/export/compile_entry.py` | deployment → `torch.compile(backend="custom_backend")` |

Nothing else crosses pillar boundaries. `models/` contains zero
GPU-specific imports and zero kernel invocations — that guarantee is
CI-linted, and its enforcement story is [09](09-subgraph-rewrite.md).

## 7. Verified Performance

Reference environment of §4; methodology and raw outputs in
[`benchmark/reports/`](../../benchmark/reports/). Every row is a CI
contract via `tests/regression/golden_*.json`.

| Metric | Result | Target | Harness |
|---|---|---|---|
| Sustained throughput | **51,240 QPS** (60-min soak, 3 shards) | ≥ 50,000 | `make bench-qps` |
| Search latency (network-inclusive) | **p50 0.29 ms · p99 0.47 ms · p99.9 0.71 ms** | p99 < 1 ms | `make bench-qps` |
| Recall@10 vs. exhaustive oracle | **0.993** @ `ef_search = 64` | ≥ 0.99 | `make bench-recall` |
| Adjacent-cell boundary recall | **100%** | 100% | `engine/tests/boundary/` |
| Spatial-attention throughput | **3.1×** vs. eager PyTorch | ≥ 3× | `make bench-gpu` |
| GPU global-memory traffic | **−42.3%** HBM bytes (Nsight `dram__bytes`) | ≥ 40% reduction | `make bench-bandwidth` |
| Index residency | **10M vectors in ~2.1 GB**, mlock'd, zero spill | RAM-only | `engine/apps/shard_inspector` |

Reproduce any row locally with the corresponding `make bench-*` target; the
same targets gate CI. The claims are contracts, not marketing.

## 8. Capacity and Memory Model

Per-cluster index composition at the current 10M × 64-dim corpus
(PQ-compressed, flat layout — derivation in [05](05-hnsw-flat-memory.md)):

| Component | Sizing rule | Footprint |
|---|---|---|
| PQ codes | 64 B/vector (64 subspaces × 8-bit) | 640 MB |
| HNSW Layer-0 adjacency | `m = 16` × 4-B integer offsets | 640 MB |
| L2 norms | FP32, pre-computed | 40 MB |
| 5-char inverted index + 6-char microhood maps | — | ~270 MB |
| Node metadata + GeoHash keys | — | ~330 MB |
| Upper layers, entry table, arena alignment | — | ~180 MB |
| **Total** | | **~2.1 GB** |

At 3 shards (~3.3M vectors each) this is ~700 MB per node against 512 GB
of RAM — roughly three orders of magnitude of headroom. The corpus could
grow ~100× before the flat-memory model is even conversationally
constrained; when it does, re-partitioning is a runbook operation
([`docs/runbooks/resharding.md`](../runbooks/resharding.md)), not an
architecture change.

## 9. Failure Modes and Degradation

| Failure | Immediate behavior | Recovery | Runbook |
|---|---|---|---|
| Shard node loss | Owned 4-char cells fail fast (gRPC `UNAVAILABLE`); the gateway's health cache marks them; queries never fan out to wrong shards | Provision replacement via Ansible, rebuild index from corpus | [`node-bringup.md`](../runbooks/node-bringup.md), [`index-rebuild.md`](../runbooks/index-rebuild.md) |
| GPU node loss | Gateway fails fast on the inference leg | Container reschedule + warm-cache restore | [`gpu-node-recovery.md`](../runbooks/gpu-node-recovery.md) |
| p99 regression | Alert fires; CI goldens block the offending change pre-production | Triage tree | [`incident-latency.md`](../runbooks/incident-latency.md) |
| Channel pool saturation | Queue-depth telemetry crosses threshold before any user-visible effect | Pool is a config knob (`channel_pool_size`) | [`incident-latency.md`](../runbooks/incident-latency.md) |

The unifying stance: **degrade loudly and deterministically, never
silently.** The same stance appears in the compiler pillar — if the spatial
pattern is expected and not found, compilation fails rather than silently
running eager.

## 10. Design Principles

The principles every deep-dive doc inherits:

1. **Determinism over raw throughput.** Thread-per-core, no GC, no
   consensus on the read path, no unbounded queues. Tail latency is the
   product; mean throughput is a side effect.
2. **Locality is the algorithm.** GeoHash prefixes, the contiguous HNSW
   arena, SRAM-resident tiles — each pillar is a locality argument wearing
   a different costume.
3. **Algebra before silicon.** The −‖Q−K‖² expansion and the Morton-coded
   GeoHash both convert a hard problem into one the hardware already
   solves natively.
4. **Oracle before optimization.** hnswlib validates engine recall, the
   eager reference validates kernel numerics, the un-compiled model
   validates the rewrite pass. No optimization lands without its oracle
   already green.
5. **The developer surface stays eager.** All performance enters at the
   compile boundary, none in the modeling code.
6. **Every claim is a harness.** Any number this doc states is one
   `make` target away, and CI owns it.

## 11. Architecture-to-Repository Map

| Concept | Code |
|---|---|
| Async server, `CallData`, CQ-per-core | `engine/src/server/` |
| GeoHash, routing table, adjacency fan-out | `engine/src/routing/` |
| HNSW build/search, flat store, PQ, SIMD | `engine/src/index/` |
| Arena, RAM locking | `engine/src/memory/` |
| Channel pool, geo router (client side) | `engine/client/` |
| Fused spatial kernel + custom op | `kernels/vectra_kernels/` |
| Eager reference (oracle + pattern source) | `kernels/reference/` |
| Pattern/replacement, rewrite pass, backend | `compiler/vectra_compiler/` |
| Pricing model (readable, eager-style) | `models/price_transformer/` |
| Gateway lifecycle | `serving/` |
| Corpus pipelines | `data/pipelines/` |
| Benchmark harnesses | `benchmark/` |
| Regression contracts | `tests/regression/` |

## 12. Non-Goals

Explicitly out of scope, so nobody "fixes" them by accident:

- **Mutation-heavy vector workloads.** The engine is read-optimized;
  ingestion is batch-oriented by design.
- **Cross-shard strong consistency.** Routing is deterministic and
  per-shard state is local; there is no distributed consensus on the read
  path — that is the point.
- **Automatic re-sharding / elastic corpus.** Re-partitioning is a runbook
  operation with human oversight.
- **CPU-only inference tier.** The Triton kernel assumes Tensor Cores.
- **Multi-tenant isolation/QoS.** Single-owner deployment model.

## 13. The Document Set

| Doc | Scope | Design doc § |
|---|---|---|
| **01 — this** | System synthesis, lifecycle, budgets, capacity | §1, §10 |
| [02](02-why-not-managed.md) | Why managed backends fail the requirements | §2 |
| [03](03-async-grpc.md) | Async gRPC, `CallData`, thread-per-core | §3 |
| [04](04-geohash-sharding.md) | GeoHash, Z-order, adjacency fan-out | §4 |
| [05](05-hnsw-flat-memory.md) | HNSW, flat layout, PQ, SIMD | §5 |
| [06](06-memory-bottleneck.md) | Eager PyTorch HBM analysis | §6 |
| [07](07-triton-kernels.md) | Fused kernel, tiling, memory hierarchy | §7 |
| [08](08-fx-compilation.md) | TorchDynamo, FX capture, `torch.cond` | §8 |
| [09](09-subgraph-rewrite.md) | Pattern matching, replacement, injection | §9 |

Companion material: [ADR index](../decisions/) (001–008),
[research notes](../research/) (parameter studies, tiling study, latency
budget), [runbooks](../runbooks/), [API reference](../api/),
[annotated bibliography](../research/bibliography.md).

Diagrams: [request-lifecycle.svg](diagrams/request-lifecycle.svg) ·
[shard-routing.svg](diagrams/shard-routing.svg) ·
[memory-hierarchy.svg](diagrams/memory-hierarchy.svg) — editable sources in
`diagrams/sources/`.

## 14. Change Control

This document is part of the Phase 0 frozen specification. Substantive
changes require an ADR in the same PR ([CONTRIBUTING.md](../../CONTRIBUTING.md)
§ Pull Request Process). Performance figures may only change through
benchmark PRs that (a) paste before/after evidence, (b) update the
corresponding goldens, and (c) update
[`docs/research/latency-budget.md`](../research/latency-budget.md). The
ledger is maintained, not archived.