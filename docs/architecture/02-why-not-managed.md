# 02 — Why Not a Managed Vector Backend

| | |
|---|---|
| **Status** | Accepted — Frozen Spec (Phase 0) |
| **Version** | 1.0.0 |
| **Synthesizes** | Design document §2 (The Limitations of Managed Backend Services) |
| **Companion decision** | [ADR-001 — Async gRPC bespoke engine over managed backend](../decisions/adr-001-async-grpc-over-managed-db.md) |
| **Implements the refutation** | [03](03-async-grpc.md) · [04](04-geohash-sharding.md) · [05](05-hnsw-flat-memory.md) |
| **Refuted-by condition** | Any managed offering passing the gates of [`benchmark/qps_latency/`](../../benchmark/qps_latency/) — see §10 |
| **Change control** | ADR-backed PRs only |

> **Verdict.** No managed vector backend satisfies R1 — 50,000 QPS over 10M
> property embeddings at sub-millisecond p99 — and none *can*, by
> architecture: the managed read path is straggler-bound (§3),
> page-cache-dependent (§4), and pause-prone (§5). The requirement does not
> merely favor a bespoke engine; it **specifies** one. Recorded as ADR-001.

**Epistemic status.** This document makes two kinds of claims, labeled
throughout:

- **[ANALYTICAL]** — exact mathematics or class-level reasoning about managed
  architectures, grounded in the literature ([1], [2], [3]; annotated
  entries in [`docs/research/bibliography.md`](../research/bibliography.md)).
  Deliberately *not* per-vendor benchmarks: the argument is structural.
- **[MEASURED]** — properties of Vectra itself, every one reproducible via a
  `make bench-*` target and enforced by the regression goldens.

---

## 1. The Decision Under Test

Requirement R1 from [01-overview](01-overview.md) §2:

> **50,000 QPS against 10M property embeddings at sub-millisecond latency.**

Any candidate backend — managed or bespoke — must answer three questions
before benchmarks are even run:

| # | Question | Why it is disqualifying if the answer is "no" |
|---|---|---|
| Q1 | Is per-query latency **O(1) in the number of shards**? | A query whose latency scales with cluster size cannot hold p99 < 1 ms as the corpus grows |
| Q2 | Is the index **guaranteed RAM-resident with no page-fault path**? | A single disk fault consumes ~10% of the entire budget (§4) |
| Q3 | Is the read path **free of stop-the-world events and per-request allocation cascades**? | At 50K QPS the pause budget is effectively zero (§5) |

Managed offerings fail all three **by architecture, not by tuning** — a
consequence of what they are built to guarantee. As the design document
states: managed backends rely on *"generalized sharding mechanisms and
consensus protocols designed for high durability across diverse workloads
rather than hyper-optimized read performance"* (§2). The read path pays
rent for machinery it never uses.

---

## 2. The Budget Arithmetic — Why Sub-Millisecond Is a Different Regime

A p99 of 1 ms is **1,000 µs, network-inclusive, end to end**. At 50,000 QPS,
one millisecond of wall time contains **50 requests in flight**. That single
ratio converts every event in the system into a request count:

| Event | Duration | Requests affected | Share of the p99 budget |
|---|---|---|---|
| Intra-DC gRPC hop (measured, step 3 of the [§5 ledger in 01](01-overview.md)) | 52 µs | 2.6 | 5% |
| **One NVMe page fault** | ~100 µs+ | 5 | **10%+** |
| One 10 ms managed-runtime pause | 10 ms | 500 | 1,000% |
| One 40 ms compaction/GC pause | 40 ms | 2,000 | 4,000% |

**[ANALYTICAL] The p99 tolerance is what makes this categorical.** Holding
p99 < 1 ms means fewer than 1 request in 100 may exceed the bound — at
50K QPS, **at most ~500 requests per second**. A single 40 ms pause pushes
2,000 requests past the bound in one event. The pause budget is not small;
it is *zero*. This is why sub-millisecond at this QPS is an architectural
invariant — every source of variance must be **designed out**, not tuned
down. Managed runtimes cannot promise that; a systems language with
deterministic memory management can.

---

## 3. Failure Mode A — Fan-Out Tail Latency (the Straggler Bound)

### The managed read path

In traditional sharded ANN systems, a query is **broadcast to all
partitions**; the coordinating node awaits every shard, deserializes the
results, performs a global merge-sort, and responds ([1], [3]). The design
document's citation of the phenomenon is exact: *"the latency of any single
query is strictly bound by the slowest responding shard in the cluster"*
(§2, citing [3]).

**[ANALYTICAL] Why spatial workloads are the worst case.** Managed engines
shard by vector ID or hash — sharding that is agnostic to query semantics.
A geo-constrained nearest-neighbor query can match vectors on *any* shard,
so for this workload the fan-out factor N equals the **total shard count,
for every query**. Hash sharding is optimal for uniform key lookup and
catastrophic for locality queries.

### The exact math

For independent shard latencies with CDF F, the coordinator's response
satisfies:

```
P(max of N shards ≤ x) = F(x)^N
```

To hold the *cluster* p99, each shard must hold the quantile
`0.99^(1/N)`:

| Fan-out N (shards) | Required single-shard quantile |
|---|---|
| 2 | 99.50% |
| 4 | 99.75% |
| 8 | 99.87% |
| 16 | **99.94%** |
| 32 | 99.97% |

A 16-shard deployment holds its p99 **only if every shard simultaneously
holds its p99.94** — four-nines behavior on every node at once, on runtimes
whose p99.9 tails are routinely ~1 ms (GC, page cache, noisy neighbors).
Independence is the *optimistic* assumption: correlated pauses from shared
infrastructure make it worse. The coordinator then adds deserialize +
merge (§5) on top. Geo-distribution closes the case categorically: a
cross-DC round trip is 10–70 ms of physics — a "geo-distributed" managed
read is double-digit milliseconds before any indexing begins.

### The refutation [MEASURED]

Vectra routes by **4-character GeoHash prefix to exactly one shard** — a
deterministic routing-table lookup ([04](04-geohash-sharding.md)). Only when
the search radius crosses a cell boundary is the query duplicated, and only
to the shards governing **intersecting** adjacent cells: median 1 shard per
query, boundary cases 2–3, never broadcast. The single-shard common case is
pass-through — there is no merge step ([01](01-overview.md) §5).

Verified: **p99 = 0.47 ms at 51,240 QPS** (`make bench-qps`; golden-gated).
Owning code: `engine/src/routing/`, `engine/client/geo_router.cpp`.

---

## 4. Failure Mode B — Memory Pressure and Disk Spill

### The managed failure mechanism [ANALYTICAL]

The design document, citing [3]: when indexing mechanisms spill to disk
under inadequate memory optimization, *"graph traversal transitions from a
CPU-bound operation to an I/O-bound operation, causing performance to
degrade non-linearly as concurrency increases."* The mechanism, made
precise:

1. Managed indexes are commonly **mmap'd files** whose residency depends on
   the OS page cache — a best-effort guarantee, not an invariant.
2. The managed runtime itself (multi-GB heap, caches, buffers) competes for
   that same memory, so the OS evicts index pages under pressure.
3. Traversal then faults — stochastically, mid-`ef_search` — and fault
   service **serializes on device queues**, which is precisely the
   non-linear collapse under rising concurrency.

The arithmetic: one NVMe fault is ~100 µs+ — **10% of the entire budget for
a single fault**, against a traversal that touches hundreds of nodes. Two
or three cold-node faults and the budget is gone before the GPU tier is
even reached.

### The footprint that invites the problem [ANALYTICAL, class-level]

Same corpus, 10M × 64-dim. Managed-class figures are typical published
footprints for full-precision, pointer-based indexes — not a specific
vendor's numbers:

| Component | Vectra (flat + PQ) | Managed-class typical |
|---|---|---|
| Vector storage | 640 MB (64 B PQ codes/vector) | 2.56 GB (FP32) |
| HNSW adjacency | 640 MB (`m = 16`, 4-byte offsets) | ~2–4 GB (8-byte pointers, per-node object overhead, higher `m` defaults) |
| L2 norms | 40 MB | 40 MB |
| Metadata, filter indexes | ~600 MB | 1–2 GB |
| **Total** | **~2.1 GB** | **~6–10 GB** |

Both fit a large node — capacity is not the argument. **Residency
discipline is**: a 4–5× larger working set, mmap'd, sharing memory with the
runtime that must serve it, is a page-fault generator under load.

### The refutation [MEASURED / STRUCTURAL]

Vectra's index is **one contiguous, pre-faulted, `mlock`'d arena** with
integer offsets instead of pointers (`engine/src/memory/arena.cpp`,
`ram_lock.cpp`; derivation in [05](05-hnsw-flat-memory.md)). PQ compression
holds the working set at ~2.1 GB. The page-fault path is not mitigated —
it is **architecturally unreachable**. Residency is an invariant, enforced
operationally: 10M vectors resident in ~2.1 GB with zero spill, verified by
`engine/apps/shard_inspector` and the RAM-lock policy.

---

## 5. Failure Mode C — Runtime Nondeterminism

### The pause-and-drain arithmetic [ANALYTICAL]

At arrival rate λ = 50,000 QPS and utilization ρ = 0.8 (service capacity
μ = 62,500 QPS), a stop-the-world pause of duration t queues λ·t requests,
which drain at rate (μ − λ):

| Pause duration | Requests queued | Drain time | Total tail excursion |
|---|---|---|---|
| 1 ms | 50 | 4 ms | ~5 ms |
| 10 ms | 500 | 40 ms | ~50 ms |
| 40 ms | 2,000 | 160 ms | ~200 ms |
| 100 ms | 5,000 | 400 ms | ~500 ms |

At 80% utilization, a pause costs **~5× its duration** in tail. Note the
reconciliation with the design document's own claim ("a transient
millisecond-level pause can queue thousands of requests"): a 1 ms pause
queues 50; the pauses that actually occur in managed runtimes — young-gen
collections, compaction, rebalancing — are the 10–100 ms regime, queueing
500–5,000. Either way, against the §2 tolerance math the pause budget is
zero.

### The feedback loop

The coordinator path makes it worse: per query it deserializes N × k
records, **allocating an object graph each time** (8 shards × 32 candidates
≈ 256 records, ~70 KB through TLS → decode → allocate → merge-sort →
re-encode, on a managed runtime). Allocation pressure feeds GC frequency;
GC frequency feeds pauses; pauses feed queues. [ANALYTICAL, arithmetic
estimates.]

### The abstraction tax (RDBMS class) [ANALYTICAL]

Vector-capable relational databases add the write path's rent to every
read: connection setup/auth, MVCC snapshot acquisition, WAL-ordered
visibility, planner, executor, extension scan, result encoding — each
10–100 µs of layers the query never needed ([2]). Durable, transactional,
general — and structurally unable to spend the whole budget on the lookup
itself.

### The refutation [MEASURED]

Vectra's read path is **C++20 with deterministic memory management**: the
`CallData` state machine with address tags, one `CompletionQueue` per
physical core, thread-per-core with zero scheduler yields, and **one
serialization boundary** ([03](03-async-grpc.md), the pattern documented in
[10]). No GC exists to pause; hot-path allocation is prohibited by
contribution policy and reviewed as such
([`CONTRIBUTING.md`](../../CONTRIBUTING.md) — "No allocation on the request
hot path"). Verified: p99.9 = 0.71 ms across a 60-minute soak at 51,240 QPS
(`make bench-qps`).

---

## 6. The Class Survey

| Managed class | Q1 — fan-out | Q2 — residency | Q3 — determinism | What it is genuinely good at |
|---|---|---|---|---|
| Vector-capable RDBMS (pgvector-class) | Often OK at small N | ✗ buffer-cache dependent | ✗ planner/MVCC/GC layers | Transactions, SQL joins, existing estates |
| Managed dedicated vector engines | ✗ generic sharding → full fan-out on spatial queries | ✗ mmap + page cache | ✗ managed-runtime GC | Zero-ops elasticity, multi-tenancy, ecosystem |
| Serverless / embedded offerings | ✗ | ✗ | ✗ + cold starts, rate ceilings | Bursty, low-traffic, zero-infra workloads |

A fourth, quieter failure applies to all three classes: **contract
control** [ANALYTICAL]. R1 is not only QPS and latency — it is SLO-grade
telemetry (`vectra_health` shard ownership, per-stage µs histograms),
deterministic routing semantics, and versioned error contracts
([`contracts/proto/`](../../contracts/proto/)). Proprietary managed APIs
cannot express these; an adapter layer to translate them becomes part of
the measured path.

**The counterfactual that closes the argument.** For a managed offering to
pass R1 it would need to: route by spatial locality to exactly one node
(its own GeoHash scheme); hold a flat, compressed, RAM-locked index; execute
reads in a systems language with per-core event loops; and expose per-shard
SLO telemetry. Each feature removes a managed generality; all four together
remove the *product*. **R1 is a build specification wearing the costume of
a buy decision.**

---

## 7. The Derived Design Contract

This analysis is the requirements specification for the deep dives. Each
failure mode mandates a property, each property has an owner, each owner
has a gate:

| Failure mode | Mandate | Owning document | Owning code | CI gate |
|---|---|---|---|---|
| A — fan-out | O(1)-in-shards routing; adjacency-only duplication; 100% boundary recall | [04](04-geohash-sharding.md) | `engine/src/routing/` | `engine/tests/boundary/`, `make bench-recall` |
| B — spill | RAM-residency as an invariant: flat arena + PQ + `mlock` | [05](05-hnsw-flat-memory.md) | `engine/src/{index,memory}/` | `shard_inspector`, soak gates |
| C — nondeterminism | Systems language; no GC; CQ-per-core; one serialization boundary | [03](03-async-grpc.md) | `engine/src/server/` | `make bench-qps` |
| Budget (§2) | Microsecond ledger maintained per stage | [`latency-budget.md`](../research/latency-budget.md) | `serving/` | `golden_latency.json` |
| Contract control | First-party proto; health/ownership telemetry | [`docs/api/`](../api/) | `contracts/proto/` | `make proto` + API tests |

---

## 8. The Cost of the Decision — Honest Accounting

A decision record that omits costs is advocacy. What "no managed backend"
actually means:

| Capability | A managed backend gives you | Vectra instead | Where it lives |
|---|---|---|---|
| Durability & backup | Vendor-managed snapshots | **Corpus-as-source-of-truth**: index is a deterministic, parallel-per-shard rebuild from checksum-pinned data | [`data/pipelines/`](../../data/pipelines/), [`runbooks/index-rebuild.md`](../runbooks/index-rebuild.md) |
| Elastic scaling | Autoscaling | Fixed topology; config-driven re-partitioning with human oversight | [`runbooks/resharding.md`](../runbooks/resharding.md) |
| Upgrades & patching | One-click | Pinned, ADR-gated version pairing (PyTorch↔Triton↔CUDA, gRPC↔protobuf) | `requirements-lock.txt`, [`decisions/`](../decisions/) |
| Observability | Vendor dashboards | First-class OTel + Prometheus, including shard-ownership telemetry a vendor cannot expose | [`deploy/observability/`](../../deploy/observability/) |
| Ops staffing | Vendor absorbs it | **We own three runbooks and the on-call** | [`docs/runbooks/`](../runbooks/) |

Why the trade is acceptable *for this workload*: the traffic is
read-dominated; the deployment is single-owner; the SLO is tail latency,
not elasticity; DR reduces to "rebuild a derived artifact from pinned
inputs" (RPO = 0 by construction, RTO bounded by rebuild); and the
observability surface we would need from a vendor we built anyway — it is
our own SLO telemetry. The real price is **systems-engineering staffing**,
and it is priced in: the runbooks exist because we wrote them, not because
a vendor did.

## 9. Re-Evaluation Triggers

This decision is conditional, not ideological. It flips — via a new ADR
superseding 001 — when any of these holds:

| Trigger | Why the decision flips |
|---|---|
| p99 SLO relaxed to ≥ 10 ms | The pause/fault budget becomes nonzero; managed tails become affordable |
| Mutation ratio exceeds ~10% of traffic | The engine is read-optimized by design ([01](01-overview.md) §12); write-first systems win |
| Multi-tenant isolation required | Vendor platforms are built for exactly this |
| Corpus outgrows flat-model headroom (~100×, i.e., working set vs. per-node RAM) | Tiered/disk-first architectures re-enter the conversation |
| Systems-engineering staffing is lost | The runbook burden is the cost; if it can't be carried, the cost is unrealized |
| **Any managed offering passes the `benchmark/qps_latency/` gates** | The verdict is empirical, not aesthetic — see §10 |

## 10. Evidence Base and Falsifiability

| Claim | Status | Verification |
|---|---|---|
| Managed read-path structure (broadcast → await-all → merge) | Literature [1], [3] | [`bibliography.md`](../research/bibliography.md) |
| max-of-N identity and quantile table | **Exact mathematics** | §3, derivable |
| Managed-runtime pause profile (10–100 ms regime) | Literature + industry baseline | §5 |
| Page-fault cost ~100 µs+ vs. budget | Arithmetic + device baselines | §4 |
| Vectra p99 = 0.47 ms @ 51,240 QPS (60-min soak) | **[MEASURED]** | `make bench-qps`, `golden_latency.json` |
| Vectra boundary recall = 100% | **[MEASURED]** | `engine/tests/boundary/` |
| Vectra residency: 10M in ~2.1 GB, zero spill | **[MEASURED]** | `shard_inspector`, soak gates |
| Working-set composition | **[MEASURED]** + derivation in [05](05-hnsw-flat-memory.md) | [01](01-overview.md) §8 |

**The falsifiability contract.** The harnesses in
[`benchmark/`](../../benchmark/) speak the gRPC contract; any backend that
implements `contracts/proto/vectra_search.proto` can be pointed at
`run_soak.py` and held to the identical gates. If a managed offering — via
adapter or native — passes, this document is revised and ADR-001 is
superseded. **The verdict is one soak away from being wrong, by design.**

## 11. Related Documents

- [01 — System Overview](01-overview.md) — requirements, lifecycle, budgets
- [03 — Async gRPC Architecture](03-async-grpc.md) — the C-dictum refutation, in depth
- [04 — GeoHash Sharding](04-geohash-sharding.md) — the A-dictum refutation, in depth
- [05 — HNSW Flat Memory](05-hnsw-flat-memory.md) — the B-dictum refutation, in depth
- [ADR-001](../decisions/adr-001-async-grpc-over-managed-db.md) — the recorded decision
- [`docs/research/latency-budget.md`](../research/latency-budget.md) — the maintained µs ledger
- [`docs/research/bibliography.md`](../research/bibliography.md) — annotated sources [1], [2], [3], [10]
- Runbooks: [`index-rebuild`](../runbooks/index-rebuild.md) · [`node-bringup`](../runbooks/node-bringup.md) · [`resharding`](../runbooks/resharding.md) · [`incident-latency`](../runbooks/incident-latency.md)

---

*Change control: this document is part of the Phase 0 frozen specification.
Substantive changes require an ADR in the same PR. Performance figures move
only through benchmark PRs that update the regression goldens and
[`latency-budget.md`](../research/latency-budget.md). Analytical claims
about managed systems move only with new citations; measured claims about
Vectra move only with new harness output.*