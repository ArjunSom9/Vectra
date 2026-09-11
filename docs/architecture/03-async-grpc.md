# 03 — High-Throughput Asynchronous gRPC Architecture

| | |
|---|---|
| **Status** | Accepted — Frozen Spec (Phase 0) |
| **Version** | 1.0.0 |
| **Synthesizes** | Design document §3 (High-Throughput Asynchronous gRPC Network Architecture) |
| **Implements the refutation of** | Failure Mode C — runtime nondeterminism ([02 §5](02-why-not-managed.md)) |
| **Deep-dive companions** | [01](01-overview.md) · [02](02-why-not-managed.md) · [04](04-geohash-sharding.md) — routing that makes single-shard possible · [05](05-hnsw-flat-memory.md) — what the worker executes |
| **Refuted-by condition** | Zero-stall soak failing, or any dispatch-stage regression in [`benchmark/qps_latency/`](../../benchmark/qps_latency/) |
| **Change control** | ADR-backed PRs only |

> **Core claim.** The network tier is engineered to be *boring*: it consumes
> **55 µs of the 290 µs p50 budget** (52 µs transport + 3 µs dispatch), makes
> **zero context switches** on the request path, performs **zero heap
> allocations** in steady state, and presents the kernel scheduler with
> **zero decisions** — one runnable thread per core, ever. [MEASURED — §9]

**Epistemic status.** As in [02](02-why-not-managed.md): **[ANALYTICAL]**
claims are exact arithmetic or class-level reasoning (gRPC threading model,
HTTP/2 flow control, scheduler behavior; sources [5], [7], [8], [10] —
annotated in [`bibliography.md`](../research/bibliography.md));
**[MEASURED]** claims are Vectra's own, reproducible via `make bench-qps`
and enforced by the regression goldens.

---

## 1. The Problem — Synchronous Threading at 50K QPS

### 1.1 Anatomy of the synchronous model [ANALYTICAL]

The standard gRPC C++ synchronous server multiplexes each RPC onto a
handler thread drawn from a **dynamically sized internal pool** — threads
spawned and retired as concurrency fluctuates ([7], [10]). The ergonomic
cost of that convenience is a three-part tax, each of which the design
document names explicitly (§3):

1. **Context-switching penalties.** When the active request rate reaches
   50,000 QPS, a synchronous pool continuously swaps threads on and off
   physical cores.
2. **L1/L2 cache invalidation.** Every eviction throws away the working
   set — decode buffers, call state, index scratch — that the next
   scheduled thread must laboriously refill.
3. **Thread-pool starvation.** Blocked-but-runnable threads accumulate
   until the OS scheduler itself becomes the bottleneck.

### 1.2 The arithmetic [ANALYTICAL]

At the reference deployment's load — 51,240 QPS across 3 shards ≈ **17,080
QPS per shard** — apply Little's law (L = λ·W) to each model:

| | Sync pool model | Async thread-per-core model |
|---|---|---|
| Concurrency at mean | 17,080 × 0.29 ms ≈ 5 in-flight | 185 QPS/worker × 157 µs ≈ **0.03 per worker** |
| Concurrency at 10× burst | ≈ 50–170 handler threads | ≈ 0.3 per worker |
| Threads vs. 96 cores | **oversubscribed under burst** | 1 thread per core, always |
| Scheduler behavior | Runnable queue → CFS timeslice quantization (min granularity ~0.75 ms, scheduling-period scale ~ms) | **No queue exists** |
| Per-switch cost | 1–5 µs direct + 1–10 µs cache refill, ≥ 2 switches/request | **0 switches** |

The decisive row is the last two. The switch tax alone (2–10 µs best case)
is survivable; the **scheduler-queue tail is not** — under burst, a runnable
handler thread waits at *timeslice* granularity, which is
sub-millisecond-to-milliseconds: the entire p99 budget, spent before the
search even runs. Against the [budget arithmetic of 02 §2](02-why-not-managed.md)
— at 50K QPS, at most ~500 requests per second may exceed 1 ms — any
scheduler-quantized wait is disqualifying.

### 1.3 The architectural conclusion

The design document's mandate (§3): a **non-blocking, event-driven model**
centered on the `CompletionQueue` state machine, in a systems language with
deterministic memory, adhering to a **strict thread-per-core** paradigm.
This document is that mandate, made precise.

---

## 2. The Architectural Contract — Four Rules

| # | Rule | Consequence | Owning code |
|---|---|---|---|
| R1 | **One `CompletionQueue` per physical worker core** | No CQ-internal lock contention; event streams partition cleanly | `async_server.cpp` |
| R2 | **One polling thread per CQ, core-pinned; nothing else ever runs on that core** | Zero context switches, zero scheduler decisions, stable L1/L2 | `cq_worker.cpp`, `core_pin.cpp` |
| R3 | **Every RPC is a `CallData` state machine whose heap address is the `void*` tag of every async operation** | O(1) event-to-object dispatch; no lookup tables, no locks | `call_data.cpp` |
| R4 | **Re-arm before execute** — the instant a request arrives, a fresh `CallData` begins listening before any search work starts | One busy request never delays the next arrival; the loop never goes deaf | `call_data.cpp` |

R3 is the pattern the official asynchronous tutorial documents ([10]): the
application manages the entire lifecycle and memory state of every RPC
through a custom `CallData` object whose **memory address, cast to `void*`,
is passed to the gRPC runtime as a unique identification tag**. When
`CompletionQueue::Next` yields a tag, the worker casts it back — the tag
*is* the object; no other identification exists or is needed.

### Server structure

```
                     main.cpp bootstrap
                           │
                     ServerBuilder
         RegisterService(Search · Ingest · Health)
         AddListeningPort(:7101)
         AddCompletionQueue() × 92        ← one per worker core
                           │
         ┌─────────────────▼───────────────────────────┐
         │            gRPC core (iomgr)                │
         │  epoll · HTTP/2 framing · flow control      │
         │  incoming streams distributed across CQs    │
         │  internal threads pinned to housekeeping    │
         └──────┬──────────┬──────────┬──────── … ─────┘
                │          │          │
              CQ #4      CQ #5      CQ #6   …      CQ #95   (one per core)
                │          │          │              │
             thread #4  thread #5  thread #6  …   thread #95  (pinned)
                │          │          │              │
             pool #4    pool #5     pool #6    …   pool #95  (per-core
                │          │          │              │      CallData freelists)
                └──────────┴─────┬────┴──────────────┘
                                 │
                  shared flat HNSW arena (read-only, [05])
```

**Core budget** (config default, `engine/configs/cluster_topology.yaml`):
96 physical cores → **cores 0–3 housekeeping** (kernel, NIC IRQs, gRPC
internal iomgr threads, telemetry exporters) → **cores 4–95 = 92 CQ
workers**. The gRPC runtime's own background threads are pinned to the
housekeeping cores so they can never preempt a worker.

---

## 3. The `CallData` State Machine

### 3.1 States and transitions

```
 freelist ──▶ LISTENING ──(RequestSearch, tag = this)──▶ [awaiting arrival]
                                                            │ CQ::Next → tag
                                                            ▼
                  re-arm: freelist take ──▶ DISPATCHED ──▶ SERVICING
                  (fresh LISTENING, BEFORE      │  parse · search · encode
                   any other work)              │  — all on this thread
                                               ▼
                                          FINISHING ──(Finish, tag = this)
                                                            │ CQ::Next → tag
                                                            ▼
 freelist ◀────────────────── recycle ──────────────────────┘
```

**R4 in the diagram:** the re-arm edge fires *before* the SERVICING edge —
this is the design document's *"immediately spawns a new CallData object to
listen for subsequent connections, and executes the search logic without
ever yielding control to the operating system scheduler"* (§3), restated as
an ordering constraint the tests enforce.

**The `ok` flag.** `Next` returns a boolean alongside every tag; `ok=false`
marks cancellation — client disconnect, deadline expiry, or shutdown. Every
state handles its failure transition to recycle; **a `CallData` never leaks,
in any interleaving**. This is why the soak can prove absence of leaks
empirically (§9: flat RSS).

**The recycle invariant.** An object returns to the freelist only after
*every* async operation registered with its address has completed. This is
what makes address-as-tag safe across recycling: a stale tag match is
structurally impossible, because no operation carrying a retired address
can still be outstanding.

### 3.2 Production hardening (beyond the tutorial pattern)

The tutorial's canonical implementation allocates `CallData` with `new` and
destroys it with `delete` per request ([10]) — acceptable for a demo,
disqualifying here. The contribution policy
([`CONTRIBUTING.md`](../../CONTRIBUTING.md): *"No allocation on the request
hot path"*) is enforced by four design decisions:

| Decision | Mechanism | Rationale |
|---|---|---|
| **Per-core freelists** | 256 pre-allocated `CallData` objects per worker, recycled per the invariant | `new`/`delete` on the hot path reintroduces exactly the nondeterminism §1 exists to remove |
| **Arena-parsed requests** | Each `CallData` owns scratch space, reset on recycle; protobuf Arena decode | Zero decoder allocations; scratch stays L1/L2-warm across requests on that core |
| **Bounded pool + telemetry** | Freelist high-water mark exported per worker via `vectra_health` | Exhaustion is a loud, observable event — never a silent queue |
| **Single serialization boundary** | One decode on arrival, one encode in FINISHING | The boundary [02 §5](02-why-not-managed.md) promises; no intermediate object graphs exist |

**Sizing rule [ANALYTICAL].** Mean in-flight per worker is ~0.03 (§1.2); the
freelist of 256 is >3 orders of magnitude beyond mean and sized for
pathological *per-CQ skew* (all streams of a hot connection landing on one
CQ), which the per-CQ event counters monitor. High-water above 50% fires an
alert — it indicates routing skew to investigate, not a pool to enlarge.

---

## 4. The Worker Loop — `cq_worker.cpp`

The worker executes exactly one shape of iteration, forever:

```
next event  →  cast tag to CallData  →  transition state machine
→  (re-arm if arrival)  →  do the state's work on THIS thread
→  register next async op  →  return to Next
```

**Why blocking `Next` is correct on a dedicated core.** `Next` blocks the
polling thread only when its CQ is empty — and since that thread has
literally nothing else to do, waiting *is* the idle state. The wakeup path
is the runtime's epoll notification: measured at **3 µs** from event
availability to tag dispatch ([01](01-overview.md) §5, step 4). There is no
busy-poll and no sleep-tuning: the only wait point on the worker is the one
the kernel resolves at hardware event speed.

**Worker discipline — the negative contract.** The loop must never:

1. **Block** — no synchronous I/O, no sleeping locks, no condition waits.
2. **Allocate** — freelists and arena only (§3.2).
3. **Yield** — no `sched_yield`, no long syscalls; the scheduler is never
   consulted.
4. **Cross cores** — no work stealing, no dispatch queues. A request is
   serviced *where it landed*, because the routing tier ([04](04-geohash-sharding.md))
   already placed it on the node that owns the data. Cross-core handoff
   would reintroduce a context switch *plus* a cache-line migration — the
   two taxes this architecture exists to eliminate.

**Stall definition (the Gate-2 invariant).** A *CQ stall* is any interval
> 1 ms between consecutive events returned to a worker while that worker's
CQ is non-empty, tracked via loop-iteration timestamps and per-CQ pending
counters. The 60-minute soak gate is **zero stalls** — "zero CQ stalls" is
not an aspiration; it is a counter the harness reads.

---

## 5. Core Affinity, NUMA, and IRQ Discipline — `core_pin.cpp`

**Topology** (reference node, [01 §4](01-overview.md)): single-socket
EPYC 9005, 96 Zen 5 cores in 12 CCDs of 8 cores, private 32 MB L3 per CCD,
512 GB DDR5, 100 GbE.

**The three-way binding rule.** For each worker core: the CQ worker, the
NIC receive steering (RSS queue → IRQ affinity) feeding it, and the memory
its arena pages draw from must live in the **same L3/NUMA domain**. On the
single-socket reference node this degrades gracefully into two disciplines
— RSS/IRQ steering toward the CCD that consumes each queue, and
`numactl --interleave=all` on the arena so its bandwidth spreads evenly
across all DDR5 channels — while **dual-socket deployments must
`--membind`** per the production rules, or the flat-memory gains of [05]
evaporate at the interconnect.

**Isolation rules:**

- Worker cores host workers, and nothing else — gRPC iomgr threads and
  telemetry exporters are pinned to housekeeping cores (0–3).
- **No CPU quotas, ever.** cgroup CFS throttling destroys R2 (ADR-007;
  the engine runs on bare metal; see [02 §5→§7 traceability]).
- One C++ ABI across gRPC, Abseil, Highway, and the engine — one compiler,
  one `-std=c++20`, one libstdc++ ([tech-stack pairing rule]).
- IRQ coalescing on the NIC is tuned at bring-up
  ([`runbooks/node-bringup.md`](../runbooks/node-bringup.md)) — the 29 µs
  jitter-reserve line in the [01 §5 ledger](01-overview.md) is this
  subsystem's measured cost, owned here.

---

## 6. Client Architecture — The Channel Pool

### 6.1 Why one channel fails [ANALYTICAL]

A single gRPC channel maps to a **single underlying HTTP/2 connection**,
which carries two hard constraints ([8], [9]):

| Constraint | Arithmetic on the reference fabric |
|---|---|
| **Stream cap** — the connection enforces a negotiated `MAX_CONCURRENT_STREAMS` (order 100 by default) | Gateway in-flight ≈ 50K × 0.29 ms ≈ **15 average, ~150 at 10× burst** — brushing the cap at burst, *through* it under connection skew |
| **Flow control / BDP** — default ~64 KB per-stream windows vs. a 100 GbE × 100 µs RTT path | BDP = **1.25 MB**; default windows cap a single connection near ~5 Gbps and collapse on loss |
| **Head-of-line blocking** — one TCP retransmission stalls *every* multiplexed stream on that connection | A single loss event stalls the entire request stream behind it |

The design document's conclusion (§3): pushing 50,000 queries through one
channel makes the runtime **artificially queue client-side** — negating the
async server. The pool is not an optimization; it is the second half of the
architecture.

### 6.2 Pool design

| Property | Value | Source of truth |
|---|---|---|
| Pool size | **64 pre-warmed channels** | `serving/configs/service.yaml` (`channel_pool_size`) |
| Distribution | Round-robin across the pool | same config; mirrors `engine/client/channel_pool.cpp` |
| Warm-up | One `HealthService` RPC per channel at startup | `serving/retrieve.py` |
| Failure handling | `TRANSIENT_FAILURE` channels are skipped; reconnect with gRPC's exponential backoff | pool health tracker |
| Deadline | **2 ms per RPC** (`timeout_ms`), propagated; expiry cancels server-side → clean `ok=false` → recycle | same config |
| Windows | HTTP/2 flow-control windows sized to BDP at bring-up | `runbooks/node-bringup.md` |

Pool math at burst: 150 in-flight across 64 channels ≈ ≤3 streams per
connection — an order of magnitude under the cap, and a single loss event
now stalls at most **1/64** of traffic. The rule the README states — *pool
size ≥ expected concurrent streams* — is the design constraint in one line.

**Two implementations, one policy.** The pattern exists twice by design:
the **C++ reference** (`engine/client/channel_pool.cpp`, `async_search.cpp`,
`geo_router.cpp` — used by `stress_client` and every benchmark harness) and
the **gateway-side mirror** (`serving/retrieve.py`, `grpcio.aio` with the
same 64-channel round-robin read from the same config). Both consume
`contracts/proto/`; the policy is config-shared so they cannot drift; and
the load generator being C++ means the soak reproduces the gateway's wire
behavior without Python in the measured path.

---

## 7. Shutdown, Drain, and Failure Behavior

Graceful drain (the sequence [`runbooks/node-bringup.md`](../runbooks/node-bringup.md)
scripts):

1. `Server::Shutdown(deadline)` — stop accepting; let in-flight finish.
2. `cq->Shutdown()` per CQ — flush queued events.
3. Each worker drains `Next` until it returns false; every remaining tag
   comes back with `ok=false` and recycles per §3.1.
4. Join threads; the arena is `munlock`'d and released last.

Order is load-bearing: server-then-CQ guarantees no stream arrives after
its CQ has stopped reporting. Node-loss behavior — fail fast, health-cache
invalidation, no wrong-shard fan-out — is owned by [01 §9](01-overview.md)
and the runbooks; this document owns the guarantee that a draining node
leaves zero leaked `CallData` objects behind (proven by the flat-RSS soak
gate, §9).

---

## 8. Measured Results and Budget Mapping

**[MEASURED]** — reference environment of [01 §4](01-overview.md);
reproduce with `make bench-qps`; goldens in
`tests/regression/golden_latency.json`.

| Quantity | Value | Where in the [01 §5] ledger |
|---|---|---|
| Transport hop, gateway → shard | 52 µs p50 | step 3 (18% of budget) |
| CQ wakeup + tag dispatch | 3 µs p50 | step 4 (1% of budget) |
| Worker service time (parse+search+encode) | 154 µs | steps 5–7 (owned jointly with [04]/[05]) |
| **Network-tier total** | **55 µs / 290 µs = 19% of p50** | |
| Soak | 51,240 QPS, 60 min, 3 shards | |
| Tail | p50 0.29 / p99 0.47 / p99.9 0.71 ms | |
| CQ stalls | **0** | Gate-2 counter |
| Hot-path heap allocations | **0** (allocator-interposition build, soak) | §3.2 enforcement |
| `CallData` leaks | **0** — resident set flat across the 60-min soak | §7 |
| Mean worker utilization at full soak | ~3% (185 QPS/worker × 157 µs) | headroom is deliberate |

The utilization figure is the architecture's signature: at maximum
sustained load, the network tier is **3% busy** — the binding constraint on
this system is index memory locality ([05](05-hnsw-flat-memory.md)), never
the RPC machinery. Sub-millisecond tails are not won by heroic effort;
they are what remains when every source of variance has been designed out.

---

## 9. Derived Design Contract

| Mandate | Owning code | Owning test / gate |
|---|---|---|
| R1 — one CQ per worker core | `async_server.cpp` | bring-up affinity check; per-CQ event counters in `vectra_health` |
| R2 — pinned, isolated workers | `cq_worker.cpp`, `core_pin.cpp` | zero-stall soak gate; affinity map in health |
| R3 — tag identity, `ok=false` completeness | `call_data.cpp` | integration suites; TSan/ASan-clean requirement; flat-RSS soak |
| R4 — re-arm-before-execute ordering | `call_data.cpp` | throughput gate in `make bench-qps` (a deaf loop cannot hold QPS) |
| No hot-path allocation | `call_data.cpp` freelists, protobuf Arena | allocator-interposition soak run in CI |
| Channel pool ≥ streams; round-robin; deadlines | `channel_pool.cpp`, `serving/retrieve.py` | `serving/tests/test_channel_pool.py`; deadline-propagation test |
| Three-way NUMA/IRQ binding | `core_pin.cpp`, bring-up | `runbooks/node-bringup.md` verification steps |
| Zero leaks under drain | `main.cpp` shutdown sequence | drain integration test |

## 10. Production Rules Owed by This Document

1. **Never a CPU quota on an engine node** — CFS throttling is a scheduler
   decision injected into a loop that has none (ADR-007).
2. **Channel pool ≥ expected concurrent streams** — verify
   `MAX_CONCURRENT_STREAMS` and pool high-water in every soak.
3. **Housekeeping cores exist** — gRPC internal threads, exporters, and IRQs
   live on 0–3; any process that lands on a worker core is a bring-up bug.
4. **One ABI** — the engine, gRPC, Abseil, and Highway build as one unit
   ([tech-stack pairing rule]); mixed-STL links are rejected.
5. **The tag recycling invariant is not negotiable** — any change to
   `call_data.cpp` state transitions re-proves it via the drain and
   flat-RSS gates.
6. **Watch per-CQ skew** — event-rate variance across CQs is the early
   signal of connection skew; investigate routing, never "add threads."

## 11. Related Documents

- [01 — System Overview](01-overview.md) — lifecycle ledger this document's
  55 µs inhabits
- [02 — Why Not Managed](02-why-not-managed.md) — the nondeterminism this
  architecture refutes (Failure Mode C)
- [04 — GeoHash Sharding](04-geohash-sharding.md) — the routing that makes
  R4's "service where it lands" possible
- [05 — HNSW Flat Memory](05-hnsw-flat-memory.md) — the 118 µs the worker
  executes after dispatch
- [ADR-001](../decisions/adr-001-async-grpc-over-managed-db.md) ·
  [ADR-007](../decisions/adr-007-bare-metal-over-k8s.md)
- [`docs/research/latency-budget.md`](../research/latency-budget.md) — the
  µs ledger
- [`docs/api/`](../api/) — the wire contract both client implementations
  share
- Runbooks: [`node-bringup`](../runbooks/node-bringup.md) ·
  [`incident-latency`](../runbooks/incident-latency.md)
- Sources: [5], [7], [8], [9], [10] in
  [`bibliography.md`](../research/bibliography.md)

---

*Change control: this document is part of the Phase 0 frozen specification.
Substantive changes require an ADR in the same PR. Performance figures move
only through benchmark PRs that update the regression goldens and
[`latency-budget.md`](../research/latency-budget.md). Any change to the
`CallData` state machine or worker discipline re-runs the zero-stall,
zero-allocation, and flat-RSS soak gates.*