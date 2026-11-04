# 03 — High-Throughput Asynchronous gRPC Architecture

| | |
|---|---|
| **Status** | Accepted (v1.1, corrected against the implementation) |
| **Synthesizes** | Design document §5.1 (asynchronous gRPC transport on a thread-per-core model) |
| **Implements the refutation of** | Failure Mode C — runtime nondeterminism ([02 §5](02-why-not-managed.md)) |
| **Deep-dive companions** | [01](01-overview.md) · [02](02-why-not-managed.md) · [04](04-geohash-sharding.md) — routing that makes single-shard possible · [05](05-hnsw-flat-memory.md) — what the worker executes |
| **Refuted-by condition** | The A1 dispatch gate failing, or any dispatch-stage regression in [`benchmark/qps_latency/`](../../benchmark/qps_latency/) |
| **Change control** | ADR-backed PRs only |

> **Core claim.** The network tier is engineered to be *boring*: it is
> budgeted at **about 55 µs of a sub-millisecond p50** (transport plus
> dispatch), makes no context switch on the request path, allocates no call
> objects in steady state, and presents the kernel scheduler with no
> decisions: one runnable thread per worker core. Deliverable A1 measures
> it: 50,000 no-op RPCs/s through the channel pool with p99 dispatch
> overhead under 100 µs [TARGET, `make bench-qps`].

**Epistemic status.** As in [02](02-why-not-managed.md): **[ANALYTICAL]**
claims are arithmetic or class-level reasoning (gRPC threading model,
HTTP/2 flow control, scheduler behavior; sources [13], [19], [20], [27],
annotated in [`bibliography.md`](../research/bibliography.md)).
**[TESTED]** claims are checked by tests in this repository, and
**[TARGET]** numbers await their harness run on the reference hardware.

---

## 1. The Problem — Synchronous Threading at 50K QPS

### 1.1 Anatomy of the synchronous model [ANALYTICAL]

The standard gRPC C++ synchronous server multiplexes each RPC onto a
handler thread drawn from a **dynamically sized internal pool**, with
threads spawned and retired as concurrency fluctuates ([19], [27]). The
ergonomic cost of that convenience is a three-part tax, which the design
document describes (§5.1):

1. **Context-switching penalties.** When the active request rate reaches
   50,000 QPS, a synchronous pool continuously swaps threads on and off
   physical cores.
2. **L1/L2 cache invalidation.** Every eviction throws away the working
   set — decode buffers, call state, index scratch — that the next
   scheduled thread must laboriously refill.
3. **Thread-pool starvation.** Blocked-but-runnable threads accumulate
   until the OS scheduler itself becomes the bottleneck.

### 1.2 The arithmetic [ANALYTICAL]

At the target load (50,000 QPS across 3 shards, about **16,700 QPS per
shard**), with a budgeted 0.3 ms service time, apply Little's law
(L = λ·W) to each model:

| | Sync pool model | Async thread-per-core model |
|---|---|---|
| Concurrency at mean | 16,700 × 0.3 ms ≈ 5 in-flight | ~180 QPS/worker × ~160 µs ≈ **0.03 per worker** |
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

The design document's mandate (§5.1): gRPC's C++ asynchronous API, in which
"the application binds a CompletionQueue to the RPC machinery, presents
every operation with a unique void* tag, and drives a poll loop on
CompletionQueue::Next, so a thread never blocks inside the runtime" [19],
deployed thread-per-core. This document is that mandate, made precise.

---

## 2. The Architectural Contract — Four Rules

| # | Rule | Consequence | Owning code |
|---|---|---|---|
| R1 | **One `CompletionQueue` per worker core** | No CQ-internal lock contention; event streams partition cleanly | `engine/src/server/async_server.cc` |
| R2 | **Polling threads pinned to that core** (`threads_per_queue`, default 1; gRPC's performance guide suggests 2 [27], so the ratio is benchmarked) | No context switches, no scheduler decisions, stable L1/L2 | `cq_worker.cc`, `core_pin.cc` |
| R3 | **Every RPC is a `CallData` state machine whose address is the `void*` tag of every async operation** | O(1) event-to-object dispatch; no lookup tables, no locks | `call_data.cc` |
| R4 | **Re-arm before execute**: the instant a request arrives, another call is armed before any search work starts | One busy request never delays the next arrival; the loop never goes deaf | `call_data.h` (`UnaryCall`) |

R3 is the pattern the official asynchronous tutorial documents ([19]): the
application manages the entire lifecycle and memory state of every RPC
through a custom `CallData` object whose **memory address, cast to `void*`,
is passed to the gRPC runtime as a unique identification tag**. When
`CompletionQueue::Next` yields a tag, the worker casts it back — the tag
*is* the object; no other identification exists or is needed.

### Server structure

```
                  engine/src/main.cc
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
         │  internal threads on housekeeping cores     │
         └──────┬──────────┬──────────┬──────── … ─────┘
                │          │          │
              CQ #4      CQ #5      CQ #6   …      CQ #95   (one per core)
                │          │          │              │
             thread #4  thread #5  thread #6  …   thread #95  (pinned)
                │          │          │              │
             pool #4    pool #5     pool #6    …   pool #95  (pre-registered,
                │          │          │              │      recycled calls)
                └──────────┴─────┬────┴──────────────┘
                                 │
                  shared shard index (lock-free reads, [05])
```

**Core budget** (`ServerOptions.worker_cores` in
`engine/configs/cluster_topology.textproto`, set per host by
`deploy/envs/prod.yaml`): 96 physical cores → **cores 0–3 housekeeping**
(kernel, NIC IRQs, gRPC's internal threads) → **cores 4–95 = 92 CQ
workers**, isolated with `isolcpus`/`nohz_full` by
`deploy/baremetal/provision.yml`, so nothing else is scheduled on them.

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

**R4 in the diagram:** the re-arm edge fires *before* the SERVICING edge.
This is the design document's "fresh CallData spawned before the current
one is processed" (§5.1), restated as an ordering constraint.

**The `ok` flag.** `Next` returns a boolean alongside every tag; `ok=false`
marks cancellation — client disconnect, deadline expiry, or shutdown. Every
state handles its failure transition to recycle; **a `CallData` never leaks,
in any interleaving**. This is why the soak can prove absence of leaks
empirically: the per-worker allocation counter stays flat.

**The recycle invariant.** An object returns to the freelist only after
*every* async operation registered with its address has completed. This is
what makes address-as-tag safe across recycling: a stale tag match is
structurally impossible, because no operation carrying a retired address
can still be outstanding.

### 3.2 Production hardening (beyond the tutorial pattern)

The tutorial's canonical implementation allocates `CallData` with `new` and
destroys it with `delete` per request ([19]). That is fine for a demo and
disqualifying here. Four design decisions keep the request path free of
call allocation:

| Decision | Mechanism | Rationale |
|---|---|---|
| **Pre-registered, recycled calls** | `requests_per_queue` (64) calls per method per completion queue, armed at start-up and recycled after `Finish` (`CallPool`) | `new`/`delete` on the hot path reintroduces exactly the nondeterminism §1 exists to remove |
| **Reused messages** | Each call keeps its request and response messages across recycles, so protobuf reuses their buffers | Decoding into warm, already-sized messages |
| **Bounded pool + telemetry** | Peak in-flight calls and post-start-up allocations exported per worker (`WorkerStats`) | Exhaustion is a loud, observable event (`CallPoolAllocating` alert), never a silent queue |
| **Single serialization boundary** | One decode on arrival, one encode at `Finish` | The boundary [02 §5](02-why-not-managed.md) promises; no intermediate object graphs exist |

**Sizing rule [ANALYTICAL].** Mean in-flight per worker is ~0.03 (§1.2); 64
pre-registered calls per method are three orders of magnitude beyond the
mean and sized for per-CQ skew (all streams of a hot connection landing on
one CQ). A late allocation means routing skew to investigate, not a pool to
enlarge blindly.

---

## 4. The Worker Loop — `cq_worker.cc`

The worker executes exactly one shape of iteration, forever:

```
next event  →  cast tag to CallData  →  transition state machine
→  (re-arm if arrival)  →  do the state's work on THIS thread
→  register next async op  →  return to Next
```

**Why blocking `Next` is correct on a dedicated core.** `Next` blocks the
polling thread only when its CQ is empty — and since that thread has
literally nothing else to do, waiting *is* the idle state. The wakeup path
is the runtime's epoll notification, budgeted at a few microseconds from
event availability to tag dispatch. There is no
busy-poll and no sleep-tuning: the only wait point on the worker is the one
the kernel resolves at hardware event speed.

**Worker discipline — the negative contract.** The loop must never:

1. **Block** — no synchronous I/O, no sleeping locks, no condition waits.
2. **Allocate calls**: pre-registered calls only (§3.2).
3. **Yield** — no `sched_yield`, no long syscalls; the scheduler is never
   consulted.
4. **Cross cores** — no work stealing, no dispatch queues. A request is
   serviced *where it landed*, because the routing tier ([04](04-geohash-sharding.md))
   already placed it on the node that owns the data. Cross-core handoff
   would reintroduce a context switch *plus* a cache-line migration — the
   two taxes this architecture exists to eliminate.

**What the soak watches.** A stalled worker shows up as a gap in its
`rpcs_completed` rate while the fleet is under load, and in the client-side
tail that `stress_client` measures from each request's scheduled send time
(open-loop, so a stall cannot hide its own queueing).

---

## 5. Core Affinity, NUMA, and IRQ Discipline — `core_pin.cc`

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
- One C++ ABI across gRPC, Abseil and the engine: one compiler (GCC 14),
  one `-std=c++20`, one libstdc++ (`scripts/bootstrap.sh` builds gRPC with
  the same toolchain).
- NIC interrupts are steered to the housekeeping cores at every boot
  (`deploy/baremetal/tune_latency.sh`); coalescing is tuned at bring-up
  ([`runbooks/node-bringup.md`](../runbooks/node-bringup.md)).

---

## 6. Client Architecture — The Channel Pool

### 6.1 Why one channel fails [ANALYTICAL]

A single gRPC channel maps to a **single underlying HTTP/2 connection**,
which carries hard constraints ([20], [27]):

| Constraint | Arithmetic on the reference fabric |
|---|---|
| **Stream cap** — the connection enforces a negotiated `MAX_CONCURRENT_STREAMS` (order 100 by default) | Gateway in-flight ≈ 50K × 0.29 ms ≈ **15 average, ~150 at 10× burst** — brushing the cap at burst, *through* it under connection skew |
| **Flow control / BDP** — default ~64 KB per-stream windows vs. a 100 GbE × 100 µs RTT path | BDP = **1.25 MB**; default windows cap a single connection near ~5 Gbps and collapse on loss |
| **Head-of-line blocking** — one TCP retransmission stalls *every* multiplexed stream on that connection | A single loss event stalls the entire request stream behind it |

The design document's conclusion (§5.1): "pushing 50,000 QPS through one
channel would silently queue requests inside the runtime and erase the
server's async advantage." The pool is not an optimization; it is the
second half of the architecture.

### 6.2 Pool design

| Property | Value | Source of truth |
|---|---|---|
| Pool size | **64 channels per shard** | `ClientOptions.channel_pool_size` in `engine/configs/cluster_topology.textproto` |
| Distinct connections | Each channel gets a distinct `vectra.channel_index` argument and a local subchannel pool, so gRPC cannot collapse the pool onto one connection [27] | `engine/client/channel_pool.cc`, `serving/retrieve.py` (`serving/tests/test_channel_pool.py` shows identical channels sharing one connection) |
| Replicas | Channel *i* connects to replica *i* mod *n* of the shard | same |
| Distribution | Round-robin across the pool | same |
| Warm-up | Every channel connects at start-up (`WaitForConnected` / `channel_ready`) | same |
| Failure handling | The fan-out is all-or-nothing; a failed RPC fails the request with its status, and gRPC reconnects with exponential backoff | `geo_router.cc`, `retrieve.py` |
| Deadline | **5 ms per RPC** (`timeout_ms`), propagated; expiry cancels server-side → `ok=false` → recycle | same config |
| Windows | gRPC's BDP probing sizes HTTP/2 flow-control windows | gRPC default |

Pool math at burst: 150 in-flight across 64 channels ≈ ≤3 streams per
connection — an order of magnitude under the cap, and a single loss event
now stalls at most **1/64** of traffic. The rule the README states — *pool
size ≥ expected concurrent streams* — is the design constraint in one line.

**Two implementations, one policy.** The pattern exists twice by design:
the **C++ client** (`engine/client/channel_pool.cc`, `async_search.cc`,
`geo_router.cc`, used by `stress_client`) and the **gateway-side mirror**
(`serving/retrieve.py`, `grpc.aio` with the same pool read from the same
configuration). Both consume `contracts/proto/`, and the policy is
config-shared so they cannot drift. Because the load generator is C++, the
soak reproduces the gateway's wire behavior without Python in the measured
path.

---

## 7. Shutdown, Drain, and Failure Behavior

Graceful drain (the sequence [`runbooks/node-bringup.md`](../runbooks/node-bringup.md)
scripts):

1. Stop re-arming calls (`CallPool::StopArming`).
2. `Server::Shutdown(deadline)`: stop accepting; let in-flight calls finish.
3. `cq->Shutdown()` per CQ: flush queued events.
4. Each worker drains `Next` until it returns false; every remaining tag
   comes back with `ok=false` and is released.
5. Join the threads, then free the call pools.

Order is load-bearing: server-then-CQ guarantees no stream arrives after
its CQ has stopped reporting. Node-loss behavior — fail fast, health-cache
invalidation, no wrong-shard fan-out — is owned by [01 §9](01-overview.md)
and the runbooks. This document owns the guarantee that a draining node
leaves no `CallData` behind (exercised by every start/stop in
`engine/tests/integration/`).

---

## 8. Targets and Budget Mapping

**[TARGET]**: the reference environment of [01 §4](01-overview.md),
reproduced with `make bench-qps`, gated by
`tests/regression/golden_latency.json` once a run is recorded.

| Quantity | Budget / target | Measured by |
|---|---|---|
| No-op RPC dispatch, 50,000 QPS through the pool | p99 < 100 µs (A1) | `run_soak.py --mode=ping` |
| Search, 50,000 QPS, 30 minutes, 3 shards | p50 < 1 ms (A3) | `run_soak.py --mode=search` |
| Transport hop, gateway → shard | ~50 µs budgeted | [`latency-budget.md`](../research/latency-budget.md) |
| CQ wakeup + tag dispatch | a few µs budgeted | same |
| Call allocations after start-up | 0 | `WorkerStats.call_pool_allocations` |

At the budgeted service time, a worker at full load is only a few percent
busy. The binding constraint is meant to be index memory locality
([05](05-hnsw-flat-memory.md)), never the RPC machinery.

## 9. Derived Design Contract

| Mandate | Owning code | Owning test / gate |
|---|---|---|
| R1 — one CQ per worker core | `async_server.cc` | per-worker `WorkerStats` in `HealthService.Report` |
| R2 — pinned, isolated workers | `cq_worker.cc`, `core_pin.cc` | `WorkerStats.core`; `make bench-qps` |
| R3 — tag identity, `ok=false` completeness | `call_data.cc` | `engine/tests/integration/`; ASan/TSan builds |
| R4 — re-arm-before-execute ordering | `call_data.h` | throughput gate in `make bench-qps` (a deaf loop cannot hold QPS) |
| No call allocation after start-up | `CallPool` | `WorkerStats.call_pool_allocations`, `CallPoolAllocating` alert |
| Channel pool, distinct connections, round-robin, deadlines | `channel_pool.cc`, `serving/retrieve.py` | `serving/tests/test_channel_pool.py` |
| NUMA/IRQ binding | `deploy/baremetal/` | `runbooks/node-bringup.md` verification steps |
| Clean drain | `async_server.cc` shutdown sequence | `engine/tests/integration/` |

## 10. Production Rules Owed by This Document

1. **Never a CPU quota on an engine node** — CFS throttling is a scheduler
   decision injected into a loop that has none (ADR-007).
2. **Channel pool ≥ expected concurrent streams.** Verify
   `MAX_CONCURRENT_STREAMS` and peak in-flight calls in every soak.
3. **Housekeeping cores exist** — gRPC internal threads, exporters, and IRQs
   live on 0–3; any process that lands on a worker core is a bring-up bug.
4. **One ABI.** The engine, gRPC and Abseil build as one unit with one
   compiler; mixed-STL links are rejected.
5. **The tag recycling invariant is not negotiable.** Any change to the
   `CallData` state transitions re-runs the integration suites under ASan and
   TSan.
6. **Watch per-CQ skew** — event-rate variance across CQs is the early
   signal of connection skew; investigate routing, never "add threads."

## 11. Related Documents

- [01 — System Overview](01-overview.md) — the request lifecycle this
  document's budget sits in
- [02 — Why Not Managed](02-why-not-managed.md) — the nondeterminism this
  architecture refutes (Failure Mode C)
- [04 — GeoHash Sharding](04-geohash-sharding.md) — the routing that makes
  R4's "service where it lands" possible
- [05 — HNSW Flat Memory](05-hnsw-flat-memory.md) — what the worker executes
  after dispatch
- [ADR-001](../decisions/adr-001-async-grpc-over-managed-db.md) ·
  [ADR-007](../decisions/adr-007-bare-metal-over-k8s.md)
- [`docs/research/latency-budget.md`](../research/latency-budget.md) — the
  µs ledger
- [`docs/api/`](../api/) — the wire contract both client implementations
  share
- Runbooks: [`node-bringup`](../runbooks/node-bringup.md) ·
  [`incident-latency`](../runbooks/incident-latency.md)
- Sources: [13], [19], [20], [27] in
  [`bibliography.md`](../research/bibliography.md)

---

*Change control: substantive changes require an ADR in the same PR. A
target becomes a result only through a recorded `make bench-qps` report.
Any change to the `CallData` state machine re-runs the integration suites
under the sanitizers.*