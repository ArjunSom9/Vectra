# ADR-001: A bespoke asynchronous gRPC engine instead of a managed vector backend

* **Status:** Accepted (January 2025)
* **Deciders:** Workstream A
* **Analysis:** [02 — Why not managed](../architecture/02-why-not-managed.md)

## Context

Objective O1 needs 50,000 searches per second over 10M property embeddings
with sub-millisecond p50 latency and recall@10 ≥ 0.95. Managed vector
backends shard by id or hash and scatter every geo-constrained query to all
partitions [1] [11]. They keep indexes in mmap'd, page-cache-dependent
storage, and serve from managed runtimes with collection pauses. Each of
these turns rare events into tail latency at this request rate [13].

## Decision

Build the retrieval tier as a C++20 service on gRPC's asynchronous API
[19]: one `CompletionQueue` per worker core, `CallData` state machines as
completion tags, pinned workers, RAM-locked flat indexes, and GeoHash
sharding that bounds every query's fan-out ([ADR-002](adr-002-geohash-precision-levels.md)).

## Consequences

* We own operations: bring-up, re-sharding, index rebuilds, on-call
  ([runbooks](../runbooks/)).
* Durability reduces to rebuilding a derived artifact: the index is a
  deterministic function of the checksum-pinned corpus
  (`data/pipelines/`, [index-rebuild](../runbooks/index-rebuild.md)).
* The wire contract is first-party (`contracts/proto/vectra/`), so the
  benchmark harnesses can hold any backend that implements it to the same
  gates. The decision is re-opened if a managed offering passes them.

## Alternatives considered

* A vector-capable RDBMS: buffer-cache residency and planner/MVCC layers on
  every read.
* A managed vector service with a geo filter: full scatter on spatial
  queries, no control over residency or runtime pauses.
