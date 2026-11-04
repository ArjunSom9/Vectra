# ADR-003: HNSW in flat, offset-addressed memory

* **Status:** Accepted (February 2025)
* **Deciders:** Workstream A
* **Specification:** [05 — The shard index](../architecture/05-hnsw-flat-memory.md)

## Context

Graph search is a chain of dependent memory accesses. A pointer-based graph
(one heap object per node, 8-byte pointers) makes every hop a likely cache
miss and spreads one node's data over several lines. The fastest public
implementations store the graph in flat arrays with integer offsets [21]
[23] [28].

## Decision

Store each layer as an array of fixed-stride records indexed by 32-bit node
id. A layer-0 record holds the node's PQ code next to its neighbor list.
Vectors, properties and the prefilter are separate flat columns, allocated
once at capacity in 64-byte-aligned buffers (huge pages where available)
and locked in RAM at start-up. Readers are lock-free. A single writer
publishes appended nodes with release/acquire ordering and updates
adjacency words through `std::atomic_ref`.

## Consequences

* A traversal step fetches one record and gets the next distance
  computation's data with it.
* Capacity is fixed at build time (`IndexParams.max_elements`). Growing past
  it is a rebuild ([index-rebuild](../runbooks/index-rebuild.md)).
* Deletion is out of scope. The workload is read-dominated, and ingestion
  appends.
