# 05 — The Shard Index: Flat-Memory HNSW with PQ

| | |
|---|---|
| **Status** | Accepted |
| **Synthesizes** | Design document §5.3 (flat-memory HNSW indexing with product quantization) |
| **Code** | `engine/src/index/`, `engine/include/vectra/index/`, `engine/src/memory/` |
| **Tests** | `engine/tests/unit/{hnsw,pq_codec,shard_index,geofilter,flat_store,simd_parity}_test.cc` |
| **Study** | [`hnsw-parameter-study.md`](../research/hnsw-parameter-study.md) |

A shard answers: the *k* nearest embeddings to a query, among the
properties within *r* km of a location that sold before a given day. This
document specifies the structures and the search algorithm.

## 1. Structures

All structures are flat: contiguous buffers addressed by 32-bit node ids,
with no per-node heap objects and no pointers. A node's graph record holds
its PQ code next to its neighbor list, so the record a traversal step
fetches brings the next distance computation with it.

| Structure | File | Layout |
|---|---|---|
| Float32 vectors | `flat_store.cc` | 64-byte-aligned rows (128 floats = 512 B) plus a squared-norm column |
| Layer-0 graph | `hnsw_graph.cc` | One record per node: `[PQ code][count][m0 neighbor ids]`, stride rounded up to 32 B (160 B at 16 subspaces, m0 = 32) |
| Upper layers | `hnsw_graph.cc` | One record per (node, level ≥ 1): `[node][down][count][m ids]`, stride rounded up to 16 B (80 B at m = 16) |
| PQ codebooks | `pq_codec.cc` | 16 subspaces × 256 centroids × 8 dimensions, float32 |
| Properties | `property_store.cc` | Columns: id, location, unit vector, 6-char cell, sale day, attributes |
| Prefilter | `geofilter.cc` | Ids sorted by 6-character cell; any 5- or 6-character cell is one contiguous range |

Large buffers come from `AlignedBuffer`: anonymous `mmap` with
`MADV_HUGEPAGE` on Linux for 2 MiB pages, 64-byte `aligned_alloc`
otherwise. At start-up the shard locks everything with `mlockall`
(`ram_lock.cc`, `--lock_memory`), and reports the result in
`MemoryStats.locked`.

Index files are written as sectioned binaries (`VECTRAIX`, version 1): a
header, then one section per structure, each with an FNV-1a checksum,
written to a temporary file and renamed into place (`binary_file.cc`).

## 2. Construction

`ShardIndex::Build` trains the product quantizer, encodes every vector, and
inserts every node into the graph, following Malkov and Yashunin [21]:

* **Levels** are drawn up front from the geometric distribution with
  mL = 1/ln M, so the graph's shape does not depend on insertion order or
  thread scheduling.
* **Degree** is M = 16 on upper layers and M0 = 2M = 32 on layer 0.
* **Neighbors** are chosen with the heuristic of [21] (Algorithm 4): a
  candidate is kept only if it is closer to the new node than to every
  neighbor already kept, which preserves long-range links.
* **ef_construction** = 200.
* **Parallelism**: `build_threads` workers insert disjoint ranges. Each
  node's adjacency is guarded by one of 65,536 striped mutexes, and the
  global entry point by its own mutex.
* **PQ** [22]: 16 subspaces of 8 dimensions, 256 centroids each (8 bits),
  so 16 bytes per vector. k-means runs per subspace in parallel on up to
  `pq_training_samples` (65,536) vectors with a seeded initialization, so
  builds are reproducible.

## 3. Search

`ShardIndex::Search(query, k, ef, radius, sold_before)`:

1. **Admission predicate.** `SearchPredicate` admits a node when its last
   sale precedes `sold_before_epoch_day` (if set), its 5-character cell is
   one of the cells covering the disk, and its exact distance is within the
   radius. The distance test compares squared chords between precomputed
   unit vectors, which orders points exactly as great-circle distance does.
2. **Population.** The prefilter counts the rows in the covering cells.
3. **Exhaustive scan** if the population is at most
   `exhaustive_scan_threshold` (2,048), or in exact mode: exact float32
   distances over the admitted rows. This is the common case for small
   radii and the oracle mode of `SearchRequest.exact`.
4. **Filtered HNSW** otherwise. Greedy descent through the upper layers,
   then a beam search on layer 0 of width `ef_search` (128), in the style of
   hnswlib's filtered search [23]. Every node is traversed, but only
   admitted nodes enter the result set. Distances are asymmetric PQ
   distances (ADC): one 16 × 256 lookup table per query, then 16 table
   reads per node. A visit budget of 256 × ef bounds pathological
   predicates.
5. **Exact re-rank.** The best `rerank_depth` (128) candidates are re-scored
   with float32 distances, and the top *k* returned.
6. **Fallback.** If the beam found fewer than *k* admitted nodes and the
   population is at most 8 × the threshold, an exhaustive scan replaces
   the result.

The re-rank is what makes PQ compatible with the recall target. Ranking by
PQ distances alone caps recall@10 well below 0.95 on clustered data, while
re-ranking the PQ beam with exact distances recovers it
([`hnsw-parameter-study.md`](../research/hnsw-parameter-study.md)). That is
why the float32 vectors stay resident.

## 4. Updates and concurrency

Readers never lock. One writer (the ingest handler, serialized by a mutex)
appends a node and links it. Then it publishes the node by advancing the
published size with a release store; readers load it with acquire and never
look past it. Adjacency words are read and written through
`std::atomic_ref`, so a reader always sees either the old or the new list
entry. The entry point (node and level) is packed into one 64-bit word.

The prefilter is rebuilt after an insert and swapped in read-copy-update
style. The old copy is retired, not freed, and freed only after a 60-second
grace period, far longer than any search. The core suites run clean under
ThreadSanitizer.

## 5. Distance kernels

`simd_distance.h` implements squared L2, inner product, squared norm and
ADC summation as fixed 16-lane loops that GCC and Clang auto-vectorize.
`engine/cmake/DetectSIMD.cmake` selects the target (`-march=znver5`,
`x86-64-v4`, `x86-64-v3`, or `-mcpu=native` on arm64). Double-precision
references check every kernel (`simd_parity_test.cc`). No SIMD library is
needed.

## 6. Memory per vector

| Structure | Bytes |
|---|---|
| Float32 row and norm | 516 |
| Layer-0 record | 160 |
| Upper layers (expected) | ~5 |
| Property columns | 96 |
| Prefilter entry | 8 |
| **Total** | **~785** |

That is 7.9 GB for 10M vectors, about 2.6 GB per shard at three shards.

## 7. Correction to the design document

The design document (§11) sizes the corpus as "PQ codes at 8–16 bytes per
128-dimensional embedding keep the 10M-vector corpus in the low hundreds of
megabytes per shard region." The codes alone do (160 MB at 16 bytes), but
an index that holds only codes cannot meet recall@10 ≥ 0.95. The
implemented design keeps PQ for traversal and the float32 vectors for the
exact re-rank, at about 785 bytes per vector in total. That is still far
inside a node's RAM, so the residency argument of
[02](02-why-not-managed.md) is unchanged.
