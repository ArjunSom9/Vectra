# ADR-004: Product-quantization parameters, and an exact re-rank

* **Status:** Accepted (February 2025; re-rank clause added April 2025)
* **Deciders:** Workstream A
* **Evidence:** [`hnsw-parameter-study.md`](../research/hnsw-parameter-study.md)

## Context

Product quantization [22] compresses a 128-dimensional float32 vector (512
bytes) into *m* one-byte codes and turns distance evaluation into *m* table
lookups (asymmetric distance computation). Compression bounds accuracy:
ranking by PQ distance alone cannot exceed the recall of an exhaustive
search over the same codes.

## Decision

* *m* = 16 subspaces of 8 dimensions, 256 centroids each (8 bits): 16 bytes
  per vector, stored inside the layer-0 graph record.
* Codebooks trained per subspace by seeded k-means on up to 65,536 vectors,
  25 iterations.
* Graph traversal uses PQ distances. The best `rerank_depth` = 128
  candidates are re-ranked with exact float32 distances, so the float32
  vectors stay resident.

## Consequences

* The re-rank is what makes recall@10 ≥ 0.95 reachable: the study shows PQ
  ranking alone capped near its exhaustive-PQ bound, with the re-rank
  lifting recall to the exact search's.
* Memory is about 785 bytes per vector, not 16 (the design document's §11
  figure counted codes only). 10M vectors take about 7.9 GB, which still
  fits a node with two orders of magnitude to spare.
* `ef_search` and `rerank_depth` are the recall/latency dials, tuned on the
  recall-latency curve (`make bench-recall`).
