# HNSW and PQ parameter study

**Question.** At the configured index parameters (M = 16, M0 = 32,
ef_construction = 200, 16 PQ subspaces × 8 bits), which `ef_search` and
`rerank_depth` meet recall@10 ≥ 0.95 (objective O1), and what does the
float32 re-rank buy?

**Tool.** `build/engine-core/index_study` (`engine/tests/tools/index_study.cc`).
It builds a real `ShardIndex` and runs unfiltered searches for 500 corpus
vectors, excluding each query's own row. It reports recall@10 against brute
force, the mean single-threaded time per query, and the mean number of
distance evaluations. A row `pq_exhaustive` gives the recall of ranking
every vector by its PQ distance: the ceiling of any search that never looks
at float32 vectors.

**Machine.** Apple M1 Pro (10 cores), macOS, Apple clang 21, Release build,
`-mcpu=native`, 8 build threads. These are development-machine numbers for
comparing settings. Production latency is measured on the reference
hardware by `make bench-qps`.

## Result 1: the property-embedding corpus

200,000 embeddings from `data.embedding_model.PropertyEmbedder` (the
synthetic corpus Vectra serves: 128-d, attribute and multi-scale
positional features):

```
build/engine-core/index_study data/corpus/study/embeddings.npy 500
```

| rerank_depth \ ef_search | 16 | 32 | 64 | **128** | 256 | 512 |
|---|---|---|---|---|---|---|
| 11 (PQ ranking only) | 0.660 | 0.655 | 0.651 | 0.650 | 0.649 | 0.649 |
| 64 | 0.792 | 0.935 | 0.987 | 0.985 | 0.984 | 0.984 |
| **128** | 0.792 | 0.935 | 0.987 | **0.998** | 0.997 | 0.997 |
| 256 | 0.792 | 0.935 | 0.987 | 0.998 | 1.000 | 1.000 |
| µs per query (rerank 128) | 30 | 40 | 60 | **104** | 171 | 313 |

`pq_exhaustive`: 0.613.

* **PQ alone cannot meet the target.** Ranking by PQ distances caps recall
  at about 0.65, the exhaustive-PQ ceiling plus the graph's own
  re-ordering. Hence ADR-004: the float32 vectors stay resident for the
  re-rank.
* **Recall is limited by min(ef_search, rerank_depth).** The beam holds
  `ef_search` candidates and the re-rank re-scores the best `rerank_depth`
  of them. Raising either alone does nothing once it exceeds the other.
* **The configured point (128, 128) gives 0.998**, well above 0.95, at
  about 104 µs single-threaded over 200,000 vectors. (64, 64) gives 0.987
  at about 60 µs and is the fallback if the latency budget tightens.

## Result 2: an adversarial Gaussian mixture

50,000 vectors from 64 isotropic Gaussian clusters in 128 dimensions
(`index_study 50000 128 64 500`). Distances inside a cluster concentrate,
so the true neighbors are barely closer than the rest of the cluster. This
is a worst case, not a model of property data.

| rerank_depth \ ef_search | 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| 11 (PQ ranking only) | 0.128 | 0.123 | 0.122 | 0.122 |
| 128 | 0.436 | 0.610 | 0.602 | 0.601 |
| 256 | 0.434 | 0.609 | 0.806 | 0.805 |

`pq_exhaustive`: 0.118. The same two effects appear, much more strongly:
PQ ranking is close to random within a cluster, and recall tracks the size
of the exactly re-ranked pool. Corpora with this geometry would need a
larger `rerank_depth` or more PQ subspaces. Run `make bench-recall` on any
new corpus before trusting the defaults.

## Filtered searches

Production searches are radius-filtered. Small populations (≤ 2,048 rows
in the covering prefilter cells) are scanned exhaustively and are exact.
Larger ones run the filtered beam with the same `ef_search` and re-rank.
The unit suites check recall@10 ≥ 0.95 for radius searches and for
sale-date-bounded searches (`shard_index_test.cc`), and ≥ 0.99 across
adjacent cells (`adjacent_recall_test.cc`).

## Reproducing

```bash
make build-engine-core
python -m data.pipelines.generate_embeddings --out_dir=data/corpus/study \
  --num_properties=200000 --num_queries=10
build/engine-core/index_study data/corpus/study/embeddings.npy 500
```

The production decision is re-made on the production corpus with
`make bench-recall`, which adds the radius filter, the network path and the
hnswlib oracle.
