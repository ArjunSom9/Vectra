# Runbook: rebuild a shard index

**When:** corpus refresh, capacity change (`max_elements`), index parameter
change (`hnsw_params.textproto`), or a corrupt index file.
**Owner:** engine on-call.

The index is a deterministic function of the corpus and the configuration:
seeded PQ training, pre-drawn HNSW levels. A rebuild is always safe.

1. Validate the corpus against its schemas and checksums:
   ```bash
   python -m data.pipelines.validate_corpus --corpus_dir=<corpus>
   ```
2. Split it by the current shard map:
   ```bash
   python -m data.pipelines.split_shards --corpus_dir=<corpus> --shards_dir=<corpus>/shards
   ```
3. Build each shard's index (one process per shard; `--build_threads`
   overrides `hnsw_params.build_threads`):
   ```bash
   indexer --config_dir=engine/configs --shard_dir=<corpus>/shards/shard-<id> \
     --out=shard-<id>.vidx
   ```
4. Check it offline:
   ```bash
   shard_inspector --index=shard-<id>.vidx
   ```
   Every section checksum must verify, and `num_vectors` must equal the
   shard's row count.
5. Gate recall before the index ships: run `make bench-recall` against a
   staging shard serving the new index. Recall@10 at the configured
   `ef_search` must be ≥ 0.95, with boundary recall 1.0 in exact mode.
6. Roll it out one replica at a time ([node-bringup](node-bringup.md),
   steps 3 to 7). Never replace every replica of a shard at once.
