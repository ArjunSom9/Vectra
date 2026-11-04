# Runbook: re-sharding

**When:** a shard approaches capacity (`ShardNearlyFull` alert), a region
is hot, or the fleet grows.
**Owner:** engine lead (the change needs review). Re-sharding is a planned
operation, never automatic.

1. Compute a balanced map from the current corpus:
   ```bash
   python -m data.pipelines.split_shards --corpus_dir=<corpus> --rebalance \
     --shards_per_region=<n> --shard_map_out=shard_map.proposed.textproto
   ```
   Within each macro-region, precision-4 cells go to the least-loaded shard,
   heaviest first. The log prints each region's per-shard loads.
2. Review the diff against `engine/configs/shard_map.textproto`: prefixes
   moved, new shard ids, endpoints. Edit the generated endpoints to the real
   hosts.
3. Build the new shards' indexes from the proposed map
   ([index-rebuild](index-rebuild.md)) and provision any new hosts
   ([node-bringup](node-bringup.md)).
4. Cut over in one change: commit the new `shard_map.textproto`, deploy it
   to every shard and gateway, then roll the shards, then the gateways. The
   gateway and the engine read the same file, so they agree on ownership
   after the roll.
5. Verify: `vectra_shard_up` for every replica, retrieval latency, and a
   recall check (`make bench-recall`) against the new layout.
6. Retire the old shards only after the new map has served a full day.
