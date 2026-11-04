# `vectra.ingest.v1.IngestService`

Source: [`contracts/proto/vectra/ingest.proto`](../../contracts/proto/vectra/ingest.proto).
Served by every shard (`engine/src/service/ingest_handler.cc`).

## `Ingest(IngestRequest) → IngestResponse`

Appends `records` (`PropertyRecord`: `property_id`, `location`,
`attributes`, `embedding`) to this shard's index. Ingest is single-writer
per shard (a mutex serializes calls), while searches continue lock-free and
see each record once it is published ([05 §4](../architecture/05-hnsw-flat-memory.md)).

`IngestResponse.accepted` is the number of records appended;
`index_size` is the shard's vector count afterwards.

## Errors

| Code | When |
|---|---|
| `INVALID_ARGUMENT` | A record has the wrong embedding dimension or an invalid location |
| `FAILED_PRECONDITION` | A record's location lies in a cell this shard does not own. Send it to the owner from the shard map |
| `RESOURCE_EXHAUSTED` | The shard has no reserved capacity left (`IndexParams.max_elements`); rebuild with more ([index-rebuild](../runbooks/index-rebuild.md)) |

Bulk loads do not use this RPC: they build indexes offline with the
indexer from `split_shards` output.
