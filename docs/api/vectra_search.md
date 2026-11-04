# `vectra.search.v1.SearchService`

Source: [`contracts/proto/vectra/search.proto`](../../contracts/proto/vectra/search.proto).
Served by every shard (`engine/src/service/search_handler.cc`). Called by
the gateway's router (`serving/retrieve.py`) and the C++ geo router
(`engine/client/geo_router.cc`).

## `Search(SearchRequest) → SearchResponse`

Returns the `top_k` nearest embeddings, by squared L2 distance, among this
shard's properties within `radius_km` of `location`.

| Field | Type | Meaning |
|---|---|---|
| `query_embedding` | repeated float | Must have the shard's dimension (128) |
| `location` | `LatLng` | Disk center, degrees |
| `radius_km` | double | Geodesic radius; 0 selects the policy default (5 km); at most `max_radius_km` (15 km) |
| `top_k` | uint32 | 1 to 1024 |
| `ef_search` | uint32 | Beam width override; 0 uses `IndexParams.ef_search`; at most 4096 |
| `include_embeddings` | bool | Return candidate embeddings (the gateway needs them) |
| `exact` | bool | Exhaustive search over the radius population (oracle mode) |
| `sold_before_epoch_day` | int32 | When positive, only properties whose last sale precedes this day |

`SearchResponse.candidates` are ordered nearest first. Each `Candidate`
carries `property_id`, `squared_l2`, `geo_distance_km`, `location`,
`attributes`, `embedding` (if requested), `squared_norm`, `geohash6` and
`shard_id`. `SearchStats` reports `distance_evaluations`,
`prefilter_population`, `used_exhaustive_scan` and `server_latency_ns`.

A shard answers only for the rows it owns. Searching a disk that crosses a
shard boundary means calling every owning shard and merging by
`squared_l2`. Routers do this ([04](../architecture/04-geohash-sharding.md));
the merge of per-shard top-k lists equals the global top-k.

## Errors

| Code | When |
|---|---|
| `INVALID_ARGUMENT` | Wrong embedding dimension; `top_k` outside 1–1024; `ef_search` > 4096; radius negative, non-finite or above the maximum; invalid coordinate; negative `sold_before_epoch_day` |
| `DEADLINE_EXCEEDED` | The client's deadline (routers use `ClientOptions.timeout_ms`, 5 ms) |
| `UNAVAILABLE` | The shard is down or draining |
