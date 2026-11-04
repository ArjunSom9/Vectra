# 04 — GeoHash Sharding and Routing

| | |
|---|---|
| **Status** | Accepted |
| **Synthesizes** | Design document §5.2 (GeoHash-based spatial sharding and deterministic routing) |
| **Code** | `engine/src/routing/` (C++), `data/geo.py` and `data/routing.py` (Python mirror), `engine/configs/{geohash_policy,shard_map,cluster_topology}.textproto` |
| **Tests** | `engine/tests/unit/geohash*_test.cc`, `routing_table_test.cc`, `fanout_policy_test.cc`, `engine/tests/boundary/`, `data/tests/test_geo.py` |

Comparable sales are local, so the corpus is partitioned by location. Each
shard owns a set of GeoHash cells, and a query is sent only to the owners of
the cells its search disk touches. This document specifies the encoding,
the disk cover, the shard map and the fan-out bound, and corrects one claim
of the design document (§6).

## 1. Encoding

A GeoHash cell of precision *p* has 5*p* bits, interleaving longitude and
latitude bisections, longitude first [17] [18]. Longitude gets ⌈5p/2⌉ bits
and latitude ⌊5p/2⌋. Each bit halves the remaining interval, and the 5-bit
groups print as the base-32 alphabet `0123456789bcdefghjkmnpqrstuvwxyz`.

Both implementations work on the integer form (`GeohashCell{bits,
precision}` in C++, `int` in Python). Neighbors are computed by
de-interleaving, offsetting the row and column, and re-interleaving.
Longitude wraps at the antimeridian; latitude does not wrap, so polar cells
have no neighbor beyond the pole (`GeohashNeighborsTest.PolarCellsHaveNoNeighborsBeyondThePole`).
The two implementations are tested against the reference examples of [17]
and against each other through the shared configuration.

| Precision | Bits (lon/lat) | Cell height | Cell width at 45° | Role |
|---|---|---|---|---|
| 3 | 8 / 7 | 156 km | 110 km | macro-region (data center) |
| 4 | 10 / 10 | 19.5 km | 27.6 km | **shard key** (`GeohashPolicy.shard_chars`) |
| 5 | 13 / 12 | 4.9 km | 3.5 km | in-shard prefilter (`kPrefilterPrecision`) |
| 6 | 15 / 15 | 0.61 km | 0.86 km | micro-neighborhood feature (`same_cell_flags`) |

Cell widths shrink with the cosine of latitude, so every geometric test uses
exact distances, never nominal cell sizes.

## 2. The disk cover

`CoverDisk(center, radius_km, precision, max_rings)` returns the cells of a
precision that intersect the geodesic disk, home cell first. It grows rings
of neighbors outward from the home cell and keeps a cell only if its exact
great-circle distance to the center is at most the radius. That distance
(`DistanceToBoxKm`) handles points inside the box, the latitude edges, and
the meridian arcs. If a ring still intersects the disk when `max_rings` is
reached, the cover is incomplete and the function returns no cover at all,
rather than a silently truncated one.

Two callers use it with different limits:

| Caller | Precision | Ring limit | Purpose |
|---|---|---|---|
| Fan-out policy (gateway and C++ router) | shard key (4) | 1 | Which shards must see the query |
| Shard index prefilter | 5 | 64 | Which rows a search may touch |

## 3. Shard map and routing table

The shard map (`engine/configs/shard_map.textproto`) assigns GeoHash
prefixes of 1 to `shard_chars` characters to shards; the longest prefix
wins. `RoutingTable::Build` expands the assignments into a dense array with
one 32-bit owner per shard-key cell. At precision 4 that is 32⁴ = 1,048,576
slots (4 MB), so a lookup is one array index, identically in C++ and in
NumPy. Conflicting or over-long prefixes are build errors.

The shipped map partitions the continental US into three regions
(`us-west`, `us-central`, `us-east`) by 2-character prefixes, matching the
three macro-regions of `cluster_topology.textproto`. For production
balance, `python -m data.pipelines.split_shards --rebalance
--shards_per_region=N` writes a map whose assignments are individual
shard-key cells. Within each region it assigns cells to the least-loaded
shard, heaviest cells first. `split_shards` then writes one directory of
columns per shard for the indexer.

## 4. The fan-out policy

`FanoutPolicy::Plan(location, radius_km)`:

1. Resolve the radius: 0 selects `default_radius_km` (5 km). Anything
   outside `[0, max_radius_km]` (15 km) is `INVALID_ARGUMENT`.
2. Cover the disk with shard-key cells and a ring limit of 1, so at most the
   3×3 block around the home cell. A disk that reaches further is rejected
   (`kRadiusExceedsBound`, `INVALID_ARGUMENT` on the wire). At 15 km this
   happens only far north, where cells are narrower than the radius.
3. Map the cells to their owners, de-duplicate them, and keep the home
   shard first.

The router sends one `SearchRequest` to each owner and merges the results by
squared L2 distance. Each shard filters its own rows by the exact radius, so
the merge of the per-shard top-k lists equals the global top-k. The fan-out
is all-or-nothing: a shard that fails fails the request, because a priced
request must not be missing the comparables on one side of a boundary.

Because shard boundaries follow prefix boundaries, most queries touch cells
of a single shard. Only queries near a boundary between differently owned
cells reach two or three shards.

## 5. Correctness at boundaries

The boundary suites prove the claim that matters: no comparable is lost
because it lives across a boundary.

* `bisecting_lines_test.cc` places queries on the hardest lines (the
  equator, the prime meridian, the antimeridian, and shard-key cell edges)
  and checks that exact mode returns every in-radius neighbor on both sides
  (recall 1.0).
* `adjacent_recall_test.cc` checks approximate recall@10 ≥ 0.99 across
  adjacent cells.
* `serving/tests/test_gateway_flow.py` splits San Francisco Bay between two
  shards and checks that the gateway's comparables equal the exact top 16
  over the whole corpus.

## 6. Correction to the design document

The design document (§3.1, §4) describes a routing layer that "selects
exactly one retrieval node" per query. That is true only for queries whose
search disk lies inside cells owned by one shard. A radius search near a
shard boundary must reach every shard whose cells intersect the disk, or it
silently loses comparables. The implemented policy bounds the fan-out (the
owners of at most nine cells) instead of eliminating it, which keeps the
tail-latency argument of [02](02-why-not-managed.md) intact: the fan-out
factor is a small constant, independent of cluster size.

## 7. Configuration

| File | Fields |
|---|---|
| `geohash_policy.textproto` | `macro_region_chars: 3`, `shard_chars: 4`, `prefilter_chars: 5`, `microhood_chars: 6`, `default_radius_km: 5`, `max_radius_km: 15` |
| `shard_map.textproto` | `shards { id, macro_region, endpoints }`, `assignments { prefix, shard_id }` |
| `cluster_topology.textproto` | `regions { name, prefixes }`, `server`, `client` |

All three are parsed by the C++ engine (`engine/src/config/engine_config.cc`)
and the Python mirror (`data/routing.py`) from the schema in
`contracts/proto/vectra/config.proto`.
