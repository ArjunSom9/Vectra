# ADR-002: GeoHash precision levels and bounded fan-out

* **Status:** Accepted (January 2025; fan-out clause amended March 2025)
* **Deciders:** Workstream A
* **Specification:** [04 — GeoHash sharding](../architecture/04-geohash-sharding.md)

## Context

Comparables are local. The partitioning must put nearby properties on the
same node, make ownership a constant-time lookup, and bound the number of
shards a radius query reaches. GeoHash prefixes nest exactly: every cell of
precision *p* contains 32 cells of precision *p*+1 [17] [18].

## Decision

| Precision | Role |
|---|---|
| 3 | macro-region (data center) |
| 4 | shard key; dense routing table of 32⁴ entries |
| 5 | in-shard prefilter |
| 6 | micro-neighborhood feature in the model |

A query reaches the owners of the precision-4 cells its disk intersects,
bounded to the 3×3 block around its home cell. The default radius is 5 km
and the maximum 15 km. A disk that would reach further is rejected with
`INVALID_ARGUMENT`.

**Amendment (March 2025).** The original text said each query goes to
"exactly one" shard. A radius query near a shard boundary must reach every
owner of the cells it touches, or it silently loses comparables. The fan-out
is bounded, not eliminated. The boundary suites
(`engine/tests/boundary/`) prove that exact mode loses nothing.

## Consequences

* Most queries reach one shard; queries near boundaries reach two or three,
  never the fleet.
* At 15 km the 3×3 bound holds up to about 67° N, where precision-4 cells
  become narrower than the radius. Larger radii need a coarser shard key.
* Hot metros are balanced by assigning individual precision-4 cells
  (`split_shards --rebalance`), not by changing the key.
