# `vectra.pricing.v1.PricingService`

Source: [`contracts/proto/vectra/pricing.proto`](../../contracts/proto/vectra/pricing.proto).
Served by the inference gateway (`serving/service.py`, port 7000). This is
the API external clients use.

## `PredictPrice(PredictPriceRequest) → PredictPriceResponse`

| Request field | Meaning |
|---|---|
| `location` | The subject property, degrees (required) |
| `attributes` | `living_area_sqft` (required, > 0), `lot_area_sqft`, `bedrooms`, `bathrooms`, `year_built`; the sale fields are ignored |
| `radius_km` | Comparable radius; 0 selects the policy default (5 km) |
| `num_candidates` | Comparables to retrieve; 0 selects the model's K (64), and it may not exceed it |
| `valuation_epoch_day` | Days since 1970-01-01; only earlier sales are comparables; 0 means today (UTC) |

| Response field | Meaning |
|---|---|
| `predicted_price_usd`, `log_price` | The prediction (the model regresses the natural log) |
| `candidates_used` | Comparables the model attended to after radius trimming |
| `comparable_ids` | The retrieved comparables, nearest first in embedding space |
| `latency` (`StageLatency`) | `retrieve_us`, `batch_wait_us`, `inference_us`, `total_us` |

## Errors

| Code | When |
|---|---|
| `INVALID_ARGUMENT` | Missing location or attributes; `num_candidates` above K; negative valuation day; a radius beyond the routing bound ([04 §4](../architecture/04-geohash-sharding.md)) |
| `NOT_FOUND` | No sale before the valuation day within the radius |
| `UNAVAILABLE`, `DEADLINE_EXCEEDED` | A shard the query must reach failed; the request is never priced from a partial comparable set |

## Example

```bash
grpcurl -plaintext -import-path contracts/proto -proto vectra/pricing.proto \
  -d '{"location": {"latitude": 37.7749, "longitude": -122.4194},
       "attributes": {"living_area_sqft": 1850, "lot_area_sqft": 4200,
                      "bedrooms": 3, "bathrooms": 2, "year_built": 1962}}' \
  localhost:7000 vectra.pricing.v1.PricingService/PredictPrice
```
