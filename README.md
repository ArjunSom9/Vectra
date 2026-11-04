# Vectra

[![CI: C++ Build](https://github.com/ArjunSom9/Vectra/actions/workflows/cpp-build.yml/badge.svg)](https://github.com/ArjunSom9/Vectra/actions/workflows/cpp-build.yml)
[![CI: Kernel Correctness](https://github.com/ArjunSom9/Vectra/actions/workflows/kernel-correctness.yml/badge.svg)](https://github.com/ArjunSom9/Vectra/actions/workflows/kernel-correctness.yml)
[![CI: Python Lint](https://github.com/ArjunSom9/Vectra/actions/workflows/python-lint.yml/badge.svg)](https://github.com/ArjunSom9/Vectra/actions/workflows/python-lint.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Geo-distributed vector search and Transformer inference for real-time
property pricing.** A request carries a property's location and attributes.
Vectra retrieves its comparable sales from a GeoHash-sharded C++ vector
search engine, scores them with a Transformer whose spatial attention runs
as a fused Triton kernel, and returns a price. The kernel is put into the
model by a compiler pass, not by hand.

The design, objectives and acceptance criteria are in
*Vectra Project Plan and Technical Design.docx* (project window January 6 to
May 30, 2025). The architecture documents in [`docs/architecture/`](docs/architecture/)
describe the implementation and record where it corrects the design.

---

## The three workstreams

| | Directory | What it is |
|---|---|---|
| **A** | [`engine/`](engine/) | C++20 search engine on gRPC's asynchronous API: a `CompletionQueue` per worker core, `CallData` tags, GeoHash sharding with bounded fan-out, flat-memory HNSW with product quantization and an exact re-rank, a sale-date bound in the search |
| **B** | [`kernels/`](kernels/) | Fused RBF-attention kernel in Triton (`‖q−k‖² = ‖q‖² + ‖k‖² − 2q·k` on tensor cores, online softmax), registered as `torch.ops.vectra.rbf_attention` |
| **C** | [`compiler/`](compiler/) | `torch.compile(backend="vectra")`: a pass at Inductor's joint-graph pre-pass that finds every formulation of RBF attention by use-def topology and substitutes the operator |

Around them: the price model ([`models/`](models/)), the inference gateway
([`serving/`](serving/)), the synthetic corpus and data pipelines
([`data/`](data/)), wire and configuration contracts
([`contracts/proto/`](contracts/proto/)), benchmarks
([`benchmark/`](benchmark/)), and deployment ([`deploy/`](deploy/)).

```
client ─ PredictPrice ─▶ gateway ─┬─ Search ─▶ shard 1 ┐  (only the shards whose
                                  └─ Search ─▶ shard 3 ┘   cells the disk touches)
                                  ◀─ merged comparables ─
                         batch ─▶ compiled model (vectra::rbf_attention) ─▶ price
```

## Objectives and status

| Objective | Target | Status |
|---|---|---|
| O1 retrieval | 50,000 QPS, p50 < 1 ms, recall@10 ≥ 0.95, 10M properties | Implemented. Recall at the configured point: **0.998** on a 200k property-embedding corpus ([study](docs/research/hnsw-parameter-study.md)). Throughput and latency: **target**, measured by `make bench-qps` on the reference cluster |
| O1 boundaries | No comparable lost across shard boundaries | **Tested**: exact mode recall 1.0 on the hardest lines (`engine/tests/boundary/`) |
| O2 kernel | ≤ 60% of eager DRAM bytes; ≥ 3× serving throughput at rtol 1e-3 | Implemented. Analytic budget: **1.4%** of eager bytes at the serving shape. Measured gates: **target** (`make bench-bandwidth`, `make bench-gpu` on H100) |
| O3 compiler | Every attention site rewritten, zero graph breaks; ≥ 65% of peak DRAM throughput | **Tested** on CPU: 3/3 sites, 0 breaks, compiled = eager within 1e-6. Bandwidth: **target**, judged at batch 1024 ([06](docs/architecture/06-memory-bottleneck.md)) |

A target becomes a result only when its harness has run on the reference
hardware and the report is recorded in `tests/regression/`. The golden files
contain the acceptance thresholds and no invented baselines.

## Toolchain

Every version was released on or before 2025-05-30, and nothing yanked is
used ([ADR-008](docs/decisions/adr-008-toolchain-pins.md)).

| Component | Version |
|---|---|
| C++ | C++20, GCC 14 (Clang 19 supported), CMake ≥ 3.28 (3.31.6 pinned), Ninja |
| RPC | gRPC 1.71.0 (vendored protobuf 29.0, Abseil 20240722.0); GoogleTest 1.17.0 |
| Python | 3.12; PyTorch 2.7.0 (CUDA 12.6 wheels); Triton 3.3.0; NumPy 2.2.6; grpcio 1.71.0; protobuf 5.29.5 |
| GPU | NVIDIA H100 SXM5, driver R570, Nsight Compute 2025.1.1 |
| Style | Google C++ and Python style guides: clang-format 19.1.7, cpplint 2.0.2, ruff 0.11.12, pylint 3.3.7 with Google's pylintrc |

`requirements-lock.txt` freezes the full Python closure (`uv pip compile
--universal --exclude-newer 2025-05-30T23:59:59Z`), and
`requirements-runtime-lock.txt` is the inference image's subset.

## Quick start

Everything below runs on a laptop without a GPU or a gRPC build. The C++
core, the compiler pass, training, the deployment compile and the gateway
all work on CPU, with the gateway talking to exact in-process shards.

```bash
scripts/bootstrap.sh --python        # .venv from the lock file + Python stubs
source .venv/bin/activate
export PYTHONPATH=.:kernels:compiler:build/gen/python

make build-engine-core test-engine   # C++ core: 72 tests, boundary suites
python -m pytest -q                  # every Python suite (about 250 tests)
```

A model, end to end:

```bash
make corpus CORPUS=data/corpus/dev   # 200k synthetic properties, shards
make train                           # eager training + temporal backtest
make export                          # deployment compile (CUDA if present)
```

The full system (needs gRPC 1.71.0; `scripts/bootstrap.sh` builds it into
`/opt/vectra/grpc-1.71.0`):

```bash
make build-engine                    # shard server, indexer, tools
scripts/run_local_cluster.sh --with-gateway
grpcurl -plaintext -import-path contracts/proto -proto vectra/pricing.proto \
  -d '{"location": {"latitude": 37.7749, "longitude": -122.4194},
       "attributes": {"living_area_sqft": 1850, "lot_area_sqft": 4200,
                      "bedrooms": 3, "bathrooms": 2, "year_built": 1962}}' \
  localhost:7000 vectra.pricing.v1.PricingService/PredictPrice
```

`make help` lists every target.

## Configuration

Engine configuration is protobuf text format, parsed against
[`config.proto`](contracts/proto/vectra/config.proto) by the engine, the
gateway and the data pipelines alike:

| File | Contents |
|---|---|
| [`engine/configs/hnsw_params.textproto`](engine/configs/hnsw_params.textproto) | dimension 128, M 16, M0 32, ef_construction 200, ef_search 128, rerank_depth 128, 16 PQ subspaces, exhaustive-scan threshold 2048 |
| [`engine/configs/geohash_policy.textproto`](engine/configs/geohash_policy.textproto) | precisions 3/4/5/6; default radius 5 km, maximum 15 km |
| [`engine/configs/shard_map.textproto`](engine/configs/shard_map.textproto) | shards, replica endpoints, prefix assignments (longest wins) |
| [`engine/configs/cluster_topology.textproto`](engine/configs/cluster_topology.textproto) | macro-regions; server options (worker cores, threads per queue, pre-registered calls, mlock); client options (64 channels per shard, 5 ms deadline) |
| [`serving/configs/service.yaml`](serving/configs/service.yaml) | gateway: listen address, bundle, device, 2 ms batching window |
| [`serving/configs/slo.yaml`](serving/configs/slo.yaml) | the acceptance thresholds the benchmark gates read |

## APIs

| Service | Served by | Reference |
|---|---|---|
| `vectra.pricing.v1.PricingService` | gateway, `:7000` | [docs/api/vectra_pricing.md](docs/api/vectra_pricing.md) |
| `vectra.search.v1.SearchService` | every shard | [docs/api/vectra_search.md](docs/api/vectra_search.md) |
| `vectra.ingest.v1.IngestService` | every shard | [docs/api/vectra_ingest.md](docs/api/vectra_ingest.md) |
| `vectra.health.v1.HealthService` | every shard | [docs/api/vectra_health.md](docs/api/vectra_health.md) |

## The deployment compile

Model authors write plain PyTorch ([authoring guide](compiler/guides/authoring_for_dynamo.md)).
At deployment:

```bash
python -m models.export.compile_entry \
  --checkpoint_dir=checkpoints/price_transformer_v1 --out=dist/compiled_model
```

This compiles with `backend="vectra"`, requires one graph and one rewritten
site per layer, checks compiled against eager at rtol 1e-3, and writes a
bundle: weights, architecture, and a manifest with checksums, match counts,
numerics and toolchain versions. The log line
`[vectra] fused spatial attention: N site(s) matched` is the evidence. A
model the pass cannot fully accelerate fails to compile instead of serving
the eager path. The gateway recompiles and re-verifies the bundle before
opening its port.

## Testing

```bash
make test               # everything that runs on this host
make test-engine        # C++ unit, integration (with gRPC) and boundary suites
make test-engine-sanitizers   # the core under ASan+UBSan and TSan
make test-kernels       # kernel parity vs a float64 oracle (Triton on GPU or TRITON_INTERPRET=1)
make test-compiler      # authoring styles, look-alikes, exactness, end-to-end compile
make test-models test-serving test-e2e
make lint               # clang-format, cpplint, ruff, pylint
```

Every optimization has an oracle: brute force and hnswlib for the index,
float64 for the kernel, the eager model for the rewrite.

## Benchmarks

| Target | Measures | Gate |
|---|---|---|
| `make bench-qps` | 30-minute open-loop soak at 50,000 QPS (search), and no-op dispatch (A1) | `tests/regression/golden_latency.json` |
| `make bench-recall` | recall@10 vs `ef_search`, with exact ground truth and the hnswlib oracle | `golden_recall.json` |
| `make bench-gpu` | eager vs Inductor vs `vectra` serving throughput, numerics first | ≥ 3.0× |
| `make bench-bandwidth` | Nsight Compute DRAM bytes and throughput, eager vs fused, batches 64 and 1024 | `serving/configs/slo.yaml` |

## Deployment

* **Engine nodes** run on bare metal ([ADR-007](docs/decisions/adr-007-bare-metal-over-k8s.md)),
  provisioned by Ansible with CPU isolation, NUMA binding, NIC IRQ steering
  and unlimited memlock ([`deploy/baremetal/`](deploy/baremetal/README.md)).
* **Inference nodes** run the pinned image
  [`deploy/gpu/Dockerfile.inference`](deploy/gpu/Dockerfile.inference)
  ([driver matrix](deploy/gpu/driver_matrix.md)).
* **Observability**: Prometheus metrics from the gateway
  (`vectra_gateway_*`) and from shard health reports
  (`serving/engine_exporter.py`, `vectra_shard_*`), with alerts and Grafana
  dashboards in [`deploy/observability/`](deploy/observability/).
* **Runbooks**: [`docs/runbooks/`](docs/runbooks/).

## Production rules

1. **Never run the engine under CPU quotas.** CFS throttling breaks
   thread-per-core determinism.
2. **Keep the channel pool at least as large as the concurrent streams.**
   One channel is one HTTP/2 connection; beyond its stream limit, gRPC
   queues inside the client.
3. **One NUMA node per shard process**: workers, index memory and NIC
   interrupts.
4. **Triton only through the pinned PyTorch wheel's dependency** (`triton`
   on PyPI). Never install `torchtriton`, the name used in the December 2022
   supply-chain attack.
5. **Driver, CUDA, PyTorch and Triton move together**, and are re-certified
   with the GPU benchmarks.
6. **No graph breaks in `models/`.** Data-dependent control flow uses
   `torch.cond`; the deployment compile enforces it.

## Documentation

* Architecture: [01 overview](docs/architecture/01-overview.md) ·
  [02 why not managed](docs/architecture/02-why-not-managed.md) ·
  [03 async gRPC](docs/architecture/03-async-grpc.md) ·
  [04 GeoHash sharding](docs/architecture/04-geohash-sharding.md) ·
  [05 shard index](docs/architecture/05-hnsw-flat-memory.md) ·
  [06 memory bottleneck](docs/architecture/06-memory-bottleneck.md) ·
  [07 fused kernel](docs/architecture/07-triton-kernels.md) ·
  [08 capture and deployment](docs/architecture/08-fx-compilation.md) ·
  [09 pattern matching](docs/architecture/09-subgraph-rewrite.md)
* [Decisions](docs/decisions/) · [Research notes](docs/research/) ·
  [Annotated bibliography](docs/research/bibliography.md)
* [Contributing](CONTRIBUTING.md)

## Citation

```bibtex
@techreport{vectra2025,
  title       = {Vectra: A High-Performance Geo-Distributed Vector Search and
                 Transformer Inference Engine},
  author      = {Arjun Som},
  year        = {2025},
  institution = {Texas A\&M University},
  url         = {https://github.com/ArjunSom9/Vectra}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
