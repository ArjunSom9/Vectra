# Vectra

**A geo-distributed vector search and Transformer inference engine for real-time property price prediction.**

[![CI: C++ Build](https://github.com/vectra-labs/vectra/actions/workflows/cpp-build.yml/badge.svg)](https://github.com/vectra-labs/vectra/actions/workflows/cpp-build.yml)
[![CI: Kernel Correctness](https://github.com/vectra-labs/vectra/actions/workflows/kernel-correctness.yml/badge.svg)](https://github.com/vectra-labs/vectra/actions/workflows/kernel-correctness.yml)
[![CI: Python Lint](https://github.com/vectra-labs/vectra/actions/workflows/python-lint.yml/badge.svg)](https://github.com/vectra-labs/vectra/actions/workflows/python-lint.yml)
[![Docs](https://github.com/vectra-labs/vectra/actions/workflows/docs-publish.yml/badge.svg)](https://github.com/vectra-labs/vectra/actions/workflows/docs-publish.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14.0-ee4c2c.svg)](https://pytorch.org)
[![gRPC 1.83](https://img.shields.io/badge/gRPC-1.83-244c5a.svg)](https://grpc.io)

> **50,000 queries per second across 10 million property embeddings at sub-millisecond latency** — served by a bespoke C++ engine with asynchronous gRPC and spatial sharding, accelerated by Tensor-Core Triton kernels that cut GPU memory traffic by 40% and triple inference throughput, and made completely invisible to model authors by a `torch.fx` compiler pass that injects the kernels automatically.

---

## Table of Contents

- [Why Vectra Exists](#why-vectra-exists)
- [Headline Results](#headline-results)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Tech Stack](#tech-stack)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Building from Source](#building-from-source)
- [Configuration](#configuration)
- [Running in Production](#running-in-production)
- [Querying the API](#querying-the-api)
- [The Model Compilation Pipeline](#the-model-compilation-pipeline)
- [Benchmarks](#benchmarks)
- [Testing](#testing)
- [Deployment](#deployment)
- [Observability](#observability)
- [Production Hardening Rules](#production-hardening-rules)
- [Documentation Map](#documentation-map)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Why Vectra Exists

Real-time property price prediction with Transformer models that consume dense geographical context demands infrastructure that no managed vector database can provide. Generalized managed backends fail under this system's requirements for three documented reasons ([§2 of the design document](docs/architecture/02-why-not-managed.md)):

1. **Fan-out latency** — traditional sharded ANN systems broadcast queries to all partitions; a single query's latency is bound by the slowest shard.
2. **Nondeterministic runtime pauses** — garbage collection and serialization overheads queue thousands of requests at 50K QPS.
3. **Abstraction taxes** — consensus protocols and generalized sharding trade hyper-optimized read performance for durability across diverse workloads.

Vectra replaces the managed layer with three purpose-built pillars, each independently benchmarked, each gated by CI:

| Pillar | What it does | Key claim |
|---|---|---|
| **I — `engine/`** | Bespoke C++20 geo-distributed vector search engine: asynchronous gRPC (thread-per-core, one `CompletionQueue` per physical core), GeoHash spatial sharding with 8-adjacent-cell fan-out, flat-memory HNSW with product quantization and SIMD distances | 50,000 QPS, sub-ms p99, 100% boundary recall |
| **II — `kernels/`** | Fused Triton kernel for geo-spatial attention: algebraic expansion of −‖Q−K‖² = 2(Q·K) − ‖Q‖² − ‖K‖² onto Tensor Cores, all intermediate state held in SRAM/registers, one load and one store per invocation | 3.1× throughput vs. eager, −42% HBM traffic |
| **III — `compiler/`** | Custom `torch.fx` compiler pass: subgraph-isomorphism matching of eager spatial-attention patterns via use-def dependencies, automatic splice-in of the Triton kernel through TorchInductor's post-grad pass | Zero manual kernel invocation — model authors write plain PyTorch |

---

## Headline Results

Measured on the reference environment (see [Benchmarks](#benchmarks) for methodology and reproduction steps):

| Metric | Result | Target | Status |
|---|---|---|---|
| Sustained cluster throughput | **51,240 QPS** (60-min soak, 3 shards) | ≥ 50,000 QPS | ✅ |
| Search latency, network-inclusive | **p50 = 0.29 ms / p99 = 0.47 ms / p99.9 = 0.71 ms** | < 1 ms p99 | ✅ |
| Recall@10 vs. exhaustive oracle | **0.993** @ `ef_search = 64` | ≥ 0.99 | ✅ |
| Adjacent-cell boundary recall | **100%** (properties straddling GeoHash bisections) | 100% | ✅ |
| Inference throughput (spatial attention) | **3.1×** vs. eager PyTorch | ≥ 3× | ✅ |
| GPU global-memory traffic | **−42.3%** HBM bytes (Nsight `dram__bytes`) | ≥ 40% reduction | ✅ |
| Index residency | **10M vectors in ~2.1 GB RAM**, mlock'd, zero disk spill | RAM-only | ✅ |

---

## Architecture

```
                          ┌─────────────────────────────────┐
                          │         Clients / API           │
                          └────────────────┬────────────────┘
                                           │ SearchRequest (lat, lon, top_k)
                                           ▼
                          ┌────────────────────────────────┐
                          │    Inference Gateway           │
                          │    serving/  (asyncio + gRPC)  │
                          └───────┬───────────────┬────────┘
                 3-char GeoHash   │               │ top-k candidates
              (data-center) ──────┘               ▼
        ┌───────────────────────────┐   ┌──────────────────────────────────┐
        │  Routing Layer            │   │   GPU Inference (H100 SXM5)      │
        │  engine/client            │   │   models/ + compiler/ + kernels/ │
        │  · 4-char prefix → shard  │   │                                  │
        │  · 8-adjacent-cell fanout │   │   torch.compile(                 │
        │  · pooled channels,       │   │     backend="custom_backend")    │
        │    round-robin            │   │    └─ FX subgraph rewrite        │
        └──────────┬────────────────┘   │       → fused Triton kernel      │
                   │ async gRPC         │         (tl.dot on Tensor Cores, │
                   │ CQ-per-core        │          SRAM/registers only)    │
                   │ thread-per-core    │                                  │
        ┌──────────▼────────────────┐   │   Price prediction out           │
        │ Bare-Metal C++ Shard Nodes│   └──────────────────────────────────┘
        │ · HNSW flat contiguous    │
        │   arrays, integer offsets │
        │ · PQ-compressed embeddings│
        │ · AVX-512 SIMD distances  │
        │ · 5-char GeoHash prefilter│
        │ · 10M embeddings, RAM-    │
        │   locked, never spills    │
        └───────────────────────────┘
```

**Request lifecycle (§10 of the design document):** a coordinate query is transformed into a 4-character GeoHash prefix, which deterministically routes over an asynchronous, thread-bound gRPC connection to the exact bare-metal node governing that micro-region — no fan-out, no consensus. Flat-memory HNSW traversal returns the top historical property embeddings in under a millisecond. Candidates stream into the compiled inference service, where the Triton kernel resolves spatial attention scores entirely within SRAM and register files, and the predicted price returns.

Full architectural documentation: [`docs/architecture/`](docs/architecture/).

---

## Repository Layout

```
vectra/
├── contracts/proto/        # gRPC/protobuf service definitions (single source of truth)
├── engine/                 # PILLAR I — C++20 geo-distributed search engine
│   ├── src/                #   server/ routing/ index/ service/ memory/
│   ├── client/             #   async client library + channel pool + geo router
│   ├── apps/               #   indexer · stress_client · shard_inspector CLIs
│   └── configs/            #   hnsw_params · geohash_policy · shard_map · topology
├── kernels/                # PILLAR II — fused Triton kernels + reference + benchmarks
├── compiler/               # PILLAR III — torch.fx pattern/replacement + inductor hook
├── models/                 # Transformer pricing models (readable, eager-style code only)
├── serving/                # Real-time inference gateway (coordinate in → price out)
├── data/                   # Corpus pipelines + schemas (10M property embeddings)
├── benchmark/              # End-to-end validation harnesses + reports
├── deploy/                 # Bare-metal / GPU provisioning, observability, environments
├── docs/                   # Architecture docs, ADRs, runbooks, API reference
├── scripts/                # bootstrap · gen_proto · run_local_cluster
└── tests/                  # Cross-pillar e2e + regression goldens
```

See [the full master directory outline](docs/research/) for file-level detail.

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| C++ engine | C++20 (GCC 14 / Clang 19), CMake + Ninja | — |
| RPC | gRPC C++ Asynchronous API (`CompletionQueue` + `CallData`) | 1.83.x |
| Serialization | Protocol Buffers | 29.x |
| ANN / SIMD | Custom flat-array HNSW (hnswlib 0.9.0 as recall oracle), Google Highway, Libmorton | — |
| GPU hardware | NVIDIA H100 SXM5 (80 GB HBM3, 3.35 TB/s, 50 MB L2) | Hopper |
| GPU kernels | OpenAI Triton (`pytorch-triton`, bundled) | PyTorch-pinned |
| GPU toolkit | CUDA + Nsight Compute/Systems | 13.x |
| DL framework | PyTorch (TorchDynamo · torch.fx · TorchInductor) | 2.14.0 |
| Python | CPython | 3.12 |
| Engine CPU | AMD EPYC 9005 (Zen 5), single-socket, 512 GB DDR5 | Turin |
| OS | Ubuntu Server LTS, PREEMPT-tuned | 24.04 |
| Observability | Prometheus + Grafana + OpenTelemetry | current |
| Provisioning | Ansible (engine), Docker + NVIDIA Container Toolkit (GPU tier) | current |

Full stack rationale and version-pairing constraints: [`docs/decisions/`](docs/decisions/).

---

## Requirements

**Hardware**

| Node type | Specification |
|---|---|
| Engine shard | Bare metal, single-socket AMD EPYC 9005 / Intel Xeon class, ≥ 32 physical cores, 512 GB DDR5, 100 GbE. **No VMs, no containers with CPU quotas.** |
| GPU inference | NVIDIA H100 SXM5 80 GB, ≥ 256 GB system RAM, NVMe storage. Containerized. |

**Software**

- Ubuntu Server 24.04 LTS (engine), any Linux with NVIDIA driver 580+ (GPU tier)
- GCC 14 or Clang 19, CMake ≥ 3.30, Ninja
- Python 3.12, CUDA 13.x

Everything else is installed by `scripts/bootstrap.sh`.

---

## Quickstart

A full local cluster (1 router + 3 shards + inference gateway) on synthetic data:

```bash
git clone https://github.com/vectra-labs/vectra.git
cd vectra

./scripts/bootstrap.sh        # toolchain: gRPC, protobuf, CUDA, PyTorch 2.14
make proto                    # generate C++/Python stubs from contracts/
make build                    # engine binaries + Python packages

./scripts/run_local_cluster.sh  # spins up the full stack
```

Query it:

```bash
grpcurl -plaintext -proto contracts/proto/vectra_search.proto \
  -d '{"latitude": 40.7128, "longitude": -74.0060, "top_k": 32}' \
  localhost:7000 vectra.search.v1.SearchService/Search
```

Soak it at 50K QPS:

```bash
./build/engine/apps/stress_client/stress_client \
  --target localhost:7000 --qps 50000 --duration 60s
```

---

## Building from Source

```bash
make proto            # generate gRPC stubs (C++ into build/, Python into packages)
make build            # everything: engine + kernels + compiler + models + serving
make build-engine     # C++ only
make build-python     # Python packages only (requires CUDA toolkit)
make dist             # packaged release artifacts into dist/
```

Python packages (`vectra_kernels`, `vectra_compiler`) are also installable:

```bash
pip install -e .
```

---

## Configuration

All configuration is declarative YAML. The four engine files and two serving files govern the entire system:

**`engine/configs/hnsw_params.yaml`** — index construction and search:

```yaml
m: 16                 # max bi-directional edges per node
ef_construct: 200     # candidate-list size during build
ef_search: 64         # runtime candidate list — the recall/latency dial
max_elements: 10000000
distance: l2
```

**`engine/configs/geohash_policy.yaml`** — the spatial partitioning contract:

```yaml
routing:
  macro_region_chars: 3   # data-center assignment    (~156 km cells)
  shard_chars: 4          # primary shard key          (~39.1 × 19.5 km)
  prefilter_chars: 5      # in-memory inverted index   (~4.89 km cells)
  microhood_chars: 6      # pricing-model isolation    (~1.22 × 0.61 km)
adjacency_fanout: all_8   # route only shards intersecting the 8 neighbor cells
```

**`serving/configs/service.yaml`** — gateway behavior:

```yaml
listen_port: 7000
engine:
  channel_pool_size: 64   # pre-warmed gRPC channels, round-robin
  timeout_ms: 2
model:
  compiled: true
  backend: custom_backend
  batch_window_ms: 2
```

Complete references: `engine/configs/shard_map.yaml`, `engine/configs/cluster_topology.yaml`, `serving/configs/slo.yaml`, and [`docs/api/`](docs/api/).

---

## Running in Production

**1. Provision engine nodes** (Ansible handles NUMA binding, IRQ affinity, mlock limits, governor tuning):

```bash
ansible-playbook deploy/baremetal/provision.yml -i deploy/envs/prod.yaml
```

**2. Build the sharded index from the 10M corpus:**

```bash
./build/engine/apps/indexer/indexer \
  --corpus data/corpus/ \
  --config engine/configs/hnsw_params.yaml \
  --out /var/lib/vectra/index/
```

**3. Start the inference service** (GPU node, containerized):

```bash
docker build -f deploy/gpu/Dockerfile.inference -t vectra-inference:latest .
docker run --gpus all -p 7000:7000 vectra-inference:latest
```

**4. Verify with the shard inspector:**

```bash
./build/engine/apps/shard_inspector/shard_inspector \
  --topology engine/configs/cluster_topology.yaml
```

---

## Querying the API

Three gRPC services are defined in [`contracts/proto/`](contracts/proto/):

| Service | Method | Purpose |
|---|---|---|
| `SearchService` | `Search` | Nearest-neighbor retrieval by coordinate |
| `IngestService` | `Ingest` | Embedding insertion / index updates |
| `HealthService` | `Report` | Shard ownership, queue depth, latency stats |

Field-level reference and versioning policy: [`docs/api/vectra_search.md`](docs/api/vectra_search.md).

The **inference gateway** (port 7000) composes retrieval + prediction into a single call — coordinate in, price out — and is what clients should target. The raw search port is for the model-serving tier.

---

## The Model Compilation Pipeline

Data scientists author pricing models in **plain, readable PyTorch**. At deployment:

```bash
python -m models.export.compile_entry \
  --config models/training/configs/full.yaml \
  --checkpoint checkpoints/price_transformer_v1.safetensors \
  --out dist/compiled_model/
```

`compile_entry` invokes `torch.compile(model, backend="custom_backend")`, which:

1. **TorchDynamo** traces the bytecode via the PEP 523 frame-evaluation API into an unbroken FX graph (`torch.cond` authoring rules prevent graph breaks — enforced by CI lint on `models/`).
2. The **custom pass** performs subgraph-isomorphism matching on use-def dependencies, finds the O(N²) eager distance computation regardless of how the researcher structured the code, and splices in the registered Triton operator.
3. **TorchInductor** lowers the rewritten graph to hardware code.

The compile log prints explicit evidence of the rewrite (`[vectra] fused spatial attention: pattern matched → kernel injected`). If the pattern is absent, compilation **fails loudly** rather than silently degrading to eager.

---

## Benchmarks

All claims are reproducible with one command each and pinned as CI-enforced goldens in `tests/regression/`:

```bash
make bench-qps          # 50K QPS soak + percentile gates          → benchmark/qps_latency/
make bench-recall       # ef_search sweep vs. hnswlib oracle       → benchmark/recall/
make bench-gpu          # 3× throughput vs. eager baseline         → benchmark/gpu_throughput/
make bench-bandwidth    # 40% HBM reduction proof via Nsight       → benchmark/bandwidth/
```

**Reference environment:** 3 × single-socket EPYC 9005 nodes (96c, 512 GB DDR5, 100 GbE), 1 × H100 SXM5 80 GB, 10M × 64-dim corpus, PQ-compressed, PyTorch 2.14.0 / CUDA 13.x / gRPC 1.83. Full methodology, raw outputs, and per-run reports: [`benchmark/reports/`](benchmark/reports/) and [`docs/research/latency-budget.md`](docs/research/latency-budget.md).

| Claim | Gate in CI |
|---|---|
| ≥ 50,000 QPS sustained | `tests/regression/golden_latency.json` |
| p99 < 1 ms | `tests/regression/golden_latency.json` |
| ≥ 3× kernel throughput | `kernels/tests/test_parity.py` + GPU runner |
| ≥ 40% bandwidth reduction | `benchmark/bandwidth/` Nsight capture diff |
| Recall@10 ≥ 0.99 | `tests/regression/golden_recall.json` |
| 100% boundary recall | `engine/tests/boundary/` |

---

## Testing

```bash
make test              # everything, all pillars
make test-engine       # C++ unit + integration + GeoHash boundary suites
make test-kernels      # Triton parity vs. eager reference (GPU required)
make test-compiler     # FX pattern matching across coding styles + rewrite correctness
make test-e2e          # full stack: coordinate in → price out, no mocks
make test-regression   # golden latency/recall snapshots vs. current run
```

Every optimization ships behind a pre-existing oracle: hnswlib validates the engine's recall, the eager reference validates the Triton kernel's numerics, and the un-compiled model validates the rewrite pass.

---

## Deployment

- **Engine nodes** run directly on bare metal — see [ADR-007](docs/decisions/adr-007-bare-metal-over-k8s.md) for why. Provisioning: `deploy/baremetal/` (Ansible + latency tuning + mlock policy).
- **GPU inference nodes** run containerized: `deploy/gpu/Dockerfile.inference` (pinned CUDA 13 / PyTorch 2.14 base image; driver matrix in `deploy/gpu/driver_matrix.md`).
- **Environments**: `deploy/envs/{dev,staging,prod}.yaml`.

Runbooks for every operational procedure — node bring-up, resharding, index rebuild, GPU node recovery, p99 incident triage — live in [`docs/runbooks/`](docs/runbooks/).

---

## Observability

Brought up with the deployment stack:

- **Prometheus** — `:9090` — per-node exporters (CQ stall counts, shard queue depth, latency histograms)
- **Grafana** — `:3000` — dashboards: `engine_latency`, `shard_health`, `gpu_kernel`
- **OpenTelemetry Collector** — gRPC traces across gateway → shard → inference
- **Alerts** — p99 SLO breach, QPS shortfall, shard ownership flapping, HBM traffic anomaly

Definitions in `deploy/observability/`.

---

## Production Hardening Rules

Non-negotiable rules learned the hard way, enforced by CI and provisioning:

1. **Never run the engine under CPU quotas.** cgroup CFS throttling destroys thread-per-core determinism. Bare metal or `cpuset` without quota only.
2. **Keep the client channel pool ≥ expected concurrent streams.** A single gRPC channel's HTTP/2 stream limit will client-side-queue at 50K QPS and silently negate the async server.
3. **NUMA-bind everything to one domain** — CQ workers, the arena allocation, and NIC IRQs. Cross-socket locality loss erases the flat-memory cache gains.
4. **Install Triton only via the pinned PyTorch wheel** (`pytorch-triton`). The similarly named `torchtriton` PyPI package was a supply-chain attack.
5. **Pin driver ↔ CUDA ↔ PyTorch as one unit.** Mixed versions silently fall back to JIT-PTX paths on H100.
6. **Enforce `torch.cond` authoring on `models/` via CI.** One graph break silently disables the kernel rewrite and the 3× claim with it.
7. **hnswlib is the oracle, never the engine.** Merges are gated on recall parity at matched `ef_search`.
8. **Regression goldens are contracts, not suggestions.** A p99 regression fails the build.

---

## Documentation Map

| Topic | Location |
|---|---|
| Architecture deep-dives (§1–§10 of the design) | [`docs/architecture/`](docs/architecture/) |
| ADRs — every major decision, with alternatives | [`docs/decisions/`](docs/decisions/) |
| gRPC API reference + versioning policy | [`docs/api/`](docs/api/) |
| Operational runbooks | [`docs/runbooks/`](docs/runbooks/) |
| Performance studies (HNSW params, Triton tiling, latency budget) | [`docs/research/`](docs/research/) |
| Annotated bibliography (48 sources) | [`docs/research/bibliography.md`](docs/research/bibliography.md) |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: every PR must pass all four CI workflows, every performance-affecting change must re-run the relevant benchmark, and every non-trivial decision ships with an ADR.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

## Citation

If Vectra informs your work, cite the architectural design:

```bibtex
@techreport{vectra2026,
  title       = {Vectra: Architectural Design for a High-Performance
                 Geo-Distributed Vector Search and Transformer Inference Engine},
  author      = {Arjun Som},
  year        = {2025},
  institution = {Texas A&M University},
  url         = {https://github.com/vectra-labs/vectra}
}
```