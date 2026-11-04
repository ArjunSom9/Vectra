# Contributing to Vectra

Thank you for investing time in Vectra, a geo-distributed vector search and
Transformer inference engine designed to serve 50,000 queries per second
across 10 million property embeddings at sub-millisecond p50 latency.

This document is the contract between you and the repository. Each rule
says how it is enforced: by a CI workflow (**CI**), by a test (**test**), or
by review (**review**).

---

## The four house rules

1. **Every claim has a harness.** A number in a PR, a document or a golden
   file comes with the command that measures it. Targets stay labelled as
   targets until a recorded run on the reference hardware says otherwise.
   (review; `tests/regression/compare_snapshots.py`)
2. **Every optimization ships behind an oracle.** Brute force and hnswlib
   validate the engine's recall; a float64 reference validates the kernel;
   the eager model validates the compiler rewrite. Never land an
   optimization whose oracle does not already exist. (test)
3. **Documentation is written at the moment of decision.** ADRs land in the
   same PR as the decision they record. (review)
4. **Paired versions move as one unit.** PyTorch, its pinned `triton` and the
   CUDA wheels; gRPC and its vendored protobuf; `.editorconfig` and the
   formatter configurations. Partial bumps are rejected
   ([ADR-008](docs/decisions/adr-008-toolchain-pins.md)). (review)

---

## Getting started

```bash
git clone https://github.com/ArjunSom9/Vectra.git
cd Vectra
git config core.autocrlf input        # Windows: keep LF endings

scripts/bootstrap.sh                  # GCC 14, .venv from the lock, gRPC 1.71.0
source .venv/bin/activate
make proto                            # Python stubs into build/gen/python (never committed)
make build                            # engine binaries + editable Python install
make test                             # every suite that runs on this host
scripts/run_local_cluster.sh --with-gateway   # three shards + the gateway
```

Without gRPC installed, `make build-engine-core test-engine` and
`python -m pytest` still build and test everything except the C++ server
layer.

**Editor setup:** EditorConfig support (the VS Code extension
`EditorConfig.EditorConfig`) and clangd for C++.

**Read before your first PR:** [`README.md`](README.md), the architecture
documents [`01`](docs/architecture/01-overview.md) to
[`09`](docs/architecture/09-subgraph-rewrite.md), and the
[ADRs](docs/decisions/).

---

## Repository invariants

| # | Invariant | Enforcement |
|---|---|---|
| 1 | **Nothing derived is committed**: stubs, corpora, reports, profiler captures, checkpoints, bundles | `.gitignore`; review |
| 2 | **Nothing that gates CI is ignored**: goldens, harness sources | `.gitignore` trailing negations |
| 3 | **No secret enters history**: `.env`, keys, certificates | review; keep local overrides in untracked `*.local.yaml` files |
| 4 | **`.gitkeep` files stay zero bytes** | `.editorconfig`; review |
| 5 | **Goldens are contracts**: `tests/regression/golden_*.json` hold the design's thresholds, and a baseline only from a recorded run | `tests/regression/compare_snapshots_test.py` (test) |

---

## Development workflow

**Branches**: short-lived, off `main`:

```
feat/engine-geofilter-cache
fix/kernels-masked-tail-parity
perf/compiler-pattern-match
docs/adr-009-...
```

**Commits**: Conventional Commits with a scope:

```
feat(engine): push the sale-date bound into the search predicate
fix(kernels): mask BLOCK_K tail loads at odd sequence lengths
perf(compiler): skip canonicalization outside logit chains
docs(decisions): record ADR-009
```

Scopes: `engine`, `kernels`, `compiler`, `models`, `serving`, `contracts`,
`data`, `bench`, `deploy`, `docs`, `ci`.

**Sign-off**: every commit carries a DCO sign-off (`git commit -s`). (review)

**Before pushing:**

```bash
make lint      # clang-format, cpplint, ruff check, ruff format --check, pylint
make test
```

---

## Code style

The Google style guides, enforced by their tools. `.editorconfig` carries
the same numbers for editors; if one changes, both change in the same
commit.

| Language | Authority | Rules | Enforcement |
|---|---|---|---|
| C++ | `.clang-format` (Google) + `CPPLINT.cfg` | C++20, 2-space, 80 columns, `.cc`/`.h`, no exceptions, `kConstant`, `trailing_` members | CI (`cpp-build.yml`) |
| Python | `ruff` + `.pylintrc` (Google's) | 4-space, 80 columns, single quotes, modules imported not names, Google docstrings | CI (`python-lint.yml`) |
| Protobuf | `.clang-format` (Proto section) | 2-space, 80 columns | CI (`cpp-build.yml`) |
| Shell | `bash -n`; `shellcheck` recommended | 2-space, `set -euo pipefail` | review |
| YAML, JSON, Markdown | `.editorconfig` | as configured | review |

**License headers.** Every new source file carries the Apache 2.0 header,
adapted to its comment syntax (review):

```python
# Copyright 2025 Texas A&M University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

**Large files.** Nothing larger than 1 MB outside `data/`. A large artifact
belongs in a report, the corpus or a release. (review)

---

## Testing requirements

**The test-with-feature rule:** behavior lands with the test that exercises
it. (review)

| Area | Suites | Oracle |
|---|---|---|
| `engine/` | `engine/tests/{unit,boundary,integration}` (`make test-engine`; integration needs gRPC) | brute force (`BruteForceKnn`); hnswlib in `make bench-recall` |
| `kernels/` | `test_parity`, `test_edges`, `test_custom_op` | float64 expansion; the eager chain |
| `compiler/` | `test_pattern_matching`, `test_rewrite_correctness`, `test_e2e_compile` | the eager model |
| `models/`, `serving/`, `data/` | `models/tests`, `serving/tests`, `data/tests` | exact fake shards; eager model |
| Cross-cutting | `tests/e2e_pipeline_test.py` (the real engine when built), `tests/regression`, `benchmark/harness_test.py` | golden thresholds |

Hard rules:

- `make test-kernels` runs everywhere: the tiled emulation and the operator's
  CPU path always, and the Triton kernel on a CUDA GPU or under
  `TRITON_INTERPRET=1` on Linux x86-64 (`kernel-correctness.yml`).
- Any change to `engine/src/routing/` runs the boundary suites. Exact-mode
  boundary recall of 1.0 is a gate, not a goal. (test)
- Memory-ownership changes in `engine/` pass `make test-engine-sanitizers`
  (ASan+UBSan and TSan). (CI matrix)
- Kernel parity tolerances tighten only with evidence.
- Index parameter changes re-run `make bench-recall` and update
  [`hnsw-parameter-study.md`](docs/research/hnsw-parameter-study.md).

---

## Performance-sensitive changes

A PR touching these areas includes benchmark evidence in its description:

| Area | Evidence | Command |
|---|---|---|
| `engine/src/{server,service,index,routing}/`, `engine/client/` | before/after p50, p99, p99.9 from a 10-minute soak | `make bench-qps` |
| `kernels/vectra_kernels/` | throughput, bytes, parity | `make bench-gpu`, `make bench-bandwidth` |
| `compiler/vectra_compiler/` | the compile log's match counts, e2e numerics | `make test-compiler` |
| Index parameters | recall sweep against the oracle | `make bench-recall` |

**Golden policy.** The thresholds in `tests/regression/golden_*.json` are
the design's acceptance criteria and move only with an ADR. Baselines are
recorded only from a real run, with `compare_snapshots.py --update
--environment="..."`, never written by hand. A regression against a recorded
baseline fails the gate. Accepting one needs maintainer sign-off and an ADR.

**Latency budget.** Changes to the request path update
[`latency-budget.md`](docs/research/latency-budget.md).

---

## Area-specific rules

### `engine/` (C++)

- **One ABI.** GCC 14, `-std=c++20`, libstdc++; gRPC is built with the same
  toolchain (`scripts/bootstrap.sh --grpc`).
- **No call allocation on the request path.** Calls are pre-registered and
  recycled; `WorkerStats.call_pool_allocations` must stay at 0 under load.
  New per-request heap allocations need justification in the PR.
- **Exceptions are not used** (Google style); errors are `absl::Status` or
  `grpc::Status`.
- **Bare metal.** Never add container defaults to `engine/`
  ([ADR-007](docs/decisions/adr-007-bare-metal-over-k8s.md)).

### `kernels/` (Triton)

- **Triton only through PyTorch's pinned dependency** (`triton==3.3.0` on
  Linux x86-64). Never install `torchtriton` (see [Security](#security)).
- The float64 oracle is authoritative.
- `tile_configs.py` changes re-run the tiling sweep
  (`python -m kernels.benchmarks.throughput --sweep`) and update
  [`triton-tiling-study.md`](docs/research/triton-tiling-study.md).

### `compiler/` (FX pass)

- **New formulation workflow**: a catalog entry, positive cases in at least
  three structurally different authoring styles, the look-alikes it must not
  match, and the per-formulation exactness test
  ([`pattern_registry.md`](compiler/guides/pattern_registry.md)).
- **Loud failure.** If the model's attention is not fully rewritten, the
  deployment compile fails; it never degrades to eager.

### `models/`

- **The readable-model guarantee.** No GPU-specific imports, no kernel
  calls, no `torch.compile` in `models/price_transformer/`. Compilation
  happens only in `models/export/compile_entry.py`. (review)
- **Data-dependent control flow uses `torch.cond`.** One graph break hides
  the attention from the rewrite.
  (test: `test_model_captures_as_a_single_graph`)

---

## Dependency and version changes

| Move together | Never split |
|---|---|
| PyTorch, its `triton`, the `nvidia-*` CUDA wheels | one lock-file unit |
| gRPC C++ tag, `grpcio`, `grpcio-tools`, protobuf runtime | same gRPC release |
| NVIDIA driver, CUDA, PyTorch | update `deploy/gpu/driver_matrix.md` in the same PR |
| `.editorconfig`, `.clang-format`, `pyproject.toml [tool.ruff]`, `.pylintrc` | one commit |

A dependency change must:

1. Stay inside the project's version policy (ADR-008): released on or before
   2025-05-30, not yanked.
2. Regenerate both lock files (`make lock`) and update `scripts/bootstrap.sh`
   if a tool moved.
3. Re-run the full suites, and the GPU suites if the ML stack moved.
4. Update `NOTICE` by hand, checking each holder's name upstream.
5. Land an ADR if it crosses a major version.

---

## Adding new files

- **Placement**: the owning area's directory (see the workstream table in
  the README).
- **New output directory**: register it in `.gitignore` in the same PR.
- **New empty scaffold**: a zero-byte `.gitkeep`.
- **New ADR**: the next number (currently 009), with Context, Decision,
  Consequences and Alternatives considered.

---

## Pull request process

- [ ] Every commit is signed off (`git commit -s`)
- [ ] Workflows green: `cpp-build`, `python-lint`, `kernel-correctness`
- [ ] New source files carry the Apache 2.0 header
- [ ] No generated artifacts in the diff
- [ ] No golden changes, or a labelled performance PR with evidence
- [ ] Tests ship with the change
- [ ] Performance-sensitive paths → benchmark evidence
- [ ] Decision-level change → ADR in the same PR
- [ ] Dependencies changed → lock files, `NOTICE`, pairing table

One approval from an area owner for ordinary changes. Two (area owner and
maintainer) for anything on the performance list, a version pairing, or a
golden change. Reviews focus on invariants, oracles and evidence; style is
the formatters' job. Keep PRs reviewable in one sitting.

---

## Issue reporting

**Bugs**: the reproduction commands and an environment block:

```
OS / driver / CUDA / PyTorch / gRPC versions
Cluster topology (node count, shard map excerpt)
Config: ef_search, rerank_depth, channel_pool_size, geohash policy
```

**Performance regressions**: p50, p99 and p99.9, the workload,
`ef_search`, and whether the cluster was under soak. "It feels slower"
cannot be triaged.

**Feature requests**: architectural features start as an ADR proposal.

---

## Security

Report vulnerabilities privately through the repository's GitHub private
vulnerability reporting, never in a public issue. We follow coordinated
disclosure with a 90-day window.

Supply-chain rules:

- Triton arrives only as PyTorch's pinned dependency. `torchtriton` was the
  name used in the December 2022 dependency-confusion attack on PyTorch's
  nightly channel. Do not install it.
- Installs pass `--exclude-newer 2025-05-30T23:59:59Z`, so even source
  builds resolve only pinned-window build dependencies.
- Model checkpoints are safetensors and JSON; loading one never executes
  code from the file.

---

## Documentation contributions

- Runbooks come from real procedures, research notes from real sweeps, ADRs
  from real decisions. A target is never written as a result.
- Architecture documents 01–09 change only through ADR-backed PRs, and
  record where the implementation corrects the design document.
- Diagrams are plain SVG with an explicit white background, in
  `docs/architecture/diagrams/`.

---

## License and sign-off

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), and that you have the right to license them
as such. The `Signed-off-by:` line certifies the
[Developer Certificate of Origin](https://developercertificate.org/).

---

## Conduct

Be rigorous, be kind, and assume competence. Review the code, not the
person; critique with evidence, and accept critique gracefully. Technical
disagreements are settled with benchmarks and ADRs.

---

## Questions

Architecture and design questions go to GitHub Discussions. Operational
questions: the relevant [runbook](docs/runbooks/) first, then an issue if
the runbook is wrong (and then fix the runbook).
