# Contributing to Vectra

Thank you for investing time in Vectra — a geo-distributed vector search and
Transformer inference engine engineered to serve 50,000 queries per second
across 10 million property embeddings at sub-millisecond latency.

This document is the contract between you and the repository. Everything in
it is **enforced by CI** — the rules below describe gates that already exist,
not aspirations.

---

## Table of Contents

- [The Four House Rules](#the-four-house-rules)
- [Getting Started](#getting-started)
- [Repository Invariants](#repository-invariants)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing Requirements](#testing-requirements)
- [Performance-Sensitive Changes](#performance-sensitive-changes)
- [Pillar-Specific Rules](#pillar-specific-rules)
- [Dependency and Version Changes](#dependency-and-version-changes)
- [Adding New Files](#adding-new-files)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Security](#security)
- [Documentation Contributions](#documentation-contributions)
- [License and Sign-off](#license-and-sign-off)
- [Code of Conduct](#code-of-conduct)

---

## The Four House Rules

These govern every decision in this repository. If a contribution conflicts
with one of them, the contribution changes — not the rule.

1. **Every claim is a gated benchmark.** If a PR asserts a number, the harness
   that measures it runs in CI. Performance claims without reproduction
   commands are review-blocked.
2. **Every optimization ships behind an oracle.** hnswlib validates the
   engine's recall; the eager reference validates the Triton kernel's
   numerics; the un-compiled model validates the FX rewrite pass. You never
   land an optimization whose correctness oracle doesn't already exist.
3. **Documentation is written at the moment of decision — never after.**
   ADRs land in the same PR as the decision they record.
4. **Two sources of truth move as one unit.** `.editorconfig` and the
   formatters change together; PyTorch, `pytorch-triton`, and CUDA change
   together; gRPC and protobuf change together. Partial bumps are rejected.

---

## Getting Started

```bash
git clone https://github.com/vectra-labs/vectra.git
cd vectra

# Windows contributors: normalize line endings on checkout
git config core.autocrlf input

./scripts/bootstrap.sh          # installs the pinned toolchain
make proto                      # generates gRPC stubs — never commit them
make build                      # engine binaries + Python packages
make test                       # all suites (see Testing Requirements)

# Smoke the full local stack: 1 router + 3 shards + inference gateway
./scripts/run_local_cluster.sh
```

**Editor setup:** VS Code requires the EditorConfig extension
(`EditorConfig.EditorConfig` — not built in) plus clangd for the C++ pillar;
JetBrains IDEs and Neovim ≥ 0.9 support everything natively.

**Read before your first PR:** [`README.md`](README.md),
[`docs/architecture/01-overview.md`](docs/architecture/01-overview.md) through
`09-subgraph-rewrite.md`, and the [ADR index](docs/decisions/) — the eight
recorded decisions explain why the repo is shaped the way it is.

---

## Repository Invariants

Five rules with no exceptions. CI catches all five:

| # | Invariant | Enforcement |
|---|---|---|
| 1 | **Nothing derived is committed** — stubs, corpus, reports, captures, checkpoints | `cpp-build.yml` diff check + `.gitignore` |
| 2 | **Nothing that gates CI is ignored** — goldens, harness sources | `.gitignore` trailing negations |
| 3 | **No secret ever enters history** — `.env`, keys, certs | secret-scanning + `*.local.yaml` convention |
| 4 | **`.gitkeep` files stay exactly zero bytes** | `editorconfig-checker` in `python-lint.yml` |
| 5 | **Goldens are contracts, not artifacts** — `tests/regression/golden_*.json` | regression suite fails on drift |

Local configuration goes in `*.local.yaml` files (untracked by design).
Committed configuration lives in the versioned `configs/` trees.

---

## Development Workflow

**Branching** — short-lived branches off `main`:

```
<type>/<scope>-<slug>
feat/engine-geofilter-cache
fix/kernels-masked-tail-parity
perf/compiler-pattern-match
docs/adr-009-…
```

**Commits** — Conventional Commits with a pillar scope:

```
feat(engine): route adjacent-cell fanout only to intersecting shards
fix(kernels): mask BLOCK_K tail loads at odd sequence lengths
perf(compiler): cache subgraph match across post-grad passes
docs(decisions): record ADR-009 for PQ sub-space count
```

Valid scopes: `engine`, `kernels`, `compiler`, `models`, `serving`,
`contracts`, `data`, `bench`, `deploy`, `docs`, `ci`.

**Sign-off** — every commit requires a DCO sign-off (`git commit -s`):

```
Signed-off-by: Your Name <you@example.com>
```

**Before pushing:**

```bash
make lint          # ruff + clang-format --dry-run + shfmt + shellcheck
make test          # full local suite
```

---

## Code Style

`.editorconfig` is the editor baseline; the formatters are the law. The two
are numerically identical by design — if one changes, both change in the
same commit.

| Language | Authority | Rules |
|---|---|---|
| C++ | `clang-format` (Google, 80 col) | 2-space, no in-source builds, presets only → `/build/` |
| Python | `ruff` lint + format (100 col) | 4-space, `target-version = "py312"` |
| Protobuf | `buf format` | 2-space, 80 col |
| Shell | `shfmt -i 2` + `shellcheck` | 2-space |
| YAML/JSON/MD | `.editorconfig` | trailing-whitespace protections as configured |

**License headers.** Every new source file carries the Apache 2.0 boilerplate
from the [`LICENSE`](LICENSE) appendix, adapted to the file's comment syntax.
Python short form:

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

The header-presence lint in `python-lint.yml` / `cpp-build.yml` fails builds
that omit it.

**Blob guard.** Any file larger than 1 MB outside `data/` fails CI. If you
genuinely need to commit a large artifact, that is almost certainly a bug —
it belongs in a benchmark report, the corpus, or a release.

---

## Testing Requirements

**The test-with-feature rule:** no file lands without a test exercising it.
This is the standing rule from the implementation plan, and it applies
retroactively to review — a PR adding behavior without its test is returned.

| Pillar | Suites (`make test-*`) | Correctness oracle |
|---|---|---|
| `engine/` | `unit`, `integration`, `boundary` | hnswlib recall parity at matched `ef_search` |
| `kernels/` | `test_parity`, `test_edges` | the eager reference implementation |
| `compiler/` | `pattern_matching`, `rewrite_correctness`, `e2e_compile` | un-compiled model numerics |
| Cross-cutting | `e2e_pipeline`, `regression` | golden snapshots |

Additional hard rules:

- `make test-kernels` requires a CUDA 13 GPU locally; CI runs it on the GPU
  runner via `kernel-correctness.yml`.
- Any change to `engine/src/routing/` requires the **boundary suites** —
  100% adjacent-cell recall is a gate, not a goal.
- Kernel parity tolerances tighten only with Nsight evidence attached.
- Any GeoHash precision change re-runs `make bench-recall` and updates
  `docs/research/hnsw-parameter-study.md`.

---

## Performance-Sensitive Changes

**Trigger list.** A PR touching any of these requires benchmark evidence in
its description:

| Area | Required evidence | Command |
|---|---|---|
| `engine/src/{server,index,routing}/`, `engine/client/` | before/after p50/p99/p99.9 + 10-min soak | `make bench-qps` |
| `kernels/vectra_kernels/` | throughput, bandwidth, parity deltas | `make bench-gpu` + `make bench-bandwidth` |
| `compiler/vectra_compiler/` | compile log showing pattern match + e2e numerics | `make test-compiler` |
| HNSW parameters | recall sweep vs. oracle | `make bench-recall` |

**Golden policy.** `tests/regression/golden_*.json` values are CI contracts:

- **Improvements** may update goldens in the same PR, with the benchmark
  output pasted into the description.
- **Regressions fail the build.** Accepting one requires maintainer sign-off
  and an ADR explaining the trade.
- Goldens never move silently, as collateral of an unrelated change, or
  without the reproducing harness in the same PR.

**Latency budget.** Changes to the request hot path update
[`docs/research/latency-budget.md`](docs/research/latency-budget.md) — the
microsecond ledger is maintained, not archived.

---

## Pillar-Specific Rules

### `engine/` (C++)

- **One ABI.** Everything builds with the pinned GCC 14, `-std=c++20`,
  libstdc++. Mixing compilers or STLs in a link is rejected.
- **No allocation on the request hot path.** The `CallData` lifecycle is
  explicit; the arena owns the graph. New per-request heap allocations need
  justification in the PR.
- **Run sanitizers** (`cmake -DVECTRA_SANITIZE=address,undefined`) before
  submitting anything that touches memory ownership.
- **No container defaults.** The engine runs thread-per-core on bare metal —
  never add Docker/Kubernetes defaults to `engine/` (see
  [ADR-007](docs/decisions/adr-007-bare-metal-over-k8s.md)).

### `kernels/` (Triton)

- **Only `pytorch-triton` from the pinned PyTorch wheel.** Never install
  standalone Triton or the similarly named `torchtriton` (supply-chain
  attack; see [Security](#security)).
- The eager reference is authoritative until a parity test says otherwise.
- `tile_configs.py` changes re-run the autotune sweep and update
  `docs/research/triton-tiling-study.md`.

### `compiler/` (FX pass)

- **New pattern workflow:** add the pattern/replacement pair, a
  `test_pattern_matching.py` case covering **at least three structurally
  different authoring styles** (matching is use-def topological, never
  name-based — prove it), a `pattern_registry.md` entry, and the
  `authoring_for_dynamo.md` update if authoring rules change.
- **Loud-failure guarantee.** If a model is expected to contain the spatial
  pattern and compilation finds none, compilation fails — never silently
  degrades to eager.

### `models/`

- **The readable-model guarantee.** `models/` contains zero GPU-specific
  imports, zero manual kernel invocations, zero `torch.compile` calls —
  compilation happens exclusively in `models/export/compile_entry.py`.
- **All data-dependent control flow uses `torch.cond`.** One graph break
  silently disables the kernel rewrite; CI lints for this and fails.

---

## Dependency and Version Changes

Dependency bumps are structural changes, treated like architecture:

| Move together | Never split |
|---|---|
| PyTorch ↔ `pytorch-triton` ↔ CUDA toolkit | the three are one lockfile unit |
| gRPC ↔ protobuf | gRPC's vendored protobuf only |
| NVIDIA driver ↔ CUDA ↔ PyTorch | update `deploy/gpu/driver_matrix.md` in the same PR |
| `.editorconfig` ↔ `.clang-format` / `ruff` config | one commit, both files |

Any dependency change must:

1. Regenerate `requirements-lock.txt` and update `scripts/bootstrap.sh`.
2. Re-run `make test-kernels` (GPU runner) if the ML stack moved.
3. Regenerate `NOTICE` via `make dist` and verify each third-party holder
   name against upstream — attribution drift is a release blocker.
4. Land an ADR if the change crosses a major version boundary.

Python stays on 3.12 until a pinned, reviewed upgrade ADR says otherwise.

---

## Adding New Files

- **Placement** follows the [repository layout](README.md#repository-layout)
  and the master directory outline — when in doubt, the pillar owns the file.
- **New benchmark output directory** → register it in the corresponding
  section of `.gitignore` in the same PR (the section banners exist to keep
  the generated/source boundary auditable).
- **New empty scaffold** → commit a zero-byte `.gitkeep` (the `.editorconfig`
  protects it from accidental newline insertion).
- **New ADR** → next number (currently **009**), format:
  *Context / Decision / Alternatives Considered / Consequences*, one file
  per decision in `docs/decisions/`.

---

## Pull Request Process

**Hard gates** — the checklist CI and reviewers apply mechanically:

- [ ] Every commit carries `Signed-off-by:` (`git commit -s`)
- [ ] All four workflows green: `cpp-build`, `python-lint`,
      `kernel-correctness`, `docs-publish`
- [ ] New source files carry the Apache 2.0 header
- [ ] Diff contains zero generated artifacts (stubs, corpus, reports,
      `.ncu-rep`, checkpoints)
- [ ] Diff contains zero goldens — **or** this is a labeled perf PR with
      before/after evidence
- [ ] The change ships with its tests
- [ ] Performance-sensitive paths touched → benchmark evidence in the
      description
- [ ] Decision-level change → ADR in the same PR
- [ ] Dependencies changed → lockfile + `NOTICE` + pairing table honored

**Review expectations:** one approval from a pillar owner for ordinary
changes; two approvals (pillar owner + maintainer) for anything on the
performance trigger list, a version pairing, or a golden update. Reviews
focus on invariants, oracles, and evidence — style is the formatters' job.

**PRs are small.** If it can't be reviewed in one sitting, split it — the
phase-gated structure of the project exists precisely to make this natural.

---

## Issue Reporting

**Bugs** — include the reproduction commands and an environment block:

```
OS / driver / CUDA / PyTorch / gRPC versions
Cluster topology (node count, shard map excerpt)
Config: ef_search, channel_pool_size, geohash policy
```

**Performance regressions** — mandatory: p50/p99/p99.9 percentiles, the
workload, `ef_search`, and whether the cluster was under soak. "It feels
slower" cannot be triaged.

**Feature requests** — architectural features open as an ADR proposal
first; the discussion happens on the document, not in comment threads.

---

## Security

**Report vulnerabilities privately to `security@vectra-labs.dev` — never
via public issues.** We follow coordinated disclosure with a 90-day window.

Supply-chain rules (the CI-enforced ones):

- Triton arrives **only** as `pytorch-triton` inside the pinned PyTorch
  wheel. The package named `torchtriton` on PyPI was a real dependency
  confusion attack against this exact ecosystem — do not install it.
- `make dist` runs `pip-audit` before packaging; a finding blocks release.
- Dependency review runs on every lockfile diff.

---

## Documentation Contributions

- **Docs-with-decision:** runbooks are written from real incidents, research
  notes from real sweeps, ADRs from real decisions. A document describing
  work that hasn't happened yet gets returned.
- Architecture docs 01–09 mirror the design document and change only via
  ADR-backed PRs.
- Diagrams: edit the `.mmd` sources in
  `docs/architecture/diagrams/sources/`, export the `.svg`, commit both.

---

## License and Sign-off

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE) and that you have the right to license them
as such. Every commit carries a `Signed-off-by:` line certifying the
[Developer Certificate of Origin](https://developercertificate.org/):

```
Developer Certificate of Origin, Version 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the
    right to submit it under the Apache License 2.0; or ...
```

New files carry the header. Dependency changes regenerate `NOTICE`. That's
the whole legal surface — Apache 2.0 keeps it that way.

---

## Code of Conduct

Be rigorous, be kind, and assume competence. Review the code, not the
person; critique with evidence, accept critique with grace. Technical
disagreements get resolved with benchmarks and ADRs — the repo's entire
philosophy — not with volume. The full
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) applies to everyone participating
in issues, reviews, and discussions.

---

## Questions

Architecture and design questions → GitHub Discussions.
Operational questions → the relevant [runbook](docs/runbooks.md) first, then
an issue if the runbook is wrong (in which case, fix the runbook — that's
the docs-with-decision rule working as intended).