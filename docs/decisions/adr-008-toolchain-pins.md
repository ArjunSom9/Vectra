# ADR-008: Toolchain pins inside the project window

* **Status:** Accepted (May 2025)
* **Deciders:** all workstreams

## Context

The project ran from January 6 to May 30, 2025. Its results must be
reproducible with the software that existed then. Several components must
also move together. PyTorch pins its Triton, Inductor's internals are what
the compiler pass relies on, gRPC vendors its protobuf and Abseil, and
generated protobuf code requires a matching runtime.

## Decision

Every pin was released on or before 2025-05-30, and nothing yanked is used.

| Component | Pin | Released |
|---|---|---|
| Python | 3.12 | |
| PyTorch | 2.7.0 (CUDA 12.6 wheels) | 2025-04-23 |
| Triton | 3.3.0 (pinned by PyTorch on Linux x86-64) | 2025-04-09 |
| gRPC (C++ tag, `grpcio`, `grpcio-tools`) | 1.71.0 | 2025-03-06 (tag), 2025-03-10 (PyPI) |
| protobuf: C++ (vendored by gRPC) / Python runtime | 29.0 / 5.29.5 | Python 2025-05-28 |
| Abseil (vendored by gRPC) | 20240722.0 | 2024-07 |
| GoogleTest | 1.17.0 | 2025-04-30 |
| NumPy | 2.2.6 | 2025-05-17 |
| safetensors / absl-py / PyYAML / prometheus-client | 0.5.3 / 2.2.2 / 6.0.2 / 0.21.1 | 2025-02-26 / 2025-04-03 / 2024-08-06 / 2024-12-03 |
| pytest / hnswlib (oracle) | 8.3.5 / 0.8.0 | 2025-03-02 / 2023-12-03 |
| CMake / Ninja (PyPI) | 3.31.6 / 1.11.1.4 | 2025-02-28 / 2025-03-22 |
| Compilers | GCC 14 (Clang 19 also supported) | Ubuntu 24.04 packages |
| clang-format / cpplint / ruff / pylint | 19.1.7 / 2.0.2 / 0.11.12 / 3.3.7 | 2025-01-14 / 2025-04-08 / 2025-05-29 / 2025-05-04 |
| uv / setuptools | 0.7.8 / 80.9.0 | 2025-05-24 / 2025-05-27 |
| ansible-core | 2.18.6 | 2025-05-19 |
| Observability | Prometheus 3.4.0, Alertmanager 0.28.1, Grafana 12.0.1, OTel Collector 0.127.0, node_exporter 1.9.1, dcgm-exporter 4.2.3-4.1.1 | 2025-03-07 to 2025-05-27 |
| CUDA base image / profiler | `nvidia/cuda:12.6.3-base-ubuntu24.04` / Nsight Compute 2025.1.1 | 2024-11 / 2025-03 |

`requirements-lock.txt` (all extras) and `requirements-runtime-lock.txt`
(the inference image) freeze the full transitive closure with
`uv pip compile --universal --exclude-newer 2025-05-30T23:59:59Z`, and
installs pass the same `--exclude-newer`, so source builds also resolve
in-window build dependencies.

**grpcio 1.72.0 is not used.** It was released inside the window
(2025-04-25) but yanked from PyPI for broken macOS wheels, and its first
fixed release (1.72.1, 2025-06-02) is outside it. The whole stack therefore
uses 1.71.0, C++ and Python alike.

## Consequences

* Upgrades are deliberate: one PR moves a whole row (for example PyTorch,
  Triton and the CUDA wheels), re-runs every suite, and re-certifies with the
  benchmark gates.
* The driver, CUDA, PyTorch and Triton matrix for GPU hosts lives in
  `deploy/gpu/driver_matrix.md`.
