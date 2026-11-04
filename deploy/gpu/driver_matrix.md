# GPU driver and toolchain matrix

The inference tier pins the driver, CUDA, PyTorch and Triton as one unit.
Mixing versions can silently change which kernels run, so every row below
is upgraded together and re-certified with `make bench-gpu` and
`make bench-bandwidth`.

| Component | Version | Notes |
|---|---|---|
| GPU | NVIDIA H100 SXM5 80 GB (sm_90) | HBM3, about 3.35 TB/s peak |
| Driver | R570 production branch (570.x) | Required by Nsight Compute 2025.1; also satisfies the CUDA 12.6 runtime |
| CUDA runtime | 12.6 (`nvidia-cuda-runtime-cu12==12.6.77`) | Shipped inside the PyTorch wheels |
| cuDNN / cuBLAS | 9.5.1.17 / 12.6.4.1 | Shipped inside the PyTorch wheels |
| PyTorch | 2.7.0 (2025-04-23) | `torch==2.7.0`, default CUDA 12.6 wheels |
| Triton | 3.3.0 | The version PyTorch 2.7.0 pins on Linux x86-64; bundles its own `ptxas` |
| Base image | `nvidia/cuda:12.6.3-base-ubuntu24.04` | `deploy/gpu/Dockerfile.inference` |
| Container toolkit | NVIDIA Container Toolkit 1.17.x | `--gpus all` |
| Profiler | Nsight Compute 2025.1.1 (CUDA 12.8 Update 1) | `kernels/benchmarks/nsight/README.md` |
| Monitoring | DCGM exporter 4.2.3-4.1.1 | `deploy/observability/prometheus.yml` |

## Rules

* Install Triton only through the pinned PyTorch wheel's dependency. Stable
  PyTorch releases depend on the PyPI package `triton`. The `pytorch-triton`
  package is the nightly build. The unrelated `torchtriton` name was the
  December 2022 supply-chain attack on the nightly channel.
* Driver updates stay within the R570 branch until the whole row is
  re-certified.
* MIG stays disabled on serving GPUs: the bandwidth acceptance assumes the
  whole HBM subsystem.
