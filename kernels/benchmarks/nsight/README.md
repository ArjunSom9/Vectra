# Profiling the kernel with Nsight Compute

The GPU acceptance numbers (O2 and O3) come from Nsight Compute counters, not
from timers. This page lists the exact profiler configuration, so that the
baseline (deliverable B1) and acceptance (B3, C3) profiles compare like with
like.

## Toolchain

* Nsight Compute 2025.1.1, from CUDA 12.8 Update 1 (March 2025). It
  profiles kernels built against the CUDA 12.6 runtime of the pinned
  PyTorch 2.7.0 wheels.
* NVIDIA driver R570 or newer (see `deploy/gpu/driver_matrix.md`).
* Profiling needs access to GPU performance counters: run as root, or set
  `NVreg_RestrictProfilingToAdminUsers=0`.

## Metrics

| Metric | Meaning |
|---|---|
| `dram__bytes_read.sum` | Bytes read from HBM |
| `dram__bytes_write.sum` | Bytes written to HBM |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | Achieved DRAM bandwidth, % of peak ("Memory Throughput" in the GPU Speed Of Light section) |
| `gpu__time_duration.sum` | Kernel time |

Bytes per query is `(dram__bytes_read.sum + dram__bytes_write.sum)` summed
over every kernel in the region, divided by the requests in the batch.

## Region profile (kernel level)

```bash
ncu --nvtx --nvtx-include "vectra_region/" \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
    --cache-control all --clock-control base \
    --csv --print-units base \
    python -m kernels.benchmarks.bandwidth --impl=eager --batch=1024 > eager.csv
# Same command with --impl=fused > fused.csv
```

`benchmark/bandwidth/run_bandwidth_proof.py` runs both commands, parses the
CSV files with `benchmark/bandwidth/collect_nsight_metrics.py`, and checks
the result against `serving/configs/slo.yaml`.

### Settings that must not change between profiles

* `--cache-control all` flushes L2 before every replayed kernel, so DRAM
  counters see each kernel's real HBM traffic. At the serving batch of 64
  the fused kernel's whole working set (about 4.3 MB) fits in the 50 MB L2;
  without flushing, back-to-back replays would read it from L2.
* `--clock-control base` locks clocks to base, which makes runs
  reproducible.
* Batch size. A fused launch at batch 64 streams its data in about 1.3 us at
  full HBM bandwidth, so ramp-up and tail effects dominate its achieved
  percentage. The bandwidth-utilization acceptance (C3) is therefore
  measured at batch 1024 (about 68 MB, larger than L2). Bytes per query (B3)
  are reported at both 64 and 1024.

## Model profile (serving level)

The same metrics over a whole compiled forward pass:

```bash
ncu --nvtx --nvtx-include "vectra_model/" ... \
    python -m benchmark.gpu_throughput.run_throughput --profile_once --mode=compiled
```

The analytic budgets these profiles are checked against come from
`python -m kernels.benchmarks.roofline`.
