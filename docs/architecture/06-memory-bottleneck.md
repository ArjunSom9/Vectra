# 06 — The Memory Bottleneck of Eager Spatial Attention

| | |
|---|---|
| **Status** | Accepted |
| **Synthesizes** | Design document §3.2 (why eager PyTorch cannot meet the inference target) and §6.1 (baseline) |
| **Model** | `kernels/benchmarks/roofline.py` (analytic), `benchmark/bandwidth/` (Nsight Compute) |

The price model's attention layer scores every token against every token
with a radial-basis kernel:

```
out = softmax(-‖q_i − k_j‖² / (2σ²) + bias_j) @ V
```

In eager PyTorch this is a chain of separate kernels, each of which reads
its inputs from HBM and writes its output back [2] [3]. This document
quantifies that traffic at the serving shape, places both formulations on
the H100 roofline [6] [16], and states how the O2 and O3 measurements must
be taken. It corrects one claim of the design document (§4).

## 1. Traffic at the serving shape

Serving shape: 64 requests per batch, 65 tokens each (the subject plus 64
comparables), 4 heads of width 32, float16 blocks. The analytic model
(`python -m kernels.benchmarks.roofline`) charges each eager operator one
round trip:

| Operator | Reads + writes |
|---|---|
| `q.unsqueeze(-2) - k.unsqueeze(-3)` (materializes [B,H,Nq,Nk,D]) | 71.4 MB |
| `.pow(2)` | 138.4 MB |
| `.sum(-1)` | 71.4 MB |
| negate, scale, bias | 13.0 MB |
| `softmax` | 4.3 MB |
| `@ v` | 4.3 MB |
| **Eager total** | **302.8 MB** |
| **Fused kernel** (Q, K, V, coefficient and bias in; output out) | **4.27 MB** |

The fused kernel moves **1.4%** of the eager chain's bytes, because the
dominant term, the five-dimensional difference tensor and its square,
never exists. O2's budget of ≤ 60% is therefore conservative, as the design
intended. The measured ratio comes from Nsight Compute's
`dram__bytes_read.sum + dram__bytes_write.sum` (B3).

## 2. Roofline placement

| | FLOP per byte |
|---|---|
| Eager chain (SIMT elementwise work) | 0.6 |
| Fused kernel (tensor-core QKᵀ and PV, online softmax) | 34.5 |
| H100 SXM5 ridge point (989 dense FP16 TFLOP/s ÷ 3.35 TB/s) | 295 |

Fusion raises arithmetic intensity about 57×, but the fused kernel is
**still below the ridge point**, so it remains bandwidth-bound at this
shape. Its attainable throughput is set by HBM bandwidth (about 115
TFLOP/s at full bandwidth), not by the tensor cores.

That is what makes O3's metric meaningful. For a bandwidth-bound kernel,
the fraction of peak DRAM throughput it achieves is the right efficiency
measure. A compute-bound kernel would show low DRAM throughput however
well it ran.

## 3. How the GPU acceptance numbers must be measured

Two properties of the serving shape decide the method.

* **The working set fits in L2.** At batch 64 the fused kernel touches 4.3
  MB, against a 50 MB L2. Nsight Compute must flush caches between replays
  (`--cache-control all`), or DRAM counters see L2 hits instead of HBM
  traffic.
* **The kernel is short.** At full bandwidth, 4.3 MB streams in about 1.3
  µs, so ramp-up and tail effects dominate any percentage measured at batch
  64.

Therefore (`kernels/benchmarks/nsight/README.md`):

| Measurement | Batch | Gate |
|---|---|---|
| Bytes per query, eager vs fused (B1, B3) | 64 and 1024 | fused ≤ 60% of eager |
| Achieved DRAM throughput of the fused kernel (C3) | 1024 (68 MB, larger than L2) | ≥ 65% of peak |
| Baseline achieved DRAM throughput of the eager chain (B1) | 1024 | recorded (the design's 40% figure) |
| Serving throughput, compiled vs eager (B3) | 64, full model | ≥ 3.0× |

Each throughput run checks numerics first (rtol 1e-3 against the eager
model) and is rejected if they fail (`benchmark/gpu_throughput/compare_runs.py`).

## 4. Correction to the design document

The design document (§6.3) argues that "because the kernel's arithmetic
intensity now sits above machine balance, the operation transitions from
memory-bound to compute-bound." At the serving shape it does not: 34.5
FLOP/byte is an order of magnitude below the H100's 295 FLOP/byte ridge.
The fused kernel remains bandwidth-bound, which is consistent with, and in
fact required by, the design's own O3 metric (achieved DRAM bandwidth).
The 3× throughput objective does not need compute-boundedness. It follows
from removing about 98% of the bytes.

The design document's 40% baseline and 65% target are DRAM-throughput
percentages of an Nsight Compute profile. They are reproducible only at a
batch whose working set exceeds L2, which §3 fixes at 1024.
