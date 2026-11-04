# 07 — The Fused RBF-Attention Kernel

| | |
|---|---|
| **Status** | Accepted |
| **Synthesizes** | Design document §6.2–6.3 (reformulation and tiled IO-aware execution) and §6.4 (Triton and the operator registration) |
| **Code** | `kernels/vectra_kernels/{spatial_rbf_attention,norm_precompute,tile_configs,custom_ops,reference}.py` |
| **Tests** | `kernels/tests/{test_parity,test_edges,test_custom_op}.py` |

## 1. What it computes

```
out[b,h,i,:] = Σ_j softmax_j(-scale[b,h] · ‖q[b,h,i] − k[b,h,j]‖² + bias[b,·,j]) · v[b,h,j,:]
```

for q `[B,H,Nq,D]`, k `[B,H,Nk,D]`, v `[B,H,Nk,Dv]`, a per-head coefficient
`scale` broadcastable to `[B,H,1,1]` (1/(2σ²) in the model), and an
optional key bias broadcastable to `[B,1|H,1,Nk]` (0, or −∞ for padding).

## 2. Algorithm

The kernel follows the IO-aware structure of FlashAttention [3] [25],
written in Triton [26] [32]:

1. **Grid**: one program per (query tile, batch × head),
   `(cdiv(Nq, BLOCK_Q), B·H)`. Arbitrary strides per batch and head are
   honored, so transposed or sliced inputs need no copy.
2. **Q tile**: loaded once with masked tails, and its squared norms
   computed in float32.
3. **Stream K/V tiles** of `BLOCK_K` rows. For each tile:
   * squared key norms: computed in the kernel, or read from a precomputed
     `[B·H, Nk]` vector when there are several query tiles per (b, h), so
     norms are not recomputed per query tile (`norm_precompute.py`);
   * `qk = tl.dot(q, kᵀ)`, a tensor-core GEMM with float32 accumulation;
   * the expansion `‖q−k‖² = ‖q‖² + ‖k‖² − 2qk`, clamped at 0 against
     rounding;
   * logits `−scale · d + bias`, with masked keys at −∞;
   * online softmax: a running maximum and running sum, the accumulator
     rescaled by `exp2((m_old − m_new) · log2 e)`. Fully masked rows keep a
     −∞ maximum without producing `inf − inf`;
   * `acc += tl.dot(p, v)`.
4. **Store** `acc / l` once, in v's dtype.

No Nq × Nk matrix, and no Nq × Nk × D difference tensor, ever reaches HBM.
Inputs are read once per (b, h) when one query tile covers all queries, which
is the case at the serving shape (65 queries ≤ BLOCK_Q = 128).

## 3. Precision

| Inputs | Dot products | Accumulation | Tolerance vs float64 oracle (× max\|V\|) |
|---|---|---|---|
| float16 | tensor cores | float32 | rtol 1e-3, atol 1e-3 |
| bfloat16 | tensor cores | float32 | rtol 1e-2, atol 8e-3 |
| float32 | `input_precision='ieee'` by default (`'tf32'` optional) | float32 | rtol 1e-4, atol 1e-5 |

The expansion loses precision when ‖q‖² and ‖k‖² are large relative to
their difference, so the parity suite includes far-apart points and
near-duplicates (`test_edges.py`). Rows whose keys are all masked produce
NaN, exactly as eager `softmax` does.

## 4. Tile selection

`tile_configs.select(Nq, Nk, D, Dv, element_bytes)` picks a query tile that
covers every query up to 128, so the serving shape reads each K/V tile from
HBM once per (b, h). It uses `BLOCK_K = 64`, 4 warps (8 above 64 rows) and 2
pipeline stages, and halves `BLOCK_K` until two programs fit in an SM's 228
KB of shared memory. `kernels.benchmarks.throughput --sweep` times every
configuration of `tile_configs.SWEEP` that fits
([`triton-tiling-study.md`](../research/triton-tiling-study.md)).

## 5. The operator

`custom_ops.py` registers `torch.ops.vectra.rbf_attention` with
`torch.library.custom_op` (deliverable B4):

| Dispatch | Implementation |
|---|---|
| CUDA | the Triton kernel |
| default (CPU, or CUDA without Triton) | `reference.matmul_rbf_attention`: the same expansion and float32 accumulation as whole-tensor operations |
| fake / meta | output shape `[B,H,Nq,Dv]` in v's dtype, for tracing |

`torch.library.opcheck` validates the schema, the fake implementation and
AOT dispatch with static and dynamic shapes (`test_custom_op.py`). The
operator is forward-only. The compiler never puts it into a graph that
needs gradients ([09](09-subgraph-rewrite.md)).

## 6. Verification

* `reference.expanded_rbf_attention` (float64) is the oracle, and
  `reference.eager_rbf_attention` is the eager chain.
* `reference.tiled_rbf_attention` emulates the kernel tile by tile (running
  max, running sum, rescaling) in PyTorch. It is the always-available
  implementation in the parity suite, together with the CPU operator path.
* The Triton kernel runs in the same suites on a GPU, or on CPU under
  `TRITON_INTERPRET=1` (Linux x86-64 CI, `kernel-correctness.yml`).
* Timing (`kernels.benchmarks.throughput`) refuses to report a time for an
  implementation whose output left its tolerance.

## 7. Installation rule

PyTorch 2.7.0 depends on `triton==3.3.0` on Linux x86-64, and that is the
only way Triton enters the environment (`requirements-lock.txt`). There are
no Triton wheels for macOS, so on macOS the operator uses its default
implementation. The `torchtriton` name was used in the December 2022
supply-chain attack on PyTorch's nightly channel, and must never be
installed.
