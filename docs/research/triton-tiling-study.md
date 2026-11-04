# Triton tiling study

**Question.** Which launch shape (query tile, key tile, warps, pipeline
stages) should the fused kernel use at the serving shape, and how sensitive
is it?

**Status.** The heuristic and the candidate set below are implemented in
`kernels/vectra_kernels/tile_configs.py`. The timing sweep needs an H100. It
has not been run as part of this repository's verification, so no timings
are reported here. To run it:

```bash
python -m kernels.benchmarks.throughput --sweep --batches=64,256,1024
# -> kernels/benchmarks/results/throughput_float16.csv
```

Every configuration's output is checked against the float64 oracle before
its time is recorded.

## Candidates

Head width 32, float16 (`tile_configs.SWEEP`). Shared memory counts Q once
and K plus V per pipeline stage (`shared_memory_bytes`). The H100 has 228 KB
per SM ([16]):

| BLOCK_Q × BLOCK_K | warps | stages | shared memory (KB) | 2 programs per SM fit |
|---|---|---|---|---|
| 16 × 64 | 4 | 2 | 17.0 | yes |
| 32 × 64 | 4 | 2 | 18.0 | yes |
| 64 × 32 | 4 | 2 | 12.0 | yes |
| 64 × 64 | 4 | 2 | 20.0 | yes |
| 64 × 64 | 4 | 3 | 28.0 | yes |
| 64 × 128 | 8 | 2 | 36.0 | yes |
| 128 × 64 | 8 | 2 | 24.0 | yes |
| 128 × 64 | 8 | 3 | 32.0 | yes |
| 128 × 128 | 8 | 2 | 40.0 | yes |

At head width 32 shared memory never binds; registers and occupancy do. For
float32 with head width 128, the heuristic halves `BLOCK_K` to 16 to fit.

## The heuristic

`select(Nq, Nk, D, Dv, element_bytes)`:

* `BLOCK_Q` is the smallest power of two ≥ 16 that covers `Nq`, capped at
  128. At the serving shape (65 tokens) that is 128, so one program covers
  all queries of a (batch, head), and each K/V tile is read from HBM once.
  With several query tiles, K and V would be re-read once per tile.
* `BLOCK_K` = 64, halved until two programs fit per SM.
* 4 warps up to 64 query rows, 8 above; 2 stages.

## What to expect, and what to measure

At batch 64 the grid is `(1, 256)`: 256 programs on 132 SMs, about two waves
of work that stream 4.3 MB in total. Launch overhead and the second,
partial wave are a large share of the run time, so the sweep's batch-64
column mostly measures latency effects. At batch 1024 (4,096 programs, about
31 waves) the kernel streams 68 MB and the configurations separate by how
well they keep HBM busy ([06](../architecture/06-memory-bottleneck.md)).
The selection is made at batch 1024 and checked at batch 64 for
regressions.

Questions the sweep answers:

1. Does `BLOCK_Q = 128` with 8 warps beat `BLOCK_Q = 64` (two query tiles,
   K/V read twice) at 65 tokens? The traffic model says it should.
2. Does a third pipeline stage hide more latency at batch 1024?
3. Does `BLOCK_K = 128` amortize the online-softmax rescaling enough to
   matter?
