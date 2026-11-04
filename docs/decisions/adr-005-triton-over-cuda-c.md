# ADR-005: Triton instead of hand-written CUDA for the fused kernel

* **Status:** Accepted (February 2025)
* **Deciders:** Workstream B
* **Specification:** [07 — The fused kernel](../architecture/07-triton-kernels.md)

## Context

The kernel is a tiled, fused, attention-shaped computation: two GEMMs per
tile pair, an elementwise transform, an online softmax. Hand-written CUDA
puts memory coalescing, shared-memory layout, tensor-core instruction
selection and software pipelining on the author.

## Decision

Write it in Triton 3.3.0 [26] [32], the version PyTorch 2.7.0 pins.
Programs reason in tiles, and the compiler handles thread-level layout and
emits tensor-core MMA for `tl.dot`. Register it as the PyTorch operator
`vectra::rbf_attention` with `torch.library.custom_op`, with a fake
implementation for tracing and a portable default implementation.

## Consequences

* The kernel is about a hundred lines of Python. It is tested on CPU with
  the Triton interpreter and on H100 in CI.
* Inductor itself generates Triton, so the operator sits in compiled graphs
  without foreign-function glue [24].
* Triton exists for Linux x86-64 only. On other hosts the operator uses its
  default implementation, the same arithmetic as whole-tensor PyTorch
  operations.
* Tile choices are heuristic (`tile_configs.select`) and validated by sweep
  (`kernels.benchmarks.throughput --sweep`), not by autotuning at run time,
  so serving latency is deterministic.
