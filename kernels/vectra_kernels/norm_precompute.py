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

"""Candidate norms ||k||^2 for the Euclidean expansion.

When several query tiles score the same candidates (Nq > BLOCK_Q), the norms
are computed once here and read by every tile, instead of being recomputed
from each K tile. For the serving shape a single query tile covers all
queries and the fused kernel computes the norms from the K tile it already
holds, which costs no extra memory traffic. Norms of raw retrieved
embeddings are also precomputed at index build time by the engine
(Candidate.squared_norm).
"""

# Triton kernels name compile-time parameters in capitals by convention.
# pylint: disable=invalid-name

import torch
import triton
import triton.language as tl


@triton.jit
def _squared_norms_kernel(
    x_ptr,
    out_ptr,
    n_heads,
    n_rows,
    n_cols,
    stride_xz,
    stride_xh,
    stride_xn,
    stride_xd,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_rows = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // n_heads
    head = pid_bh % n_heads
    rows = pid_rows * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    base = x_ptr + batch * stride_xz + head * stride_xh
    acc = tl.zeros([BLOCK_ROWS], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_COLS):
        cols = start + tl.arange(0, BLOCK_COLS)
        mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
        x = tl.load(
            base + rows[:, None] * stride_xn + cols[None, :] * stride_xd,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x * x, axis=1)
    tl.store(out_ptr + pid_bh * n_rows + rows, acc, mask=rows < n_rows)


def squared_norms(x: torch.Tensor) -> torch.Tensor:
    """Row norms of a [B, H, N, D] CUDA tensor as float32 [B, H, N]."""
    batch, heads, rows, cols = x.shape
    out = torch.empty(batch, heads, rows, dtype=torch.float32, device=x.device)
    block_rows = 64
    grid = (triton.cdiv(rows, block_rows), batch * heads)
    _squared_norms_kernel[grid](
        x,
        out,
        heads,
        rows,
        cols,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=min(128, triton.next_power_of_2(cols)),
    )
    return out
