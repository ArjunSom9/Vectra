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

"""The fused geo-spatial (RBF) attention kernel in Triton.

Computes softmax(-scale * ||q - k||^2 + key_bias) @ v in one pass:

* Algebra: -||q - k||^2 = -||q||^2 + 2 q.k - ||k||^2 [29], so the inner loop
  is one tl.dot per tile pair, which the Triton compiler lowers to
  tensor-core MMA instructions [26] [32], plus two broadcast subtractions.
* IO-awareness [3]: a BLOCK_Q tile of queries is loaded once and kept, with
  its norms, in registers; BLOCK_K tiles of candidates stream past; the
  softmax is accumulated online with a running maximum and running sum
  [3] [25], so no Nq x Nk score matrix is ever written to HBM. Each program
  writes its output tile exactly once.
* Discipline [26]: tails are handled with masked loads and stores, so any
  Nq, Nk and head dimension is legal (head dimensions are padded to a power
  of two >= 16 in registers only).

Inputs are [B, H, N, D] with arbitrary strides (head-transposed views need no
copy). Half precision multiplies on tensor cores with float32 accumulation;
float32 inputs use IEEE-exact products by default.
"""

# Triton kernels name compile-time parameters in capitals by convention.
# pylint: disable=invalid-name

import torch
import triton
import triton.language as tl

from vectra_kernels import norm_precompute
from vectra_kernels import tile_configs


@triton.jit
def _rbf_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    k_norm_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    n_heads,
    n_q,
    n_k,
    head_dim,
    value_dim,
    bias_heads,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oz,
    stride_oh,
    stride_on,
    stride_od,
    stride_bz,
    stride_bh,
    stride_bn,
    HAS_BIAS: tl.constexpr,
    PRECOMPUTED_K_NORMS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // n_heads
    head = pid_bh % n_heads

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    q_rows = offs_q < n_q
    d_cols = offs_d < head_dim
    dv_cols = offs_dv < value_dim

    q_base = q_ptr + batch * stride_qz + head * stride_qh
    k_base = k_ptr + batch * stride_kz + head * stride_kh
    v_base = v_ptr + batch * stride_vz + head * stride_vh

    # The query tile and its norms stay on chip for the whole candidate axis.
    q = tl.load(
        q_base + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd,
        mask=q_rows[:, None] & d_cols[None, :],
        other=0.0,
    )
    q32 = q.to(tl.float32)
    q_norm = tl.sum(q32 * q32, axis=1)
    scale = tl.load(scale_ptr + pid_bh)

    running_max = tl.full([BLOCK_Q], float('-inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_DV], dtype=tl.float32)
    log2e = 1.4426950408889634

    for start_k in range(0, n_k, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_rows = offs_k < n_k
        k = tl.load(
            k_base + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=k_rows[:, None] & d_cols[None, :],
            other=0.0,
        )
        if PRECOMPUTED_K_NORMS:
            k_norm = tl.load(
                k_norm_ptr + pid_bh * n_k + offs_k, mask=k_rows, other=0.0
            )
        else:
            k32 = k.to(tl.float32)
            k_norm = tl.sum(k32 * k32, axis=1)
        # Tensor-core GEMM for q.k; the norms complete the distance.
        qk = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
        sq_dist = tl.maximum(q_norm[:, None] + k_norm[None, :] - 2.0 * qk, 0.0)
        logits = -scale * sq_dist
        if HAS_BIAS:
            bias = tl.load(
                bias_ptr
                + batch * stride_bz
                + (head % bias_heads) * stride_bh
                + offs_k * stride_bn,
                mask=k_rows,
                other=float('-inf'),
            )
            logits = logits + bias[None, :]
        else:
            logits = tl.where(k_rows[None, :], logits, float('-inf'))

        # Online softmax: rescale history by exp(old_max - new_max).
        new_max = tl.maximum(running_max, tl.max(logits, axis=1))
        safe_max = tl.where(new_max == float('-inf'), 0.0, new_max)
        alpha = tl.exp2((running_max - safe_max) * log2e)
        weights = tl.exp2((logits - safe_max[:, None]) * log2e)
        running_sum = running_sum * alpha + tl.sum(weights, axis=1)
        v = tl.load(
            v_base + offs_k[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
            mask=k_rows[:, None] & dv_cols[None, :],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(
            weights.to(v.dtype), v, input_precision=INPUT_PRECISION
        )
        running_max = new_max

    out = acc / running_sum[:, None]
    out_base = out_ptr + batch * stride_oz + head * stride_oh
    tl.store(
        out_base + offs_q[:, None] * stride_on + offs_dv[None, :] * stride_od,
        out.to(out_ptr.dtype.element_ty),
        mask=q_rows[:, None] & dv_cols[None, :],
    )


def rbf_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: torch.Tensor,
    key_bias: torch.Tensor | None = None,
    config: tile_configs.TileConfig | None = None,
    input_precision: str = 'ieee',
) -> torch.Tensor:
    """Launches the fused kernel.

    Args:
      q: [B, H, Nq, D] CUDA tensor (float16, bfloat16 or float32).
      k: [B, H, Nk, D], same dtype.
      v: [B, H, Nk, Dv], same dtype.
      scale: Broadcastable to [B, H, 1, 1]; the RBF coefficient 1 / (2 sigma^2).
      key_bias: Optional additive key bias broadcastable to [B, 1|H, 1, Nk],
        typically 0 or -inf for padded candidates.
      config: Tile shape; chosen by tile_configs.select when omitted.
      input_precision: tl.dot precision for float32 inputs ('ieee', 'tf32' or
        'tf32x3'); ignored for half-precision inputs.

    Returns:
      [B, H, Nq, Dv] tensor in v's dtype.
    """
    batch, heads, n_q, head_dim = q.shape
    n_k, value_dim = k.shape[-2], v.shape[-1]
    if config is None:
        config = tile_configs.select(
            n_q, n_k, head_dim, value_dim, q.element_size()
        )
    scale_bh = (
        scale.to(torch.float32)
        .expand(batch, heads, 1, 1)
        .reshape(batch * heads)
        .contiguous()
    )
    out = torch.empty(
        batch, heads, n_q, value_dim, dtype=v.dtype, device=q.device
    )
    # With several query tiles per (batch, head), compute ||k||^2 once.
    precomputed = n_q > config.block_q
    k_norms = norm_precompute.squared_norms(k) if precomputed else scale_bh
    if key_bias is not None:
        bias = key_bias.to(torch.float32)
        while bias.dim() < 4:
            bias = bias.unsqueeze(0)
        if bias.shape[-2] != 1:
            raise ValueError('key_bias must be constant across queries')
        # Stride-0 views broadcast over batch (and heads when shape[1] == 1).
        bias = bias.expand(batch, bias.shape[1], 1, n_k)
        bias_args = (
            bias,
            bias.shape[1],
            bias.stride(0),
            bias.stride(1),
            bias.stride(3),
        )
    else:
        bias_args = (scale_bh, 1, 0, 0, 0)  # Never read.
    grid = (triton.cdiv(n_q, config.block_q), batch * heads)
    _rbf_attention_kernel[grid](
        q,
        k,
        v,
        k_norms,
        scale_bh,
        bias_args[0],
        out,
        heads,
        n_q,
        n_k,
        head_dim,
        value_dim,
        bias_args[1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        bias_args[2],
        bias_args[3],
        bias_args[4],
        HAS_BIAS=key_bias is not None,
        PRECOMPUTED_K_NORMS=precomputed,
        BLOCK_Q=config.block_q,
        BLOCK_K=config.block_k,
        BLOCK_D=tile_configs.padded_dim(head_dim),
        BLOCK_DV=tile_configs.padded_dim(value_dim),
        INPUT_PRECISION=input_precision,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return out
