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

"""Reference implementations of RBF (spatial) attention.

All three compute, for each query row i,

  out_i = sum_j softmax_j(-scale * ||q_i - k_j||^2 + key_bias_j) * v_j,

the distance-based attention variant studied in [14] and [15]. Shapes are
q [B, H, Nq, D], k [B, H, Nk, D], v [B, H, Nk, Dv]; `scale` broadcasts to
[B, H, 1, 1] and `key_bias` (0 or -inf for padding) to [B, 1|H, 1, Nk].

* eager_rbf_attention: the operator chain a researcher writes in eager
  PyTorch (subtract, square, sum, negate-scale, softmax, matmul). It
  materializes Nq x Nk x D and Nq x Nk intermediates in memory; this is the
  memory-bound baseline [3] and the pattern the compiler pass matches.
* expanded_rbf_attention: the Euclidean expansion
  -||q - k||^2 = -||q||^2 + 2 q.k - ||k||^2 [29] in float64, the numerical
  oracle for kernel parity tests.
* tiled_rbf_attention: a PyTorch emulation of the fused Triton kernel's
  algorithm, tile for tile: GEMM-form distances and online-softmax
  accumulators [3] [25], never forming the full score matrix. It is the
  custom operator's CPU implementation.
"""

import torch


def _broadcast_scale(scale: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Scale as float32 [B, H, 1, 1]."""
    return scale.to(torch.float32).expand(*q.shape[:-2], 1, 1)


def eager_rbf_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: torch.Tensor,
    key_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """The eager operator chain (memory-bound baseline)."""
    sq_dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).pow(2).sum(-1)
    logits = -sq_dist * scale
    if key_bias is not None:
        logits = logits + key_bias
    return torch.softmax(logits, dim=-1) @ v


def expanded_rbf_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: torch.Tensor,
    key_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Float64 oracle via the Euclidean expansion; returns float64."""
    q64, k64, v64 = q.double(), k.double(), v.double()
    q_norm = (q64 * q64).sum(-1, keepdim=True)
    k_norm = (k64 * k64).sum(-1).unsqueeze(-2)
    sq_dist = (q_norm + k_norm - 2.0 * q64 @ k64.transpose(-1, -2)).clamp_min(0)
    logits = -sq_dist * scale.double()
    if key_bias is not None:
        logits = logits + key_bias.double()
    return torch.softmax(logits, dim=-1) @ v64


def matmul_rbf_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: torch.Tensor,
    key_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """The kernel's arithmetic as whole-tensor operations; v's dtype.

    The same Euclidean expansion and float32 accumulation as the kernel,
    with the distance matrix materialized once: the operator's
    implementation off the GPU, where BLAS makes it fast.
    """
    qf, kf = q.float(), k.float()
    q_norm = (qf * qf).sum(-1, keepdim=True)
    k_norm = (kf * kf).sum(-1).unsqueeze(-2)
    sq_dist = (q_norm + k_norm - 2.0 * qf @ kf.transpose(-1, -2)).clamp_min(0)
    logits = -_broadcast_scale(scale, q) * sq_dist
    if key_bias is not None:
        logits = logits + key_bias.to(torch.float32)
    return (torch.softmax(logits, dim=-1) @ v.float()).to(v.dtype)


def tiled_rbf_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: torch.Tensor,
    key_bias: torch.Tensor | None = None,
    block_q: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """Kernel-algorithm emulation; result has v's dtype.

    Mirrors spatial_rbf_attention._rbf_attention_kernel: per query tile, the
    query norms stay resident, candidate tiles stream past, distances come
    from one matrix product plus norms, and a running maximum and running
    sum rescale the accumulator so no full Nq x Nk matrix exists at once.
    Products of half-precision inputs accumulate in float32, as on tensor
    cores.
    """
    out_dtype = v.dtype
    qf, kf, vf = q.float(), k.float(), v.float()
    scale4 = _broadcast_scale(scale, q)
    n_q, n_k = q.shape[-2], k.shape[-2]
    bias = None
    if key_bias is not None:
        bias = key_bias.to(torch.float32).expand(*q.shape[:-2], 1, n_k)
    out = torch.empty(
        *q.shape[:-1], v.shape[-1], dtype=torch.float32, device=q.device
    )
    for q_start in range(0, n_q, block_q):
        q_tile = qf[..., q_start : q_start + block_q, :]
        q_norm = (q_tile * q_tile).sum(-1, keepdim=True)
        rows = q_tile.shape[-2]
        running_max = torch.full(
            (*q.shape[:-2], rows, 1), float('-inf'), device=q.device
        )
        running_sum = torch.zeros_like(running_max)
        acc = torch.zeros(*q.shape[:-2], rows, v.shape[-1], device=q.device)
        for k_start in range(0, n_k, block_k):
            k_tile = kf[..., k_start : k_start + block_k, :]
            v_tile = vf[..., k_start : k_start + block_k, :]
            k_norm = (k_tile * k_tile).sum(-1).unsqueeze(-2)
            dot = q_tile @ k_tile.transpose(-1, -2)
            sq_dist = (q_norm + k_norm - 2.0 * dot).clamp_min(0)
            logits = -scale4 * sq_dist
            if bias is not None:
                logits = logits + bias[..., k_start : k_start + block_k]
            new_max = torch.maximum(running_max, logits.amax(-1, keepdim=True))
            # Rows still fully masked keep a -inf maximum; avoid inf - inf.
            safe_max = torch.where(
                torch.isinf(new_max) & (new_max < 0),
                torch.zeros_like(new_max),
                new_max,
            )
            alpha = torch.exp(running_max - safe_max)
            weights = torch.exp(logits - safe_max)
            running_sum = running_sum * alpha + weights.sum(-1, keepdim=True)
            acc = acc * alpha + weights @ v_tile
            running_max = new_max
        out[..., q_start : q_start + rows, :] = acc / running_sum
    return out.to(out_dtype)
