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

"""The fused kernel as a PyTorch operator: torch.ops.vectra.rbf_attention.

Deliverable B4. Registering the kernel with torch.library makes it a
first-class operator that TorchDynamo traces through and TorchInductor calls
from compiled graphs [24]: the fake implementation propagates shapes during
tracing, the CUDA implementation launches the Triton kernel, and the default
implementation evaluates the kernel's arithmetic (Euclidean expansion,
float32 accumulation) with whole-tensor PyTorch operations, so that compiled
models also run, and can be tested, on CPU. The compiler pass of
Workstream C substitutes this operator for the eager pattern [10].

Typical usage:

  out = torch.ops.vectra.rbf_attention(q, k, v, scale, key_bias)
"""

import torch

from vectra_kernels import reference

# Triton is published for Linux x86-64 only (PyTorch 2.7.0 pins
# triton==3.3.0 there), so on other hosts the CUDA path is unavailable and the
# operator keeps its PyTorch implementation. This is the one conditional
# import in the package.
try:
    from vectra_kernels import spatial_rbf_attention
except ImportError:
    spatial_rbf_attention = None

OP_NAME = 'vectra::rbf_attention'


def triton_available() -> bool:
    """Whether the fused Triton path can run on this host."""
    return spatial_rbf_attention is not None and torch.cuda.is_available()


@torch.library.custom_op(OP_NAME, mutates_args=())
def rbf_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: torch.Tensor,
    key_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused RBF attention: softmax(-scale * ||q - k||^2 + key_bias) @ v.

    Args:
      q: [B, H, Nq, D].
      k: [B, H, Nk, D].
      v: [B, H, Nk, Dv].
      scale: Broadcastable to [B, H, 1, 1].
      key_bias: Optional, broadcastable to [B, 1|H, 1, Nk].

    Returns:
      [B, H, Nq, Dv] in v's dtype.
    """
    return reference.matmul_rbf_attention(q, k, v, scale, key_bias)


@rbf_attention.register_kernel('cuda')
def _rbf_attention_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: torch.Tensor,
    key_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if spatial_rbf_attention is None:
        return reference.matmul_rbf_attention(q, k, v, scale, key_bias)
    return spatial_rbf_attention.rbf_attention_forward(q, k, v, scale, key_bias)


@rbf_attention.register_fake
def _rbf_attention_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: torch.Tensor,
    key_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    del k, scale, key_bias  # Only the output shape matters while tracing.
    return q.new_empty((*q.shape[:-1], v.shape[-1]), dtype=v.dtype)
