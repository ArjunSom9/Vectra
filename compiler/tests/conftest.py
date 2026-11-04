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

"""Helpers for the compiler suites: compile a function, count rewrites."""

import pytest
import torch

from vectra_compiler import backend


def compile_and_run(fn, *inputs):
    """Compiles `fn` with the vectra backend and runs it once.

    Returns:
      (compiled output, backend.CompileReport).
    """
    torch._dynamo.reset()  # pylint: disable=protected-access
    report = backend.CompileReport()
    compiled = torch.compile(
        fn, backend=backend.make_backend(report), fullgraph=True, dynamic=False
    )
    with torch.no_grad():
        out = compiled(*inputs)
    return out, report


def attention_inputs(
    batch=2,
    heads=3,
    n_q=6,
    n_k=9,
    dim=8,
    dim_v=None,
    dtype=torch.float32,
    seed=0,
):
    """Random q [B,H,Nq,D], k [B,H,Nk,D], v [B,H,Nk,Dv] and a per-head scale."""
    gen = torch.Generator().manual_seed(seed)
    dim_v = dim if dim_v is None else dim_v
    q = torch.randn(batch, heads, n_q, dim, generator=gen)
    k = torch.randn(batch, heads, n_k, dim, generator=gen)
    v = torch.randn(batch, heads, n_k, dim_v, generator=gen)
    sigma = torch.rand(1, heads, 1, 1, generator=gen) + 1.0
    return tuple(t.to(dtype) for t in (q, k, v, sigma))


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    torch._dynamo.reset()  # pylint: disable=protected-access
    yield
    torch._dynamo.reset()  # pylint: disable=protected-access
