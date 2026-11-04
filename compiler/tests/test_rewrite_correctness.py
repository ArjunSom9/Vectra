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

"""Deliverable C2 (c), per catalog entry: rewritten code computes the same.

Every catalog formulation, at awkward shapes, with tensor and Python-number
coefficients and in reduced precision, gives the eager result after the
rewrite; graphs that need gradients are left alone; and the canonicalization
the matcher relies on is bit-exact.
"""

import pytest
import torch
from torch import fx

from compiler.tests import conftest
from vectra_compiler import backend
from vectra_compiler import pattern
from vectra_compiler import rewrite_pass

aten = torch.ops.aten


def _inputs(spec, dtype=torch.float32, **shape):
    q, k, v, sigma = conftest.attention_inputs(dtype=dtype, **shape)
    coefficient = (
        2 * sigma**2 if spec.coefficient == 'divisor' else 1 / (2 * sigma**2)
    )
    inputs = [q, k, coefficient]
    if spec.has_bias:
        bias = torch.zeros(q.shape[0], 1, 1, k.shape[2], dtype=dtype)
        bias[0, ..., -1] = float('-inf')  # Mask one key of the first batch.
        inputs.append(bias)
    return inputs + [v]


@pytest.mark.parametrize('spec', pattern.CATALOG, ids=lambda s: s.name)
@pytest.mark.parametrize(
    'shape',
    [
        {},
        {'batch': 1, 'heads': 1, 'n_q': 1, 'n_k': 2, 'dim': 1, 'dim_v': 2},
        {'n_q': 5, 'n_k': 37, 'dim': 3, 'dim_v': 7},
        {'n_q': 17, 'n_k': 1, 'dim': 16, 'dim_v': 2},
    ],
    ids=['basic', 'tiny', 'ragged', 'single_key'],
)
def test_each_formulation_is_rewritten_exactly(spec, shape):
    if spec.has_bias and shape.get('n_k', 9) < 2:
        pytest.skip('masking the only key leaves nothing to attend to')
    fn = spec.eager_fn()
    inputs = _inputs(spec, **shape)
    out, report = conftest.compile_and_run(fn, *inputs)
    assert report.sites == 1, report.matches
    assert len(report.matches) == 1
    name = next(iter(report.matches))
    assert name.startswith(spec.name + '[')
    torch.testing.assert_close(out, fn(*inputs), rtol=1e-4, atol=1e-5)


def test_degenerate_cpu_matmul_keeps_the_eager_path():
    # With one query and one value column, Inductor's CPU decomposition turns
    # the batched matmul into a multiply-and-sum, so there is no bmm to
    # anchor the template on; the region compiles as ordinary code instead.
    spec = pattern.CATALOG[0]
    fn = spec.eager_fn()
    inputs = _inputs(spec, batch=1, heads=1, n_q=1, n_k=3, dim=2, dim_v=1)
    out, report = conftest.compile_and_run(fn, *inputs)
    assert report.sites == 0
    torch.testing.assert_close(out, fn(*inputs), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('spec', pattern.CATALOG, ids=lambda s: s.name)
def test_python_number_coefficients(spec):
    q, k, v, _ = conftest.attention_inputs()
    fn = spec.eager_fn()
    coefficient = 4.0 if spec.coefficient == 'divisor' else 0.25
    extra = [torch.zeros(2, 1, 1, k.shape[2])] if spec.has_bias else []

    def with_literal(q, k, *rest):
        return fn(q, k, coefficient, *rest)

    out, report = conftest.compile_and_run(with_literal, q, k, *extra, v)
    assert report.sites == 1, report.matches
    torch.testing.assert_close(
        out, with_literal(q, k, *extra, v), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize(
    'dtype', [torch.float16, torch.bfloat16], ids=['float16', 'bfloat16']
)
@pytest.mark.parametrize('spec', pattern.CATALOG, ids=lambda s: s.name)
def test_reduced_precision(spec, dtype):
    fn = spec.eager_fn()
    inputs = _inputs(spec, dtype=dtype)
    out, report = conftest.compile_and_run(fn, *inputs)
    assert report.sites == 1, report.matches
    assert len(report.matches) == 1
    name = next(iter(report.matches))
    assert name.endswith('[half]') or name.endswith('[any]')
    assert out.dtype == dtype
    reference = fn(*(t.float() for t in inputs))
    tolerance = 1e-2 if dtype == torch.float16 else 4e-2
    torch.testing.assert_close(
        out.float(), reference, rtol=tolerance, atol=tolerance
    )


def test_graphs_with_gradients_are_left_alone():
    q, k, v, sigma = (t.requires_grad_() for t in conftest.attention_inputs())
    fn = pattern.CATALOG[0].eager_fn()
    bias = torch.zeros(2, 1, 1, k.shape[2])

    torch._dynamo.reset()  # pylint: disable=protected-access
    report = backend.CompileReport()
    compiled = torch.compile(fn, backend=backend.make_backend(report))
    out = compiled(q, k, 2 * sigma**2, bias, v)
    out.square().sum().backward()
    assert report.sites == 0
    grads = [t.grad.clone() for t in (q, k, v, sigma)]
    for t in (q, k, v, sigma):
        t.grad = None
    fn(q, k, 2 * sigma**2, bias, v).square().sum().backward()
    for got, t in zip(grads, (q, k, v, sigma), strict=True):
        torch.testing.assert_close(got, t.grad, rtol=1e-4, atol=1e-5)


class _RecordValues(fx.Interpreter):
    """Runs a graph and stores each node's value in meta['val']."""

    def run_node(self, n):
        value = super().run_node(n)
        n.meta['val'] = value
        return value


def _signed_graph() -> fx.GraphModule:
    """Logit chains with negations in every position, feeding exp."""
    graph = fx.Graph()
    a, b, c = (graph.placeholder(name) for name in 'abc')
    outs = []
    for expression in (
        lambda: graph.call_function(
            aten.neg.default, (graph.call_function(aten.mul.Tensor, (a, c)),)
        ),
        lambda: graph.call_function(
            aten.mul.Tensor, (a, graph.call_function(aten.neg.default, (c,)))
        ),
        lambda: graph.call_function(
            aten.mul.Tensor, (c, graph.call_function(aten.neg.default, (a,)))
        ),
        lambda: graph.call_function(aten.mul.Tensor, (a, -0.375)),
        lambda: graph.call_function(
            aten.div.Tensor, (a, graph.call_function(aten.neg.default, (c,)))
        ),
        lambda: graph.call_function(
            aten.neg.default, (graph.call_function(aten.div.Tensor, (b, -3.0)),)
        ),
    ):
        outs.append(graph.call_function(aten.exp.default, (expression(),)))
    graph.output(tuple(outs))
    return fx.GraphModule(torch.nn.Module(), graph)


def test_sign_canonicalization_is_bit_exact():
    gen = torch.Generator().manual_seed(3)
    a = torch.randn(2, 3, 4, 5, generator=gen)
    b = torch.randn(2, 3, 4, 5, generator=gen)
    c = torch.rand(1, 3, 1, 1, generator=gen) + 0.1
    graph_module = _signed_graph()
    before = _RecordValues(graph_module).run(a, b, c)
    assert pattern.canonicalize_signs(graph_module.graph) > 0
    graph_module.recompile()
    after = graph_module(a, b, c)
    for x, y in zip(before, after, strict=True):
        assert torch.equal(x, y)
    # Every product and quotient now has its negation on the first operand
    # and a non-negative literal, if any.
    for node in graph_module.graph.nodes:
        if node.target in (aten.mul.Tensor, aten.div.Tensor):
            first, second = node.args
            assert not (
                isinstance(second, fx.Node)
                and second.target is aten.neg.default
            )
            assert not (isinstance(second, float) and second < 0)
            assert first.target is aten.neg.default or first.op == (
                'placeholder'
            )


def test_canonicalization_leaves_other_arithmetic_alone():
    # Arithmetic that does not feed exp / amax (softmax logits) is untouched.
    graph = fx.Graph()
    a, c = graph.placeholder('a'), graph.placeholder('c')
    product = graph.call_function(aten.mul.Tensor, (c, -2.0))
    graph.output(graph.call_function(aten.add.Tensor, (product, a)))
    before = str(graph)
    assert pattern.canonicalize_signs(graph) == 0
    assert str(graph) == before


def test_pass_cache_key_is_stable():
    catalog = pattern.load_catalog('cpu')
    first = rewrite_pass.RbfAttentionRewrite(catalog).uuid()
    assert first == rewrite_pass.RbfAttentionRewrite(catalog).uuid()
    assert isinstance(first, bytes) and first
