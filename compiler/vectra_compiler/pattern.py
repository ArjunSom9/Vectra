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

"""The pattern catalog: eager formulations of RBF attention to replace.

Each pattern is ordinary PyTorch reproducing the spatial scorer's operator
chain (subtract, square, sum over features, negate-scale, exponentiate and
normalize, weight the values) [10]. The pass runs on the ATen graph that
AOTAutograd hands to TorchInductor [24], at Inductor's joint-graph pre-pass:
after decomposition, so method/function/operator spellings of the same
computation have all normalized to the same ATen ops, and before Inductor's
own algebraic rewrites (such as its softmax-with-temperature rewrite), which
would otherwise make equivalent spellings diverge again. To match exactly
that form, each pattern is captured by running it through the same Inductor
front end and stopping at the same hook (`capture`), rather than by tracing
it independently: the pattern then changes in lockstep with Inductor.

The captured graph is cut at the value operand of the final batched matrix
multiply, which becomes a wildcard input. Whether V needs a contiguity copy
before the multiply depends on how the model laid V out in memory, not on
what it computes, so the cut keeps layout out of the match.

Algebraically distinct formulations are separate catalog entries:
coefficient as divisor (-d / (2 sigma^2)) or multiplier (-d * gamma), with or
without an additive key bias (padding mask), normalized by torch.softmax or
by explicit exp / sum. Equivalent spellings within an entry are reduced to
one canonical graph before matching, using only transformations that are
exact in IEEE-754 arithmetic: squares written as x * x become x.pow(2), and
negations move onto the distance operand (-(d * c), d * (-c), (-c) * d and
d * -0.5 all become (-d) * c). Python-number coefficients bind to the
coefficient wildcard. See compiler/guides/pattern_registry.md.
"""

from collections.abc import Callable
import copy
import dataclasses
import functools
import itertools

import numbers

import torch
from torch import fx
from torch._inductor import config as inductor_config
from torch._inductor import compile_fx
from torch._inductor.fx_passes import post_grad

aten = torch.ops.aten

# Placeholder names, in pattern-graph order, after the value cut.
Q, K, SCALE, BIAS, V_BMM = 'q', 'k', 'scale', 'bias', 'v_bmm'


@dataclasses.dataclass(frozen=True)
class PatternSpec:
    """One catalog entry.

    Attributes:
      coefficient: 'divisor' (-d / c) or 'multiplier' (-d * c).
      has_bias: Whether an additive key bias precedes normalization.
      normalization: 'softmax' (torch.softmax) or 'exp_sum' (exp / sum).
    """

    coefficient: str
    has_bias: bool
    normalization: str

    @property
    def name(self) -> str:
        bias = '_bias' if self.has_bias else ''
        return f'rbf_{self.coefficient}{bias}_{self.normalization}'

    @property
    def placeholder_names(self) -> tuple[str, ...]:
        if self.has_bias:
            return (Q, K, SCALE, BIAS, V_BMM)
        return (Q, K, SCALE, V_BMM)

    def eager_fn(self) -> Callable[..., torch.Tensor]:
        """The eager PyTorch formulation of this entry."""

        def logits(q, k, c, bias):
            sq_dist = (q.unsqueeze(-2) - k.unsqueeze(-3)).pow(2).sum(-1)
            scaled = (
                -sq_dist / c if self.coefficient == 'divisor' else -sq_dist * c
            )
            return scaled + bias if bias is not None else scaled

        def normalize(x):
            if self.normalization == 'softmax':
                return torch.softmax(x, dim=-1)
            e = torch.exp(x)
            return e / e.sum(-1, keepdim=True)

        def with_bias(q, k, c, bias, v):
            return normalize(logits(q, k, c, bias)) @ v

        def without_bias(q, k, c, v):
            return normalize(logits(q, k, c, None)) @ v

        return with_bias if self.has_bias else without_bias


CATALOG = tuple(
    PatternSpec(coefficient, has_bias, normalization)
    for has_bias, coefficient, normalization in itertools.product(
        (True, False), ('divisor', 'multiplier'), ('softmax', 'exp_sum')
    )
)


# Half-precision softmax decomposes with float32 upcasts, so half-precision
# blocks get their own templates. float16 and bfloat16 decompose identically
# (dtype literals are wildcards), so one half-precision set covers both.
DTYPES = (torch.float32, torch.float16)


@dataclasses.dataclass(frozen=True)
class CapturedPattern:
    """A catalog entry and its template graph for one dtype (None: any)."""

    spec: PatternSpec
    dtype: torch.dtype | None
    graph_module: fx.GraphModule

    @property
    def name(self) -> str:
        if self.dtype is None:
            precision = 'any'
        elif self.dtype == torch.float32:
            precision = 'fp32'
        else:
            precision = 'half'
        return f'{self.spec.name}[{precision}]'


class _CaptureDone(Exception):
    """Stops Inductor once the joint graph has been captured."""


def canonicalize_squares(graph: fx.Graph) -> int:
    """Rewrites mul(x, x) as pow(x, 2) in place; returns the count."""
    count = 0
    for node in list(graph.nodes):
        if (
            node.op == 'call_function'
            and node.target is aten.mul.Tensor
            and len(node.args) == 2
            and isinstance(node.args[0], fx.Node)
            and node.args[0] is node.args[1]
            and not node.kwargs
        ):
            with graph.inserting_after(node):
                square = graph.call_function(
                    aten.pow.Tensor_Scalar, (node.args[0], 2)
                )
            square.meta = dict(node.meta)
            node.replace_all_uses_with(square)
            graph.erase_node(node)
            count += 1
    return count


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _numel(node) -> int:
    val = node.meta.get('val') if isinstance(node, fx.Node) else None
    return val.numel() if isinstance(val, torch.Tensor) else 1


def _single_use(node: fx.Node) -> bool:
    return len(node.users) == 1


def _negate(graph: fx.Graph, before: fx.Node, value: fx.Node) -> fx.Node:
    """Returns neg(value), cancelling an existing negation."""
    if value.op == 'call_function' and value.target is aten.neg.default:
        return value.args[0]
    with graph.inserting_before(before):
        node = graph.call_function(aten.neg.default, (value,))
    node.meta = dict(value.meta)
    return node


# Softmax and exp / sum normalization both consume their logits through
# these; sign canonicalization is confined to the pointwise chains feeding
# them so that the rest of the graph is left exactly as Inductor expects it.
_LOGIT_SINKS = (aten.exp.default, aten.amax.default)
_LOGIT_OPS = (
    aten.neg.default,
    aten.mul.Tensor,
    aten.div.Tensor,
    aten.add.Tensor,
    aten.sub.Tensor,
)


def _logit_nodes(graph: fx.Graph) -> set[fx.Node]:
    """mul/div/add/sub/neg nodes in pointwise chains feeding exp or amax."""
    stack = [
        arg
        for node in graph.nodes
        if node.op == 'call_function' and node.target in _LOGIT_SINKS
        for arg in node.args
        if isinstance(arg, fx.Node)
    ]
    found = set()
    while stack:
        node = stack.pop()
        if (
            node in found
            or node.op != 'call_function'
            or node.target not in _LOGIT_OPS
        ):
            continue
        found.add(node)
        stack.extend(a for a in node.args if isinstance(a, fx.Node))
    return found


def _canonical_sign(graph: fx.Graph, node: fx.Node) -> bool:
    """Applies one exact sign rule at `node`; returns whether it changed."""
    if node.op != 'call_function' or node.kwargs or not node.users:
        return False
    target, args = node.target, node.args
    if target is aten.neg.default and isinstance(args[0], fx.Node):
        inner = args[0]
        # neg(neg(a)) -> a
        if inner.target is aten.neg.default:
            node.replace_all_uses_with(inner.args[0])
            return True
        # neg(a * b) -> (-a) * b and neg(a / b) -> (-a) / b
        if (
            inner.op == 'call_function'
            and _single_use(inner)
            and inner.target in (aten.mul.Tensor, aten.div.Tensor)
            and isinstance(inner.args[0], fx.Node)
            and not inner.kwargs
        ):
            inner.args = (_negate(graph, inner, inner.args[0]), inner.args[1])
            node.replace_all_uses_with(inner)
            return True
        return False
    if target not in (aten.mul.Tensor, aten.div.Tensor) or len(args) != 2:
        return False
    a, b = args
    if not isinstance(a, fx.Node):
        return False
    # Commutative multiply: the larger operand (the distances) goes first.
    if (
        target is aten.mul.Tensor
        and isinstance(b, fx.Node)
        and _numel(b) > _numel(a)
    ):
        node.args = (b, a)
        return True
    # a * (-b) -> (-a) * b and a / (-b) -> (-a) / b
    if (
        isinstance(b, fx.Node)
        and b.op == 'call_function'
        and b.target is aten.neg.default
    ):
        node.args = (_negate(graph, node, a), b.args[0])
        return True
    # a * -0.5 -> (-a) * 0.5 and a / -2 -> (-a) / 2
    if _is_number(b) and b < 0:
        node.args = (_negate(graph, node, a), -b)
        return True
    return False


def canonicalize_signs(graph: fx.Graph) -> int:
    """Moves negations onto the first operand of mul/div chains, in place.

    Every rule is exact in IEEE-754 arithmetic (negation commutes with
    rounding and multiplication is commutative), so this never changes
    numerics, even for graphs that then match nothing.

    Returns:
      The number of rewrites applied.
    """
    count = 0
    changed = True
    while changed:
        changed = False
        logit = _logit_nodes(graph)
        for node in [n for n in graph.nodes if n in logit]:
            if _canonical_sign(graph, node):
                changed = True
                count += 1
        graph.eliminate_dead_code()
    return count


def canonicalize(graph: fx.Graph) -> None:
    """The normalization applied to templates and model graphs alike."""
    canonicalize_squares(graph)
    canonicalize_signs(graph)


def _cut_at_value_operand(graph_module: fx.GraphModule) -> fx.GraphModule:
    """Promotes the final bmm's second operand to the `v_bmm` placeholder."""
    graph = graph_module.graph
    post_grad.remove_noop_ops(graph)
    bmm = next(n for n in graph.nodes if n.target is aten.bmm.default)
    placeholders = [n for n in graph.nodes if n.op == 'placeholder']
    value_input = placeholders[-1]  # The eager fn takes v last.
    with graph.inserting_after(placeholders[-1]):
        v_bmm = graph.placeholder(V_BMM)
    bmm.args = (bmm.args[0], v_bmm)
    graph.eliminate_dead_code()
    if value_input.users:
        raise RuntimeError('the value input is used outside the final bmm')
    graph.erase_node(value_input)
    canonicalize(graph)
    graph.lint()
    graph_module.recompile()
    return graph_module


def _example_inputs(
    spec: PatternSpec, device: torch.device, dtype: torch.dtype
) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    shape = (2, 3, 5, 8)
    q = torch.randn(shape, generator=gen)
    k = torch.randn(2, 3, 7, 8, generator=gen)
    v = torch.randn(2, 3, 7, 8, generator=gen)
    coefficient = torch.rand(1, 3, 1, 1, generator=gen) + 0.5
    inputs = [q, k, coefficient]
    if spec.has_bias:
        inputs.append(torch.zeros(2, 1, 1, 7))
    inputs.append(v)
    return [t.to(device=device, dtype=dtype) for t in inputs]


def capture(
    spec: PatternSpec, device: torch.device, dtype: torch.dtype
) -> fx.GraphModule:
    """Runs `spec` through Inductor's front end; returns its joint graph.

    The capture is isolated from any surrounding TorchDynamo compilation, so
    the template always has static shapes and its own fake-tensor mode.
    """
    traced = fx.symbolic_trace(spec.eager_fn())
    captured = {}

    def grab(graph: fx.Graph) -> None:
        captured['graph_module'] = copy.deepcopy(graph.owning_module)
        raise _CaptureDone()

    with torch._guards.tracing(None), torch.no_grad():  # pylint: disable=protected-access
        with inductor_config.patch(
            joint_custom_pre_pass=grab, fx_graph_cache=False
        ):
            try:
                compile_fx.compile_fx(
                    traced, _example_inputs(spec, device, dtype)
                )
            except Exception:  # pylint: disable=broad-except
                # Inductor wraps the stop signal; only a missing capture is
                # an actual failure.
                if 'graph_module' not in captured:
                    raise
    return _cut_at_value_operand(captured['graph_module'])


@functools.cache
def load_catalog(device_type: str) -> tuple[CapturedPattern, ...]:
    """Captured templates for every catalog entry and dtype on a device.

    A half-precision template identical to its float32 counterpart (entries
    without the float32-upcasting softmax) is kept once, as dtype None.
    Cached per process; the decorator is a pure memoization of an
    expensive, deterministic computation.
    """
    device = torch.device(device_type)
    entries = []
    for spec in CATALOG:
        graphs = {dtype: capture(spec, device, dtype) for dtype in DTYPES}
        if len({gm.code for gm in graphs.values()}) == 1:
            entries.append(CapturedPattern(spec, None, graphs[DTYPES[0]]))
        else:
            entries.extend(
                CapturedPattern(spec, dtype, gm) for dtype, gm in graphs.items()
            )
    return tuple(entries)
