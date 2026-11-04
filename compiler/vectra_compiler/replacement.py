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

"""Match validation and the replacement subgraph.

The subgraph rewriter matches by use-def topology with literals ignored
[10], so that one template serves every tensor shape. Everything that
literals would have pinned down is re-checked here against the matched
nodes and their fake-tensor metadata, so semantically different look-alikes
(cubes instead of squares, reductions over another axis, a scale that varies
per key, ...) are rejected: the risk register's "match filter on op targets,
not names".

The replacement has the template's input signature and a single call to the
fused operator, torch.ops.vectra.rbf_attention.
"""

from collections.abc import Callable, Sequence
import numbers

import torch
from torch import fx
from torch.fx.passes.utils import matcher_utils

from vectra_compiler import pattern

# Imported for its side effect: registering torch.ops.vectra.rbf_attention.
from vectra_kernels import custom_ops  # noqa: F401  # pylint: disable=unused-import

aten = torch.ops.aten
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

MatchFilter = Callable[[matcher_utils.InternalMatch, fx.Graph, fx.Graph], bool]
ReplacementCallback = Callable[
    [matcher_utils.InternalMatch, fx.Graph, fx.Graph], fx.Graph
]


def _val(value) -> torch.Tensor | None:
    if isinstance(value, fx.Node):
        val = value.meta.get('val')
        return val if isinstance(val, torch.Tensor) else None
    return None


def _normalized_dims(dims: Sequence[int], rank: int) -> list[int]:
    return sorted(d % rank for d in dims)


def _bound(
    spec: pattern.PatternSpec, match: matcher_utils.InternalMatch
) -> dict:
    """Graph values bound to each template placeholder, by name."""
    return dict(
        zip(spec.placeholder_names, match.placeholder_nodes, strict=True)
    )


def _broadcasts_per_head(shape: Sequence[int], lead: Sequence[int]) -> bool:
    """True for a coefficient broadcastable to [*lead, 1, 1]."""
    if len(shape) > 4:
        return False
    padded = [1] * (4 - len(shape)) + list(shape)
    return (
        padded[2] == 1
        and padded[3] == 1
        and all(
            p in (1, size) for p, size in zip(padded[:2], lead, strict=True)
        )
    )


def _ops_are_canonical(match: matcher_utils.InternalMatch) -> bool:
    """Literal checks: squares and reduction axes."""
    for pattern_node, node in match.nodes_map.items():
        if pattern_node.op != 'call_function' or not isinstance(node, fx.Node):
            continue
        if node.target is aten.pow.Tensor_Scalar and node.args[1] != 2:
            return False
        if node.target in (aten.sum.dim_IntList, aten.amax.default):
            source = _val(node.args[0])
            if source is None or len(node.args) < 2:
                return False
            if _normalized_dims(node.args[1], source.dim()) != [
                source.dim() - 1
            ]:
                return False
            want_keepdim = (
                pattern_node.args[2] if len(pattern_node.args) > 2 else False
            )
            keepdim = node.args[2] if len(node.args) > 2 else False
            if bool(keepdim) != bool(want_keepdim):
                return False
    return True


def make_filter(spec: pattern.PatternSpec) -> MatchFilter:
    """Builds the semantic validation for matches of `spec`."""

    def check(
        match: matcher_utils.InternalMatch,
        original_graph: fx.Graph,
        pattern_graph: fx.Graph,
    ) -> bool:
        del original_graph, pattern_graph  # Only the match is inspected.
        bound = _bound(spec, match)
        q, k, v_bmm = (
            _val(bound[pattern.Q]),
            _val(bound[pattern.K]),
            _val(bound[pattern.V_BMM]),
        )
        if q is None or k is None or v_bmm is None:
            return False
        if q.dim() != 4 or k.dim() != 4 or v_bmm.dim() != 3:
            return False
        lead, n_q, n_k = tuple(q.shape[:2]), q.shape[2], k.shape[2]
        if (
            tuple(k.shape[:2]) != lead
            or q.shape[3] != k.shape[3]
            or v_bmm.shape[0] != lead[0] * lead[1]
            or v_bmm.shape[1] != n_k
        ):
            return False
        if q.dtype not in _SUPPORTED_DTYPES or not (
            q.dtype == k.dtype == v_bmm.dtype
        ):
            return False
        # The difference tensor must pair every query with every key.
        for pattern_node, node in match.nodes_map.items():
            if (
                pattern_node.op == 'call_function'
                and pattern_node.target is aten.sub.Tensor
                and all(
                    isinstance(a, fx.Node)
                    and a.target is aten.unsqueeze.default
                    for a in pattern_node.args
                )
            ):
                diff = _val(node)
                if diff is None or tuple(diff.shape) != (
                    *lead,
                    n_q,
                    n_k,
                    q.shape[3],
                ):
                    return False
        if not _ops_are_canonical(match):
            return False
        coefficient = bound[pattern.SCALE]
        if isinstance(coefficient, fx.Node):
            value = _val(coefficient)
            if value is None or not _broadcasts_per_head(value.shape, lead):
                return False
        elif not isinstance(coefficient, numbers.Real) or (
            spec.coefficient == 'divisor' and coefficient == 0
        ):
            return False
        if spec.has_bias:
            bias = _val(bound[pattern.BIAS])
            if bias is None or bias.dim() > 4:
                return False
            padded = [1] * (4 - bias.dim()) + list(bias.shape)
            if (
                padded[2] != 1
                or padded[3] not in (1, n_k)
                or any(
                    p not in (1, size)
                    for p, size in zip(padded[:2], lead, strict=True)
                )
            ):
                return False
        out = _val(match.returning_nodes[0])
        return out is not None and tuple(out.shape) == (
            *lead,
            n_q,
            v_bmm.shape[2],
        )

    return check


def make_replacement(spec: pattern.PatternSpec) -> ReplacementCallback:
    """Builds the replacement-graph factory for matches of `spec`."""

    def build(
        match: matcher_utils.InternalMatch,
        original_graph: fx.Graph,
        pattern_graph: fx.Graph,
    ) -> fx.Graph:
        del original_graph, pattern_graph  # The template fixes the inputs.
        bound = _bound(spec, match)
        q, v_bmm = _val(bound[pattern.Q]), _val(bound[pattern.V_BMM])
        graph = fx.Graph()
        inputs = {
            name: graph.placeholder(name) for name in spec.placeholder_names
        }
        v = graph.call_function(
            aten.reshape.default,
            (
                inputs[pattern.V_BMM],
                [*q.shape[:2], v_bmm.shape[1], v_bmm.shape[2]],
            ),
        )
        coefficient = bound[pattern.SCALE]
        if isinstance(coefficient, fx.Node):
            scale = inputs[pattern.SCALE]
            if spec.coefficient == 'divisor':
                scale = graph.call_function(aten.reciprocal.default, (scale,))
        else:
            value = float(coefficient)
            scale = graph.call_function(
                aten.scalar_tensor.default,
                (1.0 / value if spec.coefficient == 'divisor' else value,),
                {'dtype': torch.float32, 'device': q.device},
            )
        bias = inputs[pattern.BIAS] if spec.has_bias else None
        out = graph.call_function(
            torch.ops.vectra.rbf_attention.default,
            (inputs[pattern.Q], inputs[pattern.K], v, scale, bias),
        )
        graph.output(out)
        return graph

    return build
