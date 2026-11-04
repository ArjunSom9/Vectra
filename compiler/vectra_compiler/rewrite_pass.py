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

"""The graph pass: find every eager RBF-attention region and splice in the
fused operator.

Built on torch.fx.subgraph_rewriter.replace_pattern_with_filters [10]: the
matcher walks use-def edges from the template's output, so it finds the
region however the surrounding code is structured and whatever its tensors
are named, and every non-overlapping match is replaced. Matches are
validated by replacement.make_filter before anything is rewritten.
"""

import collections

import torch
from torch import fx
from torch._inductor import custom_graph_pass
from torch._inductor.fx_passes import post_grad
from torch.fx import subgraph_rewriter

from vectra_compiler import pattern
from vectra_compiler import replacement


def _propagate_fake_values(graph: fx.Graph) -> None:
    """Fills meta['val'] for nodes the rewrite created.

    Later Inductor passes read fake-tensor metadata from every node, so the
    new nodes are evaluated under the graph's own fake mode.
    """
    known = [
        node.meta.get('val') for node in graph.nodes if node.op == 'placeholder'
    ]
    fake_mode = torch._guards.detect_fake_mode(known)  # pylint: disable=protected-access
    if fake_mode is None:
        return
    with fake_mode:
        for node in graph.nodes:
            if node.op != 'call_function' or 'val' in node.meta:
                continue
            args = fx.node.map_arg(node.args, lambda n: n.meta['val'])
            kwargs = fx.node.map_arg(node.kwargs, lambda n: n.meta['val'])
            # map_arg keeps the tuple / dict structure of its argument.
            node.meta['val'] = node.target(*args, **kwargs)  # pylint: disable=not-an-iterable,not-a-mapping


def _is_training_graph(graph: fx.Graph) -> bool:
    """True for a joint forward-backward graph (it takes tangents)."""
    return any(
        node.op == 'placeholder' and str(node.target).startswith('tangents')
        for node in graph.nodes
    )


class RbfAttentionRewrite(custom_graph_pass.CustomGraphPass):
    """Replaces eager RBF attention with torch.ops.vectra.rbf_attention.

    Installed as TorchInductor's joint-graph pre-pass. `matches` counts the
    sites rewritten per catalog entry across every graph the pass has seen.
    The fused operator is forward-only, so graphs that also compute
    gradients are left untouched; the pass is idempotent, as Inductor may
    invoke it more than once per graph.

    Attributes:
      matches: Counter of rewrites keyed by catalog entry name.
    """

    def __init__(self, catalog: tuple[pattern.CapturedPattern, ...]):
        """Initializes the pass.

        Args:
          catalog: Captured templates, e.g. from pattern.load_catalog.
        """
        self._catalog = catalog
        self.matches = collections.Counter()

    @property
    def total_matches(self) -> int:
        """Sites rewritten so far, over all catalog entries."""
        return sum(self.matches.values())

    def __call__(self, graph: fx.Graph) -> None:
        if _is_training_graph(graph):
            return
        graph_module = graph.owning_module
        # The templates are free of no-op views and in canonical form; bring
        # the model graph to the same form (both steps are exact).
        post_grad.remove_noop_ops(graph)
        pattern.canonicalize(graph)
        rewritten = 0
        for entry in self._catalog:
            replaced = subgraph_rewriter.replace_pattern_with_filters(
                graph_module,
                entry.graph_module,
                match_filters=[replacement.make_filter(entry.spec)],
                ignore_literals=True,
                replacement_callback=replacement.make_replacement(entry.spec),
            )
            self.matches[entry.name] += len(replaced)
            rewritten += len(replaced)
        if rewritten:
            graph.eliminate_dead_code()
            _propagate_fake_values(graph)
            graph.lint()

    def uuid(self) -> bytes:
        """Cache key: changes whenever the pass or its templates change."""
        return custom_graph_pass.get_hash_for_files(
            (__file__, pattern.__file__, replacement.__file__)
        )
