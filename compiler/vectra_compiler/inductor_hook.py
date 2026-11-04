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

"""Where the pass hooks into the compilation pipeline, and capture checks.

TorchDynamo hooks CPython's frame evaluation API (PEP 523 [34]) to capture
the model as a torch.fx graph [9] [24]; AOTAutograd lowers it to normalized
ATen, and TorchInductor runs its graph passes before code generation. The
rewrite is installed as Inductor's joint-graph pre-pass: the first hook that
sees the decomposed ATen graph, and the stage at which Inductor performs its
own attention fusion (fx_passes/fuse_attention.py). Running there, before
Inductor's algebraic rewrites (for example the sign trick it applies to
softmax with a temperature), means one canonical template per formulation
covers every spelling of it.
"""

import contextlib
import dataclasses

import torch
from torch._inductor import config as inductor_config

from vectra_compiler import rewrite_pass


class GraphBreakError(RuntimeError):
    """The model does not compile to a single graph."""


@contextlib.contextmanager
def installed(rewrite: rewrite_pass.RbfAttentionRewrite):
    """Installs `rewrite` as Inductor's joint-graph pre-pass for the block.

    The compile caches are disabled inside the block (the AOTAutograd cache
    requires the FX graph cache), so the pass always runs and its match
    counts are evidence for the compile at hand.
    """
    with inductor_config.patch(
        joint_custom_pre_pass=rewrite, fx_graph_cache=False
    ):
        yield


@dataclasses.dataclass(frozen=True)
class CaptureReport:
    """TorchDynamo's capture summary for one call.

    Attributes:
      graph_count: Graphs captured.
      graph_break_count: Graph breaks, counted from their recorded reasons.
        (ExplainOutput.graph_break_count is graph_count - 1, which misses a
        break whose resumed code traces no tensor operations.)
      break_reasons: Why each break happened.
    """

    graph_count: int
    graph_break_count: int
    break_reasons: tuple[str, ...]


def explain(model: torch.nn.Module, *inputs) -> CaptureReport:
    """Runs torch._dynamo.explain over one call (deliverable C1's check)."""
    explanation = torch._dynamo.explain(model)(*inputs)  # pylint: disable=protected-access
    reasons = tuple(str(r.reason) for r in explanation.break_reasons)
    return CaptureReport(
        graph_count=explanation.graph_count,
        graph_break_count=max(explanation.graph_break_count, len(reasons)),
        break_reasons=reasons,
    )


def assert_single_graph(model: torch.nn.Module, *inputs) -> CaptureReport:
    """Raises GraphBreakError unless the call captures as one graph.

    A graph break inside the spatial-scoring region would hide the pattern
    from the rewrite, so the production model must have none.
    """
    report = explain(model, *inputs)
    if report.graph_break_count or report.graph_count != 1:
        raise GraphBreakError(
            f'{report.graph_count} graph(s), {report.graph_break_count} '
            'graph break(s): ' + '; '.join(report.break_reasons)
        )
    return report
