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

"""The `vectra` TorchDynamo backend and the deployment compile entry.

  compiled = torch.compile(model, backend='vectra')

is TorchInductor with the RBF-attention rewrite installed. Deployment goes
through `compile_for_serving`, which also proves the capture has no graph
breaks and fails loudly when the expected number of attention sites was not
rewritten, instead of silently serving the slow eager path.
"""

from collections.abc import Sequence
import dataclasses

from absl import logging
import torch
from torch._inductor import compile_fx

from vectra_compiler import inductor_hook
from vectra_compiler import pattern
from vectra_compiler import rewrite_pass

BACKEND_NAME = 'vectra'


class PatternNotFoundError(RuntimeError):
    """Fewer attention sites were rewritten than the model contains."""


@dataclasses.dataclass
class CompileReport:
    """Evidence gathered while compiling one model.

    Attributes:
      sites: Attention sites rewritten to the fused operator.
      matches: Per catalog entry.
      graph_count: Graphs TorchDynamo captured.
      graph_break_count: Graph breaks TorchDynamo reported.
    """

    sites: int = 0
    matches: dict[str, int] = dataclasses.field(default_factory=dict)
    graph_count: int = 0
    graph_break_count: int = 0


def _device_type(example_inputs: Sequence[object]) -> str:
    for value in example_inputs:
        if isinstance(value, torch.Tensor):
            return value.device.type
    return 'cpu'


def make_backend(report: CompileReport | None = None):
    """A TorchDynamo backend that records its rewrites in `report`."""

    def backend(
        graph_module: torch.fx.GraphModule,
        example_inputs: Sequence[torch.Tensor],
    ):
        catalog = pattern.load_catalog(_device_type(example_inputs))
        rewrite = rewrite_pass.RbfAttentionRewrite(catalog)
        with inductor_hook.installed(rewrite):
            compiled = compile_fx.compile_fx(graph_module, example_inputs)
        logging.info(
            '[vectra] fused spatial attention: %d site(s) matched -> '
            'vectra::rbf_attention injected %s',
            rewrite.total_matches,
            {name: n for name, n in rewrite.matches.items() if n},
        )
        if report is not None:
            report.sites += rewrite.total_matches
            for name, count in rewrite.matches.items():
                if count:
                    report.matches[name] = report.matches.get(name, 0) + count
        return compiled

    return backend


if BACKEND_NAME not in torch._dynamo.list_backends(exclude_tags=()):  # pylint: disable=protected-access
    torch._dynamo.register_backend(make_backend(), name=BACKEND_NAME)  # pylint: disable=protected-access


def compile_for_serving(
    model: torch.nn.Module,
    example_inputs: Sequence[torch.Tensor],
    expected_sites: int,
) -> tuple[object, CompileReport]:
    """Compiles `model` for serving and proves the rewrite happened.

    Args:
      model: An eval-mode model written in plain PyTorch.
      example_inputs: Positional inputs with the serving shapes.
      expected_sites: Attention regions the model contains (for example,
        its number of transformer layers).

    Returns:
      (compiled model, report). The model has already been compiled for
      `example_inputs`.

    Raises:
      inductor_hook.GraphBreakError: The model breaks into several graphs.
      PatternNotFoundError: Fewer than `expected_sites` regions matched.
    """
    capture = inductor_hook.assert_single_graph(model, *example_inputs)
    report = CompileReport(
        graph_count=capture.graph_count,
        graph_break_count=capture.graph_break_count,
    )
    torch._dynamo.reset()  # pylint: disable=protected-access
    compiled = torch.compile(
        model, backend=make_backend(report), fullgraph=True, dynamic=False
    )
    with torch.no_grad():
        compiled(*example_inputs)
    if report.sites < expected_sites:
        raise PatternNotFoundError(
            f'rewrote {report.sites} of {expected_sites} expected '
            'spatial-attention site(s); refusing to serve the eager path. '
            'See compiler/guides/authoring_for_dynamo.md.'
        )
    return compiled, report
