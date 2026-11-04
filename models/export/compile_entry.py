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

"""Compiles a trained checkpoint for serving and writes a release bundle.

  python -m models.export.compile_entry \
      --checkpoint_dir=checkpoints/price_transformer_v1 \
      --out=dist/compiled_model --device=cuda

The model is compiled with torch.compile(backend='vectra', fullgraph=True):
TorchDynamo captures the forward pass as one torch.fx graph, the rewrite
pass replaces every eager RBF-attention region with vectra::rbf_attention,
and TorchInductor generates code for the rest. Compilation fails unless
there are no graph breaks and every layer's attention was rewritten, and
the compiled model must match the eager model (allclose, rtol 1e-3) before
the bundle is written.

torch.compile is a just-in-time compiler, so the bundle holds the verified
inputs to compilation (weights, architecture, serving shape and precision,
toolchain versions); `load_bundle` recompiles at service start-up and
re-runs the same checks before the service accepts traffic.
"""

from collections.abc import Sequence
import dataclasses
import hashlib
import json
import os
import platform
import shutil
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch

from models.price_transformer import model as model_lib
from models.training import checkpoint
from vectra_compiler import backend

FORMAT_VERSION = 1
MANIFEST = 'compile_manifest.json'
# The numerics gate of the design: compiled must match eager at rtol 1e-3.
RTOL = 1e-3
ATOL = 1e-3

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    'checkpoints/price_transformer_v1',
    'Checkpoint written by models.training.train.',
    allow_override=True,
)
_OUT = flags.DEFINE_string(
    'out',
    'dist/compiled_model',
    'Bundle directory to create.',
    allow_override=True,
)
_DEVICE = flags.DEFINE_string(
    'device',
    'cuda' if torch.cuda.is_available() else 'cpu',
    'Serving device.',
    allow_override=True,
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    64,
    'Serving batch size; requests are padded to it.',
    allow_override=True,
)
_BLOCK_DTYPE = flags.DEFINE_enum(
    'block_dtype',
    None,
    ['float32', 'float16', 'bfloat16'],
    'Transformer-block precision (default: float16 on CUDA, else float32).',
    allow_override=True,
)
_NUMERICS_BATCHES = flags.DEFINE_integer(
    'numerics_batches',
    4,
    'Seeded batches compared against eager.',
    allow_override=True,
)


class NumericsError(RuntimeError):
    """The compiled model disagrees with the eager model."""


@dataclasses.dataclass(frozen=True)
class ServingSpec:
    """The shape and precision the model is compiled for."""

    device: str
    batch_size: int
    block_dtype: str

    @property
    def dtype(self) -> torch.dtype:
        return getattr(torch, self.block_dtype)


def default_block_dtype(device: str) -> str:
    return 'float16' if torch.device(device).type == 'cuda' else 'float32'


def toolchain_versions() -> dict[str, str]:
    """Versions that must match between export and serving."""
    versions = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'cuda': str(torch.version.cuda),
    }
    try:
        import triton  # pylint: disable=import-outside-toplevel

        versions['triton'] = triton.__version__
    except ImportError:
        versions['triton'] = 'unavailable'
    return versions


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(
    model: model_lib.PriceTransformer, spec: ServingSpec
) -> model_lib.PriceTransformer:
    """Moves `model` to the serving device and precision (in place)."""
    model = model.to(spec.device).eval()
    if spec.dtype != torch.float32:
        model.set_block_dtype(spec.dtype)
    return model


def max_abs_error(
    compiled,
    eager: torch.nn.Module,
    config: model_lib.ModelConfig,
    spec: ServingSpec,
    batches: int,
    reference: torch.nn.Module | None = None,
) -> tuple[float, float]:
    """Gates compiled-vs-eager numerics on seeded serving-shape batches.

    Returns:
      (worst |compiled - eager|, worst |compiled - reference|), where the
      reference is the float32 eager model (0.0 if `reference` is None).

    Raises:
      NumericsError: Compiled and eager disagree beyond RTOL/ATOL.
    """
    worst, drift = 0.0, 0.0
    with torch.no_grad():
        for seed in range(batches):
            inputs = model_lib.example_inputs(
                config, spec.batch_size, spec.device, seed=1000 + seed
            )
            got = compiled(*inputs).float()
            want = eager(*inputs).float()
            error = (got - want).abs().max().item()
            if not torch.allclose(got, want, rtol=RTOL, atol=ATOL):
                raise NumericsError(
                    f'compiled model differs from eager by up to {error:.3g} '
                    f'(rtol {RTOL}, atol {ATOL}) on seeded batch {seed}'
                )
            worst = max(worst, error)
            if reference is not None:
                drift = max(
                    drift, (got - reference(*inputs).float()).abs().max().item()
                )
    return worst, drift


def compile_model(model: model_lib.PriceTransformer, spec: ServingSpec):
    """Compiles a prepared model for `spec`.

    Returns:
      (compiled model, backend.CompileReport, compile seconds).
    """
    inputs = model_lib.example_inputs(
        model.config, spec.batch_size, spec.device
    )
    start = time.perf_counter()
    compiled, report = backend.compile_for_serving(
        model, tuple(inputs), expected_sites=model.config.num_layers
    )
    return compiled, report, time.perf_counter() - start


def export_bundle(
    checkpoint_dir: str,
    out_dir: str,
    spec: ServingSpec,
    numerics_batches: int = 4,
) -> dict:
    """Compiles, verifies and writes `out_dir`; returns the manifest."""
    eager = prepare(checkpoint.load(checkpoint_dir), spec)
    compiled, report, seconds = compile_model(eager, spec)
    reference = None
    if spec.dtype != torch.float32:
        reference = checkpoint.load(checkpoint_dir, spec.device)
    worst, drift = max_abs_error(
        compiled, eager, eager.config, spec, numerics_batches, reference
    )
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for name in (checkpoint.WEIGHTS, checkpoint.CONFIG):
        shutil.copyfile(
            os.path.join(checkpoint_dir, name), os.path.join(out_dir, name)
        )
        files[name] = _sha256(os.path.join(out_dir, name))
    device_name = (
        torch.cuda.get_device_name(spec.device)
        if torch.device(spec.device).type == 'cuda'
        else platform.processor() or platform.machine()
    )
    manifest = {
        'format_version': FORMAT_VERSION,
        'files': files,
        'model': dataclasses.asdict(eager.config),
        'serving': dataclasses.asdict(spec) | {'device_name': device_name},
        'compile': {
            'backend': backend.BACKEND_NAME,
            'expected_sites': eager.config.num_layers,
            'sites': report.sites,
            'matches': report.matches,
            'graph_count': report.graph_count,
            'graph_break_count': report.graph_break_count,
            'seconds': round(seconds, 3),
        },
        'numerics': {
            'rtol': RTOL,
            'atol': ATOL,
            'batches': numerics_batches,
            'max_abs_error_vs_eager': worst,
            'max_abs_error_vs_float32_eager': drift,
        },
        'versions': toolchain_versions(),
    }
    with open(os.path.join(out_dir, MANIFEST), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')
    return manifest


class ServingModel:
    """A compiled model plus the padding that keeps its shape static.

    The model is compiled for exactly `batch_size` requests (dynamic=False),
    so smaller batches are padded with copies of their last request and
    larger ones are split; padding rows are dropped from the result.
    """

    def __init__(
        self,
        compiled,
        config: model_lib.ModelConfig,
        spec: ServingSpec,
        manifest: dict,
    ):
        self.compiled = compiled
        self.config = config
        self.spec = spec
        self.manifest = manifest

    def predict(self, inputs: model_lib.PricingInputs) -> torch.Tensor:
        """Predicted log prices, [n], for a batch of any size n >= 1."""
        n = inputs.query_embedding.shape[0]
        b = self.spec.batch_size
        outputs = []
        with torch.no_grad():
            for start in range(0, n, b):
                chunk = [
                    t[start : start + b].to(self.spec.device) for t in inputs
                ]
                size = chunk[0].shape[0]
                if size < b:
                    chunk = [
                        torch.cat([t, t[-1:].expand(b - size, *t.shape[1:])])
                        for t in chunk
                    ]
                outputs.append(self.compiled(*chunk)[:size].float())
        return torch.cat(outputs)


def load_bundle(
    bundle_dir: str, device: str | None = None, strict_versions: bool = True
) -> ServingModel:
    """Recompiles a bundle and re-verifies it before serving.

    Args:
      bundle_dir: Directory written by export_bundle.
      device: Overrides the bundle's device (its precision is then the
        default for that device).
      strict_versions: Fail if the toolchain differs from the export's.

    Raises:
      ValueError: Corrupt bundle or toolchain mismatch.
      NumericsError: The recompiled model disagrees with eager.
      backend.PatternNotFoundError, inductor_hook.GraphBreakError: The
        rewrite did not apply.
    """
    with open(os.path.join(bundle_dir, MANIFEST), encoding='utf-8') as f:
        manifest = json.load(f)
    if manifest.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'unsupported bundle format in {bundle_dir}')
    for name, digest in manifest['files'].items():
        if _sha256(os.path.join(bundle_dir, name)) != digest:
            raise ValueError(f'{name} does not match its manifest checksum')
    exported = manifest['versions']
    if strict_versions and exported != toolchain_versions():
        raise ValueError(
            f'toolchain {toolchain_versions()} differs from the export '
            f'toolchain {exported}; re-export the bundle'
        )
    serving = manifest['serving']
    spec = ServingSpec(
        device=device or serving['device'],
        batch_size=serving['batch_size'],
        block_dtype=(
            serving['block_dtype']
            if device is None
            else default_block_dtype(device)
        ),
    )
    model = prepare(checkpoint.load(bundle_dir), spec)
    compiled, report, seconds = compile_model(model, spec)
    max_abs_error(compiled, model, model.config, spec, batches=1)
    logging.info(
        '[vectra] serving model ready: %d site(s) %s in %.1fs',
        report.sites,
        report.matches,
        seconds,
    )
    return ServingModel(compiled, model.config, spec, manifest)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    spec = ServingSpec(
        device=_DEVICE.value,
        batch_size=_BATCH_SIZE.value,
        block_dtype=_BLOCK_DTYPE.value or default_block_dtype(_DEVICE.value),
    )
    manifest = export_bundle(
        _CHECKPOINT_DIR.value, _OUT.value, spec, _NUMERICS_BATCHES.value
    )
    logging.info(
        'wrote %s: %s',
        _OUT.value,
        json.dumps(manifest['compile'], sort_keys=True),
    )


if __name__ == '__main__':
    app.run(main)
