# 08 — Capture and the Deployment Compile

| | |
|---|---|
| **Status** | Accepted |
| **Synthesizes** | Design document §7.1 (graph capture with TorchDynamo and torch.fx) |
| **Code** | `compiler/vectra_compiler/{backend,inductor_hook}.py`, `models/export/compile_entry.py`, `models/price_transformer/model.py` |
| **Tests** | `compiler/tests/test_e2e_compile.py`, `models/tests/test_model.py`, `models/tests/test_train_and_export.py` |
| **Guide** | [`compiler/guides/authoring_for_dynamo.md`](../../compiler/guides/authoring_for_dynamo.md) |

## 1. The pipeline

```
model.forward ──TorchDynamo (PEP 523 frame evaluation [34])──▶ torch.fx graph
              ──AOTAutograd + Inductor decompositions──▶ ATen graph
              ──joint-graph pre-pass: RbfAttentionRewrite ([09])──▶
              ──Inductor graph passes, lowering, code generation──▶ kernels
```

TorchDynamo intercepts the bytecode of `forward`, symbolically evaluates
it, and records every tensor operation into a `torch.fx.GraphModule`
[9] [24]. AOTAutograd turns the graph into normalized ATen operators using
Inductor's decomposition table, so method, function and operator spellings
of the same computation become the same operators. Inductor then runs its
graph passes and generates code (Triton on GPU, C++ on CPU). The rewrite of
[09](09-subgraph-rewrite.md) runs at the first hook that sees the decomposed
graph.

Training and evaluation never go through this path; they run eagerly.

## 2. Graph breaks, and why there must be none

When Dynamo meets code it cannot trace (a Python branch on a tensor value,
`.item()`, printing, NumPy), it ends the graph, runs that code in Python,
and starts a new graph after it. A break inside an attention layer splits
the pattern across graphs, and the rewrite cannot see it. So the deployment
compile requires exactly one graph.

The price model's one data-dependent decision is in `trim_to_radius`:
whether every request has a comparable inside its radius. That choice is a
`torch.cond` [35], not a Python `if`. Its two branches (the in-radius mask,
or a fallback to the nearest `min_candidates`) are both traced into the
graph. `compiler/tests/test_e2e_compile.py` checks that compiled and eager
agree on both branches.

**Counting breaks.** In PyTorch 2.7.0, `torch._dynamo.explain(...)
.graph_break_count` is computed as `graph_count − 1`. That misses a break
whose resumed code traces no tensor operation, for example a Python branch
at the very end of `forward`. `inductor_hook.explain` therefore counts
recorded break reasons instead, and `assert_single_graph` requires one
graph and no reasons.

## 3. `compile_for_serving`

`vectra_compiler.backend.compile_for_serving(model, example_inputs,
expected_sites)`:

1. `assert_single_graph`, which raises `GraphBreakError` with the reasons.
2. `torch.compile(model, backend=make_backend(report), fullgraph=True,
   dynamic=False)`. The backend is `compile_fx` with the rewrite installed;
   the FX graph cache, and with it the AOTAutograd cache, is disabled
   during the compile, so the pass always runs and its match counts are
   evidence for this compile.
3. Run the example inputs once, then require `report.sites ≥
   expected_sites` (one site per Transformer layer), or raise
   `PatternNotFoundError`. A model that would serve the slow eager path
   never compiles.

The backend is also registered by name: `torch.compile(model,
backend="vectra")`.

## 4. The deployment bundle

`python -m models.export.compile_entry --checkpoint_dir=... --out=dist/compiled_model`:

1. Load the checkpoint (safetensors weights and the `model_config.json`
   architecture; loading never executes code from the file).
2. Move the model to the serving device. On CUDA, the Transformer blocks run
   in float16 (`set_block_dtype`); encoders, geometry and the read-out stay
   in float32, because half-precision coordinates would lose kilometers.
3. Compile for the serving batch (64) with `compile_for_serving`.
4. Numerics gate: on seeded serving-shape batches, the compiled model must
   match the eager model with `allclose(rtol=1e-3, atol=1e-3)` (the
   design's gate, §9). In reduced precision the drift from the float32
   eager model is also recorded.
5. Write `model.safetensors`, `model_config.json` and
   `compile_manifest.json`: SHA-256 checksums, serving shape and precision,
   per-pattern match counts, graph and break counts, numerics, and the
   toolchain versions (Python, PyTorch, Triton, CUDA, NumPy).

`torch.compile` is a just-in-time compiler, so the bundle holds the verified
inputs to compilation, not machine code. At start-up the gateway calls
`load_bundle`, which:

* verifies the checksums;
* refuses a toolchain that differs from the export's (re-export in the
  serving image instead);
* recompiles and re-runs the site count and the numerics gate before the
  port opens.

`ServingModel.predict` pads smaller batches with copies of their last
request and splits larger ones, because the model is compiled for exactly
one batch size (`dynamic=False`). Padding rows are dropped from the result.

## 5. Correction to the design document

The design document (§7.2) registers the pass "on the compilation
pipeline's post-grad hook." In PyTorch 2.7.0 the post-grad graph has
already been through Inductor's joint-graph pattern passes, which rewrite
`softmax(x · c)` into a sign-normalized form whose structure depends on how
the user wrote the negation and the coefficient. Matching there needs a
template for every spelling. The pass is installed instead at the
joint-graph pre-pass (`torch._inductor.config.joint_custom_pre_pass`), the
first hook that sees the decomposed ATen graph. That is the same stage at
which Inductor runs its own attention fusion
(`torch/_inductor/fx_passes/fuse_attention.py`). The design's intent, a
pass on the normalized ATen graph that Inductor compiles, is unchanged.
