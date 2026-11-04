# Pattern registry

The pass (`vectra_compiler.rewrite_pass.RbfAttentionRewrite`) replaces eager
RBF attention with one call to `torch.ops.vectra.rbf_attention`. This page
lists the formulations it recognizes, explains how templates are produced
and checked, and walks through adding a new one.

## Pipeline position

```
model.forward ──TorchDynamo (PEP 523)──▶ torch.fx graph (Torch IR)
              ──AOTAutograd + Inductor decompositions──▶ ATen graph
              ──joint-graph pre-pass: RbfAttentionRewrite──▶ rewritten graph
              ──Inductor joint/post-grad passes, lowering, codegen──▶ kernels
```

The pass is installed as `torch._inductor.config.joint_custom_pre_pass`
(`vectra_compiler.inductor_hook.installed`). That is the first hook on the
decomposed ATen graph, and the stage at which Inductor runs its own
attention fusion (`torch/_inductor/fx_passes/fuse_attention.py`). Running
earlier than the post-grad hooks matters. Inductor's joint-graph patterns
rewrite `softmax(x * c)` into a sign-trick form that depends on how the
negation and the coefficient were written, which would multiply the number
of templates needed. PyTorch 2.7.0 calls the joint pre-pass twice per graph,
once before and once after `remove_noop_ops`, so the pass is idempotent.

Inside the block the FX graph cache is disabled, which also bypasses the
AOTAutograd cache, so the pass runs on every compile and its match counts
are evidence for the compile at hand.

## Catalog

`vectra_compiler.pattern.CATALOG` is the product of three choices:

| Axis | Values | Eager form |
|---|---|---|
| coefficient | `divisor` | `-sq / c` |
| | `multiplier` | `-sq * c` |
| bias | with | `... + bias` before normalizing |
| | without | |
| normalization | `softmax` | `torch.softmax(x, -1)` |
| | `exp_sum` | `e = exp(x); e / e.sum(-1, keepdim=True)` |

That gives 8 formulations. `torch.softmax` decomposes with float32 upcasts
in half precision, so the four softmax entries are captured twice, as
`[fp32]` and `[half]` (float16 and bfloat16 decompose identically because
dtype literals are wildcards). The exp/sum entries are dtype-independent and
appear once, as `[any]`. In total there are 12 templates. Entry names, for
example `rbf_divisor_bias_softmax[fp32]`, are what `CompileReport.matches`
and the bundle manifest record.

## How a template is produced

`pattern.capture(spec, device, dtype)`:

1. Runs the entry's eager function through `compile_fx` with a hook at the
   same joint pre-pass position. It records the graph and aborts the compile
   inside `torch._guards.tracing(None)`, so the template never inherits
   symbolic shapes from an enclosing compile.
2. Applies `remove_noop_ops` and cuts the graph at the second operand of the
   final `bmm`, which becomes the wildcard placeholder `v_bmm`. Whether V
   needs a layout copy before the multiply depends on how the model stored
   V, not on what the region computes.
3. Canonicalizes squares and signs, the same normalization applied to model
   graphs.

Templates are captured per device type and cached for the life of the
process. Capturing all 12 takes about one second on CPU.

## How a match is checked

`subgraph_rewriter.replace_pattern_with_filters` matches by use-def topology
with `ignore_literals=True`, so one template serves every shape. All the
information that literals would have pinned down is then re-checked by
`replacement.make_filter` against fake-tensor metadata:

* ranks and shapes: q, k `[B,H,N,D]`, matching leading dimensions, the
  difference tensor `[B,H,Nq,Nk,D]`, the output `[B,H,Nq,Dv]`;
* dtypes: float32, float16 or bfloat16, the same for q, k and v;
* `pow` exponent 2; `sum` and `amax` over the last axis with the template's
  keepdim;
* coefficient broadcastable to `[B,H,1,1]`, or a Python number (non-zero for
  divisors);
* bias broadcastable to `[B,H,1,Nk]`.

`replacement.make_replacement` then emits `reshape(v_bmm)`, the coefficient
(or its reciprocal, for divisors) and one `vectra::rbf_attention` call. The
pass fills fake-tensor metadata for the new nodes so that later Inductor
passes can read it.

## Adding a formulation

1. Add the eager form to `PatternSpec.eager_fn`, plus a new axis value if
   needed, and make sure `CATALOG` produces it.
2. If the form binds new placeholders, extend `PatternSpec.placeholder_names`
   and the filter and replacement accordingly.
3. Add a positive authoring-style case to
   `compiler/tests/test_pattern_matching.py`, a per-formulation case to
   `compiler/tests/test_rewrite_correctness.py` (it is parametrized over
   `CATALOG`), and any new look-alike to the negative list.
4. Run `make test-compiler`. A formulation is only done when its
   look-alikes are proven not to match.

## Known limits

* Formulations that differ algebraically beyond sign placement and squaring
  need their own entry, for example normalizing with `exp(x - logsumexp(x))`
  or scaling by `sigma.pow(-2)` folded into q and k beforehand.
* On CPU, Inductor decomposes `bmm` with one query and one value column
  into a multiply-and-sum before the pass runs. That degenerate shape keeps
  the eager path.
* Graphs that compute gradients are never rewritten, because the operator
  has no backward formula.
