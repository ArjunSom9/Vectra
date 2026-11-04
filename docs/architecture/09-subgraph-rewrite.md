# 09 — Pattern Matching and Replacement

| | |
|---|---|
| **Status** | Accepted |
| **Synthesizes** | Design document §7.2 (pattern definition and replacement via subgraph rewriting) |
| **Code** | `compiler/vectra_compiler/{pattern,replacement,rewrite_pass}.py` |
| **Tests** | `compiler/tests/test_pattern_matching.py`, `test_rewrite_correctness.py`, `test_e2e_compile.py` |
| **Registry** | [`compiler/guides/pattern_registry.md`](../../compiler/guides/pattern_registry.md) |

The pass finds every region of a graph that computes RBF attention and
replaces it with one call to `torch.ops.vectra.rbf_attention`. It is built
on `torch.fx.subgraph_rewriter.replace_pattern_with_filters` [10], which
matches by use-def topology. Variable names, module structure and Python
spelling do not matter; only the dataflow does.

## 1. The catalog

Eight formulations, the product of three choices:

| Axis | Values |
|---|---|
| coefficient | divisor `-d / c`, multiplier `-d * c` |
| bias | with an additive key bias (the padding mask), without |
| normalization | `torch.softmax(x, -1)`, explicit `e = exp(x); e / e.sum(-1, keepdim=True)` |

`torch.softmax` decomposes with float32 upcasts in half precision, so the
four softmax formulations are captured twice (`[fp32]`, `[half]`; float16
and bfloat16 decompose identically). The exp/sum formulations are
dtype-independent (`[any]`). That makes 12 templates.

## 2. Templates are captured, not written

Each template is the eager formulation run through the same Inductor front
end as the model, stopped at the same hook (`pattern.capture`). The
template therefore always matches the decomposition Inductor actually
produces, and changes in lockstep with PyTorch. Capture runs inside
`torch._guards.tracing(None)`, so a template never inherits symbolic shapes
from an enclosing compile. All 12 capture in about a second, once per
process and device type.

Two normalizations make one template cover many spellings:

* **The value cut.** The final `bmm`'s second operand becomes a wildcard
  input. Whether V needs a layout copy before the multiply depends on how
  the model stored V, not on what the region computes.
* **Canonicalization**, applied to templates and model graphs alike:
  `x * x` becomes `x.pow(2)`, and in the pointwise chains feeding `exp` or
  `amax` the negations move onto the distance operand. `-(d * c)`,
  `d * (-c)`, `(-c) * d` and `d * -0.5` all become `(-d) * c`, with the
  larger operand first. Every rule is exact in IEEE-754 arithmetic (negation
  commutes with rounding, multiplication is commutative), so
  canonicalization never changes results. `test_sign_canonicalization_is_bit_exact`
  checks bit equality.

## 3. Matching and validation

Each template is matched with `ignore_literals=True`, so one template
serves every shape. Placeholders can bind graph nodes or Python numbers,
which is how a fixed temperature like `* 0.125` matches the multiplier
template. Everything literals would have pinned down is re-checked by
`replacement.make_filter` against fake-tensor metadata. A match is accepted
only if:

* q and k are `[B,H,N,D]` with equal leading dimensions; the difference
  tensor is `[B,H,Nq,Nk,D]` (every query against every key); v is
  `[B·H,Nk,Dv]` at the cut, and the output is `[B,H,Nq,Dv]`;
* q, k and v share a dtype, which is float32, float16 or bfloat16;
* the square is `pow(·, 2)`; `sum` and `amax` reduce the last axis with the
  template's keepdim;
* the coefficient broadcasts to `[B,H,1,1]` (per head), or is a non-zero
  Python number for divisors;
* the bias broadcasts to `[B,H,1,Nk]` (per key).

These checks are what reject the look-alikes: L1 distances, cubes, sums over
the key axis, softmax over queries, `log_softmax`, logits without the
negation, dot-product attention, per-key scales, per-query biases, and
weights without the value multiply. Each has a test proving it is left
alone and still computes its own result after compilation.

## 4. Replacement

`replacement.make_replacement` emits `reshape(v_bmm → [B,H,Nk,Dv])`, the
coefficient (its reciprocal for divisors, or a `scalar_tensor` for a Python
number), and one `vectra::rbf_attention(q, k, v, scale, bias)`. The pass
then removes dead nodes and fills fake-tensor metadata for the new nodes,
so later Inductor passes can read their shapes.

## 5. Where it runs, and when it does not

* At `joint_custom_pre_pass`, after `remove_noop_ops`
  ([08](08-fx-compilation.md) §5). PyTorch 2.7.0 invokes that hook twice
  per graph, so the pass is idempotent.
* Never on graphs that compute gradients (their placeholders include
  tangents). The operator has no backward formula, and
  `test_graphs_with_gradients_are_left_alone` checks that a training
  compile still produces eager gradients.
* On CPU only, when Nq = 1 and Dv = 1, Inductor decomposes the `bmm` into a
  multiply-and-sum before the pass runs. That degenerate shape keeps the
  eager path and stays correct (`test_degenerate_cpu_matmul_keeps_the_eager_path`).
* The pass is a `CustomGraphPass` whose `uuid()` hashes its own source
  files, so any cache that keys on it is invalidated when the pass changes.

## 6. Test matrix (deliverable C2)

| Requirement | Test |
|---|---|
| (a) Canonical patterns match, in several authoring styles (method chain, functional calls, indexing and products, literal temperature, masked, two sites per graph, transposed V) | `test_pattern_matching.py` |
| (b) Semantically different look-alikes do not match | `test_pattern_matching.py::test_lookalikes_are_not_rewritten` (10 cases) |
| (c) Rewritten code equals eager: every formulation × 4 shapes, Python-number coefficients, float16, bfloat16 | `test_rewrite_correctness.py` |
| (c) The real model: one graph, every layer rewritten, both `torch.cond` branches, reduced precision | `test_e2e_compile.py` |
| Fails loudly: an unmatchable layer, a graph break | `test_e2e_compile.py` |

## 7. Correction to the design document

The design document (§7.2) says the pass finds the pattern "regardless of
how a researcher happened to write it." Within a formulation that holds:
spelling, structure, names and sign placement are all normalized. It does
not hold across algebraically different formulations, such as normalizing
with `exp(x - logsumexp(x))`. Those need a new catalog entry, which is a
small, tested change (the registry's "Adding a formulation").
