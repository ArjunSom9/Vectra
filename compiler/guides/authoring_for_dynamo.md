# Authoring models the deployment compiler can accelerate

Data scientists write ordinary eager PyTorch. Training and evaluation run
eagerly and are never affected by the compiler. Only the deployment compile
(`models/export/compile_entry.py`) runs `torch.compile(backend="vectra",
fullgraph=True)`, and it refuses to produce a serving bundle unless:

1. TorchDynamo captures the whole forward pass as **one graph**, with no
   graph breaks, and
2. the rewrite pass replaced **every** spatial-attention site (one per
   Transformer layer) with the fused `vectra::rbf_attention` operator.

If either check fails, the compile raises `GraphBreakError` or
`PatternNotFoundError` instead of silently serving the slow eager path. This
page explains how to stay inside both rules.

## Rule 1: no graph breaks

TorchDynamo traces Python bytecode through CPython's frame-evaluation hook
(PEP 523) and records tensor operations into a `torch.fx` graph. When it
meets something it cannot trace, it ends the graph, runs that piece in
Python, and starts a new graph after it (a *graph break*). A break inside an
attention layer splits the pattern across two graphs, so the rewrite cannot
see it.

Common causes and the fix for each:

| Causes a break | Write instead |
|---|---|
| `if (dist > r).any():` (Python branch on tensor values) | `torch.cond(pred, true_fn, false_fn, operands)` |
| `x.item()`, `x.tolist()`, `float(x)` in `forward` | Keep values as tensors; use `torch.where` |
| `print(x)`, logging inside `forward` | Log outside the model |
| `numpy` calls on tensors | Use the `torch` equivalent |
| Building Python lists of tensors whose length depends on data | Use masks with a fixed shape |

`models/price_transformer/model.py:trim_to_radius` is the reference example.
Whether every request has a comparable inside its radius depends on tensor
values, so the choice between the cheap mask and the nearest-k fallback is a
`torch.cond`, not a Python `if`.

Check a model with:

```python
from vectra_compiler import inductor_hook
report = inductor_hook.explain(model, *example_inputs)
assert report.graph_break_count == 0, report.break_reasons
```

`models/tests/test_model.py::test_model_captures_as_a_single_graph` runs this
check in CI.

## Rule 2: write attention in a recognized formulation

The pass recognizes radial-basis-function (RBF) attention:

```
softmax_j( -||q_i - k_j||^2 * gamma_h  [+ bias_j] ) @ V
```

in any of the formulations listed in [pattern_registry.md](pattern_registry.md).
Within a formulation, spelling does not matter. All of these match:

```python
sq = (q.unsqueeze(-2) - k.unsqueeze(-3)).pow(2).sum(-1)    # methods
sq = torch.sum(torch.square(torch.sub(q[..., :, None, :],  # functions
                                      k[..., None, :, :])), dim=-1)
d = q[:, :, :, None, :] - k[:, :, None, :, :]; sq = (d * d).sum(4)

w = torch.softmax(-sq / (2 * sigma ** 2) + bias, dim=-1)   # divisor
w = F.softmax(torch.mul(sq, -gamma), dim=-1)               # negated coef.
w = torch.softmax(sq * -0.125, dim=-1)                     # literal
e = torch.exp(-(sq * gamma)); w = e / e.sum(-1, keepdim=True)

out = w @ v          # or torch.matmul(w, v)
```

This works because the pass runs on the ATen graph after decomposition,
where method, function and operator spellings have become the same
operators. Before matching, squares written as `x * x` are canonicalized to
`x.pow(2)`, and negations are moved onto the distance operand
(`-(d * c)`, `d * (-c)`, `(-c) * d` and `d * -0.5` all become `(-d) * c`).
Both canonicalizations are exact in IEEE-754 arithmetic.

Constraints the match filter enforces (anything else is left alone):

* q and k are `[B, H, N, D]`; the difference pairs every query with every
  key, and the reduction runs over the feature axis.
* The square is a square: `pow(2)` or `x * x`, not `abs()` or `pow(3)`.
* Normalization runs over the key axis (`dim=-1`).
* The coefficient is a per-head scale, broadcastable to `[B, H, 1, 1]`, or
  a Python number. A scale that varies per query or per key is rejected.
* The optional additive bias is a key bias, broadcastable to
  `[B, H, 1, Nk]` (the padding mask). A per-query bias is rejected.
* dtype float32, float16 or bfloat16, and the same for q, k and v.

## What is never rewritten

* Graphs that compute gradients. The fused operator is forward-only, so a
  `torch.compile` used for training compiles the eager formulation.
* Look-alikes that compute something else: L1 distances, cubes, softmax
  over queries, `log_softmax`, logits without the negation, dot-product
  attention, per-key scales and per-query biases. The negative test matrix
  in `compiler/tests/test_pattern_matching.py` keeps this list honest.
* On CPU only, the degenerate shape with one query and one value column.
  Inductor decomposes that batched matmul into a multiply-and-sum before
  the pass runs, so there is no `bmm` for the template to anchor on.
