# ADR-006: Substitute the kernel with a compiler pass, at the joint-graph pre-pass

* **Status:** Accepted (March 2025; hook clause amended April 2025)
* **Deciders:** Workstream C
* **Specification:** [08](../architecture/08-fx-compilation.md), [09](../architecture/09-subgraph-rewrite.md)

## Context

Calling the kernel from model code would couple research code to a GPU
operator, break CPU training and evaluation, and drift whenever the model
changes. The design keeps the modeling surface eager and applies
performance at deployment.

## Decision

A `torch.compile` backend (`vectra`) runs Inductor with a graph pass that
matches every catalogued formulation of RBF attention by use-def topology
and replaces it with `vectra::rbf_attention` [9] [10]. The deployment
compile fails, rather than serving eager, when the model has graph breaks
or fewer rewritten sites than layers.

**Amendment (April 2025): hook placement.** The design placed the pass on
Inductor's post-grad hook. In PyTorch 2.7.0 the post-grad graph has already
been through Inductor's joint-graph pattern passes, which rewrite
`softmax(x · c)` into a sign-normalized form whose structure depends on the
author's spelling. The pass runs instead at `joint_custom_pre_pass`, the
first hook on the decomposed ATen graph and the stage of Inductor's own
attention fusion. Templates are captured at the same hook. Exact sign and
square canonicalization then reduce each formulation to one template per
dtype class.

## Consequences

* Model authors write ordinary PyTorch
  ([authoring guide](../../compiler/guides/authoring_for_dynamo.md)).
* New formulations are catalog entries with positive, negative and exactness
  tests ([registry](../../compiler/guides/pattern_registry.md)).
* The pass depends on Inductor internals (hook name, decomposition output).
  Every PyTorch upgrade re-runs `make test-compiler`, and the pins of
  [ADR-008](adr-008-toolchain-pins.md) change only together.
