# D2.2 design: anchoring an operation the model infers

*Draft, 2026-10-05.* A partial rewrite, to settle the shape and the register before the rest is written. The lede, "The task", "The setup today", and claim (b) are written in full; the other sections are stubs that say what each will hold. The plan as it stood before this rewrite is [archived](design-2026-10.md).

D2.2 asks whether Sparse Concept Anchoring works for a concept the model has to work out from its context, rather than read off a token. The concept is an operation. Each line shows a few solved color equations under one op, with nothing to name it, and the model has to infer the op before it can answer the last equation. We anchor one op, `difference`, to the first axis of the residual stream, then remove it by editing that axis out.

So far the anchor lands, and any cost to the task is smaller than our seeds can resolve. On the best recipe so far, turning the edit up takes `difference` out gradually while the other ops stay as they were. Still to show: that this holds at fresh seeds under a preregistered plan, that the removal follows how strongly each context points to the op, and where in the depth of the model it works.

This page describes where D2.2 stands. The story of how we got here is in the [index](/docs/index.md#d22-anchoring-an-operation), and the [pivot](pivot.md) records why the op lost its word.

## The task

Each line of the corpus is one context: three solved examples of one op, then a query under the same op, written `op1 ? op2 = answer` with a constant `?` where the op word used to be. The model reads the examples, works out which op fits them, and completes the query.

Some examples show the answer another op would give instead (replacement op noise). That makes some contexts point clearly to their op and others leave it in doubt. From the op table alone we can compute how strongly each context points to each op (the posterior over ops), and from that the best score any model could reach (the Bayes ceiling). We score models against that ceiling rather than against a fixed accuracy.[^bayes]

[^bayes]: The posterior weighs each op by how well it explains the examples of a context. A predictor that holds it answers with a mix of the answers of each op, in those proportions. Its score is the ceiling, which is below 1 because noisy examples leave the op uncertain and some ops round at random.

## The setup today

What the next experiment starts from, part by part, with the report that settled each.

Ops
: Seven: `mix`, `lighten`, `darken`, `difference`, and the three HSV blend modes; answers between grid levels round at random. Settled by [ex-2.2.18](../ex-2.2.18/report.py) (the set), [ex-2.2.4](../ex-2.2.4/report.py) and [ex-2.2.13](../ex-2.2.13/report.py) (the rounding).

Contexts
: Three examples and a query; replacement op noise at 0.3, cube noise at 0.02. Settled by [ex-2.2.16](../ex-2.2.16/report.py).

Model
: d64-L4 nGPT, with a readout table of its own and attention that stops at the line break. Settled by [ex-2.2.7](../ex-2.2.7/report.py) (readout), [ex-2.2.17](../ex-2.2.17/report.py) (mask).

Training
: 200 epochs, cosine schedule peaking near 0.003. Settled by [ex-2.2.17](../ex-2.2.17/report.py) to [ex-2.2.20](../ex-2.2.20/report.py).

Anchored op
: `difference` on e₁, with no *red* anchor beside it. Settled by [ex-2.2.14](../ex-2.2.14/report.py).

Labels
: The whole context, on about one `difference` context in fifty. Settled by [ex-2.2.14](../ex-2.2.14/report.py), [ex-2.2.21](../ex-2.2.21/report.py).

Pull
: Every slice, capped at an alignment of 0.8 (the `hinge` condition); whole contexts only. Settled by [ex-2.2.21](../ex-2.2.21/report.py) (the cap, provisional), [ex-2.2.15](../ex-2.2.15/report.py) (whole contexts).

Verification lines
: Allowed, since they leave completion unchanged; the `hinge` condition trains without them. Settled by [ex-2.2.21](../ex-2.2.21/report.py).

Edit
: Projection of e₁, at every position and slice, with the share removed as the dose. Settled by [ex-2.2.21](../ex-2.2.21/report.py).

Scoring
: Expected exact match against the Bayes ceiling; removal against the target null. Settled by [ex-2.2.16](../ex-2.2.16/report.py), [ex-2.2.21](../ex-2.2.21/report.py).

Three of these are open in [ex-2.2.22](../ex-2.2.22/report.py): which slices the pull acts on, the height of the cap, and whether contexts vary in their number of examples.

<!-- REVIEW: verify each entry against its report before this leaves draft. The anchor schedule (λ_a, τ, the anti-subspace weight) is left out as settled since D2.1; add a row if it should be on the page. -->

## What we want to be able to say

Five claims. Each section says what the claim means in plain words, what we have seen, and what is still to show.

### (a) An inferred op can be anchored without costing the task

*Stub.* Where the anchor sits (the example answers, hardly at the query `=`), the task cost inside the seed band, and the open question of whether the alignment grades with the posterior.

### (b) Removing the op is graded and selective

Graded means that a stronger edit takes out more of `difference`. Selective means that the other ops are left as they were, along the whole range of doses. In M1 this was the dose-response curve and the orthogonal colors left untouched.

On the earlier grammar, where the concept was *red*, projecting the axis out removed *red*, more so the redder the line, and some of the cost fell on lines with no red in them ([ex-2.2.1](../ex-2.2.1/report.py)). That cost came through the embeddings of syntax tokens, and a readout table of their own cleaned them ([ex-2.2.7](../ex-2.2.7/report.py)). With that change, removal was clean on ten of eleven ops ([ex-2.2.11](../ex-2.2.11/report.py)).

On the in-context grammar the anchor settles on the example answers, where the context shows the most about its op ([ex-2.2.21](../ex-2.2.21/report.py)). So an edit at the query `=`, which predicts the answer, has little to act on, and none met both criteria. The edit at every position does work. On the `hinge` condition, turning it up takes `difference` out step by step, and the other ops stay within the selectivity gate at every dose. Without the cap the edit grades the same way but spills just past the gate onto another op. Editing the example answers alone takes out most of the op, so the query seems to take the op from the examples.

Still to show:

- that the `hinge` result holds at fresh seeds, under a preregistered plan;
- how selectivity changes with the cap, between 0.8 and no cap ([ex-2.2.22](../ex-2.2.22/report.py) has two caps in between);
- that the damage on each context follows how much its answer depended on `difference`, the per-context prediction the target null gives.

### (c) The removed model lands somewhere predictable

*Stub.* The target null, ex-2.2.21's post hoc finding that the edited model answers much as the Bayes predictor without `difference`, and the test of that at fresh seeds in ex-2.2.22 (H2). Whether a trained fallback is still needed.

### (d) The bound holds by construction, and we know where in depth

*Stub.* The write bound held on *red* (ex-2.2.1). The layer sweep: which slices the edit needs, and which the pull needs (ex-2.2.22's slice conditions are a first look).

### (e) Scarce, noisy labels are enough

*Stub.* Labels already cover one context in fifty. Noisy labels were tested on the op word (ex-2.2.14) and not yet on the inferred op.

## What comes next

*Stub.* Ex-2.2.22, then the preregistered experiment that anchors and removes the op at fresh seeds, then the anchor side of the layer sweep beside the SGTM baseline. One line each on the question it answers.

## Open risks

*Stub.* The live rows of the old risk table: a shortcut in place of the op, a route around the edit, and a task cost too small for five seeds to resolve.

## Out of scope

*Stub.* As before: the verification measurements (D2.3), topic markers, anchoring at fine-tune time, and several ops on separate axes.
