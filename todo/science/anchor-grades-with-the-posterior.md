---
status: open
tags: [D2.2, in-context, anchoring, ex-2.2.21]
opened: 2026-10-01
---
# Does the anchor grade with the posterior on the op?

The ex-2.2.21 draft had a hypothesis (H2 in its first review round) that the alignment at the query `=` rises with the posterior on `difference` across the middle band. The label is binary on the true op, so a context whose examples only half-fit `difference` is pulled as hard as one that names it; if the alignment grades anyway, the anchor holds the inferred op rather than the label. The review moved it out of the pilot, for two reasons found in ex-2.2.16's stored runs.

First, the anchor does not land at the query `=` on the whole-line label: it sits on the answers (the preview of ex-2.2.21). Scoring at the query `=` would measure grading where there is almost nothing to grade. The query answer is no better, since its state comes after the answer is predicted, so the measurement site has to follow wherever the anchor turns out to sit, and that is what ex-2.2.21's E1 and its two reference labels (`prompt`, `query-eq`) will show.

Second, the stimulus is coarse. At three examples and ρ = 0.3 the posterior on `difference` takes few distinct values: on the seven-op table about 0.4% of `difference` contexts fall in the 0.5 to 0.65 bin, 3% in 0.65 to 0.8, and 38% in 0.8 to 0.95, nearly all of them in one step near 0.91. (The eleven-op table is similar: 2%, 5%, and 35%.) So a test across ex-2.2.16's evidence bins has two nearly empty bins. A version that holds up treats the posterior as about three levels (near 0.3, near 0.91, and near 1) or widens the stimulus, with a mix of example counts or noise rates in the held-out set only.

Round 3 already plans a graded stimulus (the [D2.2 route](/docs/m2/d2.2/design.md#quick-route)), so this may belong there. The `sampled` arm of ex-2.2.21, which trains the grading, is the trained comparison.
