---
status: open
tags: [D2.2, anchoring, selectivity, ex-2.2.2]
opened: 2026-09-08
---
# The syntax rows carry the axis, and the tied readout may be why

Ex-2.2.2's E8 shows where the non-red cost of a full-position edit comes from: in every anchored model the embedding rows of `+` and `=` carry 0.3 to 0.4 on the anchor axis, against 0.05 in the un-anchored model, while the anti-subspace term flattens the color rows to 0.17 or below. The leak is in two token rows at slice 0, before any block runs. Under the reflection it costs three quarters of the non-red lines; under the projection, the 0.024 that ex-2.2.1 measured.

The hypothesis: nGPT ties the readout to the embedding table. The token after op1 is always `+`, and when op1 is red its state sits near e₁, so the logit for `+` is that row's dot product with an aligned state; the cheapest way to raise it is an e₁ component on the `+` row. `=` follows op2 the same way. `⏎` follows the answer, which is never anchored, and its row stays at 0.05. The anti-subspace term acts on the same positions and loses.

The test: train the recipe with an untied readout (or exclude the syntax rows from the shared table), and read the same E8 table. If the component vanishes, the projection's non-red cost should go with it, and the intervention-tuning pass would no longer need the thresholded or operand-only edits to route around it. If it stays, the component is placed by the blocks' use of the syntax states, and the anchor weight and schedule are the next dial.

Either way the rows should be clean before the D2.3 concept swap, since a rotation at every position moves them too.
