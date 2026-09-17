---
status: open
tags: [D2.2, anchoring, selectivity, ex-2.2.9]
opened: 2026-09-17
---
# The end-of-line embedding row keeps an axis component under the untied readout

[Ex-2.2.9](/docs/m2/ex-2.2.9/report.py)'s H4 found that untying the readout cleaned every syntax word's embedding row except `⏎`, which still carries an axis component on `handover` (about 0.2 in the Discussion's read, against near zero on the op words and `=`). The readout was the mechanism ex-2.2.7 named for the syntax words carrying the axis: with a tied table the model needs an axis component on `=` to predict it after a red operand. `⏎` is the one syntax word that follows the answer rather than an operand, so that mechanism does not cover it.

Two candidates. The whole-line labeller pulls the answer position, and `⏎` is the token predicted there, so the readout row for `⏎` may be where the pull lands on that position and the embedding row follows through the shared training signal; `handover-slot` (the operand-only labeller) would then show a cleaner `⏎` row. Or the answer position is where the model represents the answer's redness, and `⏎`'s embedding is the nearest place for a residual component to land. The stored embedding tables (`rows` and `rows_readout` in the metrics) let both be read without a run: `⏎` on `handover` against `handover-slot`, and its readout row beside its embedding row.

Whether it matters: the projection touches every position, so a component on `⏎` is a write at the last position of every line, after the answer is read out at `=`. That is a cost to nothing the current gates see. It is worth knowing before M3, where a syntax token following a labelled span is the usual case.

From ex-2.2.9's Discussion.
