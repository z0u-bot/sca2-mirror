---
status: open
tags: [D2.2, anchoring, labeller, ex-2.2.14]
opened: 2026-09-23
priority: high
---
# A labeller keying that draws off the op word

`sca.anchoring.LabelSpec` keys a line's label draw on its colors: `op1` reads the first operand, `either` both operands, and `line` the answer as well. The op word at position 1 never draws. [Ex-2.2.14](/docs/m2/ex-2.2.14/report.py) anchors an operation, so it needs a keying that draws off `data[line * LINE_TOKENS + 1]` against a per-op rate table, with the `span` pull covering the whole line as `line` keying does. The table is what makes the experiment's arms one labeller: the primary sets the anchored op's rate and zero elsewhere, the every-line arm sets it to one, and the noisy arm gives every other op a small rate so a fifth of the labels are wrong.

The shape that fits the existing code: a fourth `keying` value, `op`, whose draw is one uniform per line compared against `p[op_word]`, consuming the stream in its own way (it is a new labeller, so cross-labeller comparisons carry corpus-draw noise as they always have). The `slot` pull needs a role for position 1 as well: ex-2.2.14's op-word arm pulls the op word alone, so under `op` keying `slot` should mask role 1 of a drawn line and nothing else.

Two reads change beside it: the line margin takes its labelled group from the op word rather than from redness, and the per-op probe sets of ex-2.2.9 already exist, so the group is a mask over them. Worth a unit test on the mask: every position of every anchored-op line in a crop is pulled, and no position of any other line is.
