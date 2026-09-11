---
status: open
tags: [D2.2, task-grammar, representations]
opened: 2026-09-11
---
# Partial operand coverage per op: does an op generalize to the part of the cube it never saw?

Every op in every corpus so far is defined, and trained, on the whole grid: each of the 46,656 ordered pairs of grid colors appears under every op, less the holdout. So nothing yet asks whether the model learns an op as a rule that extends past the pairs it saw, or as a lookup over them. That is a generalization question in the sense M3 will need, where a document-level concept is seen with some contexts and must be recognized in others.

The proposal, from Sandy's review of [ex-2.2.4](/docs/m2/ex-2.2.4/report.py): give each op only a band of the cube. Take a continuum over the colors, cool to warm say, scaled 0 to 1, and let `multiply` cover only 0 to 0.66 while `screen` covers 0.33 to 1. Then read each op on the band it never trained on: holdout accuracy (or, under stochastic rounding, calibration) on the unseen band against the seen one, and the answer's geometry there. With more ops there are many arrangements of partially overlapping bands, and which arrangement is informative depends on what the ops share: two ops that agree on a band (op-relevance above 0) let the model transfer one's rule from the other's lines, whereas ops that are alone in their answers there cannot.

Cheapest version: un-anchored controls only, on the table the next prereg adopts, with one op banded and the rest whole, so the banded op's unseen band is the only thing that changed. Keep it separate from the anchoring conditions until the transfer read is understood on its own.
