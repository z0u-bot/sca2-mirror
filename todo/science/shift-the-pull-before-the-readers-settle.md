---
status: open
tags: [D2.2, anchoring, intervention]
opened: 2026-11-05
---
# Shift the pull before the readers settle

The [low-runs report](/docs/m2/low-runs/report.py) found that every run writes the evidence that an example fits `difference` twice: along e₁, where the anchor put it, and along one other direction that the loss made on its own. The low runs are the ones whose reader blocks take enough of that evidence from off the axis for the e₁ edit to leave a fifth of the op in place.

The review of the report suggested three schedule changes, cheaper to try first and read against the same ten low runs: a higher λ_a at the start of training, so that e₁ is in place before the readers settle on their inputs (and, the other way, a lower λ_a late: fitting examples reach α close to 1 at slice 3, so there may be room to turn the pull down until the largest α sits a little under 1); a lower λ_s̄, since under fixed labels the anti term presses e₁ down on the unlabelled `difference` contexts (49 of 50 at 1/50), which may be what leaves most contexts with only the off-axis part; and a change to the depth ramp of λ_s̄, whose direction is unclear, since a gentler ramp presses e₁ down harder at the reader's input on the unlabelled `difference` contexts too. None of them touches the reader as such, so they may move the odds without closing the gap. An anti term that spares the contexts the model itself scores as `difference` would need no labels and is the production-viable form of the second idea.

Any of these is a recipe change, so it goes through the fixed-labels screen: eight seeds at 1/50, the share of runs with removal under 0.8, against 5 of 24 on the recipe of record. A check at epoch 224 with the e₁ edit itself (the earliest epoch where the low and normal runs separate, per the report's E4) would let a sweep drop the low runs before the anneal.

Split out of [make-the-reader-depend-on-e1](./make-the-reader-depend-on-e1.md) when its edit-aware term was set aside (2026-11-05).
