---
status: done
tags: [D2.2, in-context, training, anchoring, ex-2.2.19]
opened: 2026-10-01
closed: 2026-10-13
---
# Close out D2.2 at 400 epochs

The recipe of record trains for 200 epochs, which [ex-2.2.19](/docs/m2/ex-2.2.19/report.py) adopted to halve the cost of a run: it falls short of 400 epochs by about 0.016 in held-out expected exact match, and its calibration KL matches 400 epochs. The plan is to run D2.2's rounds at 200 epochs, then repeat the final configuration at 400 epochs and fresh seeds, so the published result is on the tighter model.

The closing run is a confirmation. It reports its numbers beside the 200-epoch ones, and the two differ in more than the task score, so it re-checks every gate the 200-epoch round passed: the op margin, the site where the anchor sits, and the selectivity of the chosen edit. The anchor weights are scheduled as a fraction of training, so they stretch with the run, but twice the steps after the pull ends gives the alignment more time to drift (see [alignment decay](./alignment-decay-worth-chasing-own.md)). A lower-cost early warning: one 400-epoch run of the chosen arm beside round 3, or the op margin through training on the 200-epoch runs, which shows whether it is still falling at the end.

## Notes

**2026-10-13, holiday tidy-up** — Settled by a different route. [Ex-2.2.23](/docs/m2/ex-2.2.23/report.py) found that some anchored runs at 200 epochs missed the second rise in task skill, so the recipe moved to 400 epochs there, and every experiment since (late-pull through split-anti-seeds) trains for 400. The fresh-seed confirmation this item planned is the split-anti-seeds experiment.
