---
status: open
tags: [D2.2, anchoring, in-context, ex-2.2.21]
opened: 2026-10-02
---
# Leave slices out of the pull, at more seeds and longer training

In [ex-2.2.21](/docs/m2/ex-2.2.21/report.py) the `no-emb` arm (the whole-line pull with the embedding slice left out) had the highest held-out expected exact match of any arm, about 0.56, with its three seeds within 0.005 of each other. Its gain over the whole-line arm is inside the seed band, so three seeds cannot say whether it is real. Paired by model seed, though, there is a hint: on seed 702 every anchored arm that pulls the embedding slice took the slow path through training (EEM 0.43 to 0.54), while `no-emb` and the control did not (0.563 each). If pulling the embedding slice is what sends some seeds down the slow path, leaving it out would buy reliability more than peak score.

The reasoning for going further, from the review: an abstract concept like an op is not the property of any one token, so it probably exists only in the middle of the stack. That suggests a family of restrictions:

- no embedding slice (`no-emb`, as now);
- no last slice (the input to the readout), which [anchor-early-slices-only](./anchor-early-slices-only-tied-alternative.md) proposed for a different reason;
- both, leaving the middle blocks only.

Run them beside the whole-line arm and the control at more seeds (perhaps eight), at 200 epochs and again at 300 or 400, since the slow path shows up as a plateau that a longer run gives time to leave (ex-2.2.21's training traces). Score the task, the op margin, and the suppression pass, so the answer covers editability too.

Post hoc, the suppression pass found that `no-emb` edits less selectively than the whole-line arm. One reading is that the restriction itself makes the early blocks hold the axis less cleanly. Another, from the review, is that the anchor weight λa was set with the embedding slice in the pull, so a restricted pull may want a lower one. Crossing each restriction with a lower λa (a bracket of two or three levels) would separate the two.
