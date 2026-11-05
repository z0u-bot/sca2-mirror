---
status: open
tags: [D2.1, D2.2, post-hoc, baselines]
opened: 2026-10-30
---
# The post-hoc comparison for red on the D2.1 checkpoints

[Ex-2.2.1](/docs/m2/ex-2.2.1/report.py) compared the anchored edit for *red* with fitted directions on the D2.1 checkpoints, but gave the fitters little: three control seeds, labels from the token alone, one fit per slice, and no tuning. The [post-hoc comparison](/docs/m2/post-hoc-compare/report.py) gives the fitters a full grid, every label, and a frozen selection rule, for the op `difference`. Its code would carry over to *red* as it is, but the grid alone would most likely reproduce ex-2.2.1: in the un-anchored model *red* is a corner of the RGB cube the color embeddings form, so a direction fitted for it is an axis of that cube, and projecting it out of every operand state moves the other colors whatever the dose, slice, or site. The move that should change the picture is a gate or taper on the fitted direction's own score, so that a state is edited only where it reads as red, which is the [per-position gate item](fitters-per-position-gate.md) for the op. With it, LEACE or difference in means should come close to the anchored edit on *red*.

Run it after the D2.2 post, together with the op's gate test, as one preregistered design: removal and spill defined for *red* as in ex-2.2.1 (damage on red lines; the accuracy drop on non-red lines), a fit set drawn from the D2.1 training generator, five more `lam0` controls so each arm has eight runs, and the gate as a factor for the fitters and as an arm for the anchor (the `shaped` operator). Worth running so the paper shows the *red* result beside the op result on the same footing.

## Notes

**2026-11-05, red judgement thread** — Checked on the three stored `lam0` checkpoints: the difference-in-means component of each color embedding grades with redness (the seven red colors at 0.42 to 0.81, the 108 non-red colors from −0.50 to +0.30, averaging 0.17 in size; e₁ sits near 0.10 on every color). A threshold near 0.4 on that score separates the two groups, with the 0.6 to 0.8 bin straddling it. This was a look at the data, so a preregistration should cite it as prior evidence rather than predict it afresh.
