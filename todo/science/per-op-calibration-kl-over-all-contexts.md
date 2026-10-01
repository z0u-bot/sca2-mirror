---
status: open
tags: [D2.2, in-context, analysis, ex-2.2.19]
opened: 2026-10-01
---
# Per-op calibration KL over all held-out contexts

The op confusion matrices of [ex-2.2.17](/docs/m2/ex-2.2.17/report.py) and [ex-2.2.19](/docs/m2/ex-2.2.19/report.py) (E3) are taken on confident contexts only, where the Bayes posterior on the true op is above a threshold. The calibration KL (E2) is taken over every held-out context. Sandy, reviewing ex-2.2.19, asked whether a matrix like the confusion one can be made over the whole corpus, showing how much each op contributes to the KL.

One way: group the per-context KL by true op, for the rows, and split it by the op whose answers hold the mass the model has in excess of the Bayes answer distribution, for the columns. The row sums would then add up to the calibration KL. On unconfident contexts several ops can give the same answer, so the column split needs a rule for shared answers (perhaps sharing the mass by the Bayes posterior over ops). It would show whether the KL that remains at 200 and 400 epochs sits in the HSV-channel ops, as the confusion on confident contexts does, or in the contexts the confusion matrices leave out.
