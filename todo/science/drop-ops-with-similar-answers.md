---
status: open
tags: [D2.2, in-context, ex-2.2.17]
opened: 2026-09-28
---
# Drop ops whose answers sit close together?

Some ops of the in-context grammar give answers that are hard to tell apart by eye, such as `lighten` and `screen`, which both brighten. Sandy suggested, reviewing ex-2.2.17, that the op set could drop ops that behave similarly.

Ex-2.2.17 (E4) checked whether the mass its center control puts on other ops' answers is explained by those answers being near the true one. It is not: at the same distance from the support of the Bayes predictive, and inside the box the two operands span, other ops' answers get several times the mass of colors no op gives. Its op confusion matrix then shows where that mass goes: for every op, mostly onto the op whose answers most often coincide with its own (`lighten` onto `screen`, `darken` onto `multiply`, `hsvmix` and `mix` onto each other, `difference` onto `exclusion`). So the leak is partly a matter of similar ops, and dropping or merging one op of each of those pairs might shrink it, as well as simplifying the task.

What a pruned op set would change: similar ops make examples less informative, since an example that fits `lighten` often fits `screen` nearly as well, so fewer ops would raise the posterior on the true op for the same replacement rate. That also moves the Bayes ceiling, so every comparison with the D2.2 experiments so far would need a fresh ceiling. A first step is cheap and needs no training: for each pair of ops, how often their answers agree over all operand pairs, from the answer table of ex-2.2.16 (ex-2.2.17 has the agreement on confident contexts, as `OVERLAP`), and how much the posterior at the center condition would sharpen without each candidate. The four pairs above are the likely candidates.
