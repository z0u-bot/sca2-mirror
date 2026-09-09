---
status: open
tags: [D2.2, task-grammar, metrics]
opened: 2026-09-09
---
# Stochastic rounding of off-grid answers

Raised in the ex-2.2.3 prereg review, from the grammar's rounding rule. `screen`, `multiply`, and (off its on-grid pairs) `mix` compute a per-channel value between two grid levels and snap it to the nearer one. A deterministic snap makes the task a step function of the raw value: 10 rounds to 9 and 11 to 12, every time. The alternative is to round stochastically, so 10 rounds to 9 two-thirds of the time and to 12 one-third, drawn per line at corpus build. The model would then be trained toward the interpolated answer distribution, and a well-calibrated model would spread its answer probability over the two neighbors in proportion to the distance, which reads the effective geometry rather than the snap.

Not taken in ex-2.2.3, because it changes what the task's exact-match ceiling is. A stochastic target puts the ceiling below 1 on every rounded op (the model cannot know which way a line's coin fell), and the calibration and H1 read holdout exact match against "near 1". It would also change every op's agreement and op-relevance figures from constants to distributions.

What it would buy, and what a grammar variant would have to settle: (a) the ceiling, and whether exact match is the wrong statistic under it (the answer-distance reads ex-2.2.3 adds are the alternative); (b) whether a stochastic target makes the answer representation more graded, in the sense of the [distance-shaped answer targets](./distance-shaped-answer-targets-ex-2-1.md) item, which measures that on a deterministic corpus; (c) whether it changes the redder-than-both counts and so E3. Ex-2.2.3's rule is the nearest level with `mix`'s ties sent to the even level index, so its mean signed rounding error is near zero over the pairs, which answers the bias half of the concern without a stochastic target.
