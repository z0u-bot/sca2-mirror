---
status: done
tags: [D2.2, anchoring, schedules, ex-2.2.23]
opened: 2026-10-08
closed: 2026-10-13
---
# Where the second rise falls on the anchor schedules

The anchor and anti-subspace schedules are ex-2.1.10's, chosen when *red* was anchored in a grammar of one equation per line, and have not been revisited since the in-context grammar. One reading of the spill in ex-2.2.23 is that the model learns the HSV ops while the anti-subspace term is easing off, and a 400-epoch run spends more steps learning while it is weak.

A small look at ex-2.2.23's stored trajectories, with no compute: the anchor and anti-subspace weights against the share of training, each run's rise epoch marked on them, and the lean (the mean alignment of every state) through training, at both lengths. If the spill turns out to follow the schedules, a longer anti-subspace hold is the change to try.

Split from the withdrawn ex-2.2.24 draft (its E5; see [PR #255](https://github.com/z0u/sca2/pull/255)).

## Notes

**2026-10-13, holiday tidy-up** — Ran in [PR #262](https://github.com/z0u/sca2/pull/262) as a re-analysis of the stored ex-2.2.23 runs, and the PR was closed unmerged at the lead's request. What it found: the mean alignment over every state washes the effect out, but the peak over roles of the alignment on contexts of the other ops follows the spill (higher at 400 epochs than at 200, and higher on the runs that spill more). Much of that peak is a latched syntax token, and the runs that latched ⏎ barely spill. It also found that the anti anneal and the learning-rate cosine fall almost together (r ≈ 0.99). The question of the schedules then moved to the terms by slice (late-pull, slice-terms, split-anti-seeds), which kept the early slices off e₁ with a per-slice anti hold.
