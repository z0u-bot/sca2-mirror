---
status: done
tags: [anchoring, methodology, experiments]
opened: 2026-10-09
closed: 2026-10-13
---
# Give the anti-subspace term a weight of its own

`AntiSpec` takes its weight as ratios to the anchor peak (`peak_ratio`, `hold_ratio`), so a sweep over the anchor weight moves the anti weight with it and the ratio between the two never changes. The [τ × λ_a sweep](https://github.com/z0u/sca2/pull/260) found λ_a barely mattered, and the [lean review](/docs/m2/embedding-lean/report.py) says why: the ratio is what decides whether the anti term can hold the embedding table off e₁, and no run has moved it.

Give the anti term an absolute weight (a peak and a hold, in the same units as the anchor weight), keep the schedule shape, and update the experiments that build the spec. The next experiment on the pull is the place to land it, so the sweep that follows reads cleanly. Reviewer's request on the lean review.

## Notes

**2026-10-13, holiday tidy-up** — Done in [PR #267](https://github.com/z0u/sca2/pull/267). `AntiSpec.peak` and `AntiSpec.hold` take absolute weights, and `hold` takes one weight per slice as a tuple; the ratios are used only when those are `None`. The slice-terms and split-anti-seeds experiments build the spec this way.
