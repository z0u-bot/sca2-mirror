---
status: partial
tags: [anchoring, experiments, measurement]
opened: 2026-10-09
---
# Record α per slice and per role along the trajectory

The stored trajectories (ex-2.2.23, the τ × λ_a sweep) hold pooled measurements only, and only final checkpoints are kept, so when a latch or the lean appears during training can't be read back. The [lean review](/docs/m2/embedding-lean/report.py) found the lean in the embedding table by 200 epochs and larger at 400 on runs that lose their latch, which answers part of the question; what remains is when the ⏎ latch gives way to the lean.

Add α per slice, per role, and per op to each trajectory record, plus the e₁ component of the syntax embeddings and the lightness correlation of the color table. The trajectory already records every four epochs, so a 400-epoch run has about a hundred records; at a few hundred numbers each this is under a megabyte per run, beside checkpoints of tens of megabytes, and the probe-set forward pass is already run for the record. Ride on the next experiment rather than a scout of its own. Reviewer's request on the lean review; supersedes the lean-timing scout idea.

## Notes

**2026-10-13, holiday tidy-up** — Partly landed. Since late-pull, the trajectory of each run (the `alpha_map` records of the ex-2.2.21 experiment module) keeps the mean alignment at every slice and every position of the probe set, separately for `difference` and for the other ops pooled, and the e₁ component of every color embedding, from which the lightness slope can be computed. Still missing: alignment split by each of the other ops, and the e₁ component of the syntax embeddings, which is where a latch such as ⏎ would show.
