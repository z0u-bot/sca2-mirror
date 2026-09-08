---
status: open
tags: [D2.2, intervention, fallback, ex-2.2.2]
opened: 2026-09-08
---
# Tighten the LUNAR-style fit before comparing it with the trained fallback again

Ex-2.2.2's E6 fitted one 64×64 matrix at the embedding of each no-fallback checkpoint: fallback cross-entropy on the qualifying red lines, plus a retain term at weight 1.0 that is the mean squared distance from the unedited state at every other live position. It reaches the designed response in every seed, at a non-red deficit of 0.39 against 0.75 for the trained term under the reflection, with a seed range from 0.14 to 0.85.

Two things would tighten the spread. The retain term is state-preserving where the score is task-preserving: a KL to the clean logits at the non-qualifying `=` positions asks for what we measure. And the retain weight is a single point; a small sweep with a held-out stopping rule gives a Pareto front instead of one seed-dependent trade.

The map has to move red operand states while leaving syntax states that share the axis alone, so the fit's headroom depends on how the other 63 dimensions separate them, which is the [syntax-row leak](syntax-rows-carry-the-axis-via-tied-readout.md). Scoring-only on the stored checkpoints, a few CPU minutes per fit.
