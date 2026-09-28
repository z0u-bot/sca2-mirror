---
status: open
tags: [D2.2, training, ex-2.2.17]
opened: 2026-09-28
---
# Does a replacement-noise curriculum lift the center control past its level?

In ex-2.2.17 the d64-L4 center control (`k3-r0.3`) levelled off near 0.45 held-out expected exact match against a Bayes ceiling of 0.52, and schedule (cosine, warmup-stable-decay, staircase), length (eight and sixteen times), width (d128), and depth (L6) all left it within the seed range. The shortfall is a similar fraction of the ceiling in every op group, which suggests the shared step, inferring the op from noisy examples, as the limit (to be checked by scoring the existing checkpoints by how many examples in a context carry replacement op noise).

If that holds, a curriculum over the replacement rate is one way to help: start with clean contexts, where the op is easy to infer, and raise the rate to 0.3 over training; or the reverse, starting noisy. Sandy would rather not, since the curriculum changes the training distribution over time and so confounds every comparison made on the recipe afterward. Try it only if the noise-split scoring points at op inference, and treat it as a diagnostic of what limits the control more than as a candidate recipe.
