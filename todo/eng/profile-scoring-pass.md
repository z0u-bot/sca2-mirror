---
status: open
tags: [performance, modal, experiments]
opened: 2026-09-27
---
# Find where the scoring pass spends its GPU time

In ex-2.2.12 and ex-2.2.13 the scoring role used more L4 time than training did: 160 against 91 GPU-minutes in ex-2.2.12, and 477 against 324 in ex-2.2.13 (median 161 s per `score_one`, against 107 s per training run). Scoring is inference with interventions over the probe lines, which should be much cheaper than 4,950 training steps, so something other than arithmetic is probably dominating: recompilation per operator or per subspace (a new closure or shape each call re-traces under `filter_jit`), small batches, or host-side readout.

A profile of one `score_one` on an L4 (wall time per operator, and a count of compilations via `jax.log_compiles`) would show which. Training has a similar fixed cost to look at: about 30–50 s of each training task falls outside the step loop (compilation is 7–11 s of it; the rest is trajectory recording and the checkpoint upload).
