---
status: open
tags: [D2.2, training, performance, ex-2.2.17]
opened: 2026-09-28
---
# A cheaper recipe for the center control before building on it

The ex-2.2.17 scout reached about 0.45 held-out expected exact match with the d64-L4 model at eight times ex-2.2.16's length (105,600 steps, masked cosine, peak learning rate 0.00316). If that becomes the recipe for the following D2.2 experiments, every run pays for those steps, so it is worth bringing the cost down first.

Sandy suspects the steps can at least be halved with the wider d128-L4 model and a tuned learning rate. The evidence so far is one seed: d128 made its jump on the HSV channels at about 15k steps against 30k–45k for d64, and led through the first half of training, then ended level (0.459 against the d64 seed range of 0.436–0.464). Since the step is latency-bound, d128 costs the same per step as d64 (see `todo/eng/pack-runs-per-gpu.md`), so a shorter d128 run would be cheaper outright. One caution: its calibration KL was 0.95 against about 0.5 for d64, with a lower training loss, so a shorter run should be checked for calibration as well as accuracy.

A first sweep: d128 at two, three, and four times the length, with a short learning-rate sweep around 0.00316 at one seed, then the best condition at three seeds. Packing seeds would cut this cost further.
