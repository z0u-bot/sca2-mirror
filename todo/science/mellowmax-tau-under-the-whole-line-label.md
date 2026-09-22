---
status: open
tags: [anchoring, D2.2, ex-2.2.11]
opened: 2026-09-22
---
# Does the whole-line label want a lower mellowmax τ?

The handover recipe moved the anchor's label from the red operand's slot to the whole line, and pools the per-token alignment with mellowmax at τ = 0.1 (frozen in ex-2.2.3). Mellowmax over n positions with one clear peak is about the peak minus τ·log n, and the pull it back-propagates is a softmax over positions at that τ, so a longer span puts a little more of the pull onto tokens that are not the red operand. Sandy's question from the ex-2.2.11 review: do we need a lower τ to compensate for the longer spans?

What ex-2.2.11 says so far. Lead is 0.91 and the largest latch 0.01 on `handover`, so the pull does concentrate on the red operand. But ᾱ at op1 (the axis the non-red colors pick up at the first operand) is 0.26 on `handover` against 0.18 on `handover-slot`, at ex-2.2.9's seeds and again at fresh ones, and the whole-line label is the one change between those two conditions. That is the shape a diluted pull would leave, and it is the reason ᾱ at op1 lost its gate.

A small experiment: `handover` at τ ∈ {0.03, 0.1} (and maybe 0.01), with ᾱ at op1 as the number to watch, margin and lead as the checks that nothing else moved. The [pooled-anchor τ schedule item](/todo/science/pooled-anchor-tau-schedule-and-adaptive-tau.md) covers letting τ move during training; this one is the fixed-τ question, and it comes first.
