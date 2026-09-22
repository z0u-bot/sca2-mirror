---
status: done
tags: [anchoring, D2.2, ex-2.2.11, ex-2.2.12]
opened: 2026-09-22
closed: 2026-09-22
---
# Does the whole-line label want a lower mellowmax τ?

The handover recipe moved the anchor's label from the red operand's slot to the whole line, and pools the per-token alignment with mellowmax at τ = 0.1 (frozen in ex-2.2.3). Mellowmax over n positions with one clear peak is about the peak minus τ·log n, and the pull it back-propagates is a softmax over positions at that τ, so a longer span puts a little more of the pull onto tokens that are not the red operand. Sandy's question from the ex-2.2.11 review: do we need a lower τ to compensate for the longer spans?

What ex-2.2.11 says so far. Lead is 0.91 and the largest latch 0.01 on `handover`, so the pull does concentrate on the red operand. But ᾱ at op1 (the axis the non-red colors pick up at the first operand) is 0.26 on `handover` against 0.18 on `handover-slot`, at ex-2.2.9's seeds and again at fresh ones, and the whole-line label is the one change between those two conditions. That is the shape a diluted pull would leave, and it is the reason ᾱ at op1 lost its gate.

A small experiment: `handover` at τ ∈ {0.03, 0.1} (and maybe 0.01), with ᾱ at op1 as the number to watch, margin and lead as the checks that nothing else moved. The [pooled-anchor τ schedule item](/todo/science/pooled-anchor-tau-schedule-and-adaptive-tau.md) covers letting τ move during training; this one is the fixed-τ question, and it comes first.

## Notes

**2026-09-22, Fable** — scheduled: the `tau-0.03` and `tau-0.01` cells of [ex-2.2.12](/docs/m2/ex-2.2.12/report.py), five seeds each against ex-2.2.11's twenty `handover` seeds, with ᾱ at op1 as the number to watch and a frozen proposal line (a τ is proposed when it lowers ᾱ at op1 by more than the band at no cost to task, margin, lead, or contrast). The τ proposal stands apart from the `hue-hsv` fix, so it can be carried either way.

**2026-09-22, Opus, housekeeping** — [ex-2.2.12](/docs/m2/ex-2.2.12/report.py) ran the two cells ([#206](https://github.com/z0u/sca2/pull/206)) and the answer is no: a sharper τ does not buy back ᾱ at op1. Against the reference's 0.264 at twenty seeds, `tau-0.03` reaches 0.231 and `tau-0.01` 0.254, both drops well inside the 0.06 band the frozen proposal line asked them to clear, so no τ was proposed and the handover recipe keeps τ = 0.1. The checks stayed put (line margin 0.42 against 0.43, contrast 0.89–0.90 against 0.84), so the cells cost nothing; there was simply nothing to take.

Two readings worth carrying forward. The ladder is not monotone — 0.03 lowers ᾱ more than 0.01 does — and the gap between them is about one seed standard deviation at five seeds (sd 0.03–0.04), so what the cells mostly show is that τ is not the lever here. And the force conditions moved it about as far without being asked to: `lam-0.2` reaches 0.223 and `anti-5` 0.213, both below `tau-0.03`, which was not predicted and is also inside the band. Whatever holds ᾱ at op1 near 0.25 is not the pooling temperature.

Closing the fixed-τ question rather than reopening it at more seeds: the effect it was looking for is smaller than the band, and the mechanism is the open question, which stays with [the containment item](/todo/science/containment-rises-under-the-untied-readout.md). [Letting τ move during training](/todo/science/pooled-anchor-tau-schedule-and-adaptive-tau.md) is untouched by this and stays open on its own terms.
