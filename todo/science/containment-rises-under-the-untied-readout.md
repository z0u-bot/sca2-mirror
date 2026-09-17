---
status: open
tags: [D2.2, anchoring, containment, ex-2.2.7, ex-2.2.9]
opened: 2026-09-15
---
# Containment rises under the untied readout, and we do not know why

Ex-2.2.7 read ᾱ at op1 (the mean alignment with the axis over all 216 colors) at 0.13 on its untied condition and 0.23 on its untied whole-line condition, against 0.08 on the reference, at nine seeds. Every experiment since ex-2.1.8 gated this statistic at 0.1. The pull still landed on the red operand in those runs (lead, contrast, and latch all sat at the reference's values), so this is other colors drifting a little onto the axis at op1 rather than the pull finding a position.

The handover prereg ([ex-2.2.9](/docs/m2/ex-2.2.9/report.py)) reads containment as a prediction rather than a gate, beside H3's selectivity: a drift that costs nothing on the non-red lines is recorded for the anchored-op prereg to watch; a drift that comes with a selectivity miss is the first place to look for why. What this item asks is the mechanism. Two candidates: with a readout of its own, the embedding table is free of the logit pressure that kept non-red colors off the axis, so the anti-subspace term is doing that work alone; or the readout row for the red colors carries the axis (ex-2.2.7 saw 0.31 on `=`'s readout row) and the blocks route a little of every operand toward it. Reading ᾱ per color group, and the alignment of the readout rows, on the handover's checkpoints would separate those.

## Notes

**2026-09-17, Claude** — ex-2.2.9 ran: ᾱ at op1 is 0.28 on `handover` (readout untied) against 0.1 gated before, and the paired non-red deficit is 0.004, so the drift costs nothing the selectivity gate can see. Its H2 results write the first candidate above as the working mechanism: with a tied table the output loss holds the non-red embedding rows off e₁ (a component there would add a red logit on every line), and untying releases it, the same release H4 sees on the syntax words from the other side. The check it names is the mean axis component of the non-red embedding rows on `handover` against `handover-tied`, which the stored embedding tables allow without a run. Sandy's review asked "why?" twice on this, so the answer belongs in the next report that reads ᾱ.
