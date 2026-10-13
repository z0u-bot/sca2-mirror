---
status: open
tags: [D2.2, anchoring, schedules]
opened: 2026-10-12
---
# A principled per-slice anti weight

The recipe of record sets the anti-subspace hold by slice, falling in a straight line from 0.2 at the embedding to 0.03 at block 4 (`falloff`, from [split-anti-seeds](/docs/m2/split-anti-seeds/report.py)). It replaced the hand-set step of slice-terms, so no slice boundary has to be chosen, but the two end points are still set by hand. Two things remain:

- Tuning the falloff itself. `falloff` missed the task gate on `hue-hsv` at one seed, and a lower anti weight on the earliest slices was the suggestion from that review.
- An anti term pooled over slices with something like mellowmax, so the weight goes where the states are most aligned with e₁. It is not clear how this would work without labels, since the anti term acts on every position and line, labeled or not.

Either would be an arm against `falloff` at the split-anti-seeds seeds, scored with removal and spill at every dose. The first may fit inside the [recipe sweep](./full-recipe-sweep-after-the-lean-fix.md).

## Notes

**2026-10-13, split-anti-seeds** — The linear falloff has been tried: in [the split-anti-seeds experiment](/docs/m2/split-anti-seeds/report.py) it removed about as much as `split-anti` and kept every run within the spill criterion, and it is now the recipe of record. It missed the task gate on one op at one seed. The pooled anti term is still open, and so is tuning the falloff itself, for example a lower anti weight on the earliest slices.
