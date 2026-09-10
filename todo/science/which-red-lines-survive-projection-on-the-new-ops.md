---
status: open
tags: [D2.2, intervention, selectivity, ex-2.2.3]
opened: 2026-09-09
---
# Which red lines survive the projection on the non-`mix` ops?

On the recipe, ex-2.2.3's H4 found removal partial on `add`, `screen`, `lighten`, and `darken`: seed-mean red accuracy under the full projection stays at 0.2 to 0.32 on those ops, against 0.06 on `mix`, with seed ranges of ±0.1 to ±0.2. Yet the decoded answer moves 1.5 to 2 grid steps from the true one on every op, so the edit lands everywhere. What differs by op is whether the moved answer still counts as wrong.

Red accuracy is a coarse read on ops where the answer's red channel can come from either operand. Under `add` and `lighten` the answer's red channel saturates or takes the max, so a red operand's contribution is often invisible in the answer, and moving that operand's redness may leave the answer where it was. The seed ranges say the statistic is deciding lines near a boundary.

The read is per line: for each op, which red lines keep their true answer under the projection, and whether that answer depends on the red operand's red channel at all. If the survivors are the lines whose answer does not depend on it, the removal is complete in the sense that matters and the H4 statistic needs a dependence filter (count only lines where changing the red operand's red channel changes the answer). If the survivors are spread across dependent lines too, removal is partial for real on those ops, and the anchored-op work has to allow for it. Store contents from the ex-2.2.3 score tasks already carry per-line predictions, so this is a notebook read with no new training.

From [ex-2.2.3](/docs/m2/ex-2.2.3/report.py#suppression-transfers-h4), Discussion.

## Notes

**2026-09-10, Claude** — [ex-2.2.4](/docs/m2/ex-2.2.4/report.py) took E8's one-level rule to every op on the grid. It reads rounding as much as the op: `mix` is dependent on 0.61 of its own red probe lines and 0.49 on the shared draw, since a one-level drop moves the mean half a level and the snap sends half back. Dropping the red operand's R to zero changes the answer on 0.71 to 1.00 of red lines under every op, so "the answer never needed red" is rarely the case; what differs between ops is sensitivity to a small change. The removal statistic should carry the one-level filter in the prereg, and the proposed table has ops at 0.6 to 1.0 by it.
