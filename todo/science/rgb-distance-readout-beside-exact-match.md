---
status: open
tags: [probes, D2.2, methodology]
opened: 2026-09-22
---
# An RGB-distance readout beside expected exact match

Every removal and selectivity number in D2.2 is expected exact match on the grid: the chance the model's answer is the one right color. It gives no partial credit, so a model that lands one grid step away scores the same as one that answers black. Sandy's suggestion from the ex-2.2.11 review: report the RGB distance (or its square, an MSE) of the greedy guess, and of the median of the model's distribution over the grid, beside the exact match.

Where it would help. On ex-2.2.11's removal lines, `handover` keeps 24% of its exact match on `hue-hsv` and 12–18% on the channel-wise ops. Distance would say whether those kept answers are the right color or a near neighbour, and whether the lost ones are near misses or somewhere else entirely, which the exact match cannot distinguish. It would also show, for the near miss on `darken`, what the model answers with red taken out.

Cost: the probe already has the full distribution per line, so this is a scoring change, no re-run. The grid geometry is fixed, so the distance is well defined. Worth adding to the probe utilities so later experiments get both readouts for free, and worth a preregistered expected direction before it is used in a gate.
