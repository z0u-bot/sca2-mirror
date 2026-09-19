---
status: open
tags: [figures, reports, ex-2.2.10]
opened: 2026-09-18
---
# An interactive explorer for the probe-decoded stream figures

The stream section of [ex-2.2.10](/docs/m2/ex-2.2.10/report.py) shows the red operand and the answer decoded from the residual stream at five slices, clean against projected, for four ops: forty small cube panels. Sandy's review: "This part is hard for me to understand. I think I would need an interactive widget (JS) to explore the visualizations with controls for op, clean/projected, slice, and dataset (red lines, removal lines, all lines)."

The data is small (decoded RGB per line, slice, position, target, and pass, mean over seeds), so a single inline JS figure with the decoded coordinates embedded and the four controls above one cube panel would do it, drawn with the same wheel projection as `sca.vis.plot_rgb_cube`. The published export is static HTML, so the widget has to be self-contained; the subline strips are the precedent for an inline SVG figure with the group externalized. Two things to settle: the "all lines" dataset needs the non-red lines' decodes, which the scoring pass does not store, and the alt text for an interactive figure has to describe the whole space rather than one view.
