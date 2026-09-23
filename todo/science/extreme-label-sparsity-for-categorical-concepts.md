---
status: open
tags: [D2.2, anchoring, labeller, ex-2.2.14]
opened: 2026-09-23
---
# Extreme label sparsity for clear categorical concepts

[Ex-2.2.14](/docs/m2/ex-2.2.14/report.py) anchored the op `difference` with a label rate of 0.02 per line of that op, which is about 0.0018 of all lines, and its op margin matched the every-line arm's (1.05×) and the noisy arm's (1.00×). The pull had saturated: an op is named by one token, and once that token's embedding sits on the axis there is nothing more for extra labels to do. That suggests a much lower rate would still land a categorical concept, and a finding of that shape is worth having, since "a handful of labels anchors a clear category" is a stronger claim about the method than the rates used so far support.

The experiment is a rate sweep on the op labeller, down by decades from 0.02 until the margin falls off or the retention through anneal breaks, at a few seeds, read on the same op margin, task gap, and retention. The interesting quantity is the rate at which the anchor stops landing, and whether that rate is set by the number of labelled lines seen in training rather than the fraction. A color anchor could go beside it for contrast, since a graded property is assembled from many partial alignments and should need more.

Revisit after the op-projection (suppression) experiments, which say whether the op anchor is the one we want before we spend on sparsifying it. Related: [span variants](labeling-pull-span-variants-ex-2-1.md) and [locating the concept inside the labelled span](locating-concept-inside-labeled-span-without-being.md).
