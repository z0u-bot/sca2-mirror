---
status: open
tags: [D2.2, ex-2.2.15, anchoring]
opened: 2026-09-26
---
# The newline mask with each crop policy, and `knowable` as an oracle

Ex-2.2.15's `whole-mask` had the smallest seed spread of the first-operand lean of any arm (sd 0.021, against 0.083 for `whole`), and stayed inside the control's band. Sandy's review asks to try the mask with the other policies: `all-mask`, `half-mask`, and `knowable-mask`. `all-mask` is the informative one, since it asks whether the mask alone removes the lean the cut lines carry, the way the tied readout removed the route to whole lines but left the trailing fragments leaning (`all-tied`). The mask leaves cut lines cut, so we expect the fragment lean to stay under `all-mask`; if it does not, the mask is doing more than stopping attention at the newline.

Sandy also asks to keep `knowable` along as an oracle in later experiments. It needs to know where the evidence is, so it can't be adopted, but it removed both leans here, and it is the version on this grammar of the pivot's label variant (c). As a reference arm it bounds what any window-only policy could reach.

Both could ride in the in-context grammar pilot rather than a run of their own on this grammar, since the pilot is where the policy choice matters.
