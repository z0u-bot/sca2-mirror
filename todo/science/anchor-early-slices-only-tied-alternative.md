---
status: open
tags: [D2.2, anchoring, selectivity, ex-2.2.7]
opened: 2026-09-14
---
# Anchor the early slices only, as a tied-table alternative to untying

Ex-2.2.7 traced the anchor axis on the syntax rows to the tied readout: a red position's last-slice state is aligned with e₁, the token that follows it is a syntax token, and the cheapest way to raise that token's logit is an e₁ component on its readout row, which under tying is also its embedding row. The two arms that removed a slice from the anchor both kept the last one: `blocks-only` dropped the embedding slice and left the rows at about 0.25. Nobody has yet tried the other end, leaving the last slice (or the last few) out of the pull, which is the cheap way to weaken the path without untying the table.

Prediction, written before any run: the leak shrinks in proportion to the e₁ component the readout still sees at red positions, and stays well above zero. The residual step is α = 1/n_layer with two LERPs per block, so a state aligned at slice 3 arrives at slice 4 with about (1 − α)² ≈ 0.56 of its component unless the last block rotates it away, and with the anti-subspace term restricted to the same slices nothing asks it to. Two costs to watch beside the rows: completeness, since the last block is then free to write redness off-axis where the projection cannot reach it (the `blocks-only` completeness story, at one slice), and the last-slice placement that the eval contract reads.

Worth one or two arms in the slice sweep with the row-component table beside them, mostly to have the tied-table answer on record. The untied readout removes the path outright and keeps every slice anchored, so it stays the proposal unless this arm surprises.
