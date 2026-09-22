---
status: open
tags: [anchoring, representations, methodology]
opened: 2026-09-21
---
# Anchor *red* to a two-dimensional subspace and see whether the deep geometry comes back

The geometry-rsa reanalysis found that on the handover grammar, and for the heavier or longer-trained anchors on the six-op grammar, dropping e₁ does not return an anchored run to the control band: what differs is the geometry of the other 63 coordinates. Sandy's reading of that, from the review: a richer op set may need more than one dimension for *red*, so a single axis is too small a home and the rest of the space reorganises around it. If so, anchoring to a plane (e₁ and e₂ together, with the anchor loss on the projection onto the pair) should leave the remaining 62 coordinates closer to what a control builds.

The measurement already exists: the geometry-rsa report's e₁-dropped variant becomes an e₁₂-dropped one, and the control band is the same. A preregistered version would state, for the handover grammar, that the last-block RSA to the controls with the anchored subspace dropped is inside the control band, and would carry the single-axis arm as the comparison.

Two things to settle first. How the anchor target is defined on a plane (a fixed direction in the plane, or the whole plane, which is a different loss), and whether the extra dimension is charged to the concept's variance share, which the checks in geometry-rsa report per axis.

## Notes

**2026-09-22, from the ex-2.2.11 review** — ex-2.2.11 gives this item a second motivation. On `hue-hsv`, the one op that missed removal at fresh seeds, the stream holds part of the hue of the red operand somewhere other than the axis, on all three anchored conditions alike. Sandy's margin note there: "so, maybe 2D". Sandy also asked whether a deeper network has been tried, since the HSV ops might want more depth; that is a separate knob and would go in its own item if we take it up.

**2026-09-22, Fable** — scheduled as the `plane` and `L6-plane` cells of [ex-2.2.12](/docs/m2/ex-2.2.12/report.py), a two-by-two with depth at five seeds. The two open definitions are settled for the sweep in the design doc's decisions: alignment is the length of the projection onto the e₁–e₂ pair (unsigned), the terms and the projection take the pair where they took the axis, plane reads are compared with the control checkpoints read on the same pair, and the variance share is charged two coordinates. The reason it is the structural candidate: the axis measures how red a color is, which is the same on both sides of red, and `hue-hsv` with red at op2 is the one op that needs the side. Act one of ex-2.2.12 tests that reading before the plane cells are interpreted (kept share split by the sign of G − B on the red operand). The depth question from Sandy's note is in the same factorial rather than its own item. Implementation note: `sca.anchoring` reads one module-level `ANCHOR_AXIS` in the anchor term, the anti term, and the stream read, so the plane needs those generalized to a tuple of axes with a norm over them.
