---
status: open
tags: [D2.2, grammar, methodology, ex-2.2.3]
opened: 2026-09-10
---
# A more diverse operator set: add `hue`, `saturation`, `brightness` beside the saturating ops

The six ops of ex-2.2.3 were chosen as blend modes on the same grid, and five of them turned out to have answer distributions that get in the way of the tests. `add` sends a fifth of all pairs to white, `screen` crowds the light half, `multiply` the dark half, and `lighten` and `darken` pick one operand per channel. On the red lines that means most answers do not depend on how red the red operand is: E8 of ex-2.2.3 sorts each op's red lines by whether lowering the red operand's R by one grid level changes the answer, and only `mix` has every red line dependent. Red-line accuracy under projection then sits at the share of red lines whose answer was never sensitive, so H4's removal gate misreads a complete removal as partial on four of the six ops, and the H1 and E4 reads inherit the same skew.

The proposal from the results discussion: remove `add` and replace it with ops whose answers spread through the cube and depend on both operands, such as `hue` (take the hue of one operand at the saturation and value of the other), `saturation` and `brightness` (the HSV blend modes of the same names), and possibly `difference` (|x − y| per channel, which is symmetric and spreads well). Any candidate should be checked with the answer-cloud figure of ex-2.2.3 and with the dependence split of its E8 before it is adopted, so the diagnostic runs on the grammar rather than on the anchored model.

Two things to keep from the current set: `mix` stays, since it is D2.1's op and every reference number sits on it, and the on-grid fraction and redder-than-both counts should be recomputed per op, since the E3 reads depend on them.

## Notes

**2026-09-10, Sandy** — having seen E8, the target is both kinds rather than a swap: keep some ops that are anisotropic (answers on a face or an edge of the cube, as the saturating blend modes are) alongside ops that spread through the cube and depend on both operands. The dependence split then has both kinds to read, and the anchored-op experiments get to see whether the anchor behaves the same under each. So the item is "add hue/saturation/brightness-style ops" more than "drop the saturating ones"; `add` is still the weakest of the current set, since a fifth of its pairs go to white.

**2026-09-10, Claude** — scouted on the grid in [ex-2.2.4](/docs/m2/ex-2.2.4/report.py), no training. Nine candidates are in `sca.data.ops.CANDIDATES`: `difference`, `exclusion`, `hsvmix` (`mix` in HSV), the W3C hue/saturation/luminosity modes, and the HSV hue/sat/value modes. All spread their answers better than the current ops (entropy 6.7 to 7.4 bits against 5.6 to 7.0). By E8's one-level rule the hue/saturation/brightness modes are dependent on 0.37 to 0.55 of red lines, no better than `screen` or `multiply`, because each reads one attribute of one operand and scaling a pure red's R only changes its value; they are also the first ops where operand order carries information (1 to 2% agreement with their own reverse). The commutative three are at 1.00, 0.69, and 0.61. Proposed table A: drop `add`, add `difference`, `exclusion`, `hsvmix`. Stays open until a prereg adopts a table; the lines-per-op confound (E4) is the thing to decide there.
