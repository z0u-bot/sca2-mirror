---
status: open
tags: [figures, reports]
opened: 2026-09-17
---
# Seed spread as a box or violin, and conditions told apart by marker shape

Two asks from the review of [ex-2.2.9](/docs/m2/ex-2.2.9/report.py), both about how a per-seed figure reads on a reMarkable, where colour is gone and the page is grey.

**Spread.** Our per-seed figures draw one small dot per seed with a larger mark at the mean. That shows the spread only where the dots are sparse enough to count, and it hides the shape of it. A box-and-whisker or a violin per condition would carry the median, the quartiles and the range in one glyph, and would read in ink. Ex-2.2.9 added a thin min–max range bar behind each dot column as an interim, which is the range and nothing more. The convention would live in a `mini.vis` helper beside `smooth_step_band`, so every report draws the same glyph.

**Shape.** Conditions are told apart by colour alone. On paper `control` and `handover` are two greys. Ex-2.2.9 gives each condition a marker shape (`marker(cond)` in its report) and a legend; that mapping belongs in the shared palette so the same condition has the same shape across reports, the way `ink` gives it the same colour.

Both are conventions rather than one-off fixes, so they land in the `style-*` skills and `mini.vis` together, and the reports pick them up on their next touch.
