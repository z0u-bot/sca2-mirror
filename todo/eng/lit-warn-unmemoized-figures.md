---
status: open
tags: [reports, mini, performance]
opened: 2026-09-20
---
# Warn when a themed figure renders outside `@memo`

Eight of the m2 reports (ex-2.1.1, 2.1.2, 2.1.10, 2.1.11, 2.2.5, 2.2.8, 2.2.9, and until 2026-09-20 ex-2.2.10) use no `@memo` at all, so every render redraws every figure. Ex-2.2.10 spent 39 s of a 57 s render in `themed` wrappers; with its three heavy figures memoized, a warm render is 5 s.

Auto-memoizing every cell is not the fix: a plot function that reads a module-level dict of arrays (the ex-2.2.10 shape before the change) has no stable fingerprint for that global, so `mini.memo` skips it, and the cached figure would outlive the data. The safe version passes the data through the memoized function's arguments, and that is a per-report edit.

What `mini.lit` can do is notice. `memo`'s wrapper can set a contextvar while the wrapped call runs, and `themed`'s wrapper (or `themed_figure_html`) can log once per figure name when it runs with no memo frame above it: "figure `decoded-mix` rendered outside `@memo`; it will redraw on every render". Cheap, points at the right call, and says what to do. A `render --strict` that turns the warning into an error would suit the publish gate. Then sweep the eight reports.
