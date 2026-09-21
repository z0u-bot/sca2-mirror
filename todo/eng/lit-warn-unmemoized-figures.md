---
status: partial
tags: [reports, mini, performance]
opened: 2026-09-20
---
# Warn when a themed figure renders outside `@memo`

Until 2026-09-20, eight of the m2 reports (ex-2.1.1, 2.1.2, 2.1.10, 2.1.11, 2.2.5, 2.2.8, 2.2.9, 2.2.10) used no `@memo` at all, so every render redraws every figure. Ex-2.2.10 spent 39 s of a 57 s render in `themed` wrappers; with its three heavy figures memoized, a warm render is 5 s.

Auto-memoizing every cell is not the fix: a plot function that reads a module-level dict of arrays (the ex-2.2.10 shape before the change) has no stable fingerprint for that global, so `mini.memo` skips it, and the cached figure would outlive the data. The safe version passes the data through the memoized function's arguments, and that is a per-report edit.

What `mini.lit` can do is notice. `memo`'s wrapper can set a contextvar while the wrapped call runs, and `themed`'s wrapper (or `themed_figure_html`) can log once per figure name when it runs with no memo frame above it: "figure `decoded-mix` rendered outside `@memo`; it will redraw on every render". Cheap, points at the right call, and says what to do. A `render --strict` that turns the warning into an error would suit the publish gate. Then sweep the eight reports.

## Notes

**2026-09-20, Claude** — The sweep is done: all eight reports batch their ref reads and memoize every figure, and each warm render is 1.3–2.7 s with Markdown identical to before. The global-fingerprint half also landed: `mini.lit.memo` now tracks `ex.CONST` module attributes and array globals through `mini.memo.reachable_values`, and warns when a memoized function reads a value it cannot fingerprint (a dict of arrays, a project-typed object), so a bare `@memo` over `@themed` is safe by default. What remains is the warning this item names, a themed figure drawn with no memo frame above it, plus `render --strict` for the publish gate.

**2026-09-21, Claude** — The geometry-rsa report was written on a branch while the sweep ran on main, so it landed unmemoized: 16.5 s warm, 28 s of a cold render in `themed` wrappers. Merging main and giving it the same treatment — one batched `fetch`, `read_npz`, a `__memo_key__` on `Results` from its two artifact hashes, `@memo` over the five figures and the summary table — brings the warm render to 1.2 s with byte-identical Markdown. A ninth report that the warning this item asks for would have caught at the point of writing, rather than at the merge.
