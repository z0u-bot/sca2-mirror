---
status: open
tags: [reports, structure]
opened: 2026-09-17
---
# Put a report's helper cells in reading order

A report notebook currently declares every helper function near the top and every prose cell near the bottom, so the two halves have to be read against each other. In ex-2.2.9 the `h1_*` family sits around line 400 and the cell that calls it around line 2000; apart from `h1_verdict`, nothing else uses those functions. The same holds for `h2_*` through `h5_*` and for the exploratory tables. A reader scrolling the notebook meets eighty helper cells before the first sentence of the report, and an author changing a figure has to hunt for it.

The fix is to move each family down beside the section that uses it, so the notebook reads top to bottom: shared helpers first (loading, palette, table markup, the drawing helpers), then each section as heading, background, helpers, results. Marimo resolves cells by its dependency graph rather than by file order, so this is free at runtime and only affects how the file reads. The shared drawing helpers (`pooled_sd`, `dots`, `gate_line`, `fig_legend`) were moved above the H1 block in [ex-2.2.9](/docs/m2/ex-2.2.9/report.py) as the contained part of this; the per-hypothesis reshuffle is the rest.

Worth doing once with a script that moves whole cells rather than by hand, and worth deciding at the same time whether the report skeleton should carry the order so new reports start in it. `docs/m2/ex-2.2.9/report.py` is the natural first subject, since its sections are already split into heading / background / results cells.

## Notes

**2026-09-20, housekeeping** — The mechanism in the paragraph above is stale, and in a way that makes this item easier rather than harder. Marimo is gone ([#191](https://github.com/z0u/sca2/pull/191)) and every report is a `mini.lit` literate script, so file order *is* execution order: a helper simply has to be defined before the cell that calls it, and moving a family down to sit above its own section satisfies that on its own. There is no dependency graph to reason about and nothing to check beyond "the def still precedes the call".

The subject is unchanged: `docs/m2/ex-2.2.9/report.py` still has `h1_status` through `h1_prose` at lines 392-528 and the cells that call them at 1258 and 1900, with `h2_*` to `h5_*` in the same shape. The shared drawing helpers are already above the H1 block, so what is left is the per-hypothesis move.

One thing worth deciding along with it: whether the report skeleton in the report-writing skills should carry this order, so new reports start in it rather than being reshuffled afterwards. That is the half that stops the item recurring, and it overlaps with `todo/eng/refactor-report-skills.md`.
