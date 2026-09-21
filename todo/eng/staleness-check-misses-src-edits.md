---
status: open
tags: [publishing, tooling]
opened: 2026-09-19
priority: high
---
# `./go preview`'s staleness check misses edits under `src/`

`mini.reports.is_stale` compares the bundle's `index.html` mtime against `inputs_touched_at` — the report script plus its input directory. A report's rendering also depends on code under `src/`: `mini.lit`'s stylesheet (`src/mini/lit/lit.css`) is baked into each bundle at export time, as is the chip CSS that `mini.reports.set_provenance` injects, and any figure helper the script imports from `src/mini/vis/`. Edit one of those and `./go preview --no-serve <report>` reports success while re-exporting nothing, so the change is invisible until someone remembers `--force`.

This is a real cost during style work: the loop is edit, preview, screenshot, see the old page, and only then suspect the cache. It cost three cycles in the session that added the booktabs tables and the full-width figures.

Two candidate fixes, both cheap:

- Fold the mtimes of the bundle's *baked* inputs into `inputs_touched_at` — at minimum `src/mini/lit/lit.css` and `src/mini/reports.py`, which every bundle embeds verbatim. That covers the stylesheet case, which is the common one, without trying to trace imports.
- Or hash what gets baked (the CSS and the page shell) into a small stamp written beside `index.html`, and treat a changed stamp as stale. Sturdier than mtimes and it survives a fresh checkout, at the cost of a file per bundle.

The narrower first option is probably the right trade: a stylesheet edit is the case that bites, and tracing arbitrary imports is the general problem that mtime heuristics were chosen to avoid.

## Notes

**2026-09-20, housekeeping** — Shortlisted, into one of the two free slots. Three reasons: the narrower fix is an afternoon (fold `src/mini/lit/lit.css` and `src/mini/reports.py` mtimes into `inputs_touched_at`); the cost is measured rather than guessed, at three cycles in [#192](https://github.com/z0u/sca2/pull/192); and it is a silent wrong answer, which is the kind worth paying down early — `./go preview` reports success and shows the old page, so the failure mode is believing a screenshot.

It also gates work we have queued. What's left of `todo/style/figure-table-overflow.md` is stylesheet edits checked by eye in a browser, which is the exact loop this taxes, and `todo/eng/consolidate-css.md` is more of the same. Fixing this first makes both cheaper.
