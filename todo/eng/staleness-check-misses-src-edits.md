---
status: done
tags: [publishing, tooling]
opened: 2026-09-19
closed: 2026-09-22
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

**2026-09-22** — The baked set grew with the CSS consolidation (`consolidate-css`): `src/mini/lit/base.css` beside `lit.css`, `src/mini/chips.css` and `src/mini/lightbox.css` (read into `mini.reports` at import), and `src/subline/theme.css` inside every subline SVG. A first cut of the fix would list those with `lit.css` and `reports.py`.

**2026-09-22, closed** — Took the narrower option: `mini.reports.baked_sources()` lists what every bundle embeds verbatim, and `inputs_touched_at` folds those mtimes in. Two departures from the sketch above.

The stylesheets are globbed (`*.css` under the `mini` and `subline` packages) rather than listed. The note above exists because a hand-kept roster drifted once already, and every CSS file under those two is there to be inlined into a page, so the glob is precise as well as self-maintaining.

The Python side is listed by hand — `reports.py`, `lit/page.py`, `lit/render.py`, where the baked markup and script live — and `lit/serve.py`, `lit/caching.py` and `lit/npz.py` are left out, since they shape how a page is built rather than what lands in it. Including the three roughly triples the invalidation rate over the CSS alone (8 commits in the last six months, against 21 for the union), which is about one extra full re-export a week on a bare `./go preview`. Worth it: `--stale-only` only gates the preview path, since publishing always re-exports, so over-reporting costs one local export and says so, while under-reporting was the silent wrong answer this item is about.

Still invisible: a figure helper the report imports from `mini.vis`, and the stored results it reads. Both need the import graph or store access, which is what `PROVENANCE_ASSET` is for; `--force` remains the escape hatch.
