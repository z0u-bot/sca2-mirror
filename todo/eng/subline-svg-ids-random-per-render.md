---
status: done
tags: [vis, publishing, reports]
opened: 2026-09-22
closed: 2026-09-22
---
# Subline SVG ids are random per render, so those reports' HTML never compares equal

`utils.dom.gen_ids` seeded its id sequence with `randbytes(4)` at import, so every `clipPath` id a subline emitted (`subline.sparkline`, via `clip-{next(id_sequence)}`) changed from one render to the next. The docstring said this was on purpose — "unique within a single run, and _likely_ to be unique across runs" — so the intent was collision-avoidance, and this note never argued the ids should be predictable, only recorded what it cost.

Found while diffing the whole report corpus across a `pymdown-extensions` bump ([pymdown-12-emphasis-rewrite](./pymdown-12-emphasis-rewrite.md)): rendering the same source, against the same store data, on the same version produced different HTML for ex-2.1.1 and ex-2.1.2 and no others. Those two are the reports that draw sublines. Normalising `clip-[0-9a-f]{8}` made all 62 files compare equal, so the ids were the only nondeterminism in the weave.

Two costs, both small. Byte-comparing an export against a previous one was noisy for these reports, which is what made the version sweep need a normalising step — any future dialect or renderer bump would have paid that again. And `PdfMemo.key` in [`build_site.py`](../../scripts/build_site.py) hashes the printable page, so a republish that changed nothing substantive still moved the key for a subline report and re-printed its PDF. That one was bounded: the site builds from pinned bundles, so the HTML only moved when someone republished, and a republish is a reasonable moment to re-print.

## Notes

**2026-09-22, fixed** — Ids are now derived from content. `Sparkline` counts its clips per plot (`clip-0`, `clip-1`, …) and `Subline.plot` qualifies them with `utils.dom.content_tag`, an eight-character digest of the finished markup, through `utils.dom.stamp_ids`. Two renders of one figure now agree byte for byte; two different figures get disjoint ids; `randbytes` is gone from `dom.py`. Verified: repeat renders of ex-2.1.1 are byte-identical, the page has no duplicate or dangling ids, and against the pre-fix render it is identical once the ids are normalised. Three regression tests in `tests/test_subline.py`.

The fix this item originally suggested — seeding from the document digest — would have been wrong, and the reason is worth keeping. `mini.lit.caching` pickles memoized return values across processes, so a cached cell holding subline markup can meet a freshly rendered one in the same page. A per-document seed restarts both counters at zero and they collide; only an id that follows the figure's own content survives that. Nothing memoizes a subline today, so this was latent rather than live, but the content-derived id is robust to it either way.

Two things came out of the same trip, both now recorded on [consolidate-css](./consolidate-css.md): inlined into HTML, an SVG's `<style>` reaches the whole document rather than its own SVG, so the repeated theme blocks were doing nothing. `mini.vis.svg_figure` now keeps one copy per figure (8 blocks down to 2 on ex-2.1.1, the page 7% smaller), and the subline figures are pixel-identical before and after in both themes. The same scoping means two sublines on one page cannot theme apart, which is now a warning in `src/subline/README.md`.
