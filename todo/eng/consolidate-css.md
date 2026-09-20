---
status: open
tags: [reports, publishing, docs]
opened: 2026-09-20
---
# Consolidate the CSS: one base sheet, one print block, a map of the rest

CSS lives in six places, and it is hard to know which one to edit. Three files: `src/mini/lit/lit.css` (the woven page frame — tokens, typography, tables, figures, admonitions, swatch base; inlined at export by `mini.lit.page`), `docs/report.css` (report polish — `.sw-themed`, `.report-table`, quiet `<details>`, the draft badge — and the e-ink print block; baked at export and re-inlined from source at build by `set_report_styles`), and `scripts/md.css` (the site's Markdown pages, `<link>`ed by `build_site.py`, and also VS Code's `markdown.styles` in `.vscode/settings.json`, which is what the `.github-markdown-body` selectors are for). Three Python strings: `_CHIP_CSS` and `_LIGHTBOX_CSS` in `mini.reports` (injected at build), and the SVG `<style>` in `subline.py` (inside each SVG, so it stays there).

The duplication that matters is between `lit.css` and `md.css`: about 120 lines near-verbatim — `:root` tokens and dark overrides, `body` type, headings, `code`/`pre`, the booktabs table block, `img`/`video`, `blockquote`, anchor links. Each file's header says it follows the other, and they have drifted: `th, td` padding is `0.8em` in lit and `0.9em` in md, md has no `--mark` token, the monospace stacks differ. Smaller overlaps: the theme-detection selector (`body:not([data-theme=…]):not(.dark)…`) is written out once for `.mini-themed-figure` in lit.css and once for `.sw-themed` in report.css; and the print rules are split, with lit.css's `@media print` (`break-inside`/`break-after`, an 11pt body) a subset of report.css's (which also sets `@page`, `--measure`, `thead`, and 11pt on `html`).

Two claims in `docs/README.md` are stale and add to the confusion: "each report links it, so it shows live while editing" and "baked via `css_file`". Neither is true now — `mini.lit`'s live server (`render.py`) builds the page with `lit.css` alone, so a `report.css` edit is invisible under `./go serve` and shows only at export or build.

Proposed shape:

- **Base sheet.** Move the shared rules out of `lit.css` into `src/mini/lit/base.css`; `lit.css` keeps the flexbox frame, figures, admonitions, swatches, themed figures. `md.css` shrinks to the site-only rules (`body` column, `.fig-strip`, `.tags`, `pre.mermaid`, the pseudo-`dl`, the `.github-markdown-body` variants) and the build concatenates `base.css + md.css` into the served file. VS Code's preview reads the file from disk, so either keep a committed concatenation or point `markdown.styles` at both files (it takes a list). Where values differ, take lit.css's: the reports are what gets reviewed.
- **One print block.** Move the two print rules lit.css duplicates into `report.css`, leaving lit.css's `@media print` with what the frame needs on its own (`color-scheme: light`, flexbox off, `main.lit { padding: 0 }`).
- **Live server.** Have `./go serve` apply `report.css` too (`extra_head` in `render.py`, or `set_report_styles`), then correct the two README sentences.
- **Python strings.** Move `_CHIP_CSS` and `_LIGHTBOX_CSS` out to files beside `reports.py` so they are findable with `fd -e css`; they keep `Canvas`/`CanvasText` rather than the page tokens, since they must work on a page rendered without lit.css. The subline SVG style stays inside the SVG.
- **A map.** A header comment in `lit.css` listing where the other CSS lives and how each reaches the page.

The one-off `style=` attributes in `docs/m2/*/report.py` (a few in ex-2.1.1 and ex-2.1.2; the rest are matplotlib kwargs) are fine where they are. Check the result with a `./go preview` render in both themes and a PDF print, since the base split touches table padding and font stacks.
