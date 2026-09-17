---
status: open
tags: [reports, publishing]
opened: 2026-09-10
---

# Export to PDF during review

I like to review our reports at all stages on a reMarkable 2, so I can annotate them. It would be nice if this was a standard part of our workflow: agent makes some changes; I download and annotate the PDF; I give it back to the agent to continue with.

reMarkable supports left-right swipe gesture to navigate between pages, and up-down swipes to scroll within a page. Especially during review, I would like each section to be contained in one long page, so I can swipe U/D between paragraphs, and L/R between sections.

## Note 2026-09-15

Print styles landed in `docs/report.css` (`@media print`): page as wide as the reMarkable 2 screen and about a metre tall, so each section is one long page (sections start fresh pages), 11pt body with tall lines for handwriting, tables unscrolled and wrapped, deferred figures loaded on `beforeprint`. The workflow for now is the browser's print dialog (headers and footers off) and a manual transfer. `render.py` in the `report-render` skill prints headless (`-o report.pdf`), which is the seed for a `./go` verb if the round-trip gets automated. A page has one height, so a short section leaves white space below it; that is the price of the swipe-between-sections navigation.


## Note 2026-09-16

Printing from a phone browser does not work: it ignores the `@page` size, so the reMarkable layout only comes out of desktop Chrome. The ask now is to take the browser out of the loop: the site build prints each report's PDF itself and the page links to it. Two seams. The printing is `render.py` in the `report-render` skill (headless Chromium, `-o report.pdf`), which already honours the print styles; the site build would run it per bundle and write `<key>/report.pdf` beside `index.html`, so it needs a browser in CI (Playwright's Chromium, cached like the other pinned tools). The link belongs in the nav chip `set_banner` injects (`mini.reports`), as a third entry after `← Index` and `Source`, hidden in print like the rest of the chip. Reports are pinned by revision in `docs/publish.lock`, so the PDF is per pinned bundle and never goes stale on its own.

## Note 2026-09-17

Two paging problems from the ex-2.2.9 review. A section that runs past one page breaks mid-table and mid-admonition (the H2 "Miss" callout landed alone on its own page, and a table split across the H3/H4 boundary). `break-inside: avoid` is on `.admonition` and `figure` already, so these are blocks taller than the hint can hold or a section taller than the page; the page could be taller still (the H3 section with its two figures and four tables is the size to fit), or the hint could go on the table wrapper and the callout together with the paragraph before it. Worth a check that the headless print honours the same rules as desktop Chrome, since the phone route ignores `@page`.

## Note 2026-09-17 (Fable, with Sandy)

**Recommendation: print at publish, link at build.** The export half (`scripts/export_reports.py`, `export_one`) prints `report.pdf` beside `index.html`, the bundle sync carries it (`upload_folder` mirrors the directory, so it lands in the same commit as the HTML and is pinned by the same `publish.lock` entry), and the build adds the link to the nav chip. Since the build already bases the page at `resolve/<sha>/exports/<key>/`, the link is the relative `report.pdf`, and the dataset repo serves it inline from the extension. The page's `_assets/` follow the same route today.

Why publish rather than build. The build is read-only by design, reads only the HTML, and runs for production plus every open PR's preview on every event (`eng/publishing.md`). Printing there means Chromium in CI, fetching every report's figures on every run (the cost thumbnails were moved to export to avoid), a few minutes of print time per preview, and a few MB of PDF per report pushed to `gh-pages` on every build, which grows the repo unless the bytes are identical run to run. The export session already has the bundle on disk, Chromium preinstalled (the cloud sandbox), and `render.py` bundling the marimo runtime from `_static/`, so the print is offline and one page-load per report. It is also where the review loop lives: the same step can hand Sandy the PDF in chat (`SendUserFile`) the moment a draft is exported, before any publish, so `./go preview --no-serve` should print too and the verb should say where the file landed.

What it costs. A print-CSS edit no longer refreshes old PDFs (the same trade already accepted for figure styles baked at export; the page itself still restyles from source). One extra page render per publish, a few seconds against the minutes the notebook takes. A dev container needs Playwright's Chromium installed once (the `report-render` skill has the two commands); print should skip with a clear message when no browser is found, never fail the publish.

Two things to make it hold. Idempotency: an unchanged re-publish mints no commit today because the bytes are identical, and Chromium stamps `CreationDate`/`ModDate` and a document ID into every PDF, so strip those after printing (pikepdf or pypdf) and confirm two prints of one bundle are byte-equal; if Skia's output turns out nondeterministic beyond the metadata, fall back to keeping the previous PDF when `index.html` is unchanged. Fidelity: the export's HTML is pre-build (no lazy-loading, no lightbox, no banner), which is the better input for printing since every figure loads eagerly, but the paging check in the note above should be done on this path rather than on `_site/<key>/`, since that is what will ship.

Common ground with `markdown-publishing.md`: both are extra renditions of one bundle, made at export and linked at build, so one `exports/<key>/` directory would carry `index.html`, `report.md`, `report.pdf`, all pinned together. If that item's larger `.py → .md → .html` pipeline ever lands, the print input changes from the marimo export to the build's own HTML, and the marimo-hydration caveats in `report-render` fall away; the seam (print at export, link at build) stays the same.
