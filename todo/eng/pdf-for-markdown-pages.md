---
status: open
tags: [publishing, reports]
opened: 2026-09-23
---
# A PDF for Markdown pages, as reports have

The site build prints a `report.pdf` for every report (`mini.report_print`, through `docs/report.css`'s `@media print` block), and `./go preview --no-serve <report>` gives one locally for review on the reMarkable. Markdown pages under `docs/` get neither: `./go preview docs/m2/d2.2/pivot.md` skips the file as "not a rendered report". Design docs and proposals (`d2.2/design.md`, `d2.2/pivot.md`) are what we most often review in rounds, so they want the same path, including `--since` for the margin bars.

A stopgap that worked on 2026-09-23: render the body with `build_site.render_markdown`, inline `base.css`, `scripts/md.css`, and `docs/report.css`, load the page in Playwright, and call `print_page`. With the default `fit=True` it prints one tall 158 mm page clipped to its content, the same shape as a report's PDF. Mermaid blocks and `mini:figures` strips were not exercised.

What a proper version needs: `convert_markdown` pulling in the print sheet, a nav chip linking the PDF, the build's print memo covering Markdown pages, `./go preview` accepting a `.md` path, and `review_marks` working on Markdown sources.
