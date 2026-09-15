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

