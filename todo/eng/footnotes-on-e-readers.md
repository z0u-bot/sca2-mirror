---
status: open
tags: [reports, lit, publishing]
opened: 2026-09-22
---
# Footnotes that an e-reader can navigate

The pipeline puts footnotes at the very end of the document, and on a reMarkable the back-link from a footnote goes to the top of its page rather than to the marker, so going to a footnote and back is cumbersome. The reMarkable prints one section per tall page, so "the top of the page" can be a long way from the marker.

Sandy's idea (ex-2.2.11 review): one flexbox for the whole page, footnotes authored inline next to their marker, and on screen given `order: 1` (or similar) so they sort to the end. In print, or for the PDF export, they would stay where they are authored, next to the text that cites them, so no navigation is needed.

Things to settle: whether the PDF export path (`./go preview`, the reMarkable PDFs in `pdfs.json`) uses the same stylesheet as the site, and whether footnote numbering stays in reading order when the boxes are reordered visually.
