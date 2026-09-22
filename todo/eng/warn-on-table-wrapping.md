---
status: open
tags: [reports, lit, tables]
opened: 2026-09-22
---
# Warn when a rendered table wraps its cells excessively

Ex-2.2.11's H2 table has ten columns, and on the reMarkable's page width every cell wraps: `margin ↑` puts the arrow on its own line and each range `(0.40–0.46)` breaks twice. A reader cannot scan the row.

Two parts. The report side is done for that table: a non-breaking space before the header arrows (`margin ↑`) in ex-2.2.11. The pipeline side is the item: a check in the render or the PDF export that measures each table at the target width and warns when cells wrap beyond some threshold (say, any header wrapping mid-phrase, or a body cell wrapping more than once). The fix when it fires is usually to split the table or drop a column, which is the author's call, so a warning rather than a rewrite.
