---
status: open
tags: [publishing, tooling, reports]
opened: 2026-09-21
---
# A site build that reprints every PDF fetches figures one at a time

`route_remote` serves a print's `https://` requests from a cache that Python fills, one blocking fetch per URL inside the route handler. The fonts and KaTeX are shared and cached after the first report, but every figure is its own pinned CDN URL, so a report with twenty figures pays twenty round trips in series before its `load` event fires. On the GitHub runner a full reprint of 30 reports took six and a half minutes (the #190 preview, 2026-09-21); a fresh PR pays that on its first preview build, and again after any merge that changes `report_print.py` or `report.css`, since a preview's memo is its own and its printable page (links resolved to the preview URL) never matches production's.

Two ways to cut it. Warm the cache before `goto`: scan the printable HTML for `https://` image sources and fetch them concurrently with a thread pool, so the route handler finds every figure on disk. Or let a preview reuse production's PDFs where the printable page is byte-identical, which needs the preview's links resolved to production (an open question: a report new in the PR has no production page to link to). The first is contained and does not change what a PDF says, so it is the one to start with.
