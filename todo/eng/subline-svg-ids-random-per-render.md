---
status: open
tags: [vis, publishing, reports]
opened: 2026-09-22
---
# Subline SVG ids are random per render, so those reports' HTML never compares equal

`utils.dom.gen_ids` seeds its id sequence with `randbytes(4)` at import, so every `clipPath` id a subline emits (`subline.sparkline`, via `clip-{next(id_sequence)}`) changes from one render to the next. The docstring says this is on purpose — "unique within a single run, and _likely_ to be unique across runs" — so the intent is collision-avoidance, and this note is not arguing the ids should be predictable, only recording what it costs.

Found while diffing the whole report corpus across a `pymdown-extensions` bump ([pymdown-12-emphasis-rewrite](./pymdown-12-emphasis-rewrite.md)): rendering the same source, against the same store data, on the same version produces different HTML for ex-2.1.1 and ex-2.1.2 and no others. Those two are the reports that draw sublines. Normalising `clip-[0-9a-f]{8}` made all 62 files compare equal, so the ids are the only nondeterminism in the weave.

Two costs, both small. Byte-comparing an export against a previous one is noisy for these reports, which is what made the version sweep need a normalising step — any future dialect or renderer bump pays that again. And `PdfMemo.key` in [`build_site.py`](../../scripts/build_site.py) hashes the printable page, so a republish that changes nothing substantive still moves the key for a subline report and re-prints its PDF. That one is bounded: the site builds from pinned bundles, so the HTML only moves when someone republishes, and a republish is a reasonable moment to re-print.

The cheap fix, if it is worth one, is to derive the seed from something stable about the document (its digest, say — `mini.lit.document` already has one) rather than from `randbytes`, which keeps ids distinct between documents while making one document's render reproducible. Worth checking first whether anything actually relies on two renders of the *same* document getting different ids; nothing obvious does, since a page holds one render's output.
