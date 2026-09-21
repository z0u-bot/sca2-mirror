---
status: open
tags: [publishing, tooling, reports]
opened: 2026-09-21
---
# A PR preview that reprints every PDF loses the race in `page.goto`

`print_bundle` bounds each of its waits and documents them — up to `timeout` (8 s) for `main.lit` to appear, then `settle` (3 s) for figures and fonts — and it is careful never to raise, because "a publish must not fail on the PDF". The navigation above those waits is the exception: `page.goto(url)` takes Playwright's defaults, which are a 30 s timeout and `wait_until="load"`. `load` does not fire until every subresource has arrived, and in externalize mode the subresources are the report's figures at pinned CDN URLs, fetched one at a time through `route_remote`'s Python-side fetch. A figure-heavy report on a slow link can exceed 30 s, and then `goto` raises where everything around it would have logged and carried on.

Seen on 2026-09-21, building the preview for #190. The branch had just merged main's render sweep, which edited `src/mini/report_print.py`; `print_stamp` hashes that module, so every report's memo key changed at once and the preview had 29 PDFs to print instead of the usual zero or one. It printed `m1/ex-2.9.1`, timed out navigating to `m1/ex-2.9.2`, and `./go site` exited 1: production and the other previews deployed, that one preview did not. Main never meets this, because its memo comes from production's gh-pages and is warm by the time a sweep lands; a PR that merges a print-tooling or stylesheet change meets it every time. `cdn.jsdelivr.net` was also unreachable in that run (TLS handshake timeout), which costs one host timeout before `route_remote` gives up on it.

The tempting one-liner — `wait_until="domcontentloaded"` — is wrong on its own. `settle` is a flat `wait_for_timeout`, not a wait on the images, so `load` is currently the only thing that guarantees a figure is in the page before it prints. Dropping it trades a loud failure for PDFs that quietly lose their figures, which is worse: nobody reads the log of a build that passed.

So the fix has to keep the guarantee and lose the cliff. Roughly: navigate with `domcontentloaded` and a generous explicit timeout, then wait for the images the way `main.lit` is already waited for — `page.wait_for_load_state("load", timeout=…)`, or a poll on `Array.from(document.images).every(i => i.complete)` — and on a timeout log which figures never arrived and print anyway, matching what the function does for a missing browser. Worth pairing with a concurrent `route_remote` (the serial Python fetch is what makes a 29-report build slow in the first place), though that is the larger half and could be its own item.

Until then a PR whose merge changes `report_print.py` or `report.css` may need its preview re-run, and a re-run reprints all of them again, because the memo is only written by a build that finishes.
