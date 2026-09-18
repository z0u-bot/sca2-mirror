---
status: partial
tags: [reports, publishing, tooling]
opened: 2026-09-18
---
# Reports as literate scripts (`mini.lit`) instead of Marimo notebooks

mi-ni#86 prototyped `mini.lit`: a plain `.py` whose top-level strings are prose (f-strings, where they quote values) and whose code between them is a cell, with a `@memo` cache for the slow parts, `stop()` for a preregistration, and one self-styled HTML page. The question here is whether our reports should move to it. [Markdown publishing](./markdown-publishing.md) wants the same thing from the other end (a plain Markdown face for every report, and to drop the Marimo-native export); a literate script weaves that Markdown as a by-product.

## The trial: ex-2.1.12 ported

[`docs/m2/ex-2.1.12/report_lit.py`](../../docs/m2/ex-2.1.12/report_lit.py) is the report beside it, carried over by hand. Same prose, same helpers, same figures; the port took about an hour, most of it mechanical. What changed in the writing:

- `mo.md(rf"…{x:.2f}…")` becomes `rf"""…{x:.2f}…"""` at the top level: an f-string paragraph is evaluated as Python (added during the trial; a plain string is static prose). The names it reads are real references, so ruff and ty check them and vulture stops reporting every prose-only variable as unused, which it did for 21 names when the prose was a Jinja template. Joined lists still read better precomputed as strings in the cell (`e1_ctrl`, `ctrl_spread`).
- One namespace, so Marimo's cell-private `_x` names go. Reusing a name across cells is fine for ty (it types by flow, so `x: int = 1` then `x: str = "a"` passes), and ruff only asks that a reused loop variable be read; the distinct loop names in the port (`c_`, `t_`) are a habit from Marimo rather than a need.
- `mo.stop(cond, mo.md(…))` becomes `if cond: stop("…")`; ty narrows past it, so the `assert x is not None` after it goes too.
- A loop that appends figures becomes a joined string as the cell's last expression; `mo.vstack([md, table, md])` becomes prose, a cell whose last expression is the table HTML, prose.
- A docstring hung under a constant (`TARGET_TITLES = …` then `"""Panel titles…"""`) is a top-level string, so it renders as prose. Write it as a comment. Worth a lint, as `trailing_cell_docstrings.py` is for Marimo.
- Each figure function gets `@memo` outside `@themed`, with the alt text passed as an argument so an edit to it is cache-visible.

What came out: the prose paragraphs are identical to the Marimo render (`./go render`), every computed number included; the ten figure PNGs are byte-identical to the Marimo export's `_assets/`; ruff, ty, and the repo lints pass with the file in the tree, and the site build ignores it. Timings on the cloud sandbox:

| case | Marimo | lit |
| --- | ---: | ---: |
| fresh process, nothing cached (`./go preview --no-serve` with its PDF vs `./go lit render`) | 25.5 s | 10.0 s |
| fresh process, figures cached | (no cache) | 4.7 s |
| Markdown render (`./go render` vs the same `lit render`, which writes both) | 11.6 s | 4.7 s |
| PDF, on top of the render | included above | +1 s |
| same process, nothing changed (a prose-only edit in the live server) | | 20 ms |
| `index.html` size | 230 KB | 41 KB |

The 4.7 s warm floor is the store: two `get_refs` calls over the network at about 7 s in total on a cold connection (`get_many` is served from the local cache), plus 0.8 s of imports. `@memo` on the loader would hide it (5.8 MB of arrays pickled under `.mini/lit-cache/`) at the cost of missing a republished ref until `version=` is bumped; a short-lived ref cache in `mini.store` would be the better home.

## What a switch needs

- **Publishing.** `./go publish`, `export_reports.py`, and `build_site.py` know Marimo notebooks (`unpublished_reports.py` now also knows a literate `.py` by its `# title:` header, as a sibling document rather than a report input, which is what CI tripped on in the trial PR). A literate script weaves to the same bundle shape (`index.html` + `_assets/`), so the export step is the join: pin it in `publish.lock`, give it the banner (`set_banner`: index link and source link), thumbnails and the `mini:figures` strip, rewrite `../d2.1/report.py` links to `../d2.1/`, and print the PDF through `report.css`'s `@media print` rules (the lit print is on a default page size; `mini.report_print`'s fit-to-section pass is the other half).
- **Markdown face.** The woven `index.md` carries each figure as the `<figure>` HTML `themed` emits, a `<style>` block included; for reading, `themed` wants a plain-Markdown target (`![alt](path)`), which is mi-ni's item. Tables stay HTML, which is fine.
- **Skills and lints.** `report-render`, `style-py` (its cell conventions), the report section of `mi-ni`, `science` (freeze mechanics) and the `unannotated_cell_vars` and `trailing_cell_docstrings` lints are Marimo-shaped. The port did not need the browser path at all: `lit render` writes the page and the Markdown in one pass and the PNGs are on disk.
- **The other 25 reports.** The port is mechanical enough that a Sonnet pass per report, checked by a paragraph diff against `./go render` and a byte comparison of the PNGs (both done above), would carry them. Not worth doing until publishing works for one.

## Notes

**2026-09-18, Fable (with Sandy)** — Trial done as above; the port file stays beside `report.py` until publishing is decided, and is not published. My read: the writing experience is the win (whole-file ruff and ty, go-to-definition, no cell signatures, prose that reads as prose), the re-render speed is real, and the cost is the publishing seam, which is about a day. The Jinja spelling of numbers is the one thing that reads worse than the f-strings it replaces.

**2026-09-18, Fable (later the same day)** — Sandy's read of the PR: likely adopting. Added f-string prose to `mini.lit` (mi-ni#88) and switched the port's eight Jinja expressions to it; the woven Markdown is byte-identical to the Jinja rendering, and the vulture, ruff, and ty passes are clean on the file. Open naming question on the PR: "literate document" reads off to Sandy; "literate notebook" or "literate script" are the candidates.

**2026-09-18, Fable (evening)** — Settled with Sandy: the term is *literate script*, and Jinja is gone from `mini.lit` (mi-ni#89): prose is a plain string (static) or an f-string (evaluated field by field), and the Markdown spelling went with it. The per-field pending marks that Sandy wanted survive, since they never depended on Jinja: past a `stop()` each field that cannot be evaluated is its own mark and everything else in the paragraph renders, so a preregistration's headings and known values all show. What Jinja gave up is `{% for %}`; a comprehension inside a field (`{"\n".join(f"| {c.name} | …" for c in conds)}`) writes the same table, and a helper returning HTML covers anything larger. The publish check's helper is now `is_literate_script`.
