---
status: done
tags: [tooling, reports, vis]
opened: 2026-09-03
closed: 2026-09-06
---
# Inline SVGs are most of a Markdown render

`./go render` assembles a report as plain Markdown for a reader — a structural pass, a reviewer, anything that wants the document rather than the page. Figures drawn with matplotlib arrive as one `![alt](…)` line each, which is what that reader wants: the alt text says what the figure shows, and the link resolves if they want to look. Figures the report inlines as SVG (sublines, the swatch table) arrive as their full markup instead. On `docs/m2/ex-2.1.1/report.py` that is 8 SVGs taking 25 KB of a 45 KB document: more than half the render is path data, sitting between the paragraphs a structural pass is there to read.

The pieces of a fix are already in place. `mini.reports.externalize_html` writes each such fragment out as a sidecar under the publisher's asset dir precisely so tooling that can't run the frontend can read it — `public/.mini/report/sublines-surprisal.html` and friends are there beside the PNGs after any render. So the render could carry a link to the sidecar where it now carries the markup. What is missing is the correspondence: one sidecar holds a group of sublines, so matching an SVG in the document to the file it came from means comparing content rather than reading a name off the tag.

Worth settling alongside it: what stands in for alt text. A `![…](…)` for an SVG group needs a description, and unlike a `themed` figure these fragments carry none today — which is its own gap, since a reader of the published page has the same problem. The `alt-text` skill is the standard; `style-fig` is where the subline conventions live.

## Notes

**2026-09-06, tech debt** — Done. The correspondence turned out not to need content matching: `externalize_html` now stamps `data-mini-asset="<sidecar url>"` on the fragment's root element as it writes the file, so the render reads the name off the tag after all. The attribute is inert in the browser — nothing fetches it, and `insert_base`/`stray_links` look at `src`/`href` only — so it can carry the notebook-relative URL that `localize_links` then repoints, the same treatment a PNG gets. `clean_marimo_md.link_externalized` swaps the whole element for that link before any other pass reaches inside it.

On alt text, both existing call sites already passed an `aria_label` to `figure_html`, so the description was there and unused; the render lifts it, and warns when a fragment has none. `style-fig` now asks for one explicitly. A sidecar that is itself an image (`.svg`) is written as `![…](…)`; an `.html` one as a plain link, which is what it is — a Markdown viewer would draw the image form as a broken image.
