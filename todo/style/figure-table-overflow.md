---
status: partial
tags: [figures, reports]
opened: 2026-09-10
---
# Allow figures and tables to overflow horizontally

Figures in Marimo notebooks are by default limited to `max-width: 100%`. When reading a report, I usually have plenty of screen width available, but I use a narrow text column because it's easier to read. However, I would like the figures to be displayed at their native width when space is available. Same for tables.

There are at least two types of tables: the Markdown-type, and our own constructed type, and some of those are wrapped in a `div` with classes to allow horizontal scrolling. In practice these almost always show tabular data, and they should be styled the same. So they need normalizing. I think it makes sense to wrap the tables in `figure` tags (instead of that `div` and treat them like images. The only difference is: images should have max-width bounded by screen width, whereas tables should scroll horizontally when they're too wide.

Captions should be no wider than the containing column of text.

And all of this needs to support one level of nesting, for sub-figures with captions. They should be displayed inline where there's room, and stack vertically when there isn't.

The markup these asks were written against was Marimo's, and it is gone. `mini.lit` weaves each cell's output straight into `<main class="lit">`, so there are three shapes to work with: a themed `<figure>` (with `<figure>` children when it has sub-figures), a Markdown `<table>`, and an authored `<table class="report-table">`.

Also, our current util function that produces figure tags expects HTML for the caption, but agents often provide Markdown instead. I think it should probably accept Markdown, and render it.

For nested figures, I looked at changing the outer figure to use a flex-wrap layout. Maybe we can get that to work but the main caption needs to be on its own row. And one nice thing about the current text-wrap hack is that it balances the wrapping so the last row doesn't have fewer sub-figures than the earlier ones.

## Notes

**2026-09-17, Claude** — Sandy's review of ex-2.2.9 (on the reMarkable) added three asks to this item, all on the print/PDF side: centre each table in the column; let a wide table fill the page width with a margin of about 2 mm each side; and a caption on a themed figure was clipped at the right edge too—just half of an italic character—so captions should have their overflow visible. Done in that report instead of here, and worth making conventions: a no-break space between a value and its range (`span2`) and between `λ_a`, `=` and its value, so a cell never wraps mid-expression; and authored-table cells now render backticks as `<code>` (`cell_html` in that report), which several tables had shown as literal backticks. Both helpers belong in `mini.vis` beside `table_html`.

**2026-09-17, Claude** — The screen-side centering and scrolling landed today (rule above). Sandy confirms it works on screen and not in the PDF export, so the two print asks in the note above (centre in the column; a wide table fills the page width with about 2 mm each side) are the open part of this item, along with the figure/caption normalisation.

**2026-09-17, Sandy** — It looks like we don't need the `report-table-scroll` wrapper anymore: tables in Marimo Markdown cell outputs are scrollable. Also I think we should get rid of our own table class and just use plain `.markdown table` and `.markdown figure table`.

**2026-09-20, housekeeping** — [#192](https://github.com/z0u/sca2/pull/192) settled most of this, including the print side that the 2026-09-17 note called the open part. `main.lit` now spans the page and caps each *child* at `--measure`, so `main.lit > :is(figure, table)` opts figures and tables out and they take the full width; a narrower one still centres. Print gets the same behaviour from one variable — `@page` side margins drop to 5 mm and `--measure` becomes 118 mm, which holds the reading measure where it was and gives a figure about 30 mm more. Captions keep the measure (`min(var(--measure), 100%)` so a sub-figure's caption stays inside its own panel), and the sub-figure row is the flex-wrap layout asked for above, with the main caption on a row of its own. The stylesheets are `src/mini/lit/lit.css` (structure) and `docs/report.css` (authored-table styling and the print block); `report.css`'s Marimo-era breakout rule is gone, so the CSS that used to be quoted here has been dropped.

What's left is the normalisation, and it is smaller than it was:

- **The wrapper `div` is now harmful, not just redundant.** `docs/report.css` makes `table.report-table` its own scroll box and says a report should put the class on the table with no wrapper, but `docs/m2/ex-2.2.1/report.py` and `docs/m2/ex-2.2.6/report.py` still author `<div class="report-table-scroll">` around theirs. Reading the selectors, that div is what becomes the direct child of `main.lit`, so it is capped at `--measure` and the table inside it never reaches the full width. Worth confirming in a browser before fixing, then dropping the wrapper in both.
- **`table_html` is copied into eight reports** (ex-2.2.3 through ex-2.2.10) rather than living in `mini.vis`, which is why the two helpers from the 2026-09-17 note — the no-break space between a value and its range, and `cell_html`'s backtick-to-`<code>` — sit in one report each. Lifting `table_html` is the move that gives them a home.
- **Markdown captions** in the figure helper are untouched.

Sandy's ask to drop `.report-table` in favour of plain element selectors is partly overtaken: #192 made the *layout* rules element-based and deliberately kept the class for content semantics (`num`, `range`, `ref`). So the remaining question is narrower — whether those three cell classes are worth keeping — rather than whether the table class should exist at all.
