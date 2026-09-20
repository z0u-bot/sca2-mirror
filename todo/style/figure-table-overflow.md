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

Examples of current structure, which we can change:

```html
<div class="output block">
  <span class="markdown prose">
    <span class="paragraph">A plain mo.md() with a Markdown table</span>
    <table>...</table> <!-- if this is hard to wrap in a figure then let's not, but do let's style it like the others -->
  </span>
</div>
```

```html
<div class="output block">
  <figure class="report-figure">
    <div class="report-table-scroll"> <!-- This wrapper div seems unnecessary -->
      <table class="report-table">...</table>
    </div>
    <caption></caption>
  </figure>
</div>
```

```html
<div class="output block">
  <figure class="mini-themed-figure"> <!-- Styled by rules in mini.lit's sheet since 2026-09-20; each figure used to inline its own. -->
    <img class="mini-themed-img-light" ...>
    <img class="mini-themed-img-dark" ...>
    <figcaption>
      <span class="markdown prose dark:prose-invert contents">
        <span class="paragraph">Caption rendered from Markdown</span>
      </span>
    </figcaption>
  </figure>
</div>
```

```html
<div class="output block">
  <figure>
    <style>...</style>
    <figure class="mini-themed-figure-4uh4hsadff2">
      <!-- this inline max-width style will need to go -->
      <img class="mini-themed-img-light" ... width="414" height="414" style="max-width: 100%; height: auto;">
      <img class="mini-themed-img-dark" ... width="414" height="414" style="max-width: 100%; height: auto;">
      <figcaption>Subfigure (a)</figcaption>
    </figure>
    <style>...</style>
    <figure class="mini-themed-figure-6954f82d4dc9">
      <img>
      <img>
      <figcaption>Subfigure (b)</figcaption>
    </figure>
    <figcaption>Main caption</figcaption>
  </figure>
</div>
```

Also, our current util function that produces figure tags expects HTML for the caption, but agents often provide Markdown instead. I think it should probably accept Markdown, and render it.

For nested figures, I looked at changing the outer figure to use a flex-wrap layout. Maybe we can get that to work but the main caption needs to be on its own row. And one nice thing about the current text-wrap hack is that it balances the wrapping so the last row doesn't have fewer sub-figures than the earlier ones.

A partial implementation is in `report.css`: Markdown tables (and the authored `report-table`s inside a `figure`) break out of the column, centre on the viewport, and scroll horizontally when wider than the page. Marimo makes the table itself the scroll box (`display: block; overflow: auto`), so the centering transform sits on the table rather than on its rows; a transformed row shifts the scrollable overflow and clips the leading columns of a wide table.

```css
.output .markdown table {
  width: max-content;
  max-width: calc(100vw - 126px); /* Marimo's nav bars: 42px left, 84px right. */
  margin-left: calc(50% - 21px);
  transform: translateX(-50%);
}
```

This is a screen-only fix. On PDF export the same tables stay column-bound: the print block in `report.css` resets the breakout (`width: auto; max-width: 100%; margin-inline: 0; transform: none`) and lets cells wrap instead, which is the wrong trade for a wide numeric table. The print side still needs its own centering and page-width rule (see the note below).

## Notes

**2026-09-17, Claude** — Sandy's review of ex-2.2.9 (on the reMarkable) added three asks to this item, all on the print/PDF side: centre each table in the column; let a wide table fill the page width with a margin of about 2 mm each side; and a caption on a themed figure was clipped at the right edge too—just half of an italic character—so captions should have their overflow visible. Done in that report instead of here, and worth making conventions: a no-break space between a value and its range (`span2`) and between `λ_a`, `=` and its value, so a cell never wraps mid-expression; and authored-table cells now render backticks as `<code>` (`cell_html` in that report), which several tables had shown as literal backticks. Both helpers belong in `mini.vis` beside `table_html`.

**2026-09-17, Claude** — The screen-side centering and scrolling landed today (rule above). Sandy confirms it works on screen and not in the PDF export, so the two print asks in the note above (centre in the column; a wide table fills the page width with about 2 mm each side) are the open part of this item, along with the figure/caption normalisation.

**2026-09-17, Sandy** — It looks like we don't need the `report-table-scroll` wrapper anymore: tables in Marimo Markdown cell outputs are scrollable. Also I think we should get rid of our own table class and just use plain `.markdown table` and `.markdown figure table`.
