---
status: open
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
  <figure class="mini-themed-figure-08d4a808c42a"> <!-- Every themed figure inlines its own styles, which isn't great. -->
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

A partial implementation is in `report.css`:

```css
/* Allow tables to be full-width, overflowing the containing column. */
.output .markdown table {
  width: auto;
  max-width: 100vw;
  /* +42px is to avoid the nav bars in Marimo when it's not in fullscreen mode. Could be smaller in exported reports. */
  margin-left: calc(50% - 50vw + 42px);
  margin-right: calc(50% - 50vw + 84px);

  > thead,
  > tbody {
    transform: translateX(calc(50vw - 50% - 63px));
  }
  /* Fake border because the transform leaves the border behind. */
  > thead {
    border-bottom: none;
    > tr:last-child > :is(th, td) {
      box-shadow: inset 0px -1px 0 color-mix(in srgb, var(--input), transparent 0%);
    }
  }
}
```
