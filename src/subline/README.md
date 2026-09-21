Visualize metrics as sparklines under text.

![Screenshot of text that reads, "A long time ago, in a galaxy somewhat far away..." A sparkline beneath the text shows that the word "somewhat" is clearly out of distribution (i.e. unexpected) in this context.](../../doc/subline.svg)

## Styling

The SVG carries its own theme (light/dark aware) via CSS custom properties. To restyle without editing the library, pass `css` — appended after the built-in styles, so a later rule overrides at equal specificity:

```python
Subline(css="svg { --bg-color: light-dark(#fff, #181c1a); }").plot(text, series)
```

Overridable properties include `--bg-color`, `--col-text`, `--col-baseline`, `--col-series-1..5`, and `--blend-mode`.

One theme per page, though. Inlined into HTML, an SVG's `<style>` is not scoped to that SVG — it joins the document's stylesheets and reaches every element on the page. So two sublines on one page with different `css` do not theme apart: both blocks use the `svg` selector, and the later one in document order wins for both. Give a page's sublines the same `css`, which is also what lets `mini.vis.svg_figure` keep one copy of the block per figure instead of one per SVG.


## Citation

If you use this visualization in your research, please cite:

```bibtex
@software{text_metrics_viz,
  author = {Sandy Fraser},
  title = {Subline: A Text Metrics Visualizer},
  year = {2025},
  url = {https://github.com/z0u/subline}
}
```
