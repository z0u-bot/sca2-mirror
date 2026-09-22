Visualize metrics as sparklines under text.

![Screenshot of text that reads, "A long time ago, in a galaxy somewhat far away..." A sparkline beneath the text shows that the word "somewhat" is clearly out of distribution (i.e. unexpected) in this context.](../../doc/subline.svg)

## Styling

The SVG carries its own theme (`theme.css`, light/dark aware) as CSS custom properties, scoped to `svg.subline`. Each colour reads a page token first and falls back to the library's own value — `--bg-color: var(--bg, …)`, `--col-text: var(--fg-muted, …)`, `--col-baseline: var(--border, …)`, and the text's `font-family: var(--font-mono, …)` — so inlined in a page that defines those tokens the figure takes the page's theme, and opened on its own it takes its own.

To restyle one figure, pass `vars`: custom properties set on that SVG's root element, which reach that figure alone:

```python
Subline(vars={"--bg-color": "light-dark(#fff, #181c1a)"}).plot(text, series)
```

Settable properties include `--bg-color`, `--col-text`, `--col-baseline`, `--col-series-1..5`, and `--blend-mode`. For restyling beyond what the properties reach there is `css`, appended after the built-in theme — but inlined into HTML, an SVG's `<style>` is not scoped to that SVG: it joins the document's stylesheets and reaches every element on the page, so `css` themes every subline on the page at once, and the last one in document order wins. `mini.vis.svg_figure` keeps one copy of an identical block per figure.

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
