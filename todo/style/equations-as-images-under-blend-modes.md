---
status: done
tags: [vis, D2.2, blog]
opened: 2026-10-30
closed: 2026-11-05
---
# Show equations as images combined under the blend modes

A figure idea for the D2.2 post, noted in the paper round of the post-hoc comparison plan (2026-10-30). Draw a context as a row of small images: each example is `a ? b = y` with `a` and `b` as images and `y` their combination under the hidden op, and the query is `a ? b = y` where every pixel of `y` is what our model answers for that pixel pair, with the examples sampled from the images in the first row. A reader then sees the op the model inferred as a picture rather than as a table of colors, and an edited model as a different picture.

## Notes

**2026-11-05, equation images thread** — A working version is the [equation-images report](/docs/m2/equation-images/report.py): Kodak photos and two synthetic images as operands, posterized and dithered, one context per pixel, argmax and T = 1, on the eight `τ_ℓ fixed` runs before and after the full edit. The panels are also written as separate PNGs for an interactive figure. Still open: which panels go in the post, and whether to run at a larger scale.

**2026-11-05, equation images thread** — Settled at scale 1: the post uses the interactive widget from the report (`widget.html`, every panel at its native 192 × 128), with posterized operands, the in-step example walk and the most likely color by default. For a post, paste a copy whose image base is the absolute URL of the pinned publish, since the published `widget.html` finds its panels by relative path.
