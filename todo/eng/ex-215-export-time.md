---
status: open
tags: [performance, reports]
opened: 2026-07-28
---
# Ex-2.1.5 export time: what's left

Two fixes landed — `baselines.precision_limited_acc` now finds the nearest candidate with a matmul, and `geometry.principal_angles` batches over leading dimensions so the report's 2000-sample null is one pair of LAPACK calls. Both are bit-identical, verified. Interleaved A/B on the same machine: ~85s → ~46s.

What's left is matplotlib, and it's structural. `@themed` renders each figure twice, and `base.mplstyle` sets `figure.constrained_layout.use: True`, so every save solves a layout: 9.4s across the 18 saves, with ~13s in `get_tightbbox` overall. Measured on a synthetic multi-panel figure, the layout engine costs about as much as drawing the figure again, and it scales with panel count rather than data volume: +0.05s per panel (0.05s at 1 panel, 0.56s at 12, 1.23s at 24), and flat from 200 to 20,000 points per panel. It has to measure the rendered extent of every tick label, axis label and title to allocate margins, then iterate the solve; our figures are 12-panel grids, so we pay it 12 times over.

Two things to try, in order of bluntness:
- `themed_figure_html` passes `bbox_inches="tight"` on top of constrained layout, which makes `print_figure` draw the figure a second time to measure the crop (the profile shows 36 `figure.draw` for 18 saves). Measured ~20% off a save with no visible change in what constrained layout already produced — the two are solving nearly the same problem.
- Beyond that it's the engine itself, and the options are fewer panels per figure or a fixed layout for the grid-shaped figures that don't need solving. Both change margins, so eyeball across a few reports rather than swapping blind.

Reproduce with: `python -m cProfile -o p.prof` around a `MINI_EXPORTING=1` `runpy` of the notebook, or monkeypatch `DefaultExecutor.execute_cell` for per-cell times. Beware absolute timings across container restarts — the box this was measured on drifted ~25% between sessions, so A/B interleaved.

## Notes

**2026-08-17, grading sketch** — a third instance of the draw-twice pattern, and a fix worth knowing. `docs/m2/grading_sketch.py:fig_grid` called `fig.canvas.draw()` to settle the panel boxes before adding overlay axes, then saved: two full paints. Replacing it with `fig.get_layout_engine().execute(fig)` settles the same boxes (panel positions bit-identical, `max|Δ| = 0`) at 0.47s against 3.49s, and took the figure from 7.1s to 3.7s. Any place that freezes a constrained layout mid-build can use it.

That measurement also bounds the layout claim above: on a 30-panel figure carrying 1.8M scatter marks, `layout=None` saved nothing (savefig 3.39s vs 3.19s constrained) — the cost was marker rasterization, and it tracked mark count (60k/panel → 3.0s, 1k/panel → 0.25s). So the per-panel `get_tightbbox` cost dominates only while the panels are cheap to paint; for mark-heavy figures, paint count is the thing to cut.

**2026-09-09, housekeeping** — export has gained a step since this was profiled, and it is worth saying what kind of step before someone re-profiles and wonders whether the numbers above still mean anything. #154 added `write_thumbnails` to `scripts/export_reports.py`, which writes a small copy of every asset-served figure. It works on the bytes the report already saved — `mini.reports._thumbnail_bytes` takes a `Path` and goes through Pillow — so it costs one decode, scale and encode per figure, and no matplotlib draw or layout solve. The diagnosis here is unchanged, and so are the two leads: dropping `bbox_inches="tight"` from `themed_figure_html`, then a fixed layout for the grid-shaped figures. Un-measured, so if a fresh profile shows the thumbnail pass costing real time it is its own item rather than a correction to this one.
