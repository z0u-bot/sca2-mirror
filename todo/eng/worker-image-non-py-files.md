---
status: open
tags: [modal, mini]
opened: 2026-09-27
---
# The Modal worker image ships only the `.py` files of project packages

`modal_apparatus` builds the image with `add_local_python_source(*project_deps)`, which by default leaves out every file that isn't Python. So a data file inside a package is absent on a worker, and any module that reads one at import time fails there. `mini/reports.py` reads `lightbox.css` at module level, and `mini.vis` imports `mini.reports`, so a task that imports `mini.vis` (even indirectly) fails at setup on Modal while passing locally.

ex-2.2.17 round 3 hit this through `mini.temporal`, which imported `mini.vis` for one colour helper; that import now sits inside `plot_timeline`, so realizing a dopesheet on a worker no longer needs the CSS. The general case is still open. Two ways to close it, which could be taken together:

- Ship package data: pass an `ignore` to `add_local_python_source` that keeps the non-Python files of project packages (or add them with `add_local_dir`). This changes the image, so every worker rebuilds once.
- Read report assets lazily in `mini.reports`, at the first render that needs them, so importing the module has no file reads.

A test that imports each `mini` subpackage with the non-Python files hidden would catch the next one.
