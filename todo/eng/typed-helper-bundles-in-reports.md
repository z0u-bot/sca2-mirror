---
status: open
tags: [reports, skills]
opened: 2026-09-09
---
# Report helper cells could be typed classes

Several reports define their per-report views (`stat`, `clean`, `floor`, `per_line`, `trajectory`, `seed_row`) as closures inside one `@app.cell`, because they share the loaded `runs`, `arrays` and `traj`. Nothing in those cells is type-checked at its call sites: Marimo generates a bare parameter for each closure, so `ty` sees `Unknown` and a misspelled name or a wrong argument passes silently. Ex-2.2.1 and ex-2.2.2 both carry a bundle like this.

`@app.class_definition` covers the case. Bind the shared state as constructor arguments (`Stats(runs=..., arrays=...)`), keep the closures as methods, and annotate the instance where it is bound — Marimo then copies the class onto every downstream cell signature and `ty` checks the uses as well as the bodies. The `style-py` skill has the pattern and a worked example.

Converting a published report churns a lot of lines for no change in output, so this is worth doing as reports are next edited rather than as a sweep. Ex-2.2.3 is the first candidate: it is still a preregistration, so its bundle can be written this way from the start.
