---
status: open
tags: [tooling, typing]
opened: 2026-09-15
---
# Consider enabling ty's `unsound-assignment` rule

ty has an opt-in rule (`unsound-assignment`, off by default) that flags assignments where the value's inferred type is *assignable* to the declared type but not a *subtype* of it. In practice that means one thing: an `Any` landed in a typed slot. `Any` is assignable to everything, so the normal `invalid-assignment` check waves it through, and a wrong type can then travel a long way before anything notices.

A trial run (`uv run ty check --warn unsound-assignment`) finds 28 across the repo, spread over 19 files — a few in `src/mini/`, a few in `src/sca/`, a couple in `scripts/` and two reports.

The catch is that most of them are `Any` arriving from an untyped third-party call rather than a mistake of ours. `eqx.apply_updates` returns `Any`, so `model = eqx.apply_updates(model, updates)` is flagged even though it is right. Fixing those means `cast()` at the boundary, or a local protocol — worth it where the boundary is one we cross often, noise where it isn't.

So the question to answer before switching it on is whether the findings cluster at a handful of library boundaries (worth annotating once) or scatter (not worth it). If it's the former, enable the rule and `cast()` at those boundaries; if the latter, leave it off and note here that we looked.
