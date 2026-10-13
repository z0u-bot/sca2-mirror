---
status: open
tags: [D2.2, anchoring, intervention]
opened: 2026-10-13
priority: high
---
# The write bound and the landing on the inferred op

With a recipe of record that holds at fresh seeds (split-anti-seeds, `falloff`), two of the claims in the [D2.2 design doc](/docs/m2/d2.2/design.md) still have no experiment behind them.

- The write bound, claim (d): because we placed the axis, we can say in advance how far an edit moves each state. This held on *red* in ex-2.2.1 (H4) and has not been measured for the inferred op.
- Where the edited model lands, claim (c): the full edit covers about half of the distance to the target null, context by context (landing 0.52 on `falloff`). Is that enough, or is a trained fallback wanted to close the rest?

The design doc has the plan for both under "What comes next"; this item is the backlog entry for them, so the shortlist names what D2.2 still has to show beside the recipe sweep.
