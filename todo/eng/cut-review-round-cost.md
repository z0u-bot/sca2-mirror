---
status: open
tags: [agents, reports, review]
opened: 2026-07-31
---
# Computed comparisons and single-sourced constants in reports

The cost half of this item is answered: the skills refactor ([PR #261](https://github.com/z0u/sca2/pull/261)) replaced `results-reviewer` with `sci-review-claims`, which spot-checks the numbers an argument rests on rather than recomputing every one. What is left is the defect half, below.

The background, from ex-2.1.7: its `results-reviewer` pass spent ~97k tokens, mostly on recomputing every quoted number from the store.

Which half broke: every quoted number was already correct, because reports read them from the store through f-strings at render time, so there is no transcription step to get wrong. All four real defects were natural-language relations wrapped around correct numbers: "well under" a null the value sits above; a span/op1 seam the rendered list contradicts; "far above" a ceiling cleared by 0.05; an effect attributed to the factor with the smaller main effect. Re-deriving numbers catches none of these.

Suggestions (but think carefully before implementing):

- Make comparisons computed rather than asserted. The deep fix, and the same trick that made the numbers safe in the first place. A small helper in the report layer — `rel(a, b, noise=...)` rendering "above" / "below" / "level with", and "clears by {x}" in place of "far above" — turns an inverted comparison from a prose bug into an impossibility. Every one of the four defects above was a relation a helper could have rendered.
- Make comparisons easy to check. As above, but fail loudly if it's not what is expected; an inline test. Probably easiest to put an `assert <condition>` in the cell; for something inline in the prose we could use a matcher library, along the lines of:

  ```py
  mo.md(f"""
  Some long paragraph...
  x is above{expect(x).toBeGreaterThan(floor, noise=...)} the floor
  """)
  ```

  ... but this needs more design and a cost/benefit analysis.

- Single-source the cross-experiment constants. These are the only numbers a reviewer must visit another experiment's store to check, and they are currently pasted and duplicated: `_EX216_MARGIN` = 0.2732 appears in ex-2.1.7's report four times under three names, plus `_REFS` for ex-2.1.3. They should live in `experiment.py` once, with provenance (ref, cells, statistic) in the docstring. Then it is one block to verify instead of a hunt, and it is the natural thing for a ledger to key on.

## Notes

**2026-10-13, holiday tidy-up** — Restated after PR #261: retitled to the two suggestions that still apply (comparisons rendered or asserted in code, and cross-experiment constants defined once), since the claims review no longer recomputes every number.
