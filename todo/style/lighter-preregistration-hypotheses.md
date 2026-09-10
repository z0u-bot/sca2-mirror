---
status: partial
tags: [methodology, reports, skills]
opened: 2026-09-10
---
# Lighter preregistration: hypotheses in the Fowler register rather than the trial-protocol one

Our preregistrations have drifted toward clinical-trial precision: every hypothesis carries a gate, a partial band, a contrary reading, and a resolution rule with tie-breaks, and the report then scores each clause. That has caught real problems (a tautological hypothesis in ex-2.2.2, a selection rule that could be read three ways in ex-2.2.3), but it also costs days per experiment, makes the reports hard to read, and does not stop the misses that matter: ex-2.2.3's selection rule left out one of H2's gates and adopted a point that missed it, and its removal statistic was the wrong one for four of six ops. Precision in the wording did not buy correctness in the design.

The proposal is to keep preregistering and lower the formality: state each hypothesis the way Fowler or a Thoughtworks tech radar would, as a plain expectation with the one number we will look at and what would change our mind, and keep one or two hard gates where a decision hangs on them (which operating point is adopted, whether a risk row closes). Everything else is a prediction we write down before the run and check after it, in a sentence, with the reader's question answered first: what did we expect, what did we see, what do we make of it.

Concretely: (1) the science skill's preregistration template gets a short form alongside the long one, and the short form is the default for survey and scouting rounds; (2) the prereg-reviewer's checklist asks whether each hypothesis is one a colleague could restate from memory, and whether the selection rule carries every gate the hypotheses do; (3) the H-section shape becomes expectation, read, and what it means, with the gate arithmetic in a details block or the method.

## Notes

**2026-09-10, Sandy, ex-2.2.3 review** — "I still think we should preregister, but I wonder if we can be less precise? Hypotheses more Thoughtworks/Fowler-like and less $1B-pharma-trial-like?" Agreed on the results-discussion side; this item is the plan.

**2026-09-10, Claude** — the science skill's preregistration conventions now describe the lighter form (expectation, the one number, what would change our mind; hard gates only where a decision hangs on them), the section shape (expectation, read, what it means, with gate arithmetic in a details block or the method), and the rule that a selection rule carries every gate. The prereg-reviewer checks both. Left partial: the next preregistration is the trial, and the skill's example block still shows the long form, which is worth replacing with a short-form example once one exists.
