---
status: open
tags: [reports, skills, writing]
opened: 2026-09-10
priority: high
---
# Report register: write the explanation a colleague would get over lunch, from the first draft

More than once a report draft has been hard to follow, Sandy has asked "can you explain this report to me?", and the chat explanation that came back was what the report should have been. The writing, text-lint, prose-simplifier, and report-restructure skills have not moved the needle much on this, which suggests the fix is in the workflow rather than in another rule: the skills polish sentences, and the problem is the register the first draft is written in.

What the good chat explanations have in common: they lead with what we found in one plain sentence, they say what each number means before giving it, they use the same everyday words throughout (the model loses *red*; the answer stops depending on the red operand) instead of the statistic names, they gloss each statistic once in a phrase, and they say what we make of it and what we would do next. They do not open with the gate arithmetic, and they do not say "reads as" or "carries".

Proposed workflow change, to try on the next report: (1) before touching the notebook, write the Findings and Discussion as a message to Sandy, in the chat, as if explaining the results over lunch; (2) paste that into the report as the first draft of those sections, template expressions added afterwards; (3) only then run the polishing skills, and run them on the H sections rather than on the lunch text. A second, cheaper check: after the render, ask a fresh agent to explain the report back in plain words, and diff its explanation against the Findings; where they differ, the report is the one to change.

The skills should say this too: the writing skill's opening line becomes "write the lunch explanation first", the report-structure agent checks whether the Findings could be read aloud to a colleague, and the glossary style item (glossary of preferred terms, under eng) picks the everyday word for each statistic so the same phrase appears everywhere.

## Notes

**2026-09-10, Sandy, ex-2.2.3 review** — "the explanation you gave was excellent, and I would have loved that to be the actual report (right from the first draft). We have a lot of skills on report writing and reviewing and style, but somehow it hasn't moved the needle much." The ex-2.2.3 Discussion rewrite in the same round is the first trial of the lunch-first draft.
