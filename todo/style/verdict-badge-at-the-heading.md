---
status: open
tags: [reports, lit]
opened: 2026-09-22
---
# Show each hypothesis's verdict beside its heading

Sandy's review of ex-2.2.11 drew a circled tick and "PASS" beside the H1 and H2 headings and a cross and "MISS" beside H3 and the Decision. The Pass/Miss admonition sits at the end of each section, after a page of prose, figures, and tables, so a reader on their way through cannot see the outcome when they meet the question.

Proposal: a small badge in the heading (a tick or cross in a circle, or the word), drawn from the same status that `verdict_md` renders, so the two cannot disagree. In a literate script the heading is emitted before the section's results are computed, so either the status is computed up front (the report's `h*_status` functions already exist and are cheap) or the pipeline hoists it from the admonition at page-build time. The second keeps report authors out of it, and it works for old reports too.
