---
status: partial
tags: [D2.1, anchoring, ex-2.1.6]
---
# Labeling and pull-span variants for ex-2.1.6, queued behind the baseline read

Bumped 2026-08-05, during the ex-2.1.9 prereg review: with a pooled pull, (a) becomes the natural labeling — label any line containing *red* and let the pool find the position — and is probably the better match for what M3's document-level labels will look like. Original list follows:
(a) either-slot labels, where a red op2 can also trigger one; (b) labels drawn once at corpus build rather than per visit, so the pull comes from fixed sparse evidence, like labeled internet text; (c) a whole-span pull that includes the answer and newline — the shape a document-level label takes in natural language, where nothing marks a position as safe to exclude. Ex-2.1.6 pulls the prompt span only as a measurement instrument (the answer position has a known redness confound, and unpulled it doubles as a spillover read), so (c) is what says whether the exclusion matters rather than a premise of the method. Update 2026-08-08: (a) is now ex-2.1.10 (preregistered); (b) and
(c) remain open, and the ex-2.1.10 prereg points here for (c).

## Notes

**2026-08-18, housekeeping** — (a) has since run and published: [ex-2.1.10](https://z0u.github.io/sca2/ex-2.1.10/) found the pooled pull picks out the red operand line by line, with selectivity matching the slot oracle. So the labeling half is answered and the pull-span half isn't: (b) fixed sparse evidence drawn at corpus build, and (c) a whole-span pull including the answer and newline. Both are about what a document-level label looks like, which is M3's setting rather than D2.2's, so leaving this partial rather than promoting it.

**2026-09-09, ex-2.2.3 prereg review** — (c) is now also a D2.2 question, and a candidate fast-follow to [ex-2.2.3](/docs/m2/ex-2.2.3/report.py). The six-op grammar has lines whose answer is redder than both operands (`magenta darken yellow = red`, say), and under the operand-only labeller those answers draw no label: that is the blind span E3 reads. A labeller that reads the answer too is the M3-shaped labelling, where a document-level label says nothing about position, and the redder-than-both lines are where it would differ from the operand-only one. Left as is in ex-2.2.3, so that E3 reads the blind span first; the fast-follow would rerun the adopted point with the answer in the label's span and read E3 and the H4 statistics again on those lines.
