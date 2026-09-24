---
status: open
tags: [D2.2, anchoring, M3]
opened: 2026-09-24
---
# Label variants for the inferred op

In the [D2.2 pivot](/docs/m2/d2.2/pivot.md), a binary label on the whole context asks for the op at positions that cannot know it yet: early in the line, where attention has seen too little, and in contexts where replacement noise leaves the posterior on the true op below 0.5. The pooled anchor term softens the first case only. The whole-line label stays the default, since an M3 labeller would likely give that form. Three cheap variants should measure its effect on the task and the anchor:

- Leave out the embedding slice.
- Pull only the latter half of each line, where the posterior given the prefix is at or near its final value. This is the simplest fix along position.
- Label by a thresholded prefix posterior: a position is labelled when the posterior given the tokens before it clears a threshold. This handles both position and evidence, and is the form an M3 labeller with a confidence cutoff would give (a classifier run on each prefix of a conversation). It also leaves the contexts in the middle band of the posterior unlabelled, so alignment there shows grading the pull never touched, as D2.1 did for *red* with binary labels.

Which variant becomes the primary would be decided on the pilot. Whether the thresholded label helps is one input to the soft-label question in the pivot.
