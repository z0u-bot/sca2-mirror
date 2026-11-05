---
status: open
tags: [D2.2, anchoring, intervention]
opened: 2026-11-05
---
# Anchor the "not `difference`" end of the arc to a second axis

The [low-runs report](/docs/m2/low-runs/report.py) found that the evidence that an example fits `difference` lies on an arc. The end where an example fits sits on e₁, where the anchor put it; the end where it doesn't fit leans along one direction off the axis, which the loss chose on its own. Removing e₁ leaves that second part in place, and the low runs are the ones whose reader blocks lean on it.

So design the whole arc instead of one end of it. A second anchor would pull the "does not fit `difference`" end toward e₂, from the same labelled contexts, at a lower weight than the pull toward e₁. The arc would then lie in a known plane, and an edit could act on the plane rather than on e₁ with an unknown remainder. The lower weight is because the "not" end seems to need some freedom: it also holds what tells the other ops apart. When we anchored *red* to a plane ([anchor-red-to-a-plane](./anchor-red-to-a-plane.md), the `plane` conditions of ex-2.2.12), the constraint seemed to cost the task, which may be this same need.

Things to settle in a plan: what the second pull's target is (the "not fit" examples of labelled `difference` contexts, or something graded by the posterior); whether the edit removes the plane or only e₁ with e₂ held in place; how the anti term treats e₂; and how to tell the arc landed in the plane (the off-axis evidence measured with both axes zeroed, as low-runs did with e₁). The screen is the same as for any recipe change: eight seeds at 1/50 on fixed labels, removal and its low tail, spill, and task change.

This came from the review of the edit-aware plan (PR #346, closed). That plan trained the answer under the edit toward the target null, which trains the measurement itself and gives the model an easy way out: learning to ignore the off-axis part only when e₁ is gone. This item and [route-the-readers-learning-through-e1](./route-the-readers-learning-through-e1.md) go after the cause instead.
