---
status: done
tags: [D2.2, anchoring, intervention]
opened: 2026-11-02
closed: 2026-11-05
---
# Make the reader depend on e₁: an edit-aware term

**Set aside (2026-11-05).** We suspect this term would train the model to bury the evidence: it can lower the loss by learning to ignore the off-axis part only when e₁ is gone, and it trains on the same edit we use to measure removal. Picking it up again would need very careful design. See the note at the end; the schedule changes moved to [shift-the-pull-before-the-readers-settle](./shift-the-pull-before-the-readers-settle.md).

The [low-runs report](/docs/m2/low-runs/report.py) found that every run writes the evidence that an example fits `difference` twice: along e₁, where the anchor put it, and along one other direction that the loss made on its own. The low runs are the ones whose reader blocks take enough of that evidence from off the axis for the e₁ edit to leave a fifth of the op in place. Nothing in training sees this: the twins (same labels, seed and batches, different last bits of arithmetic) match in loss and in α for the whole run and part only in removal.

So the lever is a term that the loss does see. One shape: on the labelled contexts, run a second forward pass with e₁ cleared at the pull sites, and train its prediction toward the target null (the sans-diff reference model's answer, or a flat distribution over the op's answers). That makes the reader depend on e₁ because the model is scored on what it does without it. It costs a second forward pass on the labelled lines, which are a small share of a batch. A cheaper shape: at the pull sites, drop e₁ at random on a fraction of the labelled contexts and ask for the same null then, a dropout on the anchored coordinate. The drop must come with the null target; under the ordinary task loss it would teach the model to do without e₁, the opposite of the aim. Either shape has a way around it: a state with no e₁ is off the usual arc, so the model could learn to answer the null on that cue alone and otherwise ignore e₁. Scaling e₁ by a random factor rather than zeroing it, with a graded target, leaves no single cue to detect.

## Notes

**2026-11-05, science desk** — The edit-aware term was planned in PR #346 and set aside on review, before any training. Training the answer under the edit toward the target null trains the measurement itself, so removal would stop being evidence that the op lives on e₁. The easiest way for the model to lower that loss is to learn to ignore the off-axis part only when e₁ is gone, which would leave the evidence where it is. The review drew the problem as two parts: the evidence lies on an arc with one sharp end (e₁) and one fuzzy one, so the off-axis part stays after the edit; and the reader sometimes settles on that weaker part. Two items now go after those: [anchor-the-not-difference-end-of-the-arc](./anchor-the-not-difference-end-of-the-arc.md) and [route-the-readers-learning-through-e1](./route-the-readers-learning-through-e1.md).

