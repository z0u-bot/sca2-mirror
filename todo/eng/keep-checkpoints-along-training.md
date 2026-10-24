---
status: done
tags: [experiments, storage, measurement]
opened: 2026-10-24
closed: 2026-10-24
---
# Keep a few checkpoints along training

Training runs keep only their end checkpoint. When [spill-timing](/docs/m2/spill-timing/report.py) asked when the spill appears, it had to train `pool 1–3, soft τ` again at eight seeds to apply the edit along the way. The [loss-rate screen](/docs/m2/loss-rate/report.py) left epoch-64 checkpoints only because its runs stopped there, and those had to be pinned with refs to outlive the memo records.

A checkpoint of the d64, four-block model is under a megabyte, so keeping one at epochs 16, 32, 64, 128 and 256 beside the end checkpoint would add a few megabytes per run. Any later question about when something appears (spill, the lean, a lost anchor) could then be scored on stored runs, at a coarse spacing, without retraining. Proposed in the planning of spill-timing.

The change goes in the shared training loop (`train_one` in the [ex-2.2.21 module](/docs/m2/ex-2.2.21/experiment.py), which later experiments reuse), so it changes the memo evidence of every experiment that trains through it, unless it is an opt-in argument.

## Notes

**2026-10-24, landed** — `train_one` takes `keep_epochs` (the usual choice is `KEEP_EPOCHS`, the five epochs above) and passes it to `train_anchored`, which saves a checkpoint after each of those epochs that falls before the end. The run returns them as one artifact, `kept_checkpoints`, so pinning them at publish costs one ref per run, and `kept_checkpoint` in the ex-2.2.21 module loads one by epoch. Saving draws nothing from the random streams, and a unit test checks that the end model and trajectory are the same with and without kept checkpoints. The default is off, so earlier experiments get the same results, though editing `train_one` and `train_anchored` moves the memo evidence of every task that trains through them.
