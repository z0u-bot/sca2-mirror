---
status: open
tags: [experiments, storage]
opened: 2026-11-12
---
# Kept-checkpoint uploads that stall at the end of training

In the flat-anti experiment (48 runs on L40S cards), 10 training tasks sat in the `put` of their kept checkpoints (`put ex-2.2.21-flat-anti-…-kept`) for about eight minutes after their last step. The heartbeat was frozen, the blocking-phase budget the `put` declares had expired, and the GPU still showed about a quarter of its use. The watchdog did not abort them. Each one was cancelled and retried by hand, and every retry trained from scratch, which about doubled the cost of the run ($17.87 against an estimate of $8).

Two things would help. First, find out why the watchdog let a `put` that had passed its declared budget run on, since the budget is meant to bound the span. Second, a retry after training should start from the trained state where it can, rather than re-running 400 epochs to repeat an upload. The second one probably means storing the final checkpoint before the kept ones, or letting a task resume past its last completed phase.

## Notes

**2026-11-12, flat-anti thread** — A suggested shape for the second fix: split the upload out of training. `train_one` (ex-2.2.21) already trains in a working directory on the experiment Volume (`get_data_dir() / "cells" / label`), but it clears `along/` and `resume/` at the start of each attempt, puts the kept and final checkpoints inside the same step, and commits nothing until the step ends, so a cancelled attempt leaves nothing to resume from. Training could instead write both checkpoints to the Volume, call `volume.commit()` (as `eng/decisions.md` already advises for an expensive checkpoint), and return their names; a separate upload step on a small CPU container would then `put` them to the bucket. A stalled upload would then retry the upload alone. A Volume path in a result is the location-pinning that `eng/artifacts.md` warns about, so the handoff should stay inside one experiment run, with the upload's `Artifact` as the only thing later steps see. `train_one` is shared by many experiments, so the change moves their fingerprints, and their finished results need `--keep-stale-done`.
