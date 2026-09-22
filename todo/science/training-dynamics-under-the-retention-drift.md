---
status: open
tags: [D2.2, anchoring, schedules, ex-2.2.9, ex-2.2.10, representations]
opened: 2026-09-21
---
# Is the retention drift the model still reorganizing its representations? Training-dynamics reads beside the geometry read

[Ex-2.2.10](/docs/m2/ex-2.2.10/report.py) read the retention drop of ex-2.2.9's `handover` arm off the stored trajectories: the alignment reaches a noisy plateau by epoch 10, and on some seeds drifts slowly down under a constant anchor weight, before the anneal begins. The drift needs the whole-line labeller and the untied readout together; either half alone holds its plateau. What the drift *is*, the trajectories cannot say.

Sandy's reading, from the review of ex-2.2.10: perhaps the model is continuing to search for better representations, and the axis is carried along. The whole-geometry read ([PR #190](https://github.com/z0u/sca2/pull/190): RSA and Procrustes of anchored against control runs) sees a different latent geometry only in the presence of more ops *and* the anchor, which is the same conjunction the drift needs. So the two may be one phenomenon seen from two sides.

The proposal is to measure other training-dynamics quantities alongside the alignment trajectory, in particular the **local learning coefficient** (LLC): an estimate of the effective number of degrees of freedom the model is using near its current weights, from singular learning theory (Lau et al. 2023, *The Local Learning Coefficient: A Singularity-Aware Complexity Measure*; the `devinterp` package has an SGLD estimator). Developmental interpretability reads a change in the LLC as a phase transition: a plateau in the loss that hides a reorganization inside the model. If the LLC moves through the plateau of the alignment on `handover` and holds on `handover-slot` and `handover-tied`, the drift is a reorganization and the geometry read should find its signature at the same epochs.

What it needs: checkpoints through training rather than at the end (ex-2.2.9 kept end checkpoints only; the trajectory stride is 100 points), and an estimator run per checkpoint, which is a few hundred SGLD steps on the training loss. Cheap beside a run, and it fits the ex-2.2.11 re-run's train step if the checkpoints are stored on the trajectory stride for a few seeds.

Related: [retention under longer training](retention-under-longer-training.md) (closed; the anneal is not the cause), and the whole-geometry read in PR #190.

## Notes

**2026-09-21, ex-2.2.11 run** — the checkpoints now exist: the first three seeds of `handover`, `handover-slot`, and `handover-tied` store a checkpoint at every trajectory point under `reports/m2/ex-2.2.11/checkpoints/{label}/trajectory` (one tree per run; `TRAJ_CHECKPOINT_REF` in ex-2.2.11's `experiment.py`). At the fresh seeds the drift again shows on `handover` alone: its seed-mean alignment peaks near epoch 23 and loses about 0.05 by the end, while both references peak in or beside the anneal window (post-hoc section of the ex-2.2.11 report). Retention across the anneal is ~1.0 on every condition, so the anneal is not involved.
