---
status: open
tags: [D2.2, anchoring, schedules, ex-2.2.9]
opened: 2026-09-17
---
# Retention under longer training: is the anneal window the reason the margin drifts?

[Ex-2.2.9](/docs/m2/ex-2.2.9/report.py) missed retention on one `handover` seed of twenty (0.77 against the 0.8 gate), with a seed mean of 0.88 against 0.96 on both references and 0.99 at ex-2.2.3. The decision rule reports retention without counting it, so the miss did not decide anything, but the seed mean is low across the arm, and one seed in twenty with no band on retention does not separate a real cost of the label or the readout from one seed's luck.

The schedule is the other suspect, and the review's first guess. The recipe anneals the anchor weight over the last tenth of training, as ex-2.1.10 did. Ex-2.2.9's runs are three times longer than the reference's (4,950 steps against 1,650), so the anneal window is 495 steps against 165, and the margin has three times as long to drift after its peak while the pull fades. Every ratio against the reference carries that caveat, and retention is the ratio it bites hardest.

Three things to try, each cheap beside a re-run of the handover:

- **Read the trajectories we have.** The stored trajectories show when the margin peaks and how it drifts through the anneal on each `handover` seed. If the drift begins where the anneal begins and is proportional to its length, the schedule is the cause; if the low seed drifts earlier, it is not.
- **Anneal later, shorter, or in steps.** Set the window in steps (the reference's 165) rather than as a fraction of training, or start it later, on a few seeds, and read retention.
- **Give retention a band.** The gate is per run and the statistic is the minimum across seeds, so there is no seed-mean comparison with a band. A read that twenty seeds can resolve (the seed mean with the per-run spread, beside the minimum) would let the next report say whether `handover`'s 0.88 differs from the references' 0.96.

Related: [containment rises under the untied readout](containment-rises-under-the-untied-readout.md), where ex-2.2.9's H2 results wrote a mechanism and a check.
