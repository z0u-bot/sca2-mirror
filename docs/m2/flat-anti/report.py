# ruff: noqa: B018
# title: Flat anti weight

# An exploratory report: the recipe of record, `τ_ℓ fixed`, trained at seeds 720 to 727 under five anti-subspace
# weights (the falloff over depth, two holds that are the same at every slice, and one weight held level from the
# start), beside a control at each seed. `experiment.py` beside this script trains and scores the runs. This is the
# plan: results are TODO.

r"""
# Flat anti weight

/// tip |
<!-- lede -->
TODO: whether an anti weight that is the same at every slice does as well as the falloff over depth, and whether the high start is still needed.
///

The recipe of record anneals the anti-subspace weight from 0.25 on every slice to a hold that falls off with depth, from 0.2 at the embedding to 0.03 at slice 4. The falloff came from the [split-anti-seeds](/docs/m2/split-anti-seeds/report.py) experiment, where both holds that were the same at every slice did worse. But that experiment used an older pull: a step on slices 2 to 4, with a sharp pooling temperature from the start and no pool over slices.

Each flat hold there lost the anchor on one run, but those losses can't have come from the hold. The lost runs parted from the kept ones within the first sixty epochs ([loss-rate](/docs/m2/loss-rate/report.py)), while the anti weight was still near its start of 0.25 in every condition, and the falloff later lost one run of forty in the same way. A soft start for the pooling temperature has since stopped those losses. What remains against the flat holds is the rest: the high hold removed less on the seeds where the pull settled slowly, and the low one spilled. A sharp pool over slices 1 to 3 has since taken away the spill from block 1 ([soft-pool-scout](/docs/m2/soft-pool-scout/report.py)), so a flat hold may now do as well. That would make the recipe simpler to describe.
<!-- REVIEW: the plan first said the flat holds failed by losing the anchor. The assumptions review found the losses parted from the kept runs by about epoch 60, when the anneal had moved the weight by under 0.008 in any condition, so the hold cannot have caused them. Verify: the loss-rate report on when lost runs diverge, and AntiSpec at epochs 20 and 64. -->

This report trains the recipe at eight seeds under five anti weights, with a control at each seed: 48 runs.
"""

# %%

r"""
## Observations

- [Keeping the anchor (E1)](#keeping-the-anchor-e1): TODO.
- [Removal and spill at the end (E2)](#removal-and-spill-at-the-end-e2): TODO.
- [Along training (E3)](#along-training-e3): TODO.

## Scope and method

This is an exploratory study, with no preregistration and no gate of its own. Every run is the recipe of record, `τ_ℓ fixed` as the [fixed-confirm](/docs/m2/fixed-confirm/report.py) experiment trained it, with one change: the anti weight.

Conditions {.manual}
:   `recipe` is `τ_ℓ fixed` itself, with the falloff over depth, retrained here so that every condition shares the card and the arithmetic. `flat 0.115` anneals from 0.25 to a hold of 0.115 at every slice, the mean of the falloff over the five slices, so the term carries the same total weight spread evenly. `flat 0.2` anneals to 0.2 at every slice, the falloff value at the embedding, which kept the color embeddings from standing in for the op ([embedding-lean](/docs/m2/embedding-lean/report.py)). `level 0.115` and `level 0.2` hold their value at every slice from the first epoch, with no high start; `level 0.2` is close to `flat 0.2`, which only starts at 0.25.

Every condition keeps the end anneal it shares with the anchor, over the last 40 epochs. The anneal from 0.25 is slow to start, so the high start lasts: at epoch 64 the weight is still within a hundredth of 0.25 on every slice of every annealed condition, and at epoch 200 it is a little over halfway to the hold.

The two flat holds match the falloff on its mean and on its highest value. `flat 0.115` is lower than the falloff at slice 1, the same at slice 2, and higher at slices 3 and 4, and `flat 0.2` is higher at every slice but the embedding. No flat hold sits below the falloff on the slices the pull acts on, since 0.03 at every slice let the color embeddings lean onto e₁ in the [late-pull](/docs/m2/late-pull/report.py) experiment.
<!-- REVIEW: was "The two flat holds bracket the falloff on the slices the pull acts on". On slices 1 to 3 the falloff holds are 0.1575, 0.115 and 0.0725, so 0.115 is their middle and 0.2 is above them all; neither is below. Verify: FALLOFF_HOLDS. -->

`level 0.115` asks a separate question, and its comparison is with `flat 0.115`, which differs only in the start. In the [ex-2.1.11](/docs/m2/ex-2.1.11/report.py) experiment, an anti weight held at its low value from the start let the pull latch onto a syntax token. That low value was 0.3 times the anchor weight, where 0.115 is a little more than the anchor weight (0.1), so `level 0.115` is a milder case than that one. The soft start of the pooling temperature now spreads the early pull over positions, and may do that job instead.
<!-- REVIEW: added the scale of the ex-2.1.11 arm (`anti-hold`, ratio 0.30 against a peak ratio of 2.5), since "no high start" could be taken as a repeat of it. Verify: `lam` 0.1 and `hold_ratio` 0.3 in the anti schedule. -->

Seeds 720 to 727 have no finished run of the recipe; the loss-rate counts stopped them at epoch 64. We compare conditions as two samples of eight runs, with a 95% Welch t interval on the difference of seed means,[^welch] and Fisher's exact test for counts of runs (lost, or over the spill criterion). At eight runs a side, removal gaps under about 0.05 are hard to see, but the seed range of the recipe (about 0.08 in fixed-confirm) suggests an interval of about ±0.03 on removal. So the experiment can bound what a flat hold costs. We take a flat hold to do about as well when its removal is within 0.05 of the recipe, it loses no anchor, and no run is over the spill criterion. Every run trained and was scored on L40S cards. The earlier runs of the recipe (704 to 719) trained on L4 cards with the determinism flags on, so they appear only as context.

[^welch]: Welch t interval: a confidence interval for a difference of two means that does not assume the two samples have equal variance.

## The measurements

Every edit below is the e₁ edit. Removal and spill are each a drop under that edit, less a baseline: the median drop when the same edit is made along each of the other 63 coordinate axes of the run (the rule of the D2.2 design).

Removal
:   The drop on `difference` under the full edit at every position and slice, as a share of the way from the clean score to the target null. Higher is better, and a run below 0.5 has lost the anchor.

Spill
:   The largest drop on any other op, at the dose where it is largest. Lower is better, and a run above 0.02 is over the criterion. Along training (E3) it is at the full dose only.
    <!-- REVIEW: was "under the full edit". The criterion of the D2.2 design is at any dose, the axis scoring here stores every dose, and the checkpoints along training are scored at the full dose alone, as in fixed-confirm. Verify: `rb.axes_one` and `ac.kept_axes_one`. -->

Block-1 spill
:   Spill under the full edit at slice 1 alone, where the spill of earlier recipes entered.

Task change
:   The clean score per op, less the mean of the controls. The gate of the D2.2 design asks for a mean change over ops of at least −0.02, and at least −0.05 on any one op.

Call
:   The early verdict on whether a run kept the anchor, from mean α at the example answers of the `difference` probe contexts at slice 3, over epochs 44 to 64. A run below 0.15 is called lost, and one at 0.28 or above is called kept. The cuts were set on runs whose anti weight was near 0.25 over the window, so for the level conditions the call is descriptive and removal at the end decides.

Embedding lean
:   The slope of the e₁ component of the color embeddings against lightness, a sign of a color embedding standing in for the op.
"""

# %%

r"""
## Keeping the anchor (E1)

Does every condition keep the anchor, and does it settle as quickly? Up to the call at epoch 64 the three annealed conditions are one recipe, so losses there test the level conditions alone; a level run that aligns a syntax token or the query answer would show here. A run of any condition can still lose the anchor after epoch 64. In the loss-rate experiment, the slow runs of the older recipe hung until the anti weight at slice 3 had fallen to about half, and a hold of 0.2 never falls that far, so the speed of settling may separate `flat 0.2` from the rest.

TODO: the call at epoch 64; the epoch at which α at the example answers first crosses 0.4, by condition; removal at the end, by condition and run; α by slice and role at the end for any lost run.

"""

# %%

r"""
## Removal and spill at the end (E2)

Among the runs that keep the anchor, does a flat hold remove as much, and spill as little? Under the older pull, the high flat hold removed less on the seeds where the pull settled slowly, and the low one spilled. Block-1 spill and the embedding lean say where any spill comes from.

TODO: removal, spill, block-1 spill, task change and embedding lean per condition (seed mean and range, and the Welch interval against `recipe`, and for each level condition against the flat hold of the same value as well).
"""

# %%

r"""
## Along training (E3)

The late spill of earlier recipes rose as the anti weight annealed ([late-spill-scout](/docs/m2/late-spill-scout/report.py)), and a flat hold changes what the weight anneals to. Does spill rise late under any condition, and does α at the example answers settle as it does under the falloff?

TODO: spill and α along training, from the checkpoints every 32 epochs, with the anti weight by slice plotted underneath.
"""

# %%

r"""
## Discussion

TODO.
"""

# %%

r"""
## References

[split-anti-seeds](/docs/m2/split-anti-seeds/report.py)
:   Under a step pull on slices 2 to 4, an anti hold that falls off with depth kept the anchor on all eight runs, while holds of 0.2 and 0.1 at every slice each lost one run to the query answer, and the high one removed less.

[late-pull](/docs/m2/late-pull/report.py)
:   With the pull moved off the embedding, an anti hold of 0.03 at every slice let the color embeddings lean onto e₁ and block 1 spill, and higher holds reduced both at some cost to removal.

[embedding-lean](/docs/m2/embedding-lean/report.py)
:   The spill of the older recipe came from a color embedding standing in for the op: the e₁ component of the color embeddings followed their lightness.

[loss-rate](/docs/m2/loss-rate/report.py)
:   With a soft start for the pooling temperature, no run of forty lost the anchor, against one of forty without it.

[soft-pool-scout](/docs/m2/soft-pool-scout/report.py)
:   Keeping the pool over slices 1 to 3 at a sharp temperature removed the early spill, which entered through block 1.

[fixed-confirm](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed` kept the anchor on all eight runs at fresh seeds with removal near 0.9 and no run over the spill criterion, and became the recipe of record.

[late-spill-scout](/docs/m2/late-spill-scout/report.py)
:   Late spill on the recipe rose as the anti weight annealed, and holding the anti weight high lowered it at a cost to removal.

[ex-2.1.11](/docs/m2/ex-2.1.11/report.py)
:   In D2.1, an anti weight held at its low value from the start let the pull latch onto a syntax token, and one held at its high value throughout left the response less graded.
"""
