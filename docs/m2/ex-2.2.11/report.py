# title: Ex 2.2.11: the handover re-run

from sca.data.colors import redness
from sca.data.ops import probe_lines

# The design constants come from `experiment.py` beside this script (the script's directory is on
# sys.path while it runs). The prose quotes the frozen gates, and that module carries the same
# numbers with each gate's wording in its docstring.
import experiment as ex

RESULTS_TO_COME = "/// admonition | TODO\n    type: warning\nResults to come.\n///"


def removal_counts() -> dict[str, dict[str, int]]:
    """Per op, on its probe set: the red lines, the removal lines under the hue rule, the removal lines under
    ex-2.2.9's to-zero rule, and the saturation-and-value lines the hue rule sets aside. The order-sensitive
    ops are also split by which slot the red operand is in.
    """
    out = {}
    for op in ex.TABLE:
        both = op.name in ex.PROBE_BOTH_SLOTS
        red = [
            ln
            for ln in probe_lines(op, ex.N_PROBE, ex.PROBE_SEED, both_slots=both)
            if max(redness(ln.lhs), redness(ln.rhs)) >= ex.RED_DOSE
        ]
        hue = [ex.hue_move(op, ln.lhs, ln.rhs) >= ex.FAR_MOVE for ln in red]
        zero = [ex.ex229.to_zero_move(op, ln.lhs, ln.rhs) >= ex.FAR_MOVE for ln in red]
        row = {"red": len(red), "hue": sum(hue), "zero": sum(zero), "sv": len(red) - sum(hue)}
        if both:
            at1 = [redness(ln.lhs) >= redness(ln.rhs) for ln in red]
            row["hue_op1"] = sum(h for h, a in zip(hue, at1, strict=True) if a)
            row["hue_op2"] = sum(h for h, a in zip(hue, at1, strict=True) if not a)
        out[op.name] = row
    return out


counts = removal_counts()


def removal_md() -> str:
    head = "| op | red lines | removal (hue) | removal (to zero) | saturation-and-value | hue, red at op1 | hue, red at op2 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    rows = []
    for op, c in counts.items():
        split = (f"{c['hue_op1']}", f"{c['hue_op2']}") if "hue_op1" in c else ("", "")
        rows.append(f"| `{op}` | {c['red']} | {c['hue']} | {c['zero']} | {c['sv']} | {split[0]} | {split[1]} |")
    return head + "\n".join(rows)


def rule_disagreement(name: str) -> int:
    """How many red lines of an op one rule calls a removal line and the other does not."""
    op = next(o for o in ex.TABLE if o.name == name)
    red = [
        ln
        for ln in probe_lines(op, ex.N_PROBE, ex.PROBE_SEED, both_slots=op.name in ex.PROBE_BOTH_SLOTS)
        if max(redness(ln.lhs), redness(ln.rhs)) >= ex.RED_DOSE
    ]
    hue = {i for i, ln in enumerate(red) if ex.hue_move(op, ln.lhs, ln.rhs) >= ex.FAR_MOVE}
    zero = {i for i, ln in enumerate(red) if ex.ex229.to_zero_move(op, ln.lhs, ln.rhs) >= ex.FAR_MOVE}
    return len(hue ^ zero)


sv_share = {op: counts[op]["sv"] / counts[op]["red"] for op in ex.ORDER_SENSITIVE}
low_share = min(c["hue"] / c["red"] for op, c in counts.items() if op not in ex.ORDER_SENSITIVE)

r"""
# Ex 2.2.11: the handover re-run

/// tip |
<!-- tl;dr -->
Ex-2.2.9 turned on every change from the scouting round at once, and we did not adopt the result. Two gates missed: removal, on the three ops that take one HSV attribute from their second operand, and one seed that held the anchor a little less well by the end. Ex-2.2.10 traced both misses to how we were measuring. So this is the same experiment at fresh seeds, with three measurements fixed in advance: removal lines picked by hue, retention measured across the anneal, and the op1 alignment reported beside its references rather than gated.
///

## Findings

- [Does the model still learn the task? (H1)](#does-the-model-still-learn-the-task-h1) —
- [Does *red* land, and stay through the anneal? (H2)](#does-red-land-and-stay-through-the-anneal-h2) —
- [Does *red* come out on the lines that need its hue? (H3)](#does-red-come-out-on-the-lines-that-need-its-hue-h3) —
- [Do the two reference reads repeat? (H4)](#do-the-two-reference-reads-repeat-h4) —
- [Decision](#decision) —

/// admonition | How to read this draft
This is a preregistration. We wrote the conditions, the gates, and the decision rule before any run and froze them at commit `TODO`. Everything after that commit is either results filled into the frozen sections or exploratory. Each hypothesis section opens with what we expect and the one number we will look at, and the results go into that section in place once they exist. Anything we think of after seeing the data goes under [Exploratory analyses](#exploratory-analyses), marked as post hoc. Every count in the method is computed from `experiment.py` at render time.
///

## Why this experiment

We want one setup that every later D2.2 experiment is built on. It has two parts: the grammar, meaning which color operations the model learns, and the recipe, meaning how *red* is anchored on one axis of the residual stream.[^rs] [Ex-2.2.9](../ex-2.2.9/report.py) was the handover to that setup: it combined the four changes the scouting round had tried one at a time, and asked whether *red* still lands on its axis and still comes out cleanly.

[^rs]: The *residual stream* is the running vector of activations that each layer of a transformer reads from and writes back to.

Most of it held. The model learned the eleven ops as well as an un-anchored model did, *red* landed with the same margin as before, and neither the separate readout nor the whole-line label cost anything we could resolve. Two measurements missed their gates, so the rule said no.

[Ex-2.2.10](../ex-2.2.10/report.py) then went back to the stored runs and traced both misses to how we measured. The removal read counted a line as needing *red* when zeroing the R channel of the red operand moved its answer far. But zeroing R on pure red gives black, which has no saturation or value either, so the rule also swept in lines whose answer takes only the saturation or value of red.

The projection leaves those lines alone, because on the red operand it acts like a change of hue. On the lines whose answer does need the hue, every seed already cleared the gate.

Retention was the second miss: that read divided the final alignment by the peak alignment over the run. On the candidate, the peak is the high point of a noisy plateau reached thirty epochs before the anneal, and across the anneal itself nothing is lost. Separately, the op1 alignment rises under either half of the handover, so its old reference belongs to the old grammar.

A read chosen after looking at the data cannot then score that data. So this is ex-2.2.9 again at seeds it never trained, with the three reads fixed here first. We re-run rather than rescore because the claim we want is the preregistered one.
"""

rf"""
## Conditions

Everything about training is as ex-2.2.9 froze it: table A+ with its {ex.N_OPS} ops, {ex.N_LINES:,} lines with answers drawn stochastically, one fifth of the pairs of each op held out, and the recipe adopted in ex-2.2.3 (λ_a = {ex.LAM:g}, annealed over the last tenth of training, τ = {ex.TAU:g}, {ex.EPOCHS} epochs). The corpus is the same one at the same seed, so the held-out lines are the same. Every model is fresh: condition seed *i* trains at model seed {ex.SEED_OFFSET} + *i*, so we score no checkpoint that ex-2.2.10 looked at.

| condition | seeds | what it is | role |
| --- | ---: | --- | --- |
{chr(10).join(f"| `{c.name}` | {c.seeds} | {c.title} | {c.role} |" for c in ex.CONDS)}

**`handover`** has everything enabled. It is the one candidate, and every gate is read on it alone.

**`handover-slot`** puts the label back on the two operands only, as ex-2.2.3 had it. **`handover-tied`** puts the readout back on the shared table. In ex-2.2.9 each of them undid about half of the rise in op1 alignment, which is why both are the references for that line now. Each is also the reference for one of the reads in H4, as before. This time we score both under the removal operators, for the saturation-and-value read in H3.

**`control`** has no anchor, and sets the task bar for H1.

`handover-narrow` does not return. Its question (lines per op) was exploratory, and ex-2.2.9 answered it.

The removal operators are the ones from ex-2.2.9.
**`projection`** takes the axis out at every slice and position, and is the operator the gates are read on.
**`operands`** does the same at the two operand positions only, as the selective reference.
**`shaped-a0.4-p0`** leaves states below alignment 0.4 alone, and is reported without a gate.

## Glossary

We use the vocabulary of ex-2.2.9, and its [glossary](../ex-2.2.9/report.py#glossary) has every term. Three terms change meaning here, and one is new.

<dl>
<dt>Removal lines</dt>
<dd>The red lines whose true answer moves by at least {ex.FAR_MOVE:g} in the unit cube when we permute the channels of the red operand. Permuting the channels keeps the saturation and value of a color and changes its hue, so these are the lines whose answer needs the <em>hue</em> of red. Ex-2.2.9 zeroed the R channel instead, which also removes the saturation and value. Counts per op are in the <a href="#the-removal-lines">method</a>.</dd>
<dt>Saturation-and-value lines</dt>
<dd>The red lines the hue rule sets aside. Their answer holds still under every permutation of the red operand, so it takes only the saturation or value of red, or nothing from red at all. On the three order-sensitive ops, that is one of the two slots. We report these without a gate.</dd>
<dt>Retention</dt>
<dd>Whether the placement holds across the anneal of the anchor weight. It is the final alignment as a share of the alignment at the start of the anneal. Ex-2.2.9 divided by the peak over the run instead.</dd>
<dt>Level</dt>
<dd>The alignment at the end of training, on the <code>mix</code> lines. Every later read is taken on it, and it is the number that the drift before the anneal moves.</dd>
</dl>
"""

r"""
## Does the model still learn the task? (H1)
"""

rf"""
**What we expect.** The anchored model learns the eleven ops as well as the un-anchored one does, as it did in ex-2.2.9. The number we look at, for each op, is the `handover` seed-mean expected exact match on the held-out lines, against the same number on `control`. Gate: within {ex.TASK_GATE:g} on every op, partial to {ex.TASK_PARTIAL:g}. We report a gap smaller than a band as unresolved. What would change our mind: a gap over {ex.TASK_PARTIAL:g} on some op, which at fresh seeds would say the pass in ex-2.2.9 was luck.

/// admonition | TODO
One dot panel per op: expected exact match per seed for `handover` and `control`, the gate drawn as a band around the control's mean. A table of the seed means and gaps beside it.
///
"""

r"""
## Does *red* land, and stay through the anneal? (H2)
"""

rf"""
**What we expect.** *Red* lands where the recipe put it, with the same margin, grading, and contrast as in ex-2.2.9, and nothing is lost across the anneal. There are four gates and three ungated lines, all read on the `mix` lines of `handover` under its own labeller.

- *Margin.* Seed-mean m_line at least {ex.MARGIN_RATIO:.0%} of the {ex.REF_M_LINE:.4f} in ex-2.1.10; partial from {ex.MARGIN_PARTIAL:.0%}.
- *Grading.* Grading r² at least {ex.GRADE_R2_RATIO:.0%} of the {ex.REF_R2_SIM:.3f} in ex-2.1.11.
- *Contrast.* At least {ex.CONTRAST_GATE:g}; partial from {ex.CONTRAST_PARTIAL:g}.
- *Retention.* Every run whose alignment at the start of the anneal reaches {ex.RETENTION_FLOOR:g} ends at {ex.RETENTION_GATE:g} of that value. This is the read we changed. On the runs from ex-2.2.9, ex-2.2.10 measured it at 0.99 and above on every condition, so we expect every seed to clear it. A seed below the line would say the anneal does cost something after all, and that the old read had been right for the wrong reason.

The three lines have no gate, and we print each beside its references.

- *Level.* The final alignment, seed mean and range, beside `handover-slot`, `handover-tied`, and the point adopted in ex-2.2.3. Ex-2.2.9 read 0.66 against 0.72 and 0.70, and we expect the same ordering. The gap is the drift before the anneal. What that drift is goes to the [training-dynamics item](/todo/science/training-dynamics-under-the-retention-drift.md); here we only say how large it is at fresh seeds.
- *ᾱ at op1.* The mean alignment of the non-red colors at op1, beside the same two references and the point from ex-2.2.3. Ex-2.2.9 read 0.28 against 0.18 and 0.16. We expect the same shape, with `handover` above both and each reference about halfway down. This was a gate at 0.1 through ex-2.2.9, where it missed. It becomes a gate again once we can name a mechanism ([backlog](/todo/science/containment-rises-under-the-untied-readout.md)).
- *The `⏎` row.* The axis component of the `⏎` embedding row on every anchored condition. Ex-2.2.9 read 0.17 on `handover` and about zero on `handover-slot`, and we expect that again.

We also check lead (at least {ex.LEAD_GATE:g}) and latch (no run over {ex.LATCH_PI:g} on op1) as before, to confirm the pull did what it should. These two are manipulation checks: a miss would say the run was not the one we meant to score, and neither is part of the decision rule, as in ex-2.2.9.

/// admonition | TODO
Placement table per condition, seed mean with range: margin, grading, contrast, retention, level, ᾱ at op1, `⏎` row, with ex-2.2.9's `handover` and ex-2.2.3's point as reference rows. Two figures: the alignment trajectories per condition with the anneal window shaded, and one dot panel per line (level, ᾱ, `⏎`) by condition.
///
"""

r"""
## Does *red* come out on the lines that need its hue? (H3)
"""

rf"""
**What we expect.** With the removal lines chosen by hue, the projection takes *red* out on every op, and costs nothing on the lines that never had any *red* in them. Both gates are read under `projection`, on `handover`.

- *Removal.* On the removal lines of every op, the model keeps at most {ex.RED_KEPT_GATE:.0%} of its clean expected exact match, seed mean. Rescoring the seeds of ex-2.2.9 this way, every seed cleared it on every op; the gate here is on fresh seeds. What would change our mind: an op above the line. If that op is one of the three HSV ops, the hue rule was not the whole story. If it is a channel-wise op, the rule change was beside the point.
- *Selectivity.* The seed-mean deficit on the non-red `mix` lines is at most {ex.NONRED_DEFICIT_GATE:g}; partial to {ex.NONRED_DEFICIT_PARTIAL:g}, which is a reporting level. Ex-2.2.9 read 0.02. Bands use the per-run spread of {ex.DEFICIT_NOISE}, frozen as before. We report every other op beside `mix`, and count the red-answer lines on their own, as ex-2.2.9 did.

One more read has an expected direction but no gate.

*Saturation and value.* We take the kept share under `projection` on the saturation-and-value lines of the three order-sensitive ops ({", ".join(f"{sv_share[op]:.0%} of the red lines on `{op}`" for op in ex.ORDER_SENSITIVE)}), for `handover`, `handover-slot`, and `handover-tied`. Ex-2.2.10 saw `handover` keep about half on the two slots that take the saturation or value of red, and a fifth where the answer needs both at once, so the projection takes part of the saturation and value of a red color with it. If the two references keep the same shares, that cost belongs to the operator. If they keep more, the cost belongs to this checkpoint, and either the readout or the labeller is putting saturation and value onto the axis.

/// admonition | TODO
Removal figure: kept share per seed by op under `projection`, the gate drawn, with the ex-2.2.9 rule's values as hollow markers beside the hue rule's. Selectivity table per op. A third panel for the saturation-and-value lines: kept share by op and slot, one column per condition.
///
"""

r"""
## Do the two reference reads repeat? (H4)
"""

rf"""
**What we expect.** The two things ex-2.2.9 confirmed about the recipe hold again at fresh seeds. Neither is a gate. Each is one comparison, and a miss is something to look at rather than a reason to stop.

- *The separate readout keeps the axis off the syntax tokens.* The axis component on the syntax embeddings (`=`, the op words, `⏎`) is lower on `handover` than on `handover-tied` by more than a band, with the component appearing on the readout table instead.
- *The whole-line label costs no selectivity.* The non-red `mix` deficit under `projection` differs between `handover` and `handover-slot` by less than a band, and no more than two seeds more on `handover` sit above {ex.TAIL:g}.

Both bands take the σ sources ex-2.2.9 froze: {ex.COMPONENT_NOISE} for the embedding component, and {ex.DEFICIT_NOISE} for the deficit.

/// admonition | TODO
The embedding-component table by condition, and a dot panel of the non-red `mix` deficit per seed for `handover` and `handover-slot` with the tail level drawn.
///
"""

r"""
## Decision
"""

rf"""
{ex.DECISION}

Retention joins the rule this time. In ex-2.2.9 we reported it without letting it count, because we already suspected the read. Now that the read asks the question it was meant to ask, a loss across the anneal is something a schedule would have to answer before we carry the grammar forward.

<!-- REVIEW: retention was outside ex-2.2.9's decision rule and is inside this one. Ex-2.2.10 read it at
0.99 on every condition with the new denominator, so the gate is expected to pass with room; if that
seems a gate that cannot inform, drop it back to a reporting level. -->
"""

r"""
## Exploratory analyses

Written before the run, and not part of the decision.

### The finer hue rule

The permutation rule reaches six hues. A finer rule rotates the hue of the red operand in HSV in twelve steps, snaps each one to the grid, and asks the same question. The method counts the lines where the two rules disagree, per op. If there are any, we keep the finer rule, and say so under the method before the freeze rather than here.

### Checkpoints on the trajectory stride

The first three seeds of each anchored condition keep a checkpoint at every trajectory point. Nothing in this report reads them. The [training-dynamics item](/todo/science/training-dynamics-under-the-retention-drift.md) needs them for a local learning coefficient estimate through the plateau and for the whole-geometry read at the same epochs, and this is the cheapest place to store them.

## Discussion

*Written after the results.* What each outcome would mean, set down in advance:

If every gate clears, we adopt the handover and run the anchored-op experiments on it. The two reads ex-2.2.10 corrected were then only reads, and the grammar and recipe are what the tl;dr of ex-2.2.9 said they were.

If removal misses on an HSV op under the hue rule, then on those ops the projection takes less of the hue than the counterfactual in ex-2.2.10 suggested, and we would look next at what the stream still holds there. If it misses on a channel-wise op, the miss was never about the rule.

If retention misses across the anneal, the schedule has a real cost that the old read hid behind the drift, and a stepped anneal would be the first thing to try.

The saturation-and-value read informs the intervention work rather than this decision. If the references keep the same shares as the candidate, that cost is a property of the operator, and worth stating wherever the projection is used. If they keep more, the handover has moved some of that onto the axis, which the containment item would want to know.
"""

rf"""
## Method

### The removal lines

The table below gives, per op and on its probe set: the red lines (dose at least {ex.RED_DOSE:g}), the removal lines under the hue rule (some channel permutation of the red operand moves the true answer by at least {ex.FAR_MOVE:g}), the same count under the to-zero rule of ex-2.2.9, and the saturation-and-value lines the hue rule sets aside. The order-sensitive ops walk every color through both slots, so we split their removal lines by where the red operand sits.

{removal_md()}

On `mix` the two rules pick the same lines. On `hsvmix` they nearly agree, differing on {rule_disagreement("hsvmix")} of the {counts["hsvmix"]["red"]} red lines. On the six other channel-wise ops the hue rule counts more, because permuting a red operand moves two channels at once. An answer that zeroing R left alone, say a `screen` with a partner about as red, moves far under a permutation.

Here at least {low_share:.0%} of the red lines of every channel-wise op are removal lines. On the six ops where the rules part, the old rule took about two thirds. Ex-2.2.9 cleared the gate on those ops with room to spare, so we expect the wider set to clear it too.

<!-- REVIEW: the draft said the two rules pick the same lines on `mix` and `hsvmix`, following ex-2.2.10's
summary. The counts in the table above disagree on `hsvmix` (401 against 391; the two sets differ on
eighteen lines, fourteen of them new to the hue rule). Verify: recompute the symmetric difference of the two rules on
`hsvmix`'s red lines. The same sentence in ex-2.2.10's "what we make of it" carries the old wording. -->


On the three HSV ops the hue rule keeps the slot whose answer takes its hue from red and sets the other aside. Under `hue-hsv` that is red at op2; under `sat-hsv` and `value-hsv` it is red at op1. The old rule counted parts of both slots on each, which is where the miss in ex-2.2.9 came from.

/// admonition | TODO
The finer-rule check: lines per op where the twelve-step HSV rotation and the permutation rule disagree, computed here before the freeze.
///

### Retention

We read retention from the trajectories ({ex.TRAJ_STRIDE} points over training). The anneal starts at the first point after the peak of the anchor weight where the weight falls under {ex.ANNEAL_WEIGHT_RATIO:g} of that peak, and the alignment at the start of the anneal is the last point before that. Retention is then the final point divided by that value, per run, on the runs where that value reaches {ex.RETENTION_FLOOR:g}.

### What is stored

We store what ex-2.2.9 stored: metrics, per-run arrays, trajectories, the probe set, and the end checkpoint of every run. Three things are new. Every anchored condition is scored under all three operators. The probe arrays hold the hue move and the to-zero move per line. And the first {ex.TRAJ_CHECKPOINT_SEEDS} seeds of each anchored condition keep a checkpoint at every trajectory point. The calibration of ex-2.2.9 still stands, since the corpus and the point are unchanged.

### Budget

{ex.N_RUNS} runs of {ex.HANDOVER.steps:,} steps at d64-L4, five fewer than ex-2.2.9, plus scoring under three operators on {ex.N_OPS} probe sets for {sum(c.seeds for c in ex.SCORED_UNDER_PROJECTION)} anchored runs. That is about three minutes a run on an L4, and the trajectory checkpoints add a few hundred MB of storage.
"""
