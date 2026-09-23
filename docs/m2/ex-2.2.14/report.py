# title: Ex 2.2.14: anchoring an operation

# The design constants come from `experiment.py` beside this script (the script's directory is on
# sys.path while it runs). During preregistration that module is constants only.
import experiment as ex


def conditions_html() -> str:
    """The primary, the sweep, and the served control, one row each."""
    head = "<tr><th>condition</th><th>anchored op</th><th class=num>seeds</th><th>role</th></tr>"
    rows = [
        f"<tr><td><code>{ex.PRIMARY}</code></td><td><code>{ex.ANCHORED_OP}</code></td>"
        f"<td class=num>{ex.SEEDS}</td><td>the primary: the op's lines draw at {ex.LABEL_RATE:g}, the red labeller's share; "
        "H1 and H2 are scored on it</td></tr>",
        f"<tr><td><code>{ex.OPWORD_ARM}</code></td><td><code>{ex.ANCHORED_OP}</code></td>"
        f"<td class=num>{ex.SEEDS}</td><td>arm: the pull covers the op word alone; H3 is scored on it</td></tr>",
        f"<tr><td><code>{ex.FULL_ARM}</code></td><td><code>{ex.ANCHORED_OP}</code></td>"
        f"<td class=num>{ex.SEEDS}</td><td>arm: every line of the op draws</td></tr>",
        f"<tr><td><code>{ex.NOISY_ARM}</code></td><td><code>{ex.ANCHORED_OP}</code></td>"
        f"<td class=num>{ex.SEEDS}</td><td>arm: the primary's labeller with a fifth of its labels moved onto other ops' lines</td></tr>",
        f"<tr><td><code>anchor-&lt;op&gt;</code> ×{len(ex.SWEEP)}</td><td>each other op of table A+</td>"
        f"<td class=num>{ex.SWEEP_SEEDS}</td><td>the sweep: the same reads, reported as a description</td></tr>",
        f"<tr><td><code>{ex.CONTROL}</code></td><td>none</td><td class=num>{ex.CONTROL_SEEDS}</td>"
        "<td>ex-2.2.11's un-anchored control, served from the store; the task reference and the alignment baseline</td></tr>",
    ]
    return f'<table class="report-table dense"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


rf"""
# Ex 2.2.14: anchoring an operation

/// tip |
<!-- tl;dr -->
Every anchor so far has held a property of one token: how red a color is. Here we anchor an *operation* instead, `{ex.ANCHORED_OP}`, on the axis *red* used to have. Does it land, and does the task survive?
///

There is no *red* anchor beside it. An op is a step up in abstraction from a color: a color is defined by what one token looks like, an op by what it does to a pair of operands.

This is a few-seed smoke test, run before the many-seed equivalence experiment spends its budget. It also asks whether the blocks carry the op forward to where the answer is computed.

## Findings

- [The task survives (H1)](#the-task-survives-h1) —
- [The op lands and holds (H2)](#the-op-lands-and-holds-h2) —
- [The blocks carry it to the use sites (H3)](#the-blocks-carry-it-to-the-use-sites-h3) —
- [Containment, reported without a gate](#containment-reported-without-a-gate) —

[The rule for the follow-up](#the-rule-for-the-follow-up): —

## How to read this draft

The op, the three predictions, the containment measure, and the rule for the follow-up were fixed before any run, at commit `TODO`. Everything after that commit is either results filled into their sections or exploratory work, marked as post hoc.

The [D2.2 design](../d2.2/design.md#anchor-one-operation) asks for this smoke test before the equivalence read, scored on the alignment and task gates alone. The probe scan it names runs here as a description.

## Why this experiment

D2.2 asks whether an anchor can hold more than a token's identity at a labelled site. *Red* is a property of the token at a known position. An operation is a property of the computation: named at the op word, and used at `=` and after, deeper in the stack. It is a more abstract concept than a color, and the first rung of a ladder toward concepts in natural language.

If anchoring an op works, the suppression experiment that follows can ask the deliverable's central question: whether removing the op from the axis removes the ability to perform it, selectively and by a bounded amount.

[Ex-2.2.11](../ex-2.2.11/report.py) fixed the grammar and the recipe, and [ex-2.2.13](../ex-2.2.13/report.py) confirmed the recipe at fresh seeds with the weight unchanged. One issue stays open: the projection leaves a quarter of the red-dependent answers of `hue-hsv` in place. That belongs to the *red* anchor, which this model does not have.

The label is new in kind: until now a color decided which lines were labelled, and here the op word decides. The pull covers the whole line, so the label says which lines carry the concept but not where on the line it sits.

On the primary, no alignment measurement can tell an anchored op from an anchored token. The pull puts the op word's embedding on the axis by construction, and asks for the axis at `=` and at the answer too. So these measurements only say whether the pull landed.

One arm pulls the op word alone, so any alignment it shows at the *use sites* (`=` and the answer position, where the op is applied rather than named) is something the pull did not ask for. Whether the anchor captured the operation itself is a question for suppression, and we don't ask it here.

## Conditions

{conditions_html()}

**The op.** `{ex.ANCHORED_OP}` was chosen analytically. On three quarters of its lines no other op in the table gives the same answer (its *op-relevance*),[^op-relevance] the highest of any commutative op. It is commutative, so operand order plays no part in its read.

It is total on the grid, so each line has one answer, and the control learns it to near-perfect held-out accuracy. The ops that round stochastically are capped well below that.

[^op-relevance]: The term is the design's. The sweep reads every other op the same way, so the choice can be revisited on data.

**The labeller.** A line whose op word is the anchored op gets a label at a rate of {ex.LABEL_RATE:g}. The pull covers all six positions of a labelled line, as the handover's whole-line labeller does.

That labels a fifth of a percent of the corpus, the red labeller's share, so each labelled line gets the same pull a red line got, and the op differs from *red* in kind rather than in label share.

The primary uses this sparse rate because it resembles the labels the method will have further up the ladder: for an abstract concept in natural language, labels will be scarce and sometimes wrong. One arm below labels every line of the op instead. We record the label share for each run.

<!-- REVIEW: the rate-0.02 labeller and the every-line labeller swapped places at the human's call, the realistic labeller as the primary. The every-line arm keeps the bracket: its share is fifty times red's, and the anchor term is normalized by the mask's weight, so each of its labelled lines gets a proportionally weaker pull. -->

**The arms.** All three run at the primary's seeds. The op-word arm pulls only the op word, at the primary rate, so the use sites fall outside the mask: any contrast at `=` or the answer at the final slice was carried there by the blocks without the pull asking for it. H3 is scored on this arm.

Pulling one position instead of six also makes each of this arm's pulls about six times stronger, because the anchor term is normalized by the mask. So we report its contrast at the op position beside the primary's.

The every-line arm labels every line of the anchored op, a tenth of the corpus. If its margin agrees with the primary's, label share does not set the margin; if they differ, the ratio of the two margins shows what pulling every line of a categorical concept gains.

The noisy arm moves a fifth of the primary's labels onto lines of the other ops, so the labelled share of the corpus stays the primary's and the total pull with it. It shows what a labeller of that precision costs the margin and the task. Neither the every-line arm nor the noisy arm enters a hypothesis or the rule.

**The seeds.** {ex.SEEDS} for the primary and each arm, and {ex.SWEEP_SEEDS} for each op of the sweep, all fresh. The control is served from the store at its own five seeds; comparisons against it are between seed means, which absorbs the unpaired seed sets.

Everything else is unchanged from ex-2.2.11: [table A+](../ex-2.2.4/report.py#the-op-set), the stochastic corpus, the untied readout, λ_a = 0.1 annealed over the last tenth of training, τ = 0.1, the anti-subspace schedule, and 50 epochs at d64-L4. *Red* is not anchored: e₁ belongs to the op.

## Glossary

<dl>
<dt>Op margin</dt>
<dd>The line margin (m_line) of every experiment since ex-2.1.10, with the anchored op's lines as the labelled group: per slice, the mean alignment with e₁ over the anchored op's lines minus the mean over all eleven ops' probe lines, at the span role where that gap is largest, then averaged over every slice, the embedding included. Taking the largest role per slice is what keeps a clean <code>=</code> embedding from costing anything: at the embedding slice the op word's own role carries the gap. The quantity the anchor term optimizes, so it is a check that the treatment landed.</dd>
<dt>Contrast at a site</dt>
<dd>At one position and slice, the mean cosine with e₁ (how closely the state points along e₁, from −1 to 1, ignoring length) over the anchored op's lines minus the mean over the other ops' lines. At the op position it is the op word's own placement; at <code>=</code> and the answer it is what the blocks carried there, since those tokens are the same on every line.</dd>
<dt>Use sites</dt>
<dd><code>=</code> and the answer position: where the op is applied rather than named.</dd>
<dt>Containment</dt>
<dd>The mean alignment with e₁ at the op position over the other ten ops' lines: how much the ops that were not anchored drifted toward the axis. The categorical analogue of ᾱ at op1.</dd>
<dt>Op-identity R²</dt>
<dd>How linearly readable the op is at one site, anchored against control: the held-out R² (share of variance explained; 1 is perfect) of a ridge probe (a linear regression with a penalty on large weights) from the state to the one-hot of the op word.</dd>
</dl>

## The task survives (H1)

**What we expect.** Anchoring `{ex.ANCHORED_OP}` costs the task nothing we can measure: for each of the eleven ops, the primary's seed-mean held-out expected exact match is within {ex.TASK_GATE:g} of the control's. It holds in part when every op is within {ex.TASK_PARTIAL:g}.

A miss on the anchored op alone would say the pull on its lines competes with producing its answer. The whole-line span makes that possible for the first time on a categorical concept. A miss spread over the table would say the pull disturbs lines it never touches, at a label share of a fifth of a percent.

The design's risk table names this as the first thing an abstract anchor might cost. The red anchor on the same recipe cost nothing on any op at twenty seeds. The metric is the control's own, expected exact match on the grid; an RGB-distance readout beside it is an [open item](/todo/science/rgb-distance-readout-beside-exact-match.md) and is not scored here.

/// admonition | TODO
One figure: per op, the primary's seed-mean expected exact match beside the control's, with the gate as a band around the control and the anchored op marked. One table: per op, both means, their difference, and the verdict.
///

## The op lands and holds (H2)

**What we expect.** This is a manipulation check: the op margin is the quantity the anchor term optimizes, so it says whether the pull landed, not whether the op was captured.

On the seed mean, the op margin on the primary reaches at least {ex.MARGIN_RATIO:.0%} of what *red* reached under the same recipe ({ex.REF_M_LINE:g} on the `mix` lines in ex-2.2.11), and holds in part from {ex.MARGIN_PARTIAL:.0%}. The margin is also retained: every run whose margin reaches {ex.RETENTION_FLOOR:g} when the anchor weight starts to anneal ends training at {ex.RETENTION_GATE:g} of that value.

The two margins are computed the same way (a gap in mean cosine between a labelled group and the pool), so they are comparable in scale. But their groups and baselines differ in ways the method describes, which pull in opposite directions, so the bar is rough; the equivalence experiment sets its own from what this one measures.

A margin far under the bar with the task intact would say the op's lines are harder to pull together than red's: a first sign that a categorical concept spread over eleven contexts wants a different weight.

A margin that falls through the anneal would be the retention failure ex-2.2.9 saw on `handover` alone, and would point to the anneal as the thing to look at before the equivalence experiment.

The learning rate is low over the anneal, so a tighter share than {ex.RETENTION_GATE:g} would be defensible. But one `handover` seed in twenty ended under it in ex-2.2.11, so we keep the bar where a known failure sits and report the per-seed values.

/// admonition | TODO
One figure, two panels: the op margin over training for each primary seed with the anneal marked, and the end-of-training margin per seed against the bar. One table: margin at the anneal start, at the end, and their ratio, per seed.
///

## The blocks carry it to the use sites (H3)

**What we expect.** Some of the op reaches the use sites. On the op-word arm at the final slice, the contrast at `=` and the contrast at the answer position are each at least {ex.USE_CONTRAST_FLOOR:g} above the control's, on the seed mean. The ratio of each to the contrast at the op position is reported with no bar.

No decision depends on this prediction, the least certain of the three. We have seen the model put the answer color at the use sites; this measures whether it also holds a copy of the op there, or only what the op produced. The floor asks only for a clearly positive contrast.

At slice 0 (the embedding), the contrast at the op position is the op word's embedding projected on e₁, which the pull puts there by construction. At the use sites the token (`=` or the answer) is the same on every line, so the slice-0 contrast is zero and any later contrast came through attention, which on this arm nothing requested.

A pass would mean that once the op word is on the axis, the blocks carry some of the op to where it is used, which an intervention at the op word would rely on. A miss would mean that, at this weight, the anchor marks the op but the blocks do not carry it to the use sites; it would not mean the model computes the op somewhere else.

On the primary the whole-line pull covers the use sites too, so the contrast there is part of what the treatment optimizes and we expect a pass from the method alone. We report it beside the op-word arm as a manipulation check.

<!-- REVIEW: H3 was a manipulation check on the primary, where the whole-line span pulls positions 3 and 4 of labelled lines. It is now scored on the op-word arm, whose mask excludes the use sites, so the contrast there is carried rather than optimized. The primary stays the whole-line labeller because that is the realistic one; the narrower arm is where this hypothesis means something. -->

/// admonition | TODO
One figure: contrast over the six positions as a smooth-step chart, one series per slice, seed mean, with the three sites marked; two stacked panels, the op-word arm and the primary. One table: contrast at the op position and at the two use sites at the final slice, per seed of the arm, and the two ratios.
///

## Containment, reported without a gate

**What we expect.** The other ten ops drift a little toward e₁ at the op position. The backlog item [containment rises under the untied readout](/todo/science/containment-rises-under-the-untied-readout.md) records ᾱ at op1 at {ex.CONTAINMENT_REF:g} on the red anchor under this recipe, up from the 0.1 the earlier gates asked for, with no mechanism named for the rise. We report the categorical analogue per op, beside that number and the control.

Three things could be behind the rise:
(a) the red pull itself (a graded label on a color, at every position of the line),
(b) the two changes that arrived with it (the untied readout and the whole-line span), and
(c) the weights of the recipe.

This experiment keeps (b) and (c) and replaces (a) with a categorical pull on an op, so it can only say whether the red pull was needed. Containment near the control would say it was, alone or together with the shared factors; containment near the red value would say the shared factors produce the rise on their own.

A value between the two would mean both play a part, which seems likely, since the pull and the readout could each contribute. Either way this is one experiment's worth of evidence, and the item stays open.

/// admonition | TODO
One figure: containment per non-anchored op at the op position, seed mean over the primary, beside the control and the red anchor's ᾱ at op1. Per slice in a second panel.
///

## The rule for the follow-up

> {ex.ADOPTION}

/// admonition | TODO
One table: the primary's verdicts on H1 and H2, and per op of the sweep its seed-mean op margin, its worst task gap, and whether it qualifies.
///

## Exploratory analyses

Anything we think of after seeing the data goes here, marked as post hoc. Four analyses are planned in advance as descriptions, with no gate, and run whichever way the predictions come out. Each gets a figure as well as a table.

**The sweep.** Every other op of table A+ anchored the same way at {ex.SWEEP_SEEDS} seeds, read on the same task gap, op margin, retention, and use-site contrast. It is the design's "sweep over all ops to see whether they can all be anchored equally well", run cheaply while the machinery is warm, and the rule above falls back on it.

Whether the three order-sensitive ops behave differently is the one pattern worth looking for in advance.

**The op-identity scan.** The op-identity R² at every site, anchored against control. The design's equivalence claim is that anchoring does not change how readable the op is anywhere the anchor does not reach. The equivalence experiment has to declare a margin for that, and this scan gives it an observed spread. It carries no gate here.

**The arms.** The every-line arm and the noisy-label arm, read on the same statistics as the primary: the task gap, the op margin, retention, and the use-site contrast. The every-line arm's margin against the primary's is the price or gain of the label share; the noisy arm's margin and task gap against the primary's are the cost of a fifth of wrong labels. The op-word arm carries H3 and is read there.

**The alignment map.** The primary's mean cosine with e₁ over position and slice on the anchored op's lines and on the others, as a picture of where the anchor put the op.

/// admonition | TODO
Four figures. The sweep: per op, its seed-mean op margin and worst task gap, with the primary's marked. The scan: op-identity R² per site, anchored beside control. The arms: the op margin over training for the primary and the two arms, with the task gap per arm. The map: mean cosine over position and slice, the anchored op's lines beside the others.
///

## Discussion

/// admonition | TODO
About 200 words: whether an op anchors as readily as a color, what the use-site contrast says about where the op lives, and what the equivalence read and the suppression experiment inherit.
///

## Method

### The labeller

This experiment adds a labeller keying in which the op word draws the label, against a per-op rate table: {ex.LABEL_RATE:g} for the anchored op and zero otherwise on the primary, one for the anchored op on the every-line arm, and {ex.NOISY_TRUE_RATE:g} for the anchored op with {ex.FALSE_RATE:.4g} for each other op on the noisy arm. The pull covers the whole line, or the op word alone on the op-word arm. It consumes the random stream differently from the color labellers, so the control's batches are not this experiment's; hence the seed-mean comparison, as in ex-2.2.13.

### The measurements

The op margin is ex-2.2.3's `line_margin` with uniform line weights summing to one over the anchored op's lines and zero elsewhere, on the pooled per-op probe sets ex-2.2.9 built. The function is red's m_line, but the comparison is not like for like: red's weights grade with redness (`p_slot`) and its baseline is the `mix` lines alone, where the op's weights are binary and its baseline includes its own lines as one eleventh of the pool. The binary label concentrates the labelled mean and the pooled baseline shrinks the gap by about a tenth, so the two differences pull in opposite directions and the ratio in H2 is a rough bar rather than a matched one.

<!-- REVIEW: was "line weights set to one ... comparable with red's m_line to the definition". `line_margin` takes an einsum with the weights, so weights of one give a sum, not a mean; they must sum to one. And the labelled group and baseline differ from red's, so the comparability claim was narrowed. Verify against ex-2.2.9's probe-set layout: if the op's baseline is taken per op rather than pooled, drop the one-eleventh clause. --> Retention is the end-of-training margin over the margin at the start of the anchor weight's anneal, on the trajectory recorded every few epochs, with the anneal start located as ex-2.2.11 located it. Contrast and containment are the glossary's mean cosines, taken over the same probe lines. The op-identity scan fits a ridge probe at ridge strength {ex.PROBE_RIDGE:g} from the state at each site to the one-hot op word, with a held-out split by line.

### Budget

{ex.N_RUNS} runs at d64-L4, each as long as a run in ex-2.2.13, which trained 160 of them for about fourteen dollars on Modal. Eval adds the alignment measurements on eleven probe sets and the scan; no intervention is scored.

### What this experiment does not do

It does not anchor *red* beside the op, does not suppress anything, does not vary the weight, and does not claim equivalence on the scan. Each of those is for a later experiment.
"""
