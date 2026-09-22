# title: Ex 2.2.13: does a heavier anchor make the leftover predictable?

# The design constants come from `experiment.py` beside this script (the script's directory is on
# sys.path while it runs). During preregistration that module is constants only.
import experiment as ex

# What a result section shows while the run has not published yet.
RESULTS_TO_COME = "/// admonition | TODO\n    type: warning\nResults to come.\n///"


def ladder_html() -> str:
    """The ladder, one row per condition, with what earlier experiments saw of each."""
    head = (
        "<tr><th>condition</th><th class=num>λ_a</th><th>home of <em>red</em></th>"
        "<th class=num>seeds</th><th>seen before</th></tr>"
    )
    notes = {
        "axis-0.1": "ex-2.2.11's recipe, the reference; 20 seeds there",
        "axis-0.2": "ex-2.2.12's <code>lam-0.2</code>, 5 seeds: loose on the kept share, the tightest line margin and ᾱ in the sweep",
        "plane-0.1": "ex-2.2.12's <code>plane</code>, 5 seeds",
        "plane-0.2": "ex-2.2.12's <code>plane-lam-0.2</code>, 5 seeds: the tight kept share",
    }
    rows = [
        f"<tr><td><code>{c.name}</code></td><td class=num>{c.lam:g}</td><td>{c.subspace}</td>"
        f"<td class=num>{ex.SEEDS}</td><td>{notes.get(c.name, '')}</td></tr>"
        for c in ex.GRID
    ]
    rows += [
        f"<tr><td><code>{ex.CONTROL}</code></td><td class=num>0</td><td>none; scored on the axis</td>"
        f"<td class=num>{ex.SEEDS}</td><td>ex-2.2.11's <code>control</code>, 20 seeds there; the task reference</td></tr>",
        f"<tr><td><code>{ex.CONTROL_PLANE}</code></td><td class=num>0</td><td>none; scored on the plane</td>"
        f"<td class=num>—</td><td>the same checkpoints, a scoring pass; the ᾱ baseline for the plane</td></tr>",
    ]
    return f'<table class="report-table dense"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


rf"""
# Ex 2.2.13: does a heavier anchor make the leftover predictable?

/// tip |
<!-- tl;dr -->
The recipe from ex-2.2.11 removes *red* on ten of eleven ops. What it leaves behind on the eleventh swings widely from seed to seed. Ex-2.2.12 saw one condition where that leftover varied much less. Here we climb a ladder of anchor weights, on an axis and on a plane, at fresh seeds, and ask whether a heavier anchor gives us a leftover we can predict. That would be worth having even if the leftover never gets any smaller.
///

## Findings

- [The leftover gets more predictable (H1)](#the-leftover-gets-more-predictable-h1) —
- [The leftover does not get smaller (H2)](#the-leftover-does-not-get-smaller-h2) —
- [The worst op improves (H3)](#the-worst-op-improves-h3) —
- [What the weight spends (H4)](#what-the-weight-spends-h4) —

[The adoption rule](#the-adoption-rule): —

## How to read this draft

The ladder, the four predictions, and the adoption rule were fixed before any run, at commit `TODO`. Everything after that commit is either results filled into their sections or exploratory work, marked as post hoc.

Every run is new. The seeds that produced the observation this experiment follows up are not reused, so the reference is trained again here beside the ladder. The adoption rule summarizes a leftover by an upper confidence bound rather than by the fixed band ex-2.2.12 used; the [adoption rule](#the-adoption-rule) section gives both verdicts.

## Why this experiment

[Ex-2.2.11](../ex-2.2.11/report.py) put *red* on one axis of the residual stream of a small transformer,[^rs] taught it eleven color operations, and projected the axis out. On ten ops the model lost *red*. On `{ex.MISSED_OP}` it kept about a quarter of the answers that need the red operand, which is over the gate.

[^rs]: The *residual stream* is the running vector of activations that each layer of a transformer reads from and writes back to.

[Ex-2.2.12](../ex-2.2.12/report.py) asked why and got a clean negative result. The story about which part of a hue an axis can hold was wrong, no change to the recipe brought the leftover under the gate, and the survival turned out to come from two of the seven red colors.

It also left something unexplained. The leftover swings from seed to seed: the reference keeps anywhere from 0.06 to 0.41 of those answers, depending on which seed trained the model. One condition of the sweep, the plane at twice the anchor weight, kept 0.14 to 0.22 across its five seeds instead. A range over five draws is narrower than a range over twenty at the same spread, so the two ranges are not directly comparable; the standard deviations behind them, 0.036 against 0.125, are what H1 is built on. A standard deviation from five seeds is itself a loose estimate, which is why the gate below asks for less than that ratio.

That is the observation this experiment is built on, and it is not the one the sweep was scored on. A leftover we cannot remove is a confound for every anchored-op experiment that follows.

If it is the same size every time, we can measure it once and subtract it. If it varies three-fold across seeds, every experiment that meets it has to measure it again. So how wide the seed spread is counts as a result in its own right. Read more broadly, the spread of the leftover is one face of how reproducible training is under the recipe on this grammar, and the ladder is a test of whether a heavier anchor makes training more reproducible.

A second reason to expect something from the weight: we run at λ_a = 0.1 because [ex-2.1.6](../ex-2.1.6/report.py) chose a rung safely inside the region where the task is unhurt. The survey in [ex-2.1.11](../ex-2.1.11/report.py) later mapped the weight on the six-op grammar, putting the plateau of the margin at λ_a 0.28–0.94, with the task gate biting only from about 0.38. On that map we have been running near the bottom of the useful range the whole time, and nothing has tested the region in between.

## Conditions

One ladder, crossed with the home of *red*, and the un-anchored control beside it.

{ladder_html()}

**The weight.** λ_a takes four levels, {", ".join(f"{x:g}" for x in ex.LADDER)}, each a factor of √2 above the last. We read a weight on a log scale, so a constant ratio puts the levels evenly apart and makes the shape of any trend across them easy to see. The lowest rung is the recipe from ex-2.2.11, and {ex.LADDER[2]:g} is the one step ex-2.2.12 took. The top rung is where the margin plateau ex-2.1.11 mapped begins, and it stops short of the level where that survey saw its first task failures (about 0.38, on the six-op grammar at a warmer pooling temperature than ours). Where the useful range ends is left open here; the ladder tests the region between the recipe and that point.

**What else the weight moves.** The anti-subspace term is specified as a ratio to λ_a, peaking at 2.5× λ_a and holding at 0.3× λ_a, so a rung of the ladder raises the repulsion by the same factor as the pull. The ladder is a joint anchor-and-repulsion ladder rather than a pure anchor ladder, and a result along it belongs to the pair. Ex-2.2.12 moved the two separately, one step each (`lam-0.2` and `anti-5`), and neither moved any measurement on its own, so we do not spend runs here separating them; if the ladder moves something, an arm that holds the repulsion fixed at one rung is the follow-up.

**The home of *red*.** *Red* lives either on the first axis e₁ or on the plane spanned by e₁ and e₂. Where the plane is the home, the anchor term, the anti-subspace term, the alignment measurements, and the removal all take the pair of axes instead of the single axis, as ex-2.2.12 defined them.

The recipe has two anchoring terms and no anti-anchor term. The pull is one minus the alignment of a labelled state with the home; on the axis that alignment is the signed cosine, so the pull is toward +e₁ and a state at −e₁ is as far from home as it can be. On the plane it is the unsigned length of the projection, so the pull is toward the plane with no preferred direction within it, and −e₁ is home. The anti-subspace term is the squared alignment averaged over every live position, labelled or not, and it has no sign in either home. An anti-anchor term, the one-sided hinge that kept every state out of the hemisphere opposite the anchor so a fallback could live there, belonged to M1's fallback and last ran in M2 in ex-2.2.2's fallback arm; the recipe line from ex-2.1.6 onward has never carried it, and on the plane there is no hemisphere for it to name.

Ex-2.2.12 settled against the plane's first rationale, that a single axis is too small a home for a hue. Two things keep it here. The tight condition that prompted this experiment was a plane *and* a heavier weight, while the heavier weight on the axis was among the loosest on the kept share in that sweep even as its line margin and ᾱ were the tightest, so a ladder on the axis alone could not say whether the narrowing comes from the weight, from the subspace, or from the two together. And the plane at the recipe's own weight has been seen at five seeds only; at twenty, `plane-{ex.LADDER[0]:g}` against `{ex.REFERENCE}` is a test of the subspace on its own, at a resolution ex-2.2.12 did not have.

**The control.** The un-anchored control is trained again here at the same seeds, as ex-2.2.11 defined it. It is the reference for the task gate, which asks for a seed mean within a band of the control's, and the ᾱ baseline for the axis conditions. Every plane condition is compared against the same checkpoints scored on the plane (`{ex.CONTROL_PLANE}`): for every state, anchored or not, an unsigned two-dimensional alignment sits higher than a signed one-dimensional one, so the control has to be scored the same way.

**The seeds.** {ex.SEEDS} per condition, all fresh: condition seed *i* trains at model seed {ex.SEED_OFFSET} + *i*, where ex-2.2.11 and ex-2.2.12 used an offset of 100. Every condition pairs with every other one seed for seed within this experiment, and the reference is retrained rather than borrowed, which makes it a replication of ex-2.2.11's `handover` at seeds it never saw.

The spread question sets the count. A one-sided F-test on the ratio of variances between the top and the bottom of the ladder resolves a halved standard deviation with power 0.90 at twenty seeds a group, 0.81 at fifteen, and 0.63 at ten; a difference in means of the size we care about would be settled by half as many. {ex.N_RUNS} runs in all.

Everything else is unchanged from ex-2.2.11: [table A+](../ex-2.2.4/report.py#the-op-set), the stochastic corpus, the whole-line labeller, the untied readout, the removal lines chosen by hue, τ = 0.1, the shape of the anti-subspace schedule, and 50 epochs at d64-L4.

## Glossary

<dl>
<dt>Kept share</dt>
<dd>How much of its clean accuracy on an op's removal lines a model keeps after the projection. One means the projection did nothing; zero means every one of those answers changed, which is what the gate takes
<em>red</em> being gone to mean.</dd>
<dt>Removal lines</dt>
<dd>The red lines whose answer needs the hue of the red operand: some permutation of the channels of that operand moves the true answer far. The rule from ex-2.2.11, unchanged.</dd>
<dt>Seed spread</dt>
<dd>The standard deviation of a statistic across the seeds of one condition. This is the quantity H1 is about.</dd>
<dt>Upper bound</dt>
<dd>The one-sided {ex.UCB_LEVEL:.0%} upper confidence bound on the seed mean of a condition. One condition can have a low mean and a wide spread, and another a higher mean and a narrow spread, and the two can still have the same upper bound. That is the point of using it.</dd>
<dt>Line margin</dt>
<dd>How far the anchored states on the labelled lines sit above the rest along the home of <em>red</em>. This is the quantity the anchor term optimizes, so it is a check that the treatment landed rather than a result.</dd>
<dt>ᾱ at op1</dt>
<dd>The mean alignment with the anchored subspace over every color at the first operand position. How much the colors that are not red have drifted toward the home of <em>red</em>.</dd>
</dl>

## The leftover gets more predictable (H1)

**H1.** Across the ladder, the seed spread of the kept share on the `{ex.MISSED_OP}` removal lines narrows as λ_a rises. The number we score is the ratio of the standard deviation at the top of the ladder to the one at the lowest rung, within a subspace. H1 holds when that ratio falls to {ex.SD_RATIO_GATE:g} or below in both subspaces, and the spread falls monotonically enough that a trend contrast across the four levels has a negative slope at {ex.TREND_ALPHA:g}.[^trend]

It holds in part when one subspace does that and the other does not. On the plane alone, that would say the narrowing needs the plane, and the comparison of `plane-{ex.LADDER[0]:g}` with the reference then says whether the plane narrows the spread by itself or only once the weight rises. A flat or rising spread in both would mean the tight condition in ex-2.2.12 was five lucky seeds, and that nothing on this ladder buys predictability.

One reading of a narrowing has to be ruled out before it counts. The kept share is a proportion over a fixed set of lines, so its spread is bounded below by sampling noise that depends on where the mean sits: a condition whose mean is near zero or near one cannot vary much. H1 is therefore read together with H2. A narrowing that arrives with a mean the ladder also moved is a narrowing we get for free, and the report says so rather than claiming the recipe bought it.

[^trend]: The *trend contrast* regresses each run's absolute distance from the median of its own condition on log λ_a. Working from the median rather than the mean keeps a couple of extreme seeds from deciding it, and asking for a slope rather than for any difference between the levels means a spread that rose and then fell does not count as a pass.

The kept share is a fair thing to score here. The anchor term acts during training on how well a state aligns with the home of *red*; the kept share is taken afterwards, on held-out lines, through a projection the term never sees. Nothing in the treatment is set up to reduce its spread, so a narrowing would be telling us something.

The quantities the two terms do act on — the line margin and ᾱ at op1 — are reported under H4 as manipulation checks rather than with gates on them.

/// admonition | TODO
One figure, two panels (axis, plane): the kept share on the `hue-hsv` removal lines against λ_a on a log axis, each condition a column of 20 seed dots with its mean and a standard-deviation bar, the gate a dashed line with the failing side hatched. One table: per condition, the seed mean, standard deviation, range, and the Levene statistic per subspace, with ex-2.2.11's and ex-2.2.12's values for the four conditions they saw beside them.
///

## The leftover does not get smaller (H2)

**H2.** The seed-mean kept share on `{ex.MISSED_OP}` is flat across the ladder: no level differs from the reference by more than {ex.MEAN_BAND:g}, which is about what twenty paired seeds can resolve at the spread of the reference. H2 is a statement about what we can see at this resolution rather than a claim that the mean is unmoved, and a reviewer reading it as an equivalence test with a wide band is reading it right.

The evidence for it: the one step ex-2.2.12 took moved the mean up on the axis and down on the plane, and its tight plane condition sits almost exactly on the mean of the five reference seeds it pairs with. A level that does move the mean down by more than {ex.MEAN_BAND:g} would be a better result than we expect, and the adoption rule is written to take it.

Together, H1 and H2 give the shape of the claim: climbing the ladder leaves a leftover that is no smaller and that comes out the same size every time.

/// admonition | TODO
The seed-mean panel of the H1 figure read along the λ_a axis, with paired differences from the reference per seed in a table: per condition, the paired mean difference against `axis-0.1` and its confidence interval.
///

## The worst op improves (H3)

**H3.** On the ten ops other than `{ex.MISSED_OP}`, the highest per-op seed-mean kept share falls as λ_a rises. The number we score is that highest value at the top of the ladder against the same quantity at the reference, both at twenty seeds; H3 holds when it falls by at least {ex.WORST_OP_MARGIN:g}.

Any claim that a recipe removes *red* cleanly is limited by whichever op does worst, and that is not the same op in every condition, so a mean over the ten would hide it.

The margin is the {ex.WORST_OP_MARGIN:g} in the statement above, and it is there because a kept share is a proportion over some 330–400 removal lines per op, so on one checkpoint it carries a sampling error of about 0.02 at the values the worst op sits at, and the seed spread of the other ops in ex-2.2.12 was about 0.03. A drop smaller than that is one we could not tell from no drop. In ex-2.2.12 the worst other op was `darken` at the reference, `darken` again one step up the axis, and `lighten` on the plane at twice the weight — the only condition in that sweep with no op over the gate at all — and the spacing between those three values is close to the margin, which is why H3 asks for a margin rather than for a difference. H3 fails if the worst op is flat or rises, which would mean the ladder buys nothing on the ops that already remove *red*.

/// admonition | TODO
One figure: per condition, the per-op seed-mean kept share for all eleven ops as a strip, the worst highlighted and labelled, against the gate. One table: the worst op and its value per condition.
///

## What the weight spends (H4)

**H4.** Every cost statistic stays inside its gate from ex-2.2.11 on the seed mean through the whole ladder: {ex.COST_LIST}. The ladder stops under the level where the six-op survey saw the task give way, so we expect every cost to hold; the one we name as most likely to leave its gate first is the non-red deficit on the plane conditions, which ex-2.2.12 measured four times higher on the plane than on the axis.

If a statistic leaves its gate at or below λ_a = {ex.LADDER[-1]:g}, the useful range of the recipe on this grammar is narrower than the map in ex-2.1.11 suggested, which is worth knowing on its own. If none does, the range is at least this wide, and where it ends stays open.

We report the line margin here as the manipulation check. It is the quantity the anchor term optimizes, and a heavier pull should raise it along the ladder — one direction, with no argument available for the other. If it stays flat, the weight is not reaching the model, and the other three sections have no treatment to interpret. ᾱ at op1 goes in the same figure without a direction attached to it: it is what the anti-subspace term acts on, and that term is climbing the ladder alongside the pull.

/// admonition | TODO
One figure, one panel per cost statistic, each condition a column of seed dots against its gate, hatched on the failing side, with the ladder on a log axis. One table: every statistic per condition, gate failures marked. The line margin and ᾱ at op1 in two further panels, each subspace against the control scored the same way.
///

## The adoption rule

The rule, frozen before the run:

> {ex.ADOPTION}

What changed from the rule in ex-2.2.12, and why. {ex.OLD_RULE}

/// admonition | TODO
One table: per condition, the upper bound on `hue-hsv` and on the worst other op, the verdict for each cost statistic, and the qualifying verdict under this rule and under the one from ex-2.2.12.
///

## Exploratory analyses

Four measurements are planned as descriptions rather than tests: they carry no gate, and no finding rests on them. Anything we think of after seeing the data goes here too, marked as post hoc.

**The spread of everything else.** H1 scores the spread of one statistic. The same seed-spread table for every cost statistic, the line margin, and the task, per condition, says whether a heavier anchor makes training as a whole more reproducible on this grammar or only settles the leftover. It carries no gate because we have no prediction for the direction of most of them: a heavier pull could hold the margin to a tighter value across seeds, or could amplify whatever differs between initializations. Ex-2.2.12 gives a hint that the two spreads can move apart: one step up the axis, the line margin and ᾱ at op1 were the tightest in the sweep while the kept share stayed as wide as the reference, and on the plane at the same step it was the other way round.

**Which lines are left.** Ex-2.2.12 found that nearly all of the `{ex.MISSED_OP}` survival comes from two of the seven red colors, but it scored conditions rather than individual lines. Scoring per line across the ladder tells us whether a narrow seed spread means the same lines survive every time, which would let us characterize the leftover, or a shifting set of lines that happens to be the same size.

**The achromatic candidate.** The exploratory section of ex-2.2.12 proposed that the surviving lines are answered from how gray the second operand looks. The axis does not hold that information, and the blocks can read it off the other channels. To test it, we project at the blocks with the second operand replaced by a gray of the same value. That is a scoring pass on checkpoints this experiment trains anyway.

**What the response looks like.** A grading cloud of the α response at op1 against how red the first operand is, one per level of the ladder. The grading r² in ex-2.2.12 barely moved across its whole sweep, so this is here as a picture of what a near three-fold change in the recipe does to the response. It carries no gate.

## Discussion

/// admonition | TODO
About 200 words: what the ladder says the weight buys, whether the anchored-op experiments should inherit a different operating point, and what a predictable-but-unremovable leftover means for them.
///

## Method

### Fresh seeds

Ex-2.2.11 and ex-2.2.12 trained at model seeds 100–119. The observation this experiment follows up was read off those checkpoints, and a rule written in advance does not make data we have already seen unseen, so nothing here is served from the store: every condition, the reference and the control included, trains at seeds {ex.SEED_OFFSET}–{ex.SEED_OFFSET + ex.SEEDS - 1}. The four conditions that earlier experiments saw appear in the H1 table beside their earlier values, as a replication rather than as data.

We do not hold seeds back to quote the adopted condition from. The search is eight conditions, so the selection bias on the winner is small, and ex-2.2.12 set the pattern this experiment follows: the ladder proposes, and the anchored-op experiment that inherits the recipe confirms it at seeds of its own. Deciding the rule on half the seeds would halve the precision of the upper bound it rests on, for a correction the next experiment pays anyway.

### The plane

As ex-2.2.12 defined it. The home of *red* is the first two axes of the stream together. Alignment is the length of the projection of a state onto that pair, and the anchor term pulls that length toward one on the labelled lines. The anti-subspace term is the square of that length over every live position, and the removal projects the whole plane out.

So the concept holds two coordinates of sixty-four rather than one, and its share of the variance counts both.

### Budget

{ex.N_RUNS} runs at d64-L4, each as long as a run in ex-2.2.11, at about three minutes a run on an L4. Scoring adds the eleven ops under the operators from ex-2.2.11, the per-line pass, and the achromatic edit. That is about four times the sweep in ex-2.2.12, which cost six dollars on Modal.

### What this experiment does not vary

Depth, the pooling temperature τ, and the anti-subspace ratios stay at the values from ex-2.2.11, which means the anti-subspace weight itself climbs with the ladder. Ex-2.2.12 moved each of them one step and resolved nothing on any measurement, so a third factor here would cost runs without a prediction behind it. The weight above {ex.LADDER[-1]:g} is also left alone: the ladder tests the region ex-2.1.11's map calls useful, and finding its edge on this grammar is a different experiment.

The survey in ex-2.1.11 found that the weight and τ trade against each other, so a λ_a × τ factorial would be the natural follow-up if this ladder finds a level worth having.
"""
