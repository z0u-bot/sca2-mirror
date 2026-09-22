# title: Ex 2.2.13: does a heavier anchor make the leftover predictable?

# The design constants come from `experiment.py` beside this script (the script's directory is on
# sys.path while it runs). During preregistration that module is constants only.
import experiment as ex

# What a result section shows while the run has not published yet.
RESULTS_TO_COME = "/// admonition | TODO\n    type: warning\nResults to come.\n///"


def ladder_html() -> str:
    """The ladder, one row per condition, with the runs already in the store marked."""
    head = (
        "<tr><th>condition</th><th class=num>λ_a</th><th>home of <em>red</em></th>"
        "<th class=num>seeds</th><th class=num>in store</th><th>note</th></tr>"
    )
    notes = {
        "axis-0.1": "ex-2.2.11's recipe, the reference",
        "axis-0.2": "ex-2.2.12's <code>lam-0.2</code>",
        "plane-0.1": "ex-2.2.12's <code>plane</code>",
        "plane-0.2": "ex-2.2.12's <code>plane-lam-0.2</code>",
    }
    rows = [
        f"<tr><td><code>{c.name}</code></td><td class=num>{c.lam:g}</td><td>{c.subspace}</td>"
        f"<td class=num>{ex.SEEDS}</td><td class=num>{ex.MEMOIZED.get(c.name, 0)}</td>"
        f"<td>{notes.get(c.name, '')}</td></tr>"
        for c in ex.GRID
    ]
    rows.append(
        f"<tr><td><code>{ex.FIXED_ANTI_ARM}</code></td><td class=num>{ex.LADDER[-1]:g}</td><td>axis</td>"
        f"<td class=num>{ex.SEEDS}</td><td class=num>0</td>"
        "<td>arm: repulsion held at its strength at the foot</td></tr>"
    )
    return f'<table class="report-table dense"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


rf"""
# Ex 2.2.13: does a heavier anchor make the leftover predictable?

/// tip |
<!-- tl;dr -->
The recipe from ex-2.2.11 removes *red* on ten of eleven ops. What it leaves behind on the eleventh swings widely from seed to seed. Ex-2.2.12 saw one condition where that leftover varied much less. Here we climb a ladder of anchor weights, on an axis and on a plane, and ask whether a heavier anchor gives us a leftover we can predict. That would be worth having even if the leftover never gets any smaller.
///

## Findings

- [The leftover gets more predictable (H1)](#the-leftover-gets-more-predictable-h1) —
- [The leftover does not get smaller (H2)](#the-leftover-does-not-get-smaller-h2) —
- [The worst op improves (H3)](#the-worst-op-improves-h3) —
- [What the weight spends (H4)](#what-the-weight-spends-h4) —

[The adoption rule](#the-adoption-rule): —

## How to read this draft

The ladder, the four predictions, and the adoption rule were fixed before any run, at commit `TODO`. Everything after that commit is either results filled into their sections or exploratory work, marked as post hoc.

The rule is decided on one half of the seeds in each condition, and the numbers we quote for the adopted condition come from the other half, which the rule never sees. It also summarizes a leftover by an upper confidence bound rather than by the fixed band ex-2.2.12 used; the [adoption rule](#the-adoption-rule) section gives both verdicts.

## Why this experiment

[Ex-2.2.11](../ex-2.2.11/report.py) put *red* on one axis of the residual stream of a small transformer,[^rs] taught it eleven color operations, and projected the axis out. On ten ops the model lost *red*. On `{ex.MISSED_OP}` it kept about a quarter of the answers that need the red operand, which is over the gate.

[^rs]: The *residual stream* is the running vector of activations that each layer of a transformer reads from and writes back to.

[Ex-2.2.12](../ex-2.2.12/report.py) asked why and got a clean negative result. The story about which part of a hue an axis can hold was wrong, no change to the recipe brought the leftover under the gate, and the survival turned out to come from two of the seven red colors.

It also left something unexplained. The leftover swings from seed to seed: the reference keeps anywhere from 0.06 to 0.41 of those answers, depending on which seed trained the model. One condition of the sweep, the plane at twice the anchor weight, kept 0.14 to 0.22 across its five seeds instead. <!-- REVIEW: added the seed-count caveat to the motivating comparison. A range over 5 and a range over 20 differ by about 1.6× at equal spread (E[range] ≈ 2.33σ and 3.73σ), so the ranges as written overstate the contrast; the standard deviations recorded in `experiment.py:SD_RATIO_GATE` (0.036 against 0.125) carry it instead. Verify: recompute both from ex-2.2.12's published metrics. -->
A range over five draws is narrower than a range over twenty at the same spread, so the two ranges are not directly comparable. The standard deviations behind them, 0.036 against 0.125, are what H1 is built on.

That is the observation this experiment is built on, and it is not the one the sweep was scored on. A leftover we cannot remove is a confound for every anchored-op experiment that follows.

If it is the same size every time, we can measure it once and subtract it. If it varies three-fold across seeds, every experiment that meets it has to measure it again. So how wide the seed spread is counts as a result in its own right.

A second reason to expect something from the weight: we run at λ_a = 0.1 because [ex-2.1.6](../ex-2.1.6/report.py) chose a rung safely inside the region where the task is unhurt. The survey in [ex-2.1.11](../ex-2.1.11/report.py) later mapped the weight on the six-op grammar, putting the plateau of the margin at λ_a 0.28–0.94, with the task gate biting only from about 0.38. On that map we have been running near the bottom of the useful range the whole time, and nothing has tested the region in between.

## Conditions

One ladder, crossed with the home of *red*.

{ladder_html()}

**The weight.** λ_a takes five levels, {", ".join(f"{x:g}" for x in ex.LADDER)}, each a factor of √2 above the last. We read a weight on a log scale, so a constant ratio puts the levels evenly apart and makes the shape of any trend across them easy to see. The foot is the recipe from ex-2.2.11, and {ex.LADDER[2]:g} is the one step ex-2.2.12 took. The top reaches a level where we expect something to give way: ex-2.1.11 saw its first task failures from about 0.38 on the six-op grammar, at a warmer pooling temperature than ours.

<!-- REVIEW: added this paragraph, and changed "the anti-subspace schedule" and "the anti-subspace weight" in the two method lists to name the ratio instead. `schedules()` in ex-2.2.3 passes `lam=c.lam` into the AntiSpec, so the anti-subspace weight is λ_a × peak_ratio and climbs 4× over the ladder; the report had said it was held fixed, which the code contradicts. Verify: `docs/m2/ex-2.2.3/experiment.py:736-747`. This is a factual correction only. Resolved in the same round by adding the `axis-0.4-anti-fixed` arm and by attributing results to the pair throughout; whether the arm is worth its twenty runs is the open question for the human. -->
**What else the weight moves.** The anti-subspace term is specified as a ratio to λ_a, peaking at 2.5× it and holding at 0.3× it, so a rung of the ladder raises the repulsion by the same factor as the pull. The ladder is a joint anchor-and-repulsion ladder rather than a pure anchor ladder, and a result along it belongs to the pair. Ex-2.2.12 moved the two separately, one step each (`lam-0.2` and `anti-5`), and neither moved any measurement on its own.

**The arm that separates them.** One arm rides off the top rung of the axis ladder, at the same twenty seeds: λ_a = {ex.LADDER[-1]:g} with the anti-subspace ratios divided by four, so the repulsion sits at the absolute strength it has at the foot. If the ladder moves a measurement and this arm moves it too, the pull carries it; if the arm sits with the foot of the ladder instead, the repulsion does. The arm is outside the ladder and outside the adoption rule, and it is scored on the same measurements as everything else.

**The home of *red*.** *Red* lives either on the first axis e₁ or on the plane spanned by e₁ and e₂. Where the plane is the home, the anchor term, the anti-subspace term, the alignment measurements, and the removal all take the pair of axes instead of the single axis, as ex-2.2.12 defined them.

We no longer have a reason of its own to test the plane, since ex-2.2.12 settled against the idea that a single axis is too small a home. It is here for one job. The tight condition that prompted this experiment was a plane *and* a heavier weight, and the heavier weight on the axis was the loosest condition in that sweep. A ladder on the axis alone could not say whether the narrowing comes from the weight, from the subspace, or from neither.

Every plane condition is compared against the un-anchored control checkpoints scored on the plane (`{ex.CONTROL_PLANE}`). For every state, anchored or not, an unsigned two-dimensional alignment sits higher than a signed one-dimensional one, so the control has to be scored the same way.

**The seeds.** {ex.SEEDS} per condition, using the same seeds the `handover` run of ex-2.2.11 trained, so every condition pairs with every other one seed for seed.

The spread question sets the count: telling a halved standard deviation from an unchanged one takes about twenty runs per group, where half as many would settle a difference in means of the size we care about. {ex.NEW_RUNS} of the {(len(ex.CONDITIONS) + 1) * ex.SEEDS} runs are new; the rest come from the store.

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

**H1.** Across the ladder, the seed spread of the kept share on the `{ex.MISSED_OP}` removal lines narrows as λ_a rises. The number we score is the ratio of the standard deviation at the top of the ladder to the one at the foot, within a subspace. H1 holds when that ratio falls to {ex.SD_RATIO_GATE:g} or below in both subspaces, and the spread falls monotonically enough that a trend contrast across the five levels has a negative slope at {ex.TREND_ALPHA:g}.[^trend]

It holds in part when one subspace does that and the other does not, which would say the narrowing needs the plane. A flat or rising spread would mean the tight condition in ex-2.2.12 was five lucky seeds, and that nothing on this ladder buys predictability.

One reading of a narrowing has to be ruled out before it counts. The kept share is a proportion over a fixed set of lines, so its spread is bounded below by sampling noise that depends on where the mean sits: a condition whose mean is near zero or near one cannot vary much. H1 is therefore read together with H2. A narrowing that arrives with a mean the ladder also moved is a narrowing we get for free, and the report says so rather than claiming the recipe bought it.

[^trend]: The *trend contrast* regresses each run's absolute distance from the median of its own condition on log λ_a. Working from the median rather than the mean keeps a couple of extreme seeds from deciding it, and asking for a slope rather than for any difference between the levels means a spread that rose and then fell does not count as a pass.

The kept share is a fair thing to score here. The anchor term acts during training on how well a state aligns with the home of *red*; the kept share is taken afterwards, on held-out lines, through a projection the term never sees. Nothing in the treatment is set up to reduce its spread, so a narrowing would be telling us something.

The quantities the two terms do act on — the line margin and ᾱ at op1 — are reported under H4 as manipulation checks rather than with gates on them.

/// admonition | TODO
One figure, two panels (axis, plane): the kept share on the `hue-hsv` removal lines against λ_a on a log axis, each condition a column of 20 seed dots with its mean and a standard-deviation bar, the gate a dashed line with the failing side hatched, and the reference's spread as a band across both panels. One table: per condition, the seed mean, standard deviation, range, and the Levene statistic per subspace.
///

## The leftover does not get smaller (H2)

**H2.** The seed-mean kept share on `{ex.MISSED_OP}` is flat across the ladder: no level differs from the reference by more than 0.05, which is about what twenty seeds can resolve at the spread of the reference.

The evidence for it: the one step ex-2.2.12 took moved the mean up on the axis and down on the plane, and its tight plane condition sits almost exactly on the mean of the five reference seeds it pairs with. A level that does move the mean down by more than 0.05 would be a better result than we expect, and the adoption rule is written to take it.

Together, H1 and H2 give the shape of the claim: climbing the ladder leaves a leftover that is no smaller and that comes out the same size every time.

/// admonition | TODO
The seed-mean panel of the H1 figure read along the λ_a axis, with paired differences from the reference per seed in a table: per condition, the paired mean difference against `axis-0.1` and its confidence interval.
///

## The worst op improves (H3)

**H3.** On the ten ops other than `{ex.MISSED_OP}`, the highest per-op seed-mean kept share falls as λ_a rises. The number we score is that highest value at the top of the ladder against the same quantity at the reference, measured here at twenty seeds; H3 holds when it falls by at least {ex.WORST_OP_MARGIN:g}.

Any claim that a recipe removes *red* cleanly is limited by whichever op does worst, and that is not the same op in every condition, so a mean over the ten would hide it.

The margin is about the sampling noise on a proportion at these line counts, so a smaller drop is one we could not tell from no drop. In ex-2.2.12 the worst other op was `darken` at the reference, `darken` again one step up the axis, and `lighten` on the plane at twice the weight — the only condition in that sweep with no op over the gate at all — and the spacing between those three values is close to the margin, which is why H3 asks for a margin rather than for a difference. H3 fails if the worst op is flat or rises, which would mean the ladder buys nothing on the ops that already remove *red*.

/// admonition | TODO
One figure: per condition, the per-op seed-mean kept share for all eleven ops as a strip, the worst highlighted and labelled, against the gate. One table: the worst op and its value per condition.
///

## What the weight spends (H4)

**H4.** Through the whole ladder, {ex.COST_LIST} stay inside the gates of ex-2.2.11 on the seed mean. We name in advance which one gives way first: the non-red deficit on the plane conditions, which ex-2.2.12 measured four times higher on the plane than on the axis, and then the task at the top of the ladder.

If a statistic leaves its gate below λ_a = {ex.LADDER[-2]:g}, the useful range of the recipe is narrower than the map of the six-op grammar in ex-2.1.11 suggested, which is worth knowing on its own. The arm says whether the pull or the repulsion spent it.

We report the line margin here as the manipulation check. It is the quantity the anchor term optimizes, and a heavier pull should raise it along the ladder — one direction, with no argument available for the other. If it stays flat, the weight is not reaching the model, and the other three sections have no treatment to interpret. ᾱ at op1 goes in the same figure without a direction attached to it: it is what the anti-subspace term acts on, and that term is climbing the ladder alongside the pull.

/// admonition | TODO
One figure, one panel per cost statistic, each condition a column of seed dots against its gate, hatched on the failing side, with the ladder on a log axis. One table: every statistic per condition, gate failures marked. The line margin and ᾱ at op1 in two further panels, each subspace against the control scored the same way, and the arm marked in every panel.
///

## The adoption rule

The rule, frozen before the run:

> {ex.ADOPTION}

And what changed from the rule in ex-2.2.12, and why:

> {ex.OLD_RULE}

/// admonition | TODO
One table: per condition, the upper bound on `hue-hsv` and on the worst other op, the verdict for each cost statistic, and the qualifying verdict under this rule and under the one from ex-2.2.12. For the adopted condition, the numbers from the held-out half beside those from the deciding half.
///

## Exploratory analyses

Not part of the plan. Anything we think of after seeing the data goes here, marked as post hoc. Three measurements are planned as descriptions rather than tests, and they run whichever way the four predictions come out.

**Which lines are left.** Ex-2.2.12 found that nearly all of the `{ex.MISSED_OP}` survival comes from two of the seven red colors, but it scored conditions rather than individual lines. Scoring per line across the ladder tells us whether a narrow seed spread means the same lines survive every time, which would let us characterize the leftover, or a shifting set of lines that happens to be the same size.

**The achromatic candidate.** The exploratory section of ex-2.2.12 proposed that the surviving lines are answered from how gray the second operand looks. The axis does not hold that information, and the blocks can read it off the other channels. To test it, we project at the blocks with the second operand replaced by a gray of the same value. That is a scoring pass on checkpoints this experiment trains anyway.

**What the response looks like.** A grading cloud of the α response at op1 against how red the first operand is, one per level of the ladder. The grading r² in ex-2.2.12 barely moved across its whole sweep, so this is here as a picture of what a four-fold change in the recipe does to the response. It carries no gate.

## Discussion

/// admonition | TODO
About 200 words: what the ladder says the weight buys, whether the anchored-op experiments should inherit a different operating point, and what a predictable-but-unremovable leftover means for them.
///

## Method

### The ladder and the memoized runs

{sum(ex.MEMOIZED.values())} of the runs are served from the store: the twenty `handover` seeds from ex-2.2.11 are `{ex.REFERENCE}`, and three conditions of the sweep in ex-2.2.12 supply five seeds each. A condition that changes nothing is the same run, so we memoize those rather than retraining them.

Three of the borrowed conditions are five seeds each from ex-2.2.12, and one of those is the tight condition whose result motivated this experiment. Those seeds are seeds 1–5, which is why the deciding half is the *upper* one: writing a rule in advance does not make data we have already read unseen, so the seeds that suggested the effect are kept out of the half that decides.

### The split of the seeds

Seeds {ex.SELECT_RANGE[0]}–{ex.SELECT_RANGE[1]} decide the adoption rule, and seeds {ex.QUOTE_RANGE[0]}–{ex.QUOTE_RANGE[1]} supply the numbers we quote for the adopted condition. The upper half decides because the lower half is where the already-published ex-2.2.12 seeds sit.

The winning condition in a search wins partly on merit and partly on lucky seeds, so a number quoted from the same seeds that chose it is biased upward. Holding half the seeds back pays that correction inside this experiment instead of leaving it to the next one.

We check the four predictions on all {ex.SEEDS} seeds. They were fixed before the runs existed and no selection happens inside them, so there is no bias for a split to correct, and the spread question wants every seed it can get.

### The plane

As ex-2.2.12 defined it. The home of *red* is the first two axes of the stream together. Alignment is the length of the projection of a state onto that pair, and the anchor term pulls that length toward one on the labelled lines. The anti-subspace term is the square of that length over every live position, and the removal projects the whole plane out.

So the concept holds two coordinates of sixty-four rather than one, and its share of the variance counts both.

### Budget

{ex.NEW_RUNS} runs at d64-L4 ({ex.SEEDS} of them the fixed-repulsion arm), each as long as a run in ex-2.2.11, at about three minutes a run on an L4. Scoring adds the eleven ops under the operators from ex-2.2.11, the per-line pass, and the achromatic edit. That is about four times the sweep in ex-2.2.12, which cost six dollars on Modal.

### What this experiment does not vary

Depth, the pooling temperature τ, and the anti-subspace ratios stay at the values from ex-2.2.11, which means the anti-subspace weight itself climbs with the ladder. Ex-2.2.12 moved each of them one step and resolved nothing on any measurement, so a third factor here would cost runs without a prediction behind it.

The survey in ex-2.1.11 found that the weight and τ trade against each other, so a λ_a × τ factorial would be the natural follow-up if this ladder finds a level worth having.
"""
