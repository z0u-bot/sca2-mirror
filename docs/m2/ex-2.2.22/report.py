# title: Ex 2.2.22: where the pull acts, how far it goes, and contexts of varying length

# The design constants come from `experiment.py` beside this script (the directory of the script is on sys.path
# while it runs). This is the preregistration draft: the only computed figure is the posterior on the anchored op at
# each example count, from the seven-op table alone.
import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.vis import figure_html, light_dark, themed

P = ex.ex2216._POSTERIOR_MODULE

# --- Helpers -------------------------------------------------------------------------------------------------


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str, *, text_cols: int = 1) -> str:
    """An authored table in the shared report style; the first *text_cols* columns are text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>"
        + "".join(f"<td{' class=num' if i >= text_cols else ''}>{cell_html(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


# --- The design tables --------------------------------------------------------------------------------------


def weight_cell(a: ex.Arm) -> str:
    return "—" if not a.anchored else f"{a.weight:.2g}"


def arms_table() -> str:
    head = ["arm", "question", "reference", "slices", "λ_a", "cap", "examples", "what it changes"]
    rows = [
        [
            f"`{a.name}`",
            a.group,
            f"`{a.reference}`",
            a.slices if a.anchored else "—",
            weight_cell(a),
            "—" if a.cap is None else f"{a.cap:g}",
            ", ".join(map(str, a.counts)),
            a.note,
        ]
        for a in ex.ARMS
    ]
    return table_html(
        head,
        rows,
        f"The {len(ex.ARMS)} new arms, {ex.SEEDS} seeds each ({ex.N_RUNS} runs). Every arm trains on the seven-op set "
        f"at ρ = {ex.RHO:g}, with the newline mask, for {ex.EPOCHS} epochs, and every anchored arm anchors "
        f"`{ex.ANCHORED_OP}` on e₁ under the whole-line label. The reference is the ex-2.2.21 arm it is compared "
        "with, paired by model seed.",
        text_cols=8,
    )


# --- The posterior at each example count --------------------------------------------------------------------


TABLE7 = P.build_table(ex.ex2221.ex2218.table_of(ex.OP_NAMES))


@memo
def posterior_at_count(k: int, n: int, seed: int) -> np.ndarray:
    """The posterior on the anchored op over *n* contexts of that op with *k* examples, at the corpus noise rates."""
    d = list(ex.OP_NAMES).index(ex.ANCHORED_OP)
    rng = np.random.default_rng([seed, k])
    ctx = P.sample_contexts(TABLE7, n, k, ex.RHO, ex.ex2221.CUBE_RATE, rng, true_op=d)
    return P.posterior(TABLE7, ctx, ex.RHO, ex.ex2221.CUBE_RATE)[:, d].astype(np.float32)


N_POST = 20_000
POST_SEED = 2222
POST = {k: posterior_at_count(k, N_POST, POST_SEED) for k in ex.MIXED_COUNTS}
MID_LO, MID_HI = ex.ex2221.MIDDLE_BAND


def middle_share(post: np.ndarray) -> float:
    return float(((post >= MID_LO) & (post <= MID_HI)).mean())


MID_FIXED = middle_share(POST[ex.K])
MID_MIXED = middle_share(np.concatenate([POST[k] for k in ex.MIXED_COUNTS]))


def counts_figure() -> str:
    alt = f"""
        Cumulative distributions of the posterior on `{ex.ANCHORED_OP}` over `{ex.ANCHORED_OP}` contexts, one curve
        per example count from one to five. Each curve rises in a few steps, at different places for each count: one
        example puts most contexts near 0.7, three put them near 0.9 and 1, and five put most at 1. Mixing the counts
        puts {MID_MIXED:.0%} of contexts in the middle band, against {MID_FIXED:.0%} at three examples alone.
    """
    return counts_draw({k: POST[k] for k in ex.MIXED_COUNTS}, alt)


@memo
def counts_draw(post: dict[int, np.ndarray], alt_text: str) -> str:
    @themed(
        name="posterior-by-count",
        alt_text=alt_text,
        caption=f"""
            **The posterior on `{ex.ANCHORED_OP}` at each example count.** The share of `{ex.ANCHORED_OP}` contexts
            whose posterior is at most the value on the x-axis, on the seven-op table at ρ = {ex.RHO:g}. The heavy
            curve is three examples, the corpus of ex-2.2.21. The shaded band is the middle band of ex-2.2.16.
            {N_POST:,} sampled contexts per curve.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.0, 2.8), layout="constrained")
        ax.axvspan(MID_LO, MID_HI, facecolor=light_dark("#000", "#fff"), alpha=0.06, lw=0)
        x = np.linspace(0, 1, 401)
        shades = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(post)))
        for (k, p), color in zip(post.items(), shades, strict=True):
            s = np.sort(p)
            lw = 2.0 if k == ex.K else 1.0
            ax.plot(
                x,
                np.searchsorted(s, x, side="right") / len(s),
                color=color,
                lw=lw,
                label=f"{k} example{'s' if k > 1 else ''}",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"posterior on {ex.ANCHORED_OP}")
        ax.set_ylabel("share of contexts")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        return fig

    return _plot()


# %%

rf"""
# Ex 2.2.22: Where the pull acts, how far it goes, and contexts of varying length

/// tip |
<!-- lede -->
A scout before round 3 of the D2.2 route. We retrain the anchored `{ex.ANCHORED_OP}` arms of ex-2.2.21 with the pull kept off the first or last slices of the residual stream, with the hinge cap raised, and on contexts whose number of examples varies, and score each as ex-2.2.21 did.
///

Ex-2.2.21 found that an edit applied at every position can take `{ex.ANCHORED_OP}` out of an anchored model gradually while the other ops stay as they were, at least on the arm whose pull is capped. It left three questions about how round 3 should anchor. This scout trains {len(ex.ARMS)} new arms at {ex.SEEDS} seeds each, and reuses ex-2.2.21's runs as references, paired by model seed. It chooses nothing by itself; the choice for round 3 is made with the results in hand, and confirmed there at fresh seeds.
"""

# %%

r"""
## Findings

- [Where the pull acts (E1)](#where-the-pull-acts-e1) —
- [How far the pull goes (E2)](#how-far-the-pull-goes-e2) —
- [Mixed counts keep the control near its ceiling (H1)](#mixed-counts-keep-the-control-near-its-ceiling-h1) — *verdict*.
- [The anchor and the posterior (E3)](#the-anchor-and-the-posterior-e3) —
- [The edit lands on the target null (H2)](#the-edit-lands-on-the-target-null-h2) — *verdict*.
- [Decision](#decision) —

/// admonition | How to read this report
This is a preregistration draft. Once it is agreed, the predictions and the criteria for the decision are frozen at a commit quoted here, before any run of this experiment. Each section opens with what we expect, and the results will replace the placeholders in place.
///
"""

# %%

rf"""
## Why this experiment

In the in-context grammar the model never sees the name of an op. It reads a few solved examples, works out which op they follow, and applies that op to the query. When we anchor `{ex.ANCHORED_OP}`, we ask the residual stream to lie along one direction, e₁, on contexts whose examples follow `{ex.ANCHORED_OP}`. Ex-2.2.21 found that the anchor settles on the example answers, where the context shows the most about its op, and that an edit removing the e₁ component at every position takes `{ex.ANCHORED_OP}` out. Looked at after the fact, the edited model seemed to answer `{ex.ANCHORED_OP}` contexts much as an ideal predictor would if it no longer knew that op; H2 scores that.

Three things about that result could change how round 3 anchors.

First, which slices the pull acts on. A *slice* is one of the points along the residual stream where we can read the state: the token embedding, then the output of each block. The recipe pulls every slice. An arm that left the embedding slice out had the best task score of any arm, but its edit spilled onto other ops. An op is inferred from the whole context, so it seems likely to live in the middle of the stack, which suggests leaving out the last slice too. But leaving slices out changes the anchor weight in effect as well, since the term averages over the slices it pulls, so each new restriction comes with an arm at the weight that keeps the pull on each slice as it was.

Second, how hard the pull is. The hinge arm stops pulling once a state reaches an alignment of 0.8 with e₁, and its edit stayed within the selectivity gate; the uncapped arm just missed it. The cap of 0.8 was never searched, so two caps between 0.8 and no cap ask whether the selectivity falls off gradually.

Third, how many examples a context has. Round 3 may test whether the anchor grades with the evidence: whether a context whose examples only partly fit `{ex.ANCHORED_OP}` sits less far along e₁ than one that fits it plainly. At three examples the posterior on `{ex.ANCHORED_OP}` takes only a few distinct values, so there is little to grade against. Varying the count from one context to the next spreads it, and training on the mix lets the model learn that contexts vary in length. That changes the recipe, so the control has to show it still gets near its ceiling.
"""

# %%

rf"""
## Parameters

Every new arm changes one setting from an ex-2.2.21 arm. The recipe is ex-2.2.21's: the seven-op set, ρ = {ex.RHO:g}, the newline mask, {ex.EPOCHS} epochs, {ex.MODEL}, and the whole-line label. Verification lines are left out of every arm, so each pairs with its reference in one setting alone; ex-2.2.21 found they leave completion unchanged.

{arms_table()}

**The slice arms.** `no-last` leaves out the last slice, the input to the readout. `middle` leaves out the embedding slice and the last slice, so only the three slices between them are pulled. The anchor term and the anti-subspace term both average over the slices they act on, so at a fixed weight, pulling fewer slices pulls each one harder: by a quarter for `no-emb` and `no-last`, and by two thirds for `middle`. Each `-matched` arm scales the weight down by that factor. So the plain arm and its matched twin bracket the two readings of a restriction: a change in where the anchor is, and a change in how strongly each slice is pulled. Leaving a slice out also drops the anti-subspace term there, so the states of other ops at that slice are free to sit on e₁. No arm separates that from the anchor term, so E1 compares the two terms left out together.

**The cap arms.** The whole-line pull with the hinge at a cap of 0.9 and of 0.95. With ex-2.2.21's hinge arm at 0.8 and its whole-line arm uncapped, that makes four levels.

**The count arms.** The control and the hinge arm on a corpus where each context draws its number of examples uniformly from {", ".join(map(str, ex.MIXED_COUNTS))}. The mean is three, as in the fixed corpus, so an epoch has about as many tokens and steps. The held-out set has every count. The figure below shows what each count adds to the posterior.

{counts_figure()}

At three examples, {MID_FIXED:.0%} of `{ex.ANCHORED_OP}` contexts sit in the middle band; with the counts mixed, {MID_MIXED:.0%} do. More useful than the share is the number of distinct steps: one example puts most contexts near 0.7, two near 0.4 and 0.97, three near 0.9 and 1, four near 0.86 and 0.99, and five mostly near 1.

**The seeds.** The model seeds of ex-2.2.21's three-seed arms, {ex.SEED_OFFSET} to {ex.SEED_OFFSET + ex.SEEDS - 1}, so every comparison is paired by seed. In ex-2.2.21 the last of these took a slow path through training on most anchored arms, so the pairing also shows whether a setting changes that.
"""

# %%

rf"""
## Measurements

The measurements are ex-2.2.21's, with two additions.

**Task score.** Expected exact match (EEM) on held-out contexts: the probability the model puts on the right answer under the true op. Each run is compared with the control at the same seed. For the count arms it is reported at each count, beside the Bayes ceiling at that count (the score of an ideal predictor that weighs every op by how well it fits the examples).

**Where the anchor sits.** The alignment of the state with e₁ (their cosine) by role and slice, on `{ex.ANCHORED_OP}` contexts and on the others, and the op margin: how far `{ex.ANCHORED_OP}` contexts sit along e₁ beyond the rest. Ex-2.2.21 measured the margin at the last slice, which `no-last` and `middle` leave unpulled, so here it is measured at every slice and summarized as the mean over all five, for every arm alike. The anchor term asks for this, so it checks that the pull landed and tests nothing.

**The edit.** The projection at every position, at doses γ = {", ".join(f"{g:g}" for g in ex.DOSE_GAMMAS)} (the share of the e₁ component removed). Ex-2.2.21 (E2) set two criteria: the drop on `{ex.ANCHORED_OP}` grows with the dose and reaches at least {ex.GRADING_MIN_DAMAGE:.0%} of the way to the target null at full dose, and no other op drops by more than {ex.SELECTIVITY_GATE:g} at any dose (net of the control under the same edit). Beside the two criteria, the *selective reach*: how far toward the target null the strongest dose that stays within the gate goes.

**Landing (new).** The *target null* is the answer distribution of an ideal predictor that has lost `{ex.ANCHORED_OP}` and nothing else: it weighs the other ops by how well they fit the examples. For each held-out `{ex.ANCHORED_OP}` context, we take the KL divergence KL(target null ‖ model) of the model's answer distribution from the target null, on the clean model and under the full edit. (KL divergence measures how much one probability distribution differs from another; it is zero when they match.) Only this direction is finite: the target null puts no weight on colors that no remaining op gives, and the model puts some weight on every color.
The landing is the share of the clean gap the edit closes, taken as a ratio of means over contexts: one less the mean edited divergence over the mean clean divergence. A mean of per-context shares would be dominated by the contexts the clean model already answers like the target null. We also compare runs with one another: the Jensen-Shannon divergence between the answer distributions of each pair of seeds, a symmetric version of the KL divergence, under the edit and on the clean model.

**Grading (new).** At the answer of each example in a `{ex.ANCHORED_OP}` context, the alignment against the posterior on `{ex.ANCHORED_OP}` given the examples up to and including that answer (an answer is one color token). Every context gives several points this way, one per example, so even the fixed-count corpus has some spread; the mixed counts add more.
"""

# %%

r"""
## Where the pull acts (E1)

We look at the slice arms beside the whole-line and `no-emb` arms of ex-2.2.21, on four measures: the task score net of the control, the op margin, whether the edit meets the two criteria, and the selective reach. The question is whether one restriction holds the task score of `no-emb` while editing as selectively as the whole-line arm, and whether the matched weight changes the answer.

/// admonition | TODO
A table of the four measures for each slice arm and its reference, seed means with the seed range. A figure of the edit for each arm: the drop on `difference` and the worst other op against the dose, one panel per slice set, the plain and matched weights as two lines. A figure of the alignment by slice at the example answers, for `difference` contexts and the others.
///
"""

# %%

r"""
## How far the pull goes (E2)

The same four measures over the four caps: 0.8 (ex-2.2.21's hinge arm), 0.9, 0.95, and no cap (ex-2.2.21's whole-line arm). If the selectivity falls off gradually with the cap, a cap near where it crosses the gate gives the most anchor that still edits cleanly. If it drops at one cap, the step is where to stop. At three seeds a gradual fall smaller than the seed range would look flat, and the uncapped arm only just missed the gate in ex-2.2.21, so a flat result would say the cap matters less than the seeds vary.

/// admonition | TODO
The worst other op at full dose, the selective reach, the op margin, and the task score net of the control, each against the cap, with seeds as points and the seed mean as a line.
///
"""

# %%

rf"""
## Mixed counts keep the control near its ceiling (H1)

**What we expect.** On ex-2.2.21's held-out set, which has three examples per context, we expect the control trained on mixed counts to fall short of the ex-2.2.21 control by less than {ex.REGRESSION_TOL:g} in EEM, paired by seed: a pass. A shortfall above {ex.REGRESSION_TOL:g} but inside the seed band would be a partial pass, since three seeds could not tell it from noise, and a larger one a miss. A gain would also be a pass. If the seeds disagree about the direction by more than the tolerance each way, the result would be outside the plan, and the verdict would be Unresolved.

At each count we also report the skill of the mixed-count control, the share of the way from the floor (a predictor that ignores the examples) to the ceiling, with no gate. A count where the skill falls well below the others would say the model has not learned that length, which matters for the grading measurement.

/// admonition | TODO
EEM on the three-example held-out set for the two controls, per seed, with the tolerance marked. Beside it, the skill of the mixed-count control at each count, with the skill of ex-2.2.21's control at three examples as a reference.
///
"""

# %%

r"""
## The anchor and the posterior (E3)

Whether the alignment at the example answers rises with the posterior on `difference`. The label is the same for every `difference` context however well its examples fit, so if the alignment grades anyway, the anchor follows what the model has inferred and not just the label. We look at it on the hinge arm with mixed counts, and on ex-2.2.21's hinge and whole-line arms at three examples. This is a first look that shapes the prediction round 3 makes, so it has no gate.

/// admonition | TODO
The seed-mean alignment at the example answers against the posterior on `difference` given the examples so far, in bins, with the share of points in each bin. One panel per slice, one line per arm, and the control as a reference.
///
"""

# %%

rf"""
## The edit lands on the target null (H2)

**What we expect.** In ex-2.2.21, after the fact, the edited hinge arm answered `{ex.ANCHORED_OP}` contexts about as the target null does. If that holds, round 3 would not need to train a designed fallback, since the anchor alone gives the edit a predictable destination. We score it on the hinge arm, three seeds from ex-2.2.21's stored runs, since it is the arm round 3 is most likely to build on, and report every other arm beside it.

Two criteria, both at full dose with the edit at every position, on held-out `{ex.ANCHORED_OP}` contexts:
**(a)** the edit closes at least {ex.LANDING_FRACTION:.0%} of the KL divergence from the target null that the clean model has, as a ratio of means over contexts, in every seed;
**(b)** the edited runs agree with one another about as well as the clean runs do: their mean pairwise Jensen-Shannon divergence is at most {ex.SEED_AGREEMENT_RATIO:g} times that of the clean runs.

Both would be a pass, one a partial pass, and neither a miss, where (a) holds only if it holds in every seed and fails only if it fails in every seed. If it holds in some seeds and not others, the result would be outside the plan, and the verdict would be Unresolved.

/// admonition | Open decision
Three things to settle before the freeze.

**Which runs H2 scores.** The prediction came from looking at ex-2.2.21's hinge runs, and scoring those same runs would show little. Three more hinge runs at fresh seeds (about $0.40) would give H2 runs it has never seen; the cap and mixed-count arms each change one setting, so they are less clean as a test. Proposed: add the replicate and score H2 on it, with the stored runs beside it.

**Criterion (b).** The clean runs sit near the Bayes ceiling, so they agree closely, and a ratio over a divergence near zero can fail on tiny absolute differences, or swing from one pair of seeds to the next (three seeds make three pairs). Options: add an absolute floor below which (b) passes whatever the ratio; or drop (b), since (a) already says how close each run lands to one shared destination.

**The thresholds.** {ex.LANDING_FRACTION:.0%} in (a) is stricter than ex-2.2.21's half the way in EEM, since this criterion is about where the answers land and not only how much of the op is gone. To check: whether ex-2.2.21's post hoc matrices for the hinge arm suggest a closure near that value.
///

/// admonition | TODO
Per seed: the KL divergence from the target null on the clean model and under the full edit, as paired points, with the {ex.LANDING_FRACTION:.0%} line. Beside it, the mean pairwise Jensen-Shannon divergence between seeds, clean and edited, for each arm.
///
"""

# %%

rf"""
## Decision

Three choices for round 3: which slices the pull acts on, and at what weight; the cap; and whether the corpus mixes example counts. The candidates are the arms of this scout and the ex-2.2.21 arms they pair with.

The criteria, reported for every candidate in one table:

- task score net of the control (a hard gate: a candidate whose seed-mean shortfall exceeds both {ex.TASK_COST_TOL:g} and the seed band cannot be adopted);
- the op margin;
- whether the edit at every position meets ex-2.2.21's two criteria;
- the selective reach;
- the landing measurements of H2.

The counts choice also weighs H1 and how much spread E3 finds. The choice is made with the results in hand, and round 3 confirms it at fresh seeds before its numbers are quoted as a result.

/// admonition | TODO
The table of every criterion for every candidate, then what we chose and why.
///
"""

# %%

r"""
## Discussion

/// admonition | TODO
Written once the results are in.
///
"""

# %%

rf"""
## Method

**The corpus.** The fixed-count arms train on ex-2.2.21's corpus of three-example contexts. The count arms train on a corpus of the same number of contexts, built by the same generator, with the number of examples drawn uniformly per context from {", ".join(map(str, ex.MIXED_COUNTS))}. Two whole contexts at five examples take 76 tokens, inside the window of 96. The held-out set for the count arms has {ex.ex2221.HOLDOUT_CONTEXTS:,} contexts per op at each count. Every arm is also scored on ex-2.2.21's held-out set, so the count arms compare with their references on the same contexts.

**The edit and the landing.** The suppression pass is ex-2.2.21's, at every position only. The landing measurements use the predictive the edited model gives over the color vocabulary at the query answer, and the target null from the posterior over the other ops (`sca.data.incontext.target_null`).

**Budget.** {ex.N_RUNS} runs at about $0.13 each on an L4, the cost of ex-2.2.21's runs. Under $5 in all, with the scoring passes.
"""
