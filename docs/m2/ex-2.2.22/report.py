# title: Ex 2.2.22: localized by depth, various pull caps, and contexts of varying length

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


SETTING_CELLS = {
    "slices": lambda c: c.slices if c.anchored else "—",
    "weight": lambda c: f"{c.weight:.2g}" if c.anchored else "—",
    "cap": lambda c: "—" if c.cap is None else f"{c.cap:g}",
    "counts": lambda c: "–".join(map(str, (c.counts[0], c.counts[-1]))) if len(c.counts) > 1 else str(c.counts[0]),
}


def conditions_table() -> str:
    """One row per new condition. The setting it changes from its reference is in bold."""
    head = ["condition", "reference", "slices", "λ_a", "cap", "examples"]
    rows = []
    for c in ex.CONDITIONS:
        ref = ex.by_name(c.reference)
        cells = [
            f"<b>{fmt(c)}</b>" if ex.settings(c)[k] != ex.settings(ref)[k] else fmt(c)
            for k, fmt in SETTING_CELLS.items()
        ]
        rows.append([f"`{c.name}`", f"`{c.reference}`", *cells])
    return table_html(
        head,
        rows,
        f"The {len(ex.CONDITIONS)} new conditions, {ex.SEEDS} seeds each ({ex.N_RUNS} runs). Each changes the setting "
        "in bold from its reference, and is paired with it by model seed. The base is the hinge condition of "
        f"ex-2.2.21 (`{ex.BASE}`): every slice pulled, λ_a = {ex.LAMBDA_A:g}, the pull capped at "
        f"{ex.HINGE_CAP:g}, and {ex.K} examples per context.",
        text_cols=6,
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
# Ex 2.2.22: Localized by depth, various pull caps, and contexts of varying length

/// tip |
<!-- lede -->
A scout. We retrain the anchored `{ex.ANCHORED_OP}` condition with the pull kept off the embedding or readout, with the hinge cap raised, and on contexts whose number of examples varies.
///

Ex-2.2.21 found that an edit applied at every position can take `{ex.ANCHORED_OP}` out of an anchored model gradually while the other ops stay as they were, at least when the pull is capped. This scout trains {len(ex.CONDITIONS)} new conditions at {ex.SEEDS} seeds each, and reuses ex-2.2.21's runs as references, paired by model seed.
"""

# %%

r"""
## Findings

- [Localized by depth (E1)](#localized-by-depth-e1) —
- [Various pull caps (E2)](#various-pull-caps-e2) —
- [Mixed counts keep the recipe near its ceiling (H1)](#mixed-counts-keep-the-recipe-near-its-ceiling-h1) — *verdict*.
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

In the in-context grammar the model never sees the name of an op. It reads a few solved examples, works out which op must be in use, and applies that op to the query. So when we anchor `{ex.ANCHORED_OP}`, we ask the residual stream to lie along one direction, e₁, on contexts whose examples follow `{ex.ANCHORED_OP}`. Ex-2.2.21 found that the anchor settles on the example answers, where the context shows the most about its op, and that an edit removing the e₁ component at every position takes `{ex.ANCHORED_OP}` out. The edited model seemed to answer `{ex.ANCHORED_OP}` contexts much as an ideal predictor would if it no longer knew that op; H2 scores that more formally.

We now consider three changes to our recipe.

First, which slices the pull acts on. A *slice* is one of the depths along the residual stream where we can read the state: the token embedding, then the output of each block, the last of which feeds the readout. The current recipe pulls every slice. A condition that left the embedding slice out had the best task score of any, but its edit spilled onto other ops. An op is *inferred* from the whole context, so it seems likely to live in the middle of the stack, which suggests leaving out the last slice too. But leaving slices out changes the anchor weight in effect as well, since the term averages over the slices it pulls, so each new restriction comes with a condition at the weight that keeps the effective pull as it was.

Second, how hard the pull is on nearly aligned states. The `hinge` condition stops pulling once a state reaches an alignment of 0.8 with e₁, and its edit stayed within the selectivity gate; the uncapped condition *just* missed it. The cap of 0.8 was arbitrary, so two caps between 0.8 and no cap ask whether the selectivity falls off gradually.

Third, how many examples a context has. We would like to be able to test whether the anchor grades with the evidence. At three examples the posterior on `{ex.ANCHORED_OP}` takes only a few distinct values, so there is little to grade against. Varying the count from one context to the next spreads it out, and training on that lets the model learn that contexts vary in length. That changes the recipe, so it has to show it still gets near its ceiling.
"""

# %%

rf"""
## Parameters

Every new condition changes one setting from the `hinge` condition of ex-2.2.21, our best recipe so far. The base recipe is the same as in ex-2.2.21: the seven-op set; replacement op noise of ρ = {ex.RHO:g}, the chance that an example follows some other op; the newline mask; {ex.EPOCHS} epochs; {ex.MODEL}; and the whole-line label. Like the `hinge` condition, no condition has verification lines (examples whose answer the model judges as right or wrong); ex-2.2.21 found they leave completion unchanged.

{conditions_table()}

**The slice conditions.** `no-emb` leaves out the embedding slice, `no-last` the last slice, the input to the readout, and `middle` both, so only the three slices between them are pulled. The anchor term and the anti-subspace term both average over the slices they act on, so at a fixed weight, pulling fewer slices pulls each one harder: by a quarter for `no-emb` and `no-last`, and by two thirds for `middle`. Each `-matched` twin scales the weight down by that factor. So the plain condition and its matched twin bracket the two readings of a restriction: a change in where the anchor is, and a change in how strongly each slice is pulled. Leaving a slice out also drops the anti-subspace term there, so the states of other ops at that slice are free to sit on e₁. No condition separates that from the anchor term, so E1 compares the two terms left out together.

**The cap conditions.** The pull capped at 0.9 and at 0.95. With the `hinge` condition at {ex.HINGE_CAP:g} and ex-2.2.21's whole-line condition uncapped, that makes four levels.

**The count condition.** `mixed` trains on a corpus where each context draws its number of examples uniformly from {", ".join(map(str, ex.MIXED_COUNTS))}. The mean is three, as in the fixed corpus, so an epoch has about as many tokens and steps. The held-out set has every count. The figure below shows what each count adds to the posterior.

{counts_figure()}

At three examples, {MID_FIXED:.0%} of `{ex.ANCHORED_OP}` contexts sit in the middle band; with the counts mixed, {MID_MIXED:.0%} do. More useful than the share is the number of distinct steps: one example puts most contexts near 0.7, two near 0.4 and 0.97, three near 0.9 and 1, four near 0.86 and 0.99, and five mostly near 1.

**The seeds.** The model seeds of ex-2.2.21's three-seed conditions, {ex.SEED_OFFSET} to {ex.SEED_OFFSET + ex.SEEDS - 1}, so every comparison is paired by seed. In ex-2.2.21 the last of these took a slow path through training on most anchored conditions, so the pairing also shows whether a setting changes that.
"""

# %%

rf"""
## Measurements

The measurements are ex-2.2.21's, with two additions.

**Task score.** Expected exact match (EEM) on held-out contexts: the probability the model puts on the right answer under the true op. Each run is compared with the control at the same seed. For `mixed` it is reported at each count, beside the Bayes ceiling at that count (the score of an ideal predictor that weighs every op by how well it fits the examples).

**Where the anchor sits.** The alignment of the state with e₁ (their cosine) by role and slice, on `{ex.ANCHORED_OP}` contexts and on the others, and the op margin: how far `{ex.ANCHORED_OP}` contexts sit along e₁ beyond the rest. Ex-2.2.21 measured the margin at the last slice, which `no-last` and `middle` leave unpulled, so here it is measured at every slice and summarized as the mean over all five, for every condition alike. The anchor term asks for this, so it checks that the pull landed and tests nothing.

**The edit.** The projection at every position, at doses γ = {", ".join(f"{g:g}" for g in ex.DOSE_GAMMAS)} (the share of the e₁ component removed). Ex-2.2.21 (E2) set two criteria: the drop on `{ex.ANCHORED_OP}` grows with the dose and reaches at least {ex.GRADING_MIN_DAMAGE:.0%} of the way to the target null at full dose, and no other op drops by more than {ex.SELECTIVITY_GATE:g} at any dose (net of the control under the same edit). Beside the two criteria, the *selective reach*: how far toward the target null the strongest dose that stays within the gate goes.

**Landing (new).** The *target null* is the answer distribution of an ideal predictor that has lost `{ex.ANCHORED_OP}` and nothing else: it weighs the other ops by how well they fit the examples. For each held-out `{ex.ANCHORED_OP}` context, we take the KL divergence KL(target null ‖ model) of the model's answer distribution from the target null, on the clean model and under the full edit. (KL divergence measures how much one probability distribution differs from another; it is zero when they match.) Only this direction is finite: the target null puts no weight on colors that no remaining op gives, and the model puts some weight on every color.
The landing is the share of the clean gap the edit closes, taken as a ratio of means over contexts: one less the mean edited divergence over the mean clean divergence. A mean of per-context shares would be dominated by the contexts the clean model already answers like the target null. We also compare runs with one another: the Jensen-Shannon divergence between the answer distributions of each pair of seeds, a symmetric version of the KL divergence, under the edit and on the clean model.

**Grading (new).** At the answer of each example in a `{ex.ANCHORED_OP}` context, the alignment against the posterior on `{ex.ANCHORED_OP}` given the examples up to and including that answer (an answer is one color token). Every context gives several points this way, one per example, so even the fixed-count corpus has some spread; the mixed counts add more.
"""

# %%

r"""
## Localized by depth (E1)

We look at the slice conditions beside the `hinge` condition, on four measures: the task score net of the control, the op margin, whether the edit meets the two criteria, and the selective reach. The question is whether a restriction raises the task score, as leaving out the embedding did on the uncapped pull in ex-2.2.21, while editing as selectively as the `hinge` condition, and whether the matched weight changes the answer. Ex-2.2.21's uncapped `no-emb` condition is shown beside its capped twin.

/// admonition | TODO
A table of the four measures for each slice condition and its reference, seed means with the seed range. A figure of the edit for each condition: the drop on `difference` and the worst other op against the dose, one panel per slice set, the plain and matched weights as two lines. A figure of the alignment by slice at the example answers, for `difference` contexts and the others.
///
"""

# %%

r"""
## Various pull caps (E2)

The same four measures over the four caps: 0.8 (the `hinge` condition), 0.9, 0.95, and no cap (ex-2.2.21's whole-line condition). If the selectivity falls off gradually with the cap, a cap near where it crosses the gate gives the most anchor that still edits cleanly. If it drops at one cap, the step is where to stop. At three seeds a gradual fall smaller than the seed range would look flat, and the uncapped condition only just missed the gate in ex-2.2.21, so a flat result would say the cap matters less than the seeds vary.

/// admonition | TODO
The worst other op at full dose, the selective reach, the op margin, and the task score net of the control, each against the cap, with seeds as points and the seed mean as a line.
///
"""

# %%

rf"""
## Mixed counts keep the recipe near its ceiling (H1)

**What we expect.** On ex-2.2.21's held-out set, which has three examples per context, we expect the `mixed` condition to fall short of the `hinge` condition by less than {ex.REGRESSION_TOL:g} in EEM, paired by seed: a pass. A shortfall above {ex.REGRESSION_TOL:g} but inside the seed band would be a partial pass, since three seeds could not tell it from noise, and a larger one a miss. A gain would also be a pass. If the seeds disagree about the direction by more than the tolerance each way, the result would be outside the plan, and the verdict would be Unresolved.

At each count we also report the skill of `mixed`, the share of the way from the floor (a predictor that ignores the examples) to the ceiling, with no gate. A count where the skill falls well below the others would say the model has not learned that length, which matters for the grading measurement.

/// admonition | TODO
EEM on the three-example held-out set for `mixed`, `hinge`, and the control, per seed, with the tolerance marked. Beside it, the skill of `mixed` at each count, with the skill of `hinge` at three examples as a reference.
///
"""

# %%

r"""
## The anchor and the posterior (E3)

Whether the alignment at the example answers rises with the posterior on `difference`. The label is the same for every `difference` context however well its examples fit, so if the alignment grades anyway, the anchor follows what the model has inferred and not just the label. We look at it on `mixed`, and on the `hinge` and whole-line conditions of ex-2.2.21 at three examples. This is a first look that would shape a later prediction, so it has no gate.

/// admonition | TODO
The seed-mean alignment at the example answers against the posterior on `difference` given the examples so far, in bins, with the share of points in each bin. One panel per slice, one line per condition, and the control as a reference.
///
"""

# %%

rf"""
## The edit lands on the target null (H2)

**What we expect.** In ex-2.2.21, after the fact, the edited `hinge` condition answered `{ex.ANCHORED_OP}` contexts about as the target null does. If that holds, we would not need to train a designed fallback, since the anchor alone gives the edit a predictable destination. We score it on the `hinge` condition, the base of this scout, and report every other condition beside it.

Two criteria, both at full dose with the edit at every position, on held-out `{ex.ANCHORED_OP}` contexts:
**(a)** the edit closes at least {ex.LANDING_FRACTION:.0%} of the KL divergence from the target null that the clean model has, as a ratio of means over contexts, in every seed;
**(b)** the edited runs agree with one another about as well as the clean runs do: their mean pairwise Jensen-Shannon divergence is at most {ex.SEED_AGREEMENT_RATIO:g} times that of the clean runs.

Both would be a pass, one a partial pass, and neither a miss, where (a) holds only if it holds in every seed and fails only if it fails in every seed. If it holds in some seeds and not others, the result would be outside the plan, and the verdict would be Unresolved.

/// admonition | Open decision
Three things to settle before the freeze.

**Which runs H2 scores.** The prediction came from looking at ex-2.2.21's hinge runs, and scoring those same runs would show little. Three more `hinge` runs at fresh seeds (about $0.40) would give H2 runs it has never seen; the other conditions each change one setting, so they are less clean as a test. Proposed: add the replicate and score H2 on it, with the stored runs beside it.

**Criterion (b).** The clean runs sit near the Bayes ceiling, so they agree closely, and a ratio over a divergence near zero can fail on tiny absolute differences, or swing from one pair of seeds to the next (three seeds make three pairs). Options: add an absolute floor below which (b) passes whatever the ratio; or drop (b), since (a) already says how close each run lands to one shared destination.

**The thresholds.** {ex.LANDING_FRACTION:.0%} in (a) is stricter than ex-2.2.21's half the way in EEM, since this criterion is about where the answers land and not only how much of the op is gone. To check: whether ex-2.2.21's post hoc matrices for the `hinge` condition suggest a closure near that value.
///

/// admonition | TODO
Per seed: the KL divergence from the target null on the clean model and under the full edit, as paired points, with the {ex.LANDING_FRACTION:.0%} line. Beside it, the mean pairwise Jensen-Shannon divergence between seeds, clean and edited, for each condition.
///
"""

# %%

rf"""
## Decision

Three choices for the recipe: which slices the pull acts on, and at what weight; the cap; and whether the corpus mixes example counts. The candidates are the conditions of this scout and the ex-2.2.21 conditions they pair with.

The criteria, reported for every candidate in one table:

- task score net of the control (a hard gate: a candidate whose seed-mean shortfall exceeds both {ex.TASK_COST_TOL:g} and the seed band cannot be adopted);
- the op margin;
- whether the edit at every position meets ex-2.2.21's two criteria;
- the selective reach;
- the landing measurements of H2.

The counts choice also weighs H1 and how much spread E3 finds. The choice is made with the results in hand, and confirmed at fresh seeds before its numbers are quoted as a result.

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

**The corpus.** The fixed-count conditions train on ex-2.2.21's corpus of three-example contexts. `mixed` trains on a corpus of the same number of contexts, built by the same generator, with the number of examples drawn uniformly per context from {", ".join(map(str, ex.MIXED_COUNTS))}. Two whole contexts at five examples take 76 tokens, inside the window of 96. The held-out set for `mixed` has {ex.ex2221.HOLDOUT_CONTEXTS:,} contexts per op at each count. Every condition is also scored on ex-2.2.21's held-out set, so `mixed` compares with its reference on the same contexts.

**The edit and the landing.** The suppression pass is ex-2.2.21's, at every position only. The landing measurements use the predictive the edited model gives over the color vocabulary at the query answer, and the target null from the posterior over the other ops (`sca.data.incontext.target_null`).

**Budget.** {ex.N_RUNS} runs at about $0.13 each on an L4, the cost of ex-2.2.21's runs. Under $5 in all, with the scoring passes.
"""
