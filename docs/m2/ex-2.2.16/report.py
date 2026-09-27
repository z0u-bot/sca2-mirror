# title: Ex 2.2.16: the in-context grammar pilot

# The design constants come from `experiment.py` beside this script and the posterior computation from
# `posterior.py` (the script's directory is on sys.path while it runs). Nothing here reads the store yet: the method
# section is computed from the op table alone, with no training.
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
import posterior as P
from mini.lit import memo
from mini.vis import AxesRow, figure_html, light_dark, themed

# --- Helpers -------------------------------------------------------------------------------------------------


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(
    head: list[str], rows: list[list[str]], caption: str, *, ref_rows: frozenset[int] = frozenset(), text_cols: int = 1
) -> str:
    """An authored result table in the shared report style; the first *text_cols* columns are text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        f"<tr{' class=ref' if r in ref_rows else ''}>"
        + "".join(f"<td{' class=num' if i >= text_cols else ''}>{cell_html(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for r, row in enumerate(rows)
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def cond_name(k: int, rho: float) -> str:
    """How a grammar condition is named in tables and figures: `k3-r0.3` is three examples at ρ = 0.3."""
    return f"k{k}-r{rho:g}"


def rho_colors(n: int) -> list:
    """Ordered shades for the replacement rates, from a colormap whose ends both survive the page background."""
    lo, hi = light_dark((0.0, 0.85), (0.3, 1.0))
    return [plt.get_cmap("viridis")(t) for t in np.linspace(lo, hi, n)]


def rule_color() -> str:
    return light_dark("#333", "#ddd")


# --- The posterior scan ---------------------------------------------------------------------------------------


@memo
def answer_table() -> P.AnswerTable:
    return P.build_table(ex.TABLE)


@memo
def table_constants(table: P.AnswerTable) -> tuple[float, float, float]:
    """The told-op ceiling, its mode form, and the analytic floor; the floor is a few seconds of lookups."""
    return table.told_op(), table.mode_match(), table.floor()


TABLE = answer_table()
TOLD_OP, TOLD_OP_MODE, FLOOR = table_constants(TABLE)


@memo
def scan(table: P.AnswerTable, k_grid: tuple, rho_grid: tuple, n: int, seed: int) -> dict:
    """For every grid point: the posterior on the true op over *n* sampled contexts, and the mean Bayes ceiling,
    its hard-accuracy form, and the floor over their queries. Each point draws from its own stream, so the scan
    does not depend on the order the grid is walked.
    """
    out = {}
    for i, k in enumerate(k_grid):
        for j, rho in enumerate(rho_grid):
            rng = np.random.default_rng([seed, i, j, 0])
            ctx = P.sample_contexts(table, n, k, rho, 0.0, rng)
            post = P.posterior(table, ctx, rho, 0.0)
            out[(k, rho)] = {
                "post_true": post[np.arange(n), ctx.true_op].astype(np.float32),
                "ceiling": float(P.expected_match(table, ctx, post).mean()),
                "mode_ceiling": float(P.mode_match(table, ctx, post).mean()),
                "floor": float(P.floor(table, ctx).mean()),
            }
    return out


@memo
def cube_scan(table: P.AnswerTable, conditions: tuple, cube_grid: tuple, n: int, seed: int) -> dict:
    """The ceiling at each proposed condition under cube noise at each rate: for the posterior that knows the rate,
    and for the one that ignores it (computed as if κ were zero on the same contexts). The stream at κ = 0 is the
    one `scan` uses at the same grid point, so the two tables agree there.
    """
    out = {}
    for k, rho in conditions:
        for kappa in cube_grid:
            rng = np.random.default_rng([seed, ex.K_GRID.index(k), ex.RHO_GRID.index(rho), round(kappa * 1000)])
            ctx = P.sample_contexts(table, n, k, rho, kappa, rng)
            out[(k, rho, kappa)] = {
                "ceiling": float(P.ceiling(table, ctx, rho, kappa).mean()),
                "ignoring": float(P.ceiling(table, ctx, rho, 0.0).mean()),
            }
    return out


@memo
def per_op(table: P.AnswerTable, k: int, rho: float, n: int, seed: int) -> dict:
    """The ceiling, the told-op ceiling, the floor, and the spread of the posterior on the true op, for contexts of
    each op in turn at one grid point.
    """
    out = {}
    for o, name in enumerate(table.names):
        rng = np.random.default_rng([seed, 200, o])
        ctx = P.sample_contexts(table, n, k, rho, 0.0, rng, true_op=o)
        post = P.posterior(table, ctx, rho, 0.0)
        pt = post[:, o]
        out[name] = {
            "ceiling": float(P.expected_match(table, ctx, post).mean()),
            "told": float((table.prob[o] ** 2).sum(-1).mean()),
            "floor": float(P.floor(table, ctx).mean()),
            "sd": float(pt.std()),
            "mid": float(((pt >= ex.MIDDLE_BAND[0]) & (pt <= ex.MIDDLE_BAND[1])).mean()),
        }
    return out


SCAN = scan(TABLE, ex.K_GRID, ex.RHO_GRID, ex.N_CONTEXTS, ex.POSTERIOR_SEED)
CUBE = cube_scan(TABLE, ex.GRAMMAR_CONDITIONS, ex.CUBE_GRID, ex.N_CONTEXTS, ex.POSTERIOR_SEED)
PER_OP = per_op(TABLE, *ex.CENTRE, ex.N_CONTEXTS // 4, ex.POSTERIOR_SEED)


def spread(k: int, rho: float) -> float:
    """The standard deviation of the posterior on the true op across contexts: the one-number spread."""
    return float(SCAN[(k, rho)]["post_true"].std())


def band_shares(k: int, rho: float) -> tuple[float, float, float]:
    """Shares of contexts below, inside, and above the middle band of the posterior on the true op."""
    pt = SCAN[(k, rho)]["post_true"]
    lo, hi = ex.MIDDLE_BAND
    return float((pt < lo).mean()), float(((pt >= lo) & (pt <= hi)).mean()), float((pt > hi).mean())


def ceiling(k: int, rho: float) -> float:
    return SCAN[(k, rho)]["ceiling"]


# --- Figures ------------------------------------------------------------------------------------------------


def posterior_figure() -> str:
    data = {kr: v["post_true"] for kr, v in SCAN.items()}
    lo3, mid3, hi3 = band_shares(*ex.CENTRE)
    alt = f"""
        Six panels, one per example count from {ex.K_GRID[0]} to {ex.K_GRID[-1]}, each showing the cumulative share
        of contexts against the posterior on the true op, one curve per replacement rate. With no noise the curves
        hug the right edge from three examples on; as the rate rises they spread across the range. At the center
        condition, {ex.CENTRE[0]} examples and ρ = {ex.CENTRE[1]:g}, {lo3:.0%} of contexts sit below the middle
        band, {mid3:.0%} inside it, and {hi3:.0%} above.
    """
    return posterior_draw(data, ex.K_GRID, ex.RHO_GRID, ex.MIDDLE_BAND, alt)


@memo
def posterior_draw(data: dict, k_grid: tuple, rho_grid: tuple, band: tuple, alt_text: str) -> str:
    @themed(
        name="posterior-ecdf",
        alt_text=alt_text,
        caption=f"""
            **The posterior on the true op, across contexts.** Each panel is one example count; each curve is one
            replacement rate ρ, shaded light to dark from 0 to {rho_grid[-1]:g}. A curve shows the share of contexts
            whose posterior on the true op is at most the value on the x-axis. The shaded band is the middle band,
            {band[0]:g} to {band[1]:g}: a curve that rises steeply inside it has many graded contexts, one that stays
            flat until the right edge has contexts that all but name the op. Sampled contexts, {ex.N_CONTEXTS:,}
            per curve.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.6), layout="constrained", sharex=True, sharey=True)
        axes = cast(np.ndarray, axes).ravel()
        colors = rho_colors(len(rho_grid))
        x = np.linspace(0, 1, 401)
        for ax, k in zip(axes, k_grid, strict=True):
            ax.axvspan(band[0], band[1], facecolor=light_dark("#000", "#fff"), alpha=0.06, lw=0, zorder=0)
            for rho, c in zip(rho_grid, colors, strict=True):
                pt = np.sort(data[(k, rho)])
                ax.plot(x, np.searchsorted(pt, x, side="right") / len(pt), color=c, lw=1.2, label=f"ρ = {rho:g}")
            ax.set_title(f"{k} example{'s' if k > 1 else ''}", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        for ax in axes[3:]:
            ax.set_xlabel("posterior on the true op")
        for ax in axes[::3]:
            ax.set_ylabel("share of contexts")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


def ceiling_figure() -> str:
    ceil = {kr: v["ceiling"] for kr, v in SCAN.items()}
    sd = {kr: spread(*kr) for kr in SCAN}
    best_k, best_rho = max(sd, key=lambda kr: sd[kr])
    alt = f"""
        Two line charts against the example count. Left, the Bayes ceiling in expected exact match: every curve rises
        with more examples and sits lower at a higher replacement rate, toward a dashed line at {TOLD_OP:.2f} for a
        model told the op, with the floor as a dotted line at {FLOOR:.2f}. Right, the spread of the posterior on the
        true op, which peaks at {sd[(best_k, best_rho)]:.2f} for {best_k} examples at ρ = {best_rho:g} and falls
        toward zero as the examples pin the op down.
    """
    return ceiling_draw(ceil, sd, ex.K_GRID, ex.RHO_GRID, TOLD_OP, FLOOR, ex.GRAMMAR_CONDITIONS, alt)


@memo
def ceiling_draw(
    ceil: dict, sd: dict, k_grid: tuple, rho_grid: tuple, told: float, floor: float, proposed: tuple, alt_text: str
) -> str:
    @themed(
        name="ceiling-grid",
        alt_text=alt_text,
        caption=f"""
            **The Bayes ceiling and the spread of the stimulus, over the grid.** One curve per replacement rate ρ,
            shaded as in the figure above. **Left:** the ceiling, the expected exact match of the predictor that
            answers with the posterior-weighted answer distribution. The dashed line is the same predictor told the
            op ({told:.3f}), the dotted line the floor, with no evidence ({floor:.3f}). **Right:** the standard
            deviation of the posterior on the true op across contexts. Ringed marks are the three proposed
            conditions.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained")
        axes = cast(AxesRow, axes)
        colors = rho_colors(len(rho_grid))
        for ax, series, label in zip(
            axes, (ceil, sd), ("Bayes ceiling (EEM)", "spread of the posterior (sd)"), strict=True
        ):
            for rho, c in zip(rho_grid, colors, strict=True):
                ax.plot(k_grid, [series[(k, rho)] for k in k_grid], "-o", color=c, lw=1.2, ms=3.5, label=f"ρ = {rho:g}")
            for k, rho in proposed:
                ax.plot(k, series[(k, rho)], "o", ms=9, mfc="none", mec=rule_color(), mew=0.9, zorder=5)
            ax.set_xticks(list(k_grid))
            ax.set_xlabel("examples per context")
            ax.set_ylabel(label)
        axes[0].axhline(told, color=rule_color(), lw=0.9, ls="--", zorder=1)
        axes[0].axhline(floor, color=rule_color(), lw=0.7, ls=":", zorder=1)
        axes[0].set_ylim(0, 0.8)
        axes[1].set_ylim(0, 0.4)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


def cube_figure() -> str:
    k0, r0 = ex.CENTRE
    drop = CUBE[(k0, r0, ex.CUBE_GRID[-1])]["ceiling"] - CUBE[(k0, r0, 0.0)]["ceiling"]
    alt = f"""
        A line chart of the Bayes ceiling against the cube-noise rate from 0 to {ex.CUBE_GRID[-1]:g}, one line per
        proposed condition. Every line falls gently; at the center condition the ceiling drops by {abs(drop):.3f}
        at the highest rate. Hollow marks, for a posterior that ignores the cube noise, sit a few thousandths above
        the filled ones.
    """
    return cube_draw(CUBE, ex.GRAMMAR_CONDITIONS, ex.CUBE_GRID, alt)


@memo
def cube_draw(cube: dict, conditions: tuple, cube_grid: tuple, alt_text: str) -> str:
    @themed(
        name="cube-noise",
        alt_text=alt_text,
        caption="""
            **What cube noise costs the ceiling.** One line per proposed condition. Filled marks: the ceiling for
            the posterior that knows the cube-noise rate κ. Hollow marks: the ceiling for the posterior that treats
            every odd example as replacement noise, which is what a model that has not learned about cube noise
            would hold.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.6, 3.0), layout="constrained")
        colors = [light_dark(*c) for c in (("#2b6cb0", "#7fb3ff"), ("#c0392b", "#ff8a76"), ("#2e8b57", "#7fd8a4"))]
        for (k, rho), c in zip(conditions, colors, strict=True):
            ys = [cube[(k, rho, kap)]["ceiling"] for kap in cube_grid]
            ys_ign = [cube[(k, rho, kap)]["ignoring"] for kap in cube_grid]
            ax.plot(cube_grid, ys, "-o", color=c, lw=1.2, ms=4, label=cond_name(k, rho))
            ax.plot(cube_grid, ys_ign, "o", color=c, ms=4, mfc="none", mew=0.9)
        ax.set_xlabel("cube-noise rate κ")
        ax.set_ylabel("Bayes ceiling (EEM)")
        ax.set_xticks(list(cube_grid))
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


# --- Tables -------------------------------------------------------------------------------------------------


def grid_table(stat, caption: str, fmt: str = "{:.3f}") -> str:
    head = ["examples", *(f"ρ = {rho:g}" for rho in ex.RHO_GRID)]
    rows = [[str(k), *(fmt.format(stat(k, rho)) for rho in ex.RHO_GRID)] for k in ex.K_GRID]
    return table_html(head, rows, caption)


def conditions_table() -> str:
    head = [
        "condition",
        "examples",
        "ρ",
        "tokens per line",
        "ceiling (EEM) ↑",
        "floor",
        "ceiling − floor",
        "hard-EM ceiling",
        "spread (sd)",
        "below band",
        "in band",
        "above band",
    ]
    rows = []
    for k, rho in ex.GRAMMAR_CONDITIONS:
        lo, mid, hi = band_shares(k, rho)
        c = ceiling(k, rho)
        rows.append(
            [
                f"`{cond_name(k, rho)}`",
                str(k),
                f"{rho:g}",
                str(ex.context_tokens(k)),
                f"{c:.3f}",
                f"{SCAN[(k, rho)]['floor']:.3f}",
                f"{c - SCAN[(k, rho)]['floor']:.3f}",
                f"{SCAN[(k, rho)]['mode_ceiling']:.3f}",
                f"{spread(k, rho):.3f}",
                f"{lo:.0%}",
                f"{mid:.0%}",
                f"{hi:.0%}",
            ]
        )
    return table_html(
        head,
        rows,
        f"The three proposed grammar conditions. The ceiling and floor are the analytic bounds the control at each "
        f"condition is scored between; the hard-EM ceiling is the same predictor naming the mode, for comparison "
        f"with the pivot. The band columns are shares of contexts by posterior on the true op, below, inside, and "
        f"above {ex.MIDDLE_BAND[0]:g}–{ex.MIDDLE_BAND[1]:g}. The center condition is `{cond_name(*ex.CENTRE)}`.",
        ref_rows=frozenset({ex.GRAMMAR_CONDITIONS.index(ex.CENTRE)}),
    )


def per_op_table() -> str:
    head = ["op", "ceiling (EEM)", "told the op", "floor", "spread (sd)", "in band"]
    rows = [
        [
            f"`{name}`",
            f"{v['ceiling']:.3f}",
            f"{v['told']:.3f}",
            f"{v['floor']:.3f}",
            f"{v['sd']:.3f}",
            f"{v['mid']:.0%}",
        ]
        for name, v in PER_OP.items()
    ]
    return table_html(
        head,
        rows,
        f"Per true op at the center condition, `{cond_name(*ex.CENTRE)}`: the ceiling, the ceiling for a model told "
        f"the op (which is the rounding cap), the floor, the spread of the posterior on the op across its contexts, "
        f"and the share of its contexts in the middle band. Ops total on the grid have a told-op ceiling of 1. "
        f"`{ex.ANCHORED_OP}` is the anchored op.",
        ref_rows=frozenset({ex.OP_NAMES.index(ex.ANCHORED_OP)}),
    )


def cube_table() -> str:
    head = ["condition", *(f"κ = {kap:g}" for kap in ex.CUBE_GRID)]
    rows = [
        [f"`{cond_name(k, rho)}`", *(f"{CUBE[(k, rho, kap)]['ceiling']:.3f}" for kap in ex.CUBE_GRID)]
        for k, rho in ex.GRAMMAR_CONDITIONS
    ]
    return table_html(
        head,
        rows,
        "The Bayes ceiling at each proposed condition under cube noise at rate κ, for the posterior that knows κ.",
    )


# --- The numbers the prose quotes -------------------------------------------------------------------------------

N = {
    "clean1_hi": band_shares(1, 0.0)[2],
    "clean3_hi": band_shares(3, 0.0)[2],
    "centre": dict(zip(("lo", "mid", "hi"), band_shares(*ex.CENTRE), strict=True)),
    "r35": dict(zip(("lo", "mid", "hi"), band_shares(3, 0.35), strict=True)),
    "ceil_clean3": ceiling(3, 0.0),
    "mode_clean3": SCAN[(3, 0.0)]["mode_ceiling"],
    "ceil_r35": ceiling(3, 0.35),
    "mode_r35": SCAN[(3, 0.35)]["mode_ceiling"],
    "ceil_centre": ceiling(*ex.CENTRE),
    "sd_centre": spread(*ex.CENTRE),
    "sd_max": max(spread(*kr) for kr in SCAN),
    "sd_max_at": max(SCAN, key=lambda kr: spread(*kr)),
    "floor_range": (min(v["floor"] for v in SCAN.values()), max(v["floor"] for v in SCAN.values())),
    "cube_cost": {
        kap: CUBE[(*ex.CENTRE, 0.0)]["ceiling"] - CUBE[(*ex.CENTRE, kap)]["ceiling"] for kap in ex.CUBE_GRID[1:]
    },
    "cube_ignoring": {
        kap: CUBE[(*ex.CENTRE, kap)]["ignoring"] - CUBE[(*ex.CENTRE, kap)]["ceiling"] for kap in ex.CUBE_GRID[1:]
    },
    "diff": PER_OP[ex.ANCHORED_OP],
}
COND_NAMES = [cond_name(k, rho) for k, rho in ex.GRAMMAR_CONDITIONS]
ALT_34 = (3, 0.4)
ALT_435 = (4, 0.35)

rf"""
# Ex 2.2.16: the in-context grammar pilot

/// tip |
<!-- tl;dr -->
The first experiment on the grammar from the [D2.2 pivot](/docs/m2/d2.2/pivot.md): the model infers the op from a few solved examples, with no word to name it. We train the control at three conditions of example count and replacement rate, anchor `{ex.ANCHORED_OP}` under the whole-line label and its variants, and score everything against a ceiling computed from the op table. Frozen rules say what round 3 adopts.
///

This is round 2 of the [quick route](/docs/m2/d2.2/design.md#quick-route). The method section folds in the [posterior scouting report](/todo/science/scout-posterior-in-context-grammar.md): it computes the posterior over ops and the Bayes ceiling from the op table, with no training, and uses them to pick the three grammar conditions for the control.

## Findings

/// admonition | TODO
One line per rule and per prediction, linking to its section, verdict blank. The pilot gates nothing; each of the frozen rules (a)–(e) from the design gets a line saying what it proposed. Sandy to shape.
///

## How to read this draft

The method section on the posterior and the ceiling is complete and computed; the sections it feeds are stubs. The rules and any predictions are not yet frozen. When they are, this note quotes the commit.

## Why this experiment

[Ex-2.2.14](/docs/m2/ex-2.2.14/report.py) anchored an op and every gate passed, but the anchor went to the word that names the op, so the anchored concept was an attribute of one token. The [pivot](/docs/m2/d2.2/pivot.md) takes op words out of the grammar.

Each line is now one context: a few solved examples of one op, written with `?` in place of the op word, then a query under the same op. The op is inferred from the examples, and the anchored concept is "the op in this context is `{ex.ANCHORED_OP}`", which no token names.

The pilot does four jobs the old plan spread over four rounds: it scouts the posterior (the method section below), trains the new-grammar control, smoke-tests anchoring the inferred op, and pilots the label variants.

The control is the regression check; a grammar this different needs its own, as ex-2.2.3 did. No model trained on the corpus can know more about the op than the examples say, and the posterior captures that. So instead of fixed accuracy numbers, the task gate is how far the control falls short of the ceiling the posterior sets, with an analytic floor beside it.

The crop policy starts from `{ex.CROP_POLICY}`, which pulls only labelled lines wholly inside the training window. [Ex-2.2.15](/docs/m2/ex-2.2.15/report.py) found that lines cut short by the window show the first-operand lean. Its review chose `whole` over the rule's `half` because a whole line always shows its evidence, which matters more once the evidence is a set of examples, and its discussion asks the pilot to watch the trailing-fragment lean as well.

## Glossary

<dl>
<dt>Context</dt>
<dd>One line of the corpus: <em>k</em> examples of one op, then a query. Its op is inferred from the examples.</dd>
<dt>Example</dt>
<dd>A solved equation inside a context, <code>a ? b = y,</code>: the evidence the op is inferred from.</dd>
<dt>Query</dt>
<dd>The last equation of a context, whose answer the model completes. It is always clean.</dd>
<dt>Replacement noise</dt>
<dd>Showing, in some examples, the answer another op would give, at a rate ρ per example. It spreads the posterior over ops, so it grades the stimulus.</dd>
<dt>Cube noise</dt>
<dd>Showing, in some examples, a color drawn from the whole grid, at a rate κ. Usually no op produces it, so it removes an example's evidence without pointing anywhere else.</dd>
<dt>Posterior on the true op</dt>
<dd>How strongly the examples of a context point to the op that generated it, computed from the op table under the rounding and noise the corpus uses. For <code>{ex.ANCHORED_OP}</code> contexts it is the graded stimulus.</dd>
<dt>Bayes ceiling</dt>
<dd>The expected exact match on the query of the predictor that holds the posterior over ops and answers with the posterior-weighted answer distribution: the score of a calibrated model that has learned everything the examples say, which is where cross-entropy training aims.</dd>
<dt>Floor</dt>
<dd>The same predictor with no evidence: uniform over ops, so its answer distribution is the op-marginal. It is what a model that ignores the examples can score.</dd>
<dt>Skill score</dt>
<dd>Where a score sits between the floor (0) and the ceiling (1): (score − floor) / (ceiling − floor).</dd>
</dl>

## Conditions

/// admonition | TODO
The conditions table: the control at `{"`, `".join(COND_NAMES)}`; one larger control at the center; anchored `{ex.ANCHORED_OP}` at the center under the whole-line label and each of the four label variants; the hinge-capped pull; the control and the whole-line arm with verification lines; the newline mask on the control and the whole-line arm; and `knowable` as an oracle. Three seeds per arm at {ex.MODEL}, crop policy `{ex.CROP_POLICY}`. The list is in the [design](/docs/m2/d2.2/design.md#the-pilot); the table and its prose go here once the arms are settled.
///

## The grammar rule (a)

/// admonition | TODO
The frozen rule: the condition with the widest spread of posteriors, among those where the {ex.MODEL} control comes within a margin of the ceiling, goes forward. The margin, the spread statistic (the method proposes the standard deviation of the posterior on the true op), and the larger-control branch to be set. Figure: control EEM per condition as seed dots against the ceiling and floor drawn as dashed lines per condition, with the skill score as a table column.
///

<!-- REVIEW: two open points for setting this rule. (1) The calibrated ceiling is not an upper bound on EEM: a control
that sharpens past calibration can score above it, up to the hard-EM ceiling. "Within a margin of the ceiling" should
say whether it is one-sided, and whether a control above the ceiling passes. (2) The standard deviation grows with
bimodality as well as with grading: on the three-example row it peaks at k3-r0.4, which has no more middle-band
contexts than the center. Among the three proposed conditions the two statistics agree, so the choice matters only if
the grid changes. -->


## The label rule (b)

/// admonition | TODO
The whole-line label stays the primary unless a variant clears the task gate by more than the seed band with the anchor held; variant (d) is reported and not promoted. Figure: task and margin per label arm.
///

## The hinge rule (c)

/// admonition | TODO
The capped pull goes forward if the uncapped arm saturates at the query `?`. The saturation level and the figure (alignment by role and slice) to be set.
///

## The verification rule (d)

/// admonition | TODO
Verification lines stay in the corpus if they move completion by less than the seed band on the control and the anchored arm.
///

## The operator rule (e)

/// admonition | TODO
The scoring-only suppression pass on the pilot's own checkpoints: projection, reflection, and repulsion, each at the query `?`, the query `=`, and every position; the operator whose damage grades with dose and stays within the selectivity gate on the other ops goes forward.
///

## Predictions

/// admonition | TODO
Any ungated predictions the pilot writes down before the run: the control's distance from the ceiling at {ex.MODEL}; alignment grading with the posterior on `{ex.ANCHORED_OP}` across the middle band; the first-operand and trailing-fragment leans under `{ex.CROP_POLICY}`; calibration of the answer distribution against the posterior-weighted one. Sandy to shape; one section each if any is gated.
///

## Exploratory analyses

/// admonition | TODO
Anything conceived after seeing the data, marked as post hoc.
///

## Discussion

/// admonition | TODO
What the pilot proposes to round 3, and what it leaves open.
///

## Method

### The posterior over ops

Each example is a pair and an answer, and under stochastic rounding the answer is a draw from up to eight colors. So the likelihood of a shown answer *y* under an op *o* is the probability that rounding the result of *o* on that pair gives *y*, written $P_o(y \mid a, b)$ (`answer_dist` in `sca.data.ops`).

The corpus rounds each channel independently, in proportion to where the raw value sits between grid levels. The posterior uses the same rounding. Nearest rounding would make it sharper than the corpus supports.
"""

r"""
Replacement noise enters the likelihood as a mixture. With rate ρ, each example shows a draw from another op's distribution instead, the other op uniform over the ten, so

$$
L_o(y \mid a, b) = (1 - \rho - \kappa)\, P_o(y \mid a, b) + \rho \cdot \frac{1}{10} \sum_{o' \neq o} P_{o'}(y \mid a, b) + \frac{\kappa}{216},
$$

where κ is the cube-noise rate. The posterior over the eleven ops is the product of the example likelihoods under a uniform prior. This is the posterior a model trained on the noisy corpus can reach at best. It also keeps every op above zero whenever ρ or κ is, which is what lets the designed null weight the other ops by how nearly they fit on a context that fits one op alone.
"""

rf"""
Of two possible readings of ρ, we take the one whose numbers match the pivot: per example, with the replacing op uniform over the other ten. A replacement is invisible when the replacing op agrees with the true op on the pair, so slightly fewer than a fraction ρ of examples mislead, and the posterior is a smooth weighting rather than a count of the ops that fit.

Under this reading, the shares of contexts in each band of the posterior reproduce the scratch simulation in the pivot: one clean example puts the posterior above {ex.MIDDLE_BAND[1]:g} on {N["clean1_hi"]:.0%} of contexts and three do so on {N["clean3_hi"]:.0%}; at three examples and ρ = 0.35, {N["r35"]["hi"]:.0%} are above the band, {N["r35"]["mid"]:.0%} inside it, and {N["r35"]["lo"]:.0%} below.

The figure covers the whole grid, {len(ex.K_GRID)} example counts by {len(ex.RHO_GRID)} replacement rates, with the true op uniform over the table.

{posterior_figure()}

Clean examples pin the op down fast, so the example count alone grades very little; replacement noise is what spreads the posterior across the range. The spread is widest, at a standard deviation of {N["sd_max"]:.2f}, for {N["sd_max_at"][0]} examples at ρ = {N["sd_max_at"][1]:g}; at the center condition it is {N["sd_centre"]:.2f}, with {N["centre"]["mid"]:.0%} of contexts in the middle band and {N["centre"]["lo"]:.0%} below it.

### The Bayes ceiling and the floor

A model that has learned everything the examples say answers the query with the answer distribution weighted by the posterior, $q(y) = \sum_o \pi_o P_o(y \mid c, d)$ for the query pair $(c, d)$. That is the distribution cross-entropy training converges to, and a calibrated model holds it.[^calibrated]

Our reports score expected exact match: the probability mass the model puts on the answers the true op can give, weighted by how often it gives them. So the ceiling is $\sum_y q(y)\, P_t(y \mid c, d)$ under the true op *t*, averaged over contexts.

Told the op, the same predictor scores $\sum_y P_t(y)^2$, which is {TOLD_OP:.3f} over the table. The shortfall from 1 comes from stochastic rounding, and it does not depend on the examples or the noise.

The floor (no evidence, uniform over ops) is the same at every grid point, {FLOOR:.3f} (the sampled values run from {N["floor_range"][0]:.3f} to {N["floor_range"][1]:.3f}).

[^calibrated]: A calibrated model is one whose stated probabilities match how often things actually happen: of the answers it gives 30% to, about 30% are right.

{ceiling_figure()}

With three clean examples the ceiling is {N["ceil_clean3"]:.3f}, within 0.02 of a model told the op, so almost all of the shortfall is rounding. Replacement noise adds a shortfall that comes from inference: at three examples and ρ = 0.35 the ceiling falls to {N["ceil_r35"]:.3f}, and at the center condition it is {N["ceil_centre"]:.3f}.

<details markdown="1"><summary>The grid as tables</summary>

{grid_table(ceiling, "The Bayes ceiling in expected exact match at each grid point.")}

{grid_table(spread, "The spread of the posterior on the true op across contexts (standard deviation) at each grid point.")}

</details>

**The numbers in the pivot are the hard-accuracy form.** The pivot quotes a ceiling of about 0.74 at three clean examples, 0.58 at ρ = 0.35, and 0.75 for a model told the op. Those are what the same predictor scores when it puts all its mass on the mode of $q$: {N["mode_clean3"]:.3f}, {N["mode_r35"]:.3f}, and {TOLD_OP_MODE:.3f} here.

That form is higher because expected exact match is linear in the model distribution, so the metric itself is maximized by a model that names one color. The calibrated predictor spreads its mass over every rounding of the answer, and scores about 0.05 less at three clean examples and 0.1 less at ρ = 0.35.

The ceiling of record is the calibrated one, since that is where training aims. A control scoring above it would have sharpened past calibration, and the calibration check in the measurements would show that. The tables keep the hard form as the maximum of the metric.

Per op, the ceiling is capped by how much the op rounds. `{ex.ANCHORED_OP}` is total on the grid, so its told-op ceiling is 1 and its ceiling at the center condition is {N["diff"]["ceiling"]:.3f}, with the posterior on it spread about as widely as on any op (sd {N["diff"]["sd"]:.2f}, {N["diff"]["mid"]:.0%} in the middle band). That is the graded stimulus the anchored arms are scored against.

{per_op_table()}

### Cube noise

The pivot allows cube noise at a low rate, so that the model learns to discount examples that fit nothing. At the center condition it costs {N["cube_cost"][ex.CUBE_GRID[1]]:.3f} of ceiling at κ = {ex.CUBE_GRID[1]:g}, {N["cube_cost"][ex.CUBE_GRID[2]]:.3f} at κ = {ex.CUBE_GRID[2]:g}, and {N["cube_cost"][ex.CUBE_GRID[3]]:.3f} at κ = {ex.CUBE_GRID[3]:g}.

A posterior that does not know about cube noise treats a cube color as replacement noise. It is a little sharper, and the metric rewards sharpness (as above), so it scores {N["cube_ignoring"][ex.CUBE_GRID[3]]:.3f} higher at the highest rate. So the ceiling barely depends on whether the model has learned about cube noise. Whether the corpus carries it is left open (`CUBE_RATE`).

{cube_figure()}

{cube_table()}

### The three grammar conditions

Rule (a) picks the widest spread among the conditions where the control nears its ceiling, so the three should differ in spread and in ceiling, and bracket the pivot's working point of three examples at ρ near 0.3. We propose `{COND_NAMES[0]}`, `{COND_NAMES[1]}`, and `{COND_NAMES[2]}`.

{conditions_table()}

`{COND_NAMES[1]}` is the center. `{COND_NAMES[0]}` steps ρ down on the same line length: it gives up spread ({spread(*ex.GRAMMAR_CONDITIONS[0]):.2f} against {spread(*ex.CENTRE):.2f}) for a higher ceiling, and is the fallback within the same block size if the control falls short at the center.

`{COND_NAMES[2]}` adds one example at the center ρ, which raises the ceiling by {ceiling(*ex.GRAMMAR_CONDITIONS[2]) - ceiling(*ex.CENTRE):.2f} and keeps most of the spread ({spread(*ex.GRAMMAR_CONDITIONS[2]):.2f}), though fewer contexts sit in the middle band ({band_shares(*ex.GRAMMAR_CONDITIONS[2])[1]:.0%} against {band_shares(*ex.CENTRE)[1]:.0%}). It asks whether more evidence buys a higher ceiling without flattening the stimulus, at {ex.context_tokens(4)} tokens per line rather than {ex.context_tokens(3)}.

Two alternatives were weighed. `{cond_name(*ALT_34)}` has the widest spread on the three-example row, but its middle-band share ({band_shares(*ALT_34)[1]:.0%}) is no larger than at the center and its ceiling is {ceiling(*ex.CENTRE) - ceiling(*ALT_34):.2f} lower, so it grades no better and leaves less room between floor and ceiling. `{cond_name(*ALT_435)}` comes closer to the spread at the center ({spread(*ALT_435):.2f}, with {band_shares(*ALT_435)[1]:.0%} in the middle band) at a ceiling of {ceiling(*ALT_435):.3f}; `{COND_NAMES[2]}` gives up a little of that spread for {ceiling(*ex.GRAMMAR_CONDITIONS[2]) - ceiling(*ALT_435):.2f} more ceiling.

<!-- REVIEW: the three conditions are a proposal from this scan, for Sandy to confirm or move. The spread statistic
proposed for rule (a) is the standard deviation of the posterior on the true op; the middle-band share is reported
beside it. Verify: the conditions table and the grid tables above.
REVIEW: the k4-r0.35 sentence said k4-r0.3 "beats it on ceiling at about the same spread"; the scan gives k4-r0.35 the
larger spread (0.32 against 0.29) and middle-band share (35% against 31%), so the choice is a trade, and the prose now
says so. The k4-r0.3 sentence also gains its middle-band share, which falls from the center more than its sd does. -->

### Reading a score against its ceiling

The ceiling differs between conditions, so a score is shown two ways. Figures keep raw expected exact match on the y-axis and draw the ceiling and floor of each condition as dashed lines beside its seed marks, so the reader sees without arithmetic how much of the available room a model takes.

A table that compares across conditions adds the skill score as a column (0 is a model that ignores the examples, 1 is the Bayes predictor). The figure for the control also draws the ceiling for a model told the op ({TOLD_OP:.3f}) beside the Bayes ceiling, so that the part of the gap due to inference can be seen.

<!-- REVIEW: this convention is provisional (Sandy is deciding); raw EEM on the axis with dashed bounds per condition,
the skill score as a table column or summary panel and never the main axis. -->

### The corpus and the training

/// admonition | TODO
The data spec: one context per line, ops uniform, pairs uniform over ordered pairs, the query clean, the block size (a context of *k* examples is 6*k* + 6 tokens, and the block should fit at least two whole contexts), the holdouts, the labeller keyed per context, and the recipe inherited from ex-2.2.14's primary under crop policy `{ex.CROP_POLICY}`. The generator's noise model must match `posterior.py`: a per-example replacement rate with the replacing op uniform over the other ten, and cube noise uniform over the grid, or the ceiling here is not the ceiling of the corpus.
///

### The measurements

/// admonition | TODO
Held-out expected exact match per condition against its ceiling and floor; the calibration check (the model's answer distribution against the posterior-weighted one); alignment by role and slice with the query `?` and the query `=` measured separately; alignment against the posterior on `{ex.ANCHORED_OP}`; the first-operand and trailing-fragment leans from ex-2.2.15; the op margin; and the scoring-only suppression pass.
///

### Budget

/// admonition | TODO
Runs, seeds, and the Modal estimate, once the arm list is settled.
///
"""
