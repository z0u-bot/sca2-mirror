# title: Ex 2.2.21: the in-context grammar pilot, on the reworked recipe

# The design constants come from `experiment.py` beside this script (the directory of the script is on sys.path
# while it runs). Nothing here has been trained yet: the preview section scores the eval ex-2.2.16 published for its
# own anchored arms, and the method section computes the posterior from the seven-op table alone.
import json
import tempfile
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import AxesRow, figure_html, light_dark, themed

P = ex.ex2216._POSTERIOR_MODULE
D16 = ex.ex2216.OP_NAMES.index(ex.ANCHORED_OP)

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


def rule_color() -> str:
    return light_dark("#333", "#ddd")


def fetch_json(ref_name: str) -> dict:
    store = project_store()
    ref = store.get_refs([ref_name])[ref_name]
    assert ref is not None, f"{ref_name} is not published"
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(store.get_many([(ref, Path(tmp) / "x.json")])[0].read_text())


@memo
def stored_evals() -> tuple[dict, dict, dict]:
    """Ex-2.2.16's eval and suppression pass, and ex-2.2.19's eval, as each published them."""
    return (
        fetch_json(ex.ex2216.EVAL_REF),
        fetch_json(ex.ex2216.SUPPRESSION_REF),
        fetch_json(ex.ex2219.EVAL_REF),
    )


EVAL16, SUPP16, EVAL19 = stored_evals()

# --- The recipe the control has to reproduce ------------------------------------------------------------------

REF19 = [r for r in EVAL19["runs"] if r["label"].startswith(ex.REGRESSION_REF + "-")]
REF19_EEM = [r["task"]["eem"]["all"] for r in REF19]
REF19_CEIL = REF19[0]["task"]["ceiling"]["all"]
REF19_FLOOR = REF19[0]["task"]["floor"]["all"]
REF19_KL = float(np.mean([r["task"]["kl"]["all"] for r in REF19]))
REF19_MEAN = float(np.mean(REF19_EEM))
REF19_SD = float(np.std(REF19_EEM, ddof=1))
REF19_SKILL = (REF19_MEAN - REF19_FLOOR) / (REF19_CEIL - REF19_FLOOR)


def seed_band(n1: int, n2: int, sd: float = REF19_SD) -> float:
    return ex.SEED_BAND_SD * sd * float(np.sqrt(1 / n1 + 1 / n2))


BAND_55 = seed_band(5, 5)
BAND_53 = seed_band(5, 3)

# --- The preview: ex-2.2.16's anchored arms ------------------------------------------------------------------

PREVIEW_ARMS = ("control-k3-r0.3", "anchor-whole", "anchor-hinge")
ROLE_LABELS = [
    *(t for i in "₁₂₃" for t in (f"a{i}", "?", f"b{i}", "=", f"y{i}", ",")),
    "a",
    "?",
    "b",
    "=",
    "y",
    "⏎",
]
Q_EQ = EVAL16["runs"][0]["roles"]["query ="]
Q_Q = EVAL16["runs"][0]["roles"]["query ?"]
Q_ANS = Q_EQ + 1
EX_ANS = [i for i, t in enumerate(ROLE_LABELS[:18]) if t.startswith("y")]


def runs16(arm: str) -> list[dict]:
    return [r for r in EVAL16["runs"] if r["arm"] == arm]


def alignment16(arm: str) -> np.ndarray:
    """Seed-mean alignment, ops × slices × positions, for the k = 3 contexts of one arm."""
    return np.mean([r["alignment"] for r in runs16(arm)], axis=0)


ALIGN16 = {a: alignment16(a) for a in PREVIEW_ARMS}


def last_block(arm: str, anchored: bool) -> np.ndarray:
    a = ALIGN16[arm][:, -1, :]
    return a[D16] if anchored else np.delete(a, D16, axis=0).mean(axis=0)


MARGIN16 = {
    a: float(np.mean([r["margin"]["value"] for r in runs16(a)]))
    for a in ("control-k3-r0.3", "anchor-whole", "anchor-hinge")
}
EEM16 = {
    a: float(np.mean([r["task"]["eem"]["all"] for r in runs16(a)]))
    for a in ("control-k3-r0.3", "anchor-whole", "control-mask", "anchor-mask")
}
CTRL16_EEM = float(np.mean([r["task"]["eem"]["all"] for r in runs16("control-k3-r0.3")]))
CTRL16_SKILL = float(
    np.mean(
        [
            (r["task"]["eem"]["all"] - r["task"]["floor"]["all"])
            / (r["task"]["ceiling"]["all"] - r["task"]["floor"]["all"])
            for r in runs16("control-k3-r0.3")
        ]
    )
)


def supp_mean(arm: str) -> dict:
    """Seed means of the suppression pass for one arm: clean and target-null scores per op, and each edit."""
    rs = [r for r in SUPP16["runs"] if r["arm"] == arm]
    keys = [(e["operator"], e["dose"], e["site"]) for e in rs[0]["edits"]]
    return {
        "clean": np.mean([r["clean"]["eem"] for r in rs], axis=0),
        "null": np.mean([r["null"]["eem"] for r in rs], axis=0),
        "edits": {k: np.mean([r["edits"][i]["eem"] for r in rs], axis=0) for i, k in enumerate(keys)},
    }


SUPP = {a: supp_mean(a) for a in PREVIEW_ARMS}


def net_drop(arm: str, key: tuple) -> np.ndarray:
    """Per op, the drop in expected exact match under an edit, less the drop the control shows under the same edit."""
    s, c = SUPP[arm], SUPP["control-k3-r0.3"]
    return (s["clean"] - s["edits"][key]) - (c["clean"] - c["edits"][key])


def to_null(arm: str, key: tuple) -> float:
    """The net drop on `difference` as a share of the way from the clean score to the target null."""
    s = SUPP[arm]
    return float(net_drop(arm, key)[D16] / (s["clean"][D16] - s["null"][D16]))


def worst_other(arm: str, key: tuple) -> float:
    return float(np.delete(net_drop(arm, key), D16).max())


def preview_figure() -> str:
    data = {a: {"anchored": last_block(a, True).tolist(), "other": last_block(a, False).tolist()} for a in PREVIEW_ARMS}
    w = data["anchor-whole"]["anchored"]
    h = data["anchor-hinge"]["anchored"]
    alt = f"""
        Three line charts side by side, one per arm of ex-2.2.16 (the control, the whole-line arm, and the hinge arm),
        each plotting the alignment with the anchored axis at the last block against the 24 positions of a
        three-example context, with one line for `difference` contexts and one for the mean of the other ops. The
        control is flat near zero. On the whole-line arm the `difference` line peaks at the answer positions: about
        {min(w[i] for i in EX_ANS):.2f} to {max(w[i] for i in EX_ANS):.2f} at the three example answers and
        {w[Q_ANS]:.2f} at the query answer, with {w[Q_EQ]:.2f} at the query `=`. The hinge arm moves part of it to the
        query `=` ({h[Q_EQ]:.2f}).
    """
    return preview_draw(data, alt)


@memo
def preview_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="preview-alignment",
        alt_text=alt_text,
        caption=f"""
            **Where ex-2.2.16 put the anchor.** Alignment with e₁ at the last block, by position in a context of
            three examples, seed means over three runs on held-out contexts. Solid: `{ex.ANCHORED_OP}` contexts.
            Dashed: the mean over the other ten ops. The shaded column is the query `=`, where the answer is
            computed. The labels are the roles: a, b, and y for the operands and the answer, numbered by example and
            bare for the query.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), layout="constrained", sharey=True)
        axes = cast(AxesRow, axes)
        ink = rule_color()
        accent = light_dark("#c0392b", "#ff8a76")
        x = np.arange(len(ROLE_LABELS))
        titles = {"control-k3-r0.3": "control", "anchor-whole": "whole-line", "anchor-hinge": "hinge"}
        for ax, (arm, d) in zip(axes, data.items(), strict=True):
            ax.axvspan(Q_EQ - 0.5, Q_EQ + 0.5, facecolor=light_dark("#000", "#fff"), alpha=0.08, lw=0)
            ax.plot(x, d["anchored"], "-o", color=accent, lw=1.2, ms=2.5, label=ex.ANCHORED_OP)
            ax.plot(x, d["other"], "--", color=ink, lw=1.0, label="other ops")
            ax.set_title(titles[arm], fontsize=9)
            ax.set_xticks(x, ROLE_LABELS, fontsize=6)
            ax.set_ylim(-0.1, 1.0)
        axes[0].set_ylabel("alignment, last block")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False, fontsize=7)
        return fig

    return _plot()


PREVIEW_EDITS = (
    ("projection", 1.0, "query ="),
    ("projection", 1.0, "every position"),
    ("projection", 0.5, "every position"),
    ("repulsion", 0.0, "every position"),
)


def preview_table() -> str:
    head = ["arm", "edit", "site", "drop on `difference`", "share of the way to the null", "worst other op"]
    rows = []
    for arm in ("anchor-whole", "anchor-hinge"):
        for op, dose, site in PREVIEW_EDITS:
            k = (op, dose, site)
            dose_txt = f"γ = {dose:g}" if op == "projection" else f"landing {dose:g}"
            rows.append(
                [
                    f"`{arm}`",
                    f"{op}, {dose_txt}",
                    site,
                    f"{net_drop(arm, k)[D16]:.3f}",
                    f"{to_null(arm, k):.0%}",
                    f"{worst_other(arm, k):.3f}",
                ]
            )
    return table_html(
        head,
        rows,
        f"Ex-2.2.16's suppression pass on its two scored anchored arms, a few of its edits. Every number is a seed "
        f"mean of the drop in held-out expected exact match, less the drop the control shows under the same edit. "
        f"The share is of the way from the clean score on `{ex.ANCHORED_OP}` to the target null; the operator rule "
        f"asked for {ex.GRADING_MIN_DAMAGE:.0%} at full dose and at most {ex.SELECTIVITY_GATE:g} on every other op.",
        text_cols=3,
    )


PV = {
    "whole_full": ("projection", 1.0, "every position"),
    "whole_eq": ("projection", 1.0, "query ="),
}

# --- The posterior on the seven-op table -------------------------------------------------------------------------


@memo
def tables() -> tuple:
    return P.build_table(ex.ex2216.TABLE), P.build_table(ex.ex2218.table_of(ex.OP_NAMES))


TABLE11, TABLE7 = tables()


@memo
def posterior_on_anchored(table, names: tuple, k: int, rho: float, kappa: float, n: int, seed: int) -> dict:
    """At one condition: the posterior on the anchored op over contexts of that op, and the ceiling and floor over
    contexts of every op.
    """
    d = list(names).index(ex.ANCHORED_OP)
    rng = np.random.default_rng([seed, k, round(rho * 100), 0])
    ctx = P.sample_contexts(table, n, k, rho, kappa, rng, true_op=d)
    post = P.posterior(table, ctx, rho, kappa)
    rng = np.random.default_rng([seed, k, round(rho * 100), 1])
    allc = P.sample_contexts(table, n, k, rho, kappa, rng)
    return {
        "post": post[:, d].astype(np.float32),
        "ceiling": float(P.ceiling(table, allc, rho, kappa).mean()),
        "floor": float(P.floor(table, allc).mean()),
    }


GRID_K = (2, 3, 4)
GRID_RHO = (0.2, 0.3, 0.4)
N_POST = 20_000
POST_SEED = 2221

POST7 = {
    (k, r): posterior_on_anchored(TABLE7, ex.OP_NAMES, k, r, ex.CUBE_RATE, N_POST, POST_SEED)
    for k in GRID_K
    for r in GRID_RHO
}
POST11 = posterior_on_anchored(TABLE11, ex.ex2216.OP_NAMES, *ex.CENTRE, ex.CUBE_RATE, N_POST, POST_SEED)


def band_shares(post: np.ndarray) -> tuple[float, float, float]:
    lo, hi = ex.MIDDLE_BAND
    return float((post < lo).mean()), float(((post >= lo) & (post <= hi)).mean()), float((post > hi).mean())


C7 = POST7[ex.CENTRE]
BANDS7 = band_shares(C7["post"])
BANDS11 = band_shares(POST11["post"])


def posterior_figure() -> str:
    alt = f"""
        A cumulative distribution of the posterior on `{ex.ANCHORED_OP}` over `{ex.ANCHORED_OP}` contexts at three
        examples and ρ = {ex.CENTRE[1]:g}, one curve for the eleven-op table and one for the seven-op table. Both
        rise across the middle band from {ex.MIDDLE_BAND[0]:g} to {ex.MIDDLE_BAND[1]:g}; on the seven-op table
        {BANDS7[1]:.0%} of contexts sit inside it, against {BANDS11[1]:.0%} on eleven ops.
    """
    return posterior_draw(C7["post"], POST11["post"], alt)


@memo
def posterior_draw(post7: np.ndarray, post11: np.ndarray, alt_text: str) -> str:
    @themed(
        name="posterior-anchored",
        alt_text=alt_text,
        caption=f"""
            **The stimulus on the seven-op table.** The share of `{ex.ANCHORED_OP}` contexts whose posterior on
            `{ex.ANCHORED_OP}` is at most the value on the x-axis, at three examples, ρ = {ex.CENTRE[1]:g}, and
            cube noise at κ = {ex.CUBE_RATE:g}. The shaded band is the middle band. {N_POST:,} sampled contexts per
            curve.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.0, 2.8), layout="constrained")
        ink = rule_color()
        accent = light_dark("#2b6cb0", "#7fb3ff")
        lo, hi = ex.MIDDLE_BAND
        ax.axvspan(lo, hi, facecolor=light_dark("#000", "#fff"), alpha=0.06, lw=0)
        x = np.linspace(0, 1, 401)
        for post, color, label in ((post11, ink, "eleven ops"), (post7, accent, "seven ops")):
            s = np.sort(post)
            ax.plot(x, np.searchsorted(s, x, side="right") / len(s), color=color, lw=1.3, label=label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"posterior on {ex.ANCHORED_OP}")
        ax.set_ylabel("share of contexts")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        return fig

    return _plot()


def grid_table() -> str:
    head = ["examples", *(f"ρ = {r:g}" for r in GRID_RHO)]
    rows = [
        [str(k), *(f"{POST7[(k, r)]['ceiling']:.3f} · {band_shares(POST7[(k, r)]['post'])[1]:.0%}" for r in GRID_RHO)]
        for k in GRID_K
    ]
    return table_html(
        head,
        rows,
        f"The seven-op table near the center condition, at κ = {ex.CUBE_RATE:g}: the Bayes ceiling in expected exact "
        f"match over contexts of every op, then the share of `{ex.ANCHORED_OP}` contexts in the middle band of the "
        f"posterior on `{ex.ANCHORED_OP}`.",
    )


# --- The design tables --------------------------------------------------------------------------------------


def arms_table() -> str:
    head = ["arm", "group", "anchored", "label", "corpus", "what it asks", "seeds"]
    rows = [
        [
            f"`{a.name}`",
            a.group,
            "yes" if a.anchored else "—",
            (f"`{a.label}`" if a.anchored else "—") + (" + hinge" if a.hinge else ""),
            "verification" if a.verify else "—",
            a.note,
            str(a.seeds),
        ]
        for a in ex.ARMS
    ]
    return table_html(
        head,
        rows,
        f"The {len(ex.ARMS)} arms, {ex.N_RUNS} runs. Every arm trains on the seven-op set at three examples and "
        f"ρ = {ex.CENTRE[1]:g}, with the newline mask, for {ex.EPOCHS} epochs; every anchored arm anchors "
        f"`{ex.ANCHORED_OP}` on e₁. `{ex.CONTROL}` and `{ex.PRIMARY}` are the references.",
        ref_rows=frozenset({i for i, a in enumerate(ex.ARMS) if a.name in (ex.CONTROL, ex.PRIMARY)}),
        text_cols=6,
    )


rf"""
# Ex 2.2.21: the in-context grammar pilot, on the reworked recipe

/// tip |
<!-- tl;dr -->
A second try at round 2 of the D2.2 route, on the recipe ex-2.2.17 to ex-2.2.20 reworked. We retrain the control and the anchored `{ex.ANCHORED_OP}` arms under the whole-line label and its variants, then score the rules ex-2.2.16 never reached, to decide how round 3 anchors and edits the inferred op.
///

## Findings

- [The recipe holds, and the anchor costs nothing (H1)](#the-recipe-holds-and-the-anchor-costs-nothing-h1) —
- [The label rule (S1)](#the-label-rule-s1) —
- [Where the anchor sits (E1)](#where-the-anchor-sits-e1) —
- [The site and operator rule (S2)](#the-site-and-operator-rule-s2) —
- [The verification rule (S3)](#the-verification-rule-s3) —
- [The anchor follows the evidence (H2)](#the-anchor-follows-the-evidence-h2) —
- [The leans stay with the control (H3)](#the-leans-stay-with-the-control-h3) —

## How to read this draft

This is a preregistration draft: nothing in it has been trained. The rules and predictions will be frozen at a commit quoted here once the plan is agreed, and results will replace the `TODO` boxes in place. The [preview](#what-ex-2216s-anchored-arms-already-show) scores ex-2.2.16's stored runs; it predates this plan and shaped it.

## Why this experiment

[Ex-2.2.16](/docs/m2/ex-2.2.16/report.py) was round 2 of the [D2.2 route](/docs/m2/d2.2/design.md#quick-route): train the control on the in-context grammar, anchor `{ex.ANCHORED_OP}` on the same grammar, and let rules frozen in advance choose the label, the site, the operator, and whether verification lines stay in the corpus for round 3. It stopped at its first rule. The control learned less than half of what its examples allow ({CTRL16_SKILL:.0%} of the way from the floor to the Bayes ceiling), and an anchor compared with a model that far from finished would say little about round 3.

Four reports then worked on the control alone: [ex-2.2.17](/docs/m2/ex-2.2.17/report.py) added the newline mask, a lower peak learning rate, and more steps; [ex-2.2.18](/docs/m2/ex-2.2.18/report.py) dropped four ops whose answers often coincide with another op; [ex-2.2.19](/docs/m2/ex-2.2.19/report.py) settled the length at {ex.EPOCHS} epochs; and [ex-2.2.20](/docs/m2/ex-2.2.20/report.py) kept the plain schedule.

On the seven-op set the control now scores {REF19_MEAN:.3f} against a ceiling of {REF19_CEIL:.3f}, {REF19_SKILL:.0%} of the way from the floor. This pilot reruns round 2 on that recipe. It keeps ex-2.2.16's label, verification, and operator rules, with one change each where ex-2.2.16's stored runs or the new recipe ask for it, and adds two label variants. The [design](/docs/m2/d2.2/design.md#the-pilot) names the rules (a) to (e); here they are H1 and S1 to S3.

## What ex-2.2.16's anchored arms already show

Ex-2.2.16's anchored arms were trained and evaluated, but the eval was never scored. These numbers come from the half-trained model, so they show what to look for and decide nothing.

The anchor landed. The op margin, which measures how far `{ex.ANCHORED_OP}` contexts sit along the anchored axis beyond the rest, was {MARGIN16["anchor-whole"]:.2f} on the whole-line arm against {MARGIN16["control-k3-r0.3"]:.2f} on the control. But it landed in a place round 3 cannot use as it is:

{preview_figure()}

Most of the alignment sits on the answers: on each example answer, and most of all on the query answer, which reaches {last_block("anchor-whole", True)[Q_ANS]:.2f} at the last block. The query `=`, where the model computes the answer, holds {last_block("anchor-whole", True)[Q_EQ]:.2f}. The pooled anchor term asks a labelled context to align somewhere in its span; the answers are where the context has shown the most evidence about its op, so the pull seems to have settled there.

The query `?` stayed clean ({last_block("anchor-whole", True)[Q_Q]:.2f}). Ex-2.2.16 worried that it would saturate, but the position that came close was the query answer. The hinge arm, which caps the pull, moved part of the alignment to the query `=` ({last_block("anchor-hinge", True)[Q_EQ]:.2f}).

The suppression pass agrees. An edit at the query `=` barely touches the answer. The projection at every position grades with dose, but it stops short of the target null and spills onto other ops:

{preview_table()}

The preview changes three things in this plan. Two new label variants pull the positions up to the query `=` and the query `=` alone, giving the site rule an arm anchored where the answer is computed. The saturation half of ex-2.2.16's hinge rule goes, since the query `?` did not saturate, though the hinge arm stays a candidate site. The operator rule is scored at the query `=` as well as at every position.

## Glossary

Terms follow [ex-2.2.16](/docs/m2/ex-2.2.16/report.py#glossary); these are the ones the rules here use.

<dl>
<dt>Alignment</dt>
<dd>The cosine between a state and e₁, the anchored axis, per position and slice. (Cosine similarity compares direction only: 1 when two vectors point the same way, 0 when they are perpendicular.)</dd>
<dt>Op margin</dt>
<dd>How far the contexts of the anchored op sit along e₁ beyond the rest. At each slice, take the mean alignment over `{ex.ANCHORED_OP}` contexts less the mean over all contexts, at the role where that gap is largest; then average over slices. The anchor term optimizes this quantity, so it checks that the pull landed.</dd>
<dt>Expected exact match (EEM)</dt>
<dd>The probability mass the model puts on the colors the query answer can be under the true op, weighted by how often the op gives each, averaged over held-out contexts. The task measurement.</dd>
<dt>Bayes ceiling and floor</dt>
<dd>The expected exact match of a predictor that holds the posterior over ops given the examples, and of one that ignores the examples.</dd>
<dt>Target null</dt>
<dd>The answer a model that has lost `{ex.ANCHORED_OP}` and nothing else would give: the posterior-weighted answer distribution with `{ex.ANCHORED_OP}` removed and the other ops renormalized.</dd>
<dt>Seed band</dt>
<dd>The smallest difference between two seed means a comparison resolves: {ex.SEED_BAND_SD:g}σ√(1/n₁ + 1/n₂) for arms at n₁ and n₂ seeds, with σ the seed standard deviation pooled over the two. At the spread of ex-2.2.19's four runs it is about {BAND_55:.3f} for five seeds against five and {BAND_53:.3f} for five against three.</dd>
</dl>

## Conditions

Every arm trains at the center condition of ex-2.2.16 (three examples, ρ = {ex.CENTRE[1]:g}) on the seven-op set, under ex-2.2.19's recipe: the newline mask, a peak rate of {ex.PEAK_LR:g}, {ex.EPOCHS} epochs, {ex.MODEL}. The anchored arms add ex-2.2.14's anchor recipe, each weight scheduled as a fraction of training as before, and crop policy `{ex.CROP_POLICY}`.

{arms_table()}

Ex-2.2.16's arms at other corpus conditions, its larger control, and its two newline-mask arms are gone: the condition is settled, the larger control was for a shortfall the recipe has since closed, and the mask is now in every arm.

**The new label variants.** Both keep ex-2.2.16's labeller: a `{ex.ANCHORED_OP}` context draws a label with probability {ex.LABEL_RATE:g}, keyed on the op array beside the corpus. Variant (e), `prompt`, pulls every position of a labelled context up to and including the query `=`, leaving out the query answer and the line break; an M3 labeller that marks the prompt and leaves the response alone would give this label. Variant (f), `query-eq`, pulls the query `=` alone: a slot pull at the position round 3 edits.

Neither variant leaves out the example answers, because they are part of the evidence. If the pull still settles on them under `prompt`, that is a result about pooling.

**The seeds.** Fresh model seeds from {ex.SEED_OFFSET}. The control and the whole-line arm get {ex.SEEDS_REFERENCE} each, since every comparison runs through one of them; the others get three.

## The recipe holds, and the anchor costs nothing (H1)

**What we expect.** Round 3 starts from this recipe only if both of these criteria hold:

- **(a)** The control reproduces ex-2.2.19: its seed-mean held-out expected exact match is within {ex.REGRESSION_TOL:g} of {REF19_MEAN:.3f}, the mean of ex-2.2.19's four runs at {ex.EPOCHS} epochs.
- **(b)** The whole-line arm is not worse than the control by more than the seed band.

The arms differ from ex-2.2.19 only in their seeds and in the anchor, so (a) checks that nothing else moved.

We expect (b) to hold too. Ex-2.2.14 found that the anchor costs the task nothing when {ex.LABEL_RATE:.0%} of contexts are labelled, and the stored runs of ex-2.2.16 agree: the whole-line arm scored {EEM16["anchor-whole"]:.3f} against {EEM16["control-k3-r0.3"]:.3f} for the control, and {EEM16["anchor-mask"]:.3f} against {EEM16["control-mask"]:.3f} with the newline mask. A miss on (b) would mean the anchor and the task compete on this recipe, and would leave the site rule with no qualifying arm, since it asks the same of each candidate.

This replaces corpus rule (a) of ex-2.2.16, which asked the control to come within 0.03 of the calibrated ceiling. The seven-op control misses that: it sits {REF19_CEIL - REF19_MEAN:.3f} below the ceiling at {ex.EPOCHS} epochs, and still misses at 400. Ex-2.2.17 found the remaining gap on contexts whose examples settle the op, where the model keeps mass on the answers of a similar op.

We take the control as it stands. Its skill score and calibration KL[^kl] are reported beside every comparison, but no criterion depends on them. In ex-2.2.19 the KL was about {REF19_KL:.2f} nats; ex-2.2.16 called a model calibrated below 0.05.

[^kl]: A KL divergence: a non-negative measure, in nats, of how far one probability distribution sits from another, 0 when they match. Here it compares the model's answer distribution with the Bayes predictor's.
<!-- REVIEW: (b) uses the seed band, about 0.009 at ex-2.2.19's spread, so it would also catch a real but small cost we might accept. A fixed margin (0.015, as in (a)) is the alternative; a miss on (b) leaves S2 with no candidates. Verify: the old-recipe gaps above, inside the seed spread of that recipe. -->
<!-- REVIEW: dropping the absolute ceiling margin is a decision for Sandy. The alternative is to keep 0.03 and treat the pilot as blocked on the control, which ex-2.2.17 to ex-2.2.20 suggest no cheap change to the recipe will clear. Verify: ex-2.2.19's E4 and ex-2.2.17's answer scoring. -->

/// admonition | TODO
A figure of held-out expected exact match per seed for the control and the whole-line arm, seed means as bars, with ex-2.2.19's four runs as a reference column; the ceiling and floor as rules and the band of (a) shaded. Beside it, the calibration KL for the same runs. A table with the seed means, the seed band, and the verdict on each criterion.
///

## The label rule (S1)

**The rule.** Ex-2.2.16's rule (b), unchanged. The whole-line label stays the primary unless a variant clears the task gate: its seed-mean held-out expected exact match exceeds that of the whole-line arm by more than the seed band, while it holds the anchor, with an op margin of at least {ex.MARGIN_KEEP:.0%} of the whole-line margin. If more than one variant qualifies, the one with the higher op margin goes forward.

The candidates are `no-emb`, `latter`, and `prefix`. Variant (d), `sampled`, trains the grading that [H2](#the-anchor-follows-the-evidence-h2) measures, so it is reported and not promoted. The two new variants go to the site rule.

**What we expect.** No variant clears the gate, for the reasons ex-2.2.16 gave: the label share barely moved the margin in ex-2.2.14, and the variants change less than that.

/// admonition | TODO
A figure with one column per label arm: held-out expected exact match (seeds and seed mean, the whole-line seed band shaded) above, the op margin as a share of the whole-line margin below, with the {ex.MARGIN_KEEP:.0%} line. A table of the same numbers and the verdict.
///

## Where the anchor sits (E1)

The site rule depends on where along the context each anchored arm puts its alignment. E1 measures, for every anchored arm, the seed-mean alignment at each position of a held-out context and each slice, on `{ex.ANCHORED_OP}` contexts and on the other ops. The preview figure is the last-block row of this for ex-2.2.16.

/// admonition | TODO
One heatmap per anchored arm, positions across and slices down, of the alignment on `{ex.ANCHORED_OP}` contexts less the alignment on the other ops. Beside them, the last-block profile of every arm on one set of axes, as in the preview figure. A table of the alignment at the query `?`, the query `=`, the query answer, and the example answers, at the last block.
///

## The site and operator rule (S2)

**The rule.** It picks the arm to anchor with in round 3 and the edit to suppress with, merging hinge rule (c) and operator rule (e) of ex-2.2.16.

The suppression pass of ex-2.2.16 (scoring only, unchanged) runs on the four candidate arms (`{"`, `".join(ex.SITE_ARMS)}`) and on the control. It applies three edits: the projection at γ in {{{", ".join(f"{g:g}" for g in ex.DOSE_GAMMAS)}}}, the repulsion to a landing at {", ".join(f"{b:g}" for b in ex.REPULSION_LANDINGS)} for states above an alignment of {ex.REPULSION_THRESHOLD:g}, and the reflection as a one-dose reference. Each edit is applied at every slice and at three sites: the query `?`, the query `=`, and every position.

Every drop below is a net drop: the seed-mean fall in held-out expected exact match, minus the fall the control shows under the same edit. An operator at a site *qualifies* on an arm when:

- it grades with dose: the net drop on `{ex.ANCHORED_OP}` contexts never shrinks as the dose rises, and at full dose it covers at least {ex.GRADING_MIN_DAMAGE:.0%} of the distance from the clean score to the target null;
- it is selective: on each of the other six ops, the net drop is at most {ex.SELECTIVITY_GATE:g} at every dose.

The rule is scored at the query `=` and at every position. The query `?` gates nothing; with the query `=`, it gives the first bypass measurement.

An arm is a *candidate* when it passes H1 (b) against the control. Candidates are taken in the order listed, and the first on which some operator qualifies goes forward, with that operator and site. The reflection has one dose, so it cannot grade; it is a reference only.

If several operators qualify on the same arm, the tie breaks by site first, then by operator. The query `=` beats every position, since round 3 edits the narrower site first; at the same site, the projection beats the repulsion.
<!-- REVIEW: made the tie-break order explicit (site first, then operator) and stated that the reflection cannot qualify, as ex-2.2.16's REFLECT_GAMMA docstring has it. The earlier wording left a projection at every position against a repulsion at the query `=` undecided. Verify: if operator should take precedence, swap the order. --> If nothing qualifies on any candidate, round 3's operator stays open, and the report says which criterion failed where.
<!-- REVIEW: "does not fall along the dose axis" is ex-2.2.16's non-decreasing criterion, which a seed-mean dip of a thousandth at a small dose would fail. A slack of 0.005 per step is an option. Verify against the preview: the whole-line projection at every position rises by 0.04 to 0.06 per step. -->

The order favors the labels an M3 labeller could give: the whole context, then the prompt. The hinge and the slot pull need the cap or the site chosen by hand, which M3 would have to justify.

**What we expect.** On the whole-line arm nothing qualifies, as in the preview. On `query-eq` and `prompt` the projection at the query `=` grades and is selective, and `prompt` goes forward. We are least sure of `prompt`: the pooled term may still put the alignment on the example answers, which E1 will show.

/// admonition | TODO
One figure per candidate arm: the net drop on `{ex.ANCHORED_OP}` against dose, one line per operator and site, with the target null as a rule; beneath it, the worst net drop over the other ops against dose, with the selectivity gate as a rule. A table of each operator and site on each arm against the two criteria, and the verdict.
///

## The verification rule (S3)

**The rule.** Ex-2.2.16's rule (d), unchanged. Verification lines stay in the corpus from round 3 on if they leave completion unchanged: the seed-mean held-out expected exact match on completion contexts is within the seed band of the arm without them, on both pairs (`control-verify` against `{ex.CONTROL}`, `anchor-verify` against `{ex.PRIMARY}`). The verification accuracy and the op margin of `anchor-verify` are reported with no gate.

**What we expect.** Completion is unchanged on both pairs. The verification arms see {1 - ex.VERIFY_RATE:.0%} of the completion lines the others do. On the old recipe that cost the control nothing measurable; at {ex.EPOCHS} epochs each line is seen more often, so the cost should be smaller still.

/// admonition | TODO
A figure of held-out expected exact match per seed for the two pairs, with the seed band of each shaded, and verification accuracy beside it. A table of the same and the verdict.
///

## The anchor follows the evidence (H2)

**What we expect.** We score the arm the site rule sends forward, or the whole-line arm if it sends none, on the alignment at the query `=` at the last block of held-out `{ex.ANCHORED_OP}` contexts.

We expect it to rise with the posterior on `{ex.ANCHORED_OP}` across the middle band ({ex.MIDDLE_BAND[0]:g} to {ex.MIDDLE_BAND[1]:g}): in the bins with edges at {", ".join(f"{b:g}" for b in ex.MIDDLE_BINS)}, the seed-mean alignment in each bin is higher than in the bin below. On contexts of the other ops it should stay low and flat.

The label is binary on the true op, so a context whose examples half-fit `{ex.ANCHORED_OP}` is pulled as hard as one that names it. If the alignment grades anyway, the anchor holds the inferred op rather than the op label.

The same measurement is reported at the query answer, and on every anchored arm beside the one scored. In the preview, the whole-line arm showed little grading at the query `=`, where it held little alignment to grade.

/// admonition | TODO
A figure of alignment at the query `=` against the posterior on `{ex.ANCHORED_OP}`, binned, for the scored arm (seed means, seeds faded) with the other ops' contexts beside it, and `sampled` as the trained comparison. A second panel at the query answer.
///

## The leans stay with the control (H3)

**What we expect.** On the whole-line arm, the first-operand lean and the trailing-fragment lean stay within the seed band of the control, as ex-2.2.16 defined them. The newline mask now keeps a context from reading the one before it, which should make a cut-off fragment look less like a whole context than it did in ex-2.2.15.

/// admonition | TODO
A figure of both leans per seed for the control and every anchored arm, with the seed band of the control shaded.
///

## Discussion

/// admonition | TODO
After the run. The outcomes that change round 3: which arm and edit go forward (S2), and if none, whether the anchor can be moved to the query `=` without choosing the site by hand.
///

## Method

### The corpus and the posterior

The corpus is ex-2.2.18's seven-op corpus at the center condition, which ex-2.2.19 also trained on: one context per line, 300,000 contexts, three examples and ρ = {ex.CENTRE[1]:g}, cube noise at κ = {ex.CUBE_RATE:g}, and stochastic rounding. The held-out and probe sets are ex-2.2.18's. The verification arms train on the same generator with {ex.VERIFY_RATE:.0%} of contexts written as verification lines.

Dropping four ops changes the posterior, but the center condition still grades the stimulus: on the seven-op table {BANDS7[1]:.0%} of `{ex.ANCHORED_OP}` contexts sit in the middle band, against {BANDS11[1]:.0%} on eleven ops, with {BANDS7[0]:.0%} below it and {BANDS7[2]:.0%} above. On these sampled contexts the Bayes ceiling is {C7["ceiling"]:.3f} and the floor {C7["floor"]:.3f}; on the held-out set of ex-2.2.19, the ceiling is {REF19_CEIL:.3f}.

{posterior_figure()}

<details markdown="1"><summary>The neighboring conditions</summary>

{grid_table()}

</details>

### The measurements

Ex-2.2.16's measurements, scored on its held-out sets at the end of training, with the mask on in every arm: expected exact match against the ceiling and floor, the calibration KL, the alignment by position and slice, the alignment against the posterior, the op margin, the two leans, the suppression pass, and verification accuracy. The op margin and the leans are also recorded through training on a fixed probe set. Comparisons are between seed means, with the seed band as the resolution; the seed standard deviation is pooled over the two arms compared.

### The new variants

Both are masks, by role within the context, on the positions a labelled context pulls (roles as listed under [Conditions](#conditions)). They run through the same pooled term as every other variant, which gives each labelled context the same total pull however many positions share it. On `query-eq` the pool has one position, so the whole pull lands on the query `=`: the variant changes how hard that position is pulled as well as where.

S2 chooses an arm without saying which of the two changes made the difference; E1 shows the alignment each arm reaches at each position.

### Budget

{ex.N_RUNS} runs at about \${ex.cost_per_run():.2f} each on an L4, from ex-2.2.19's unanchored runs at {ex.EPOCHS} epochs; we expect the anchored step to cost about the same and will check on the first runs. The suppression pass is scoring only, on {sum(ex.arm(a).seeds for a in ex.SUPPRESSION_ARMS)} checkpoints. Under \$10 in all.
"""
