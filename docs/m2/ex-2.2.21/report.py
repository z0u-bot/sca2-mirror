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
BAND_33 = seed_band(3, 3)

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


QUERY_START = len(ROLE_LABELS) - 6
SLICE_NAMES = ["emb", *(f"block {i}" for i in range(1, ALIGN16["anchor-whole"].shape[1]))]
STACK_GAP = 1.1


def smooth_step(y: list[float], flat: float = 0.55, n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """A per-token profile drawn as plateaus joined by smoothstep ramps, so it reads as a sequence of tokens."""
    xs, ys = [], []
    for i, v in enumerate(y):
        xs += [i - flat / 2, i + flat / 2]
        ys += [v, v]
        if i + 1 < len(y):
            t = np.linspace(0, 1, n)[1:-1]
            xs += list(i + flat / 2 + t * (1 - flat))
            ys += list(v + (y[i + 1] - v) * (3 * t**2 - 2 * t**3))
    return np.array(xs), np.array(ys)


def profiles(arm: str) -> dict:
    """Per slice, the alignment on `difference` contexts and the mean over the other ops."""
    a = ALIGN16[arm]
    return {
        "anchored": a[D16].tolist(),
        "other": np.delete(a, D16, axis=0).mean(axis=0).tolist(),
    }


def preview_figure() -> str:
    data = {a: profiles(a) for a in PREVIEW_ARMS}
    w = data["anchor-whole"]["anchored"]
    h = data["anchor-hinge"]
    ex_mid = [w[2][i] for i in EX_ANS]
    alt = f"""
        Three panels side by side, one per arm of ex-2.2.16 (the control, the whole-line arm, and the hinge arm). Each
        stacks five step-shaped traces, one per slice from the embedding at the bottom to the last block at the top,
        plotting the alignment with the anchored axis against the 24 positions of a three-example context: a solid
        trace for `difference` contexts and a dashed one for the mean of the other ops, with the other two arms as
        faint lines. A vertical line marks where the query starts. The control is flat near zero at every slice. On
        the whole-line arm the `difference` trace peaks on the answers: the example answers reach
        {min(ex_mid):.2f} to {max(ex_mid):.2f} at block 2, and the query answer climbs from {w[1][Q_ANS]:.2f} at
        block 1 to {w[-1][Q_ANS]:.2f} at the last block, while the query `=` stays below {max(r[Q_EQ] for r in w):.2f}
        at every slice. On the hinge arm the query `=` holds about {h["anchored"][-1][Q_EQ]:.2f} at the last block,
        and so does the dashed trace for the other ops ({h["other"][-1][Q_EQ]:.2f}). At the embedding the solid and
        dashed traces coincide: the whole-line arm lifts `?`, `,`, and the line break to about {w[0][Q_Q]:.2f}, and
        the hinge arm lifts `=` and `,` to about {h["anchored"][0][Q_EQ]:.2f}. A bar at the bottom right spans an
        alignment of 0 to 1.
    """
    return preview_draw(data, alt)


@memo
def preview_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="preview-alignment-stack",
        alt_text=alt_text,
        caption=f"""
            **Where ex-2.2.16 put the anchor.** Alignment with e₁ by position in a context of three examples, one
            trace per slice, from the embedding at the bottom to the last block at the top; seed means over three
            runs on held-out contexts. Solid: `{ex.ANCHORED_OP}` contexts. Dashed: the mean over the other ten ops.
            Faint: the `{ex.ANCHORED_OP}` trace of the other two arms. The shaded column is the query `=`, whose state
            predicts the answer; the vertical line marks the start of the query. The labels are the roles: a, b, and
            y for the operands and the answer, numbered by example and bare for the query. Each trace's baseline is a
            hairline at zero, and the bar at bottom right spans an alignment of 0 to 1.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 5.2), layout="constrained", sharey=True)
        axes = cast(AxesRow, axes)
        ink = rule_color()
        accent = light_dark("#c0392b", "#ff8a76")
        ghost = light_dark("#999", "#777")
        titles = {"control-k3-r0.3": "control", "anchor-whole": "whole-line", "anchor-hinge": "hinge"}
        n_slices = len(data["anchor-whole"]["anchored"])
        for ax, (arm, d) in zip(axes, data.items(), strict=True):
            ax.axvspan(Q_EQ - 0.5, Q_EQ + 0.5, facecolor=light_dark("#000", "#fff"), alpha=0.07, lw=0)
            ax.axvline(QUERY_START - 0.5, color=ink, lw=0.8, ls=(0, (2, 2)))
            for sl in range(n_slices):
                base = sl * STACK_GAP
                ax.axhline(base, color=ink, lw=0.3, alpha=0.5)
                for other, od in data.items():
                    if other != arm:
                        ax.plot(*_lift(smooth_step(od["anchored"][sl]), base), color=ghost, lw=0.6, alpha=0.6)
                ax.plot(*_lift(smooth_step(d["other"][sl], flat=0.8), base), "--", color=ink, lw=0.9)
                ax.plot(*_lift(smooth_step(d["anchored"][sl], flat=0.8), base), color=accent, lw=1.3)
            ax.set_title(titles[arm], fontsize=9)
            ax.set_xticks(np.arange(len(ROLE_LABELS)), ROLE_LABELS, fontsize=6)
            ax.set_xlim(-0.6, len(ROLE_LABELS) - 0.4)
            ax.text(QUERY_START - 0.3, (n_slices - 0.05) * STACK_GAP, "query", fontsize=6, color=ink, va="top")
        bar_x = len(ROLE_LABELS) - 0.1
        axes[-1].plot([bar_x, bar_x], [0, 1], color=ink, lw=1.2, clip_on=False)
        for v in (0, 1):
            axes[-1].text(bar_x + 0.35, v, f"{v}", fontsize=6, color=ink, va="center", clip_on=False)
        axes[0].set_yticks([sl * STACK_GAP for sl in range(n_slices)], SLICE_NAMES, fontsize=7)
        axes[0].set_ylim(-0.15, n_slices * STACK_GAP)
        handles = [
            plt.Line2D([], [], color=accent, lw=1.3, label=ex.ANCHORED_OP),
            plt.Line2D([], [], color=ink, lw=0.9, ls="--", label="other ops"),
            plt.Line2D([], [], color=ghost, lw=0.6, label="the other arms"),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncols=3, frameon=False, fontsize=7)
        return fig

    return _plot()


def _lift(xy: tuple[np.ndarray, np.ndarray], base: float) -> tuple[np.ndarray, np.ndarray]:
    return xy[0], xy[1] + base


SUPP_SITES = ("query ?", "query =", "every position")
SUPP_EDITS = (
    *(("projection", g, f"γ {g:g}") for g in (0.25, 0.5, 0.75, 1.0)),
    ("repulsion", 0.25, "rep. 0.25"),
    ("repulsion", 0.0, "rep. 0"),
    ("reflection", 2.0, "refl."),
)


def suppression_figure() -> str:
    data: dict[str, dict] = {
        arm: {
            "half": float(ex.GRADING_MIN_DAMAGE * (SUPP[arm]["clean"][D16] - SUPP[arm]["null"][D16])),
            "sites": {
                site: {
                    "anchored": [float(net_drop(arm, (o, g, site))[D16]) for o, g, _ in SUPP_EDITS],
                    "worst": [worst_other(arm, (o, g, site)) for o, g, _ in SUPP_EDITS],
                }
                for site in SUPP_SITES
            },
        }
        for arm in ("anchor-whole", "anchor-hinge")
    }
    wf = data["anchor-whole"]["sites"]["every position"]
    he = data["anchor-hinge"]["sites"]["query ="]
    alt = f"""
        A grid of six panels: rows for the whole-line arm and the hinge arm of ex-2.2.16, columns for the three edit
        sites (the query `?`, the query `=`, and every position). Each panel plots the net drop in expected exact
        match against the edit: the projection at four doses, the repulsion at two, and the reflection, with a solid
        trace for `difference` contexts and a dashed one for the worst other op, a rule at the selectivity gate of
        {ex.SELECTIVITY_GATE:g}, and a rule halfway to the target null. At the query `?` both arms stay at zero. At
        the query `=` the whole-line arm stays at zero, and on the hinge arm the drop on `difference`
        ({min(he["anchored"][:4]):.2f} to {max(he["anchored"][:4]):.2f} under the projection) is matched by the worst
        other op. At every position the whole-line projection rises to {max(wf["anchored"][:4]):.2f}, below the
        halfway rule at {data["anchor-whole"]["half"]:.2f}, and the worst other op passes the gate from γ = 0.75.
    """
    return suppression_draw(data, alt)


@memo
def suppression_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="preview-suppression",
        alt_text=alt_text,
        caption=f"""
            **Ex-2.2.16's suppression pass on its two scored anchored arms.** The net drop in held-out expected exact
            match under each edit: the seed-mean drop, less the drop the control shows under the same edit. Solid:
            on `{ex.ANCHORED_OP}` contexts. Dashed: the worst of the other ten ops. The dotted rule is the selectivity
            gate ({ex.SELECTIVITY_GATE:g}), and the dash-dot rule is {ex.GRADING_MIN_DAMAGE:.0%} of the way from the
            clean score on `{ex.ANCHORED_OP}` to the target null. The projection runs from γ = 0.25 to full removal;
            the repulsion lands states at 0.25 and then 0; the reflection has one dose.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.6), layout="constrained", sharey=True, sharex=True)
        ink = rule_color()
        accent = light_dark("#c0392b", "#ff8a76")
        x = np.arange(len(SUPP_EDITS))
        groups = (slice(0, 4), slice(4, 6), slice(6, 7))
        titles = {"anchor-whole": "whole-line", "anchor-hinge": "hinge"}
        for r, (arm, d) in enumerate(data.items()):
            for c, site in enumerate(SUPP_SITES):
                ax = axes[r, c]
                ax.axhline(ex.SELECTIVITY_GATE, color=ink, lw=0.8, ls=":")
                ax.axhline(d["half"], color=ink, lw=0.8, ls="-.")
                ax.axhline(0, color=ink, lw=0.3, alpha=0.5)
                for g in groups:
                    ax.plot(x[g], d["sites"][site]["anchored"][g], "-o", color=accent, lw=1.3, ms=3)
                    ax.plot(x[g], d["sites"][site]["worst"][g], "--o", color=ink, lw=0.9, ms=2.5, mfc="none")
                if r == 0:
                    ax.set_title(f"at {site}", fontsize=9)
                if c == 0:
                    ax.set_ylabel(f"{titles[arm]}\nnet drop", fontsize=8)
                ax.set_xticks(x, [e[2] for e in SUPP_EDITS], fontsize=6, rotation=45, ha="right")
        handles = [
            plt.Line2D([], [], color=accent, lw=1.3, marker="o", ms=3, label=ex.ANCHORED_OP),
            plt.Line2D([], [], color=ink, lw=0.9, ls="--", marker="o", ms=2.5, mfc="none", label="worst other op"),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncols=2, frameon=False, fontsize=7)
        return fig

    return _plot()


PV = {"whole_full": ("projection", 1.0, "every position")}
WHOLE = profiles("anchor-whole")["anchored"]
HINGE = profiles("anchor-hinge")
EX_EQ = [i for i, t in enumerate(ROLE_LABELS[:18]) if t == "="]
COMMA = ROLE_LABELS.index(",")

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
STEP7 = float(((C7["post"] >= 0.85) & (C7["post"] < 0.95)).mean() / BANDS7[1])


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


def label_cell(a: ex.Arm) -> str:
    """The label an arm pulls with; `no-emb` is the whole-line label with the embedding slice left out."""
    if not a.anchored:
        return "—"
    if a.label == "no-emb":
        return "`whole` + no-emb"
    return f"`{a.label}`" + (" + hinge" if a.hinge else "")


def arms_table() -> str:
    head = ["arm", "group", "anchored", "label", "corpus", "what it asks", "seeds"]
    rows = [
        [
            f"`{a.name}`",
            a.group,
            "yes" if a.anchored else "—",
            label_cell(a),
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
- [Editing the op out (E2)](#editing-the-op-out-e2) —
- [The verification rule (S2)](#the-verification-rule-s2) —
- [The leans stay with the control (H2)](#the-leans-stay-with-the-control-h2) —

## How to read this draft

This is a preregistration: nothing in it has been trained yet. The rules and predictions were frozen at commit `2d84d2b`, before any run of this experiment, and results will replace the `TODO` boxes in place. The [preview](#what-ex-2216s-anchored-arms-already-show) scores ex-2.2.16's stored runs; it predates this plan and shaped it.

## Why this experiment

[Ex-2.2.16](/docs/m2/ex-2.2.16/report.py) was round 2 of the [D2.2 route](/docs/m2/d2.2/design.md#quick-route). It trained a control and several arms that anchor `{ex.ANCHORED_OP}` on the in-context grammar, and froze rules to select how round 3 would anchor and edit: the label, the site, the operator, and whether verification lines stay in the corpus. Round 3 would then anchor the inferred op at fresh seeds and suppress it.

But ex-2.2.16 stopped at its first rule. Its control had learned less than half of what the examples allow, {CTRL16_SKILL:.0%} of the way from the floor to the Bayes ceiling. An anchor compared with a model that far from finished would say little about round 3.

Four reports then worked on the control alone: [ex-2.2.17](/docs/m2/ex-2.2.17/report.py) added the newline mask, a lower peak learning rate, and more steps; [ex-2.2.18](/docs/m2/ex-2.2.18/report.py) dropped four ops whose answers often coincide with another op; [ex-2.2.19](/docs/m2/ex-2.2.19/report.py) settled the length at {ex.EPOCHS} epochs; and [ex-2.2.20](/docs/m2/ex-2.2.20/report.py) kept the plain schedule.

On the seven-op set the control now scores {REF19_MEAN:.3f} against a ceiling of {REF19_CEIL:.3f}, {REF19_SKILL:.0%} of the way from the floor. This pilot reruns round 2 on that recipe.

It keeps the label, verification, and operator rules of ex-2.2.16, changing each only where the stored runs of ex-2.2.16 or the new recipe call for it, and adds three label variants as references. The [design](/docs/m2/d2.2/design.md#the-pilot) names the rules (a) to (e); here they are H1, S1, and S2, and the site and operator rules become an exploratory analysis, E2.

## What ex-2.2.16's anchored arms already show

Ex-2.2.16's anchored arms were trained and evaluated, but the eval was never scored. These numbers come from the half-trained model, so they only show what to look for.

It turns out the anchor landed. The op margin, which measures how far `{ex.ANCHORED_OP}` contexts sit along the anchored axis beyond the rest, was {MARGIN16["anchor-whole"]:.2f} on the whole-line arm against {MARGIN16["control-k3-r0.3"]:.2f} on the control. But most of it landed where an edit cannot reach the answer:

{preview_figure()}

On the whole-line arm the alignment (the cosine of a state with e₁) sits on the answers. It builds up on the example answers by block 2 ({min(WHOLE[2][i] for i in EX_ANS):.2f} to {max(WHOLE[2][i] for i in EX_ANS):.2f}) and fades a little after; on the query answer it climbs with depth, to {WHOLE[-1][Q_ANS]:.2f} at the last block. The query `=`, whose state predicts the answer, stays below {max(r[Q_EQ] for r in WHOLE):.2f} at every slice.

The pooled anchor term asks a labelled context to align somewhere in its span. The answers are where the context has shown the most evidence about its op, so the pull seems to have settled there.

For an edit, the two kinds of answer differ. The state at the query answer comes after the answer is predicted (the model reads left to right) and predicts only the line break, so an edit there cannot change the answer. The query can read the example answers through the blocks above them, so an edit there may still reach it, though not an edit at the last slice, which no block reads.

The query `?` stayed clean ({WHOLE[-1][Q_Q]:.2f}). Ex-2.2.16 worried that it would saturate, but the position that came close was the query answer. The hinge arm, which caps the pull, holds {HINGE["anchored"][-1][Q_EQ]:.2f} at the query `=` at the last block, but the other ops hold nearly as much there ({HINGE["other"][-1][Q_EQ]:.2f}), so that alignment says little about `{ex.ANCHORED_OP}`.

The suppression pass agrees:

{suppression_figure()}

At the query `=`, an edit on the whole-line arm barely touches the answer, and on the hinge arm it lowers every op about as much as `{ex.ANCHORED_OP}`.

With the edit at every position and every slice, the projection on the whole-line arm grows with dose up to γ = 0.75, then levels off {to_null("anchor-whole", PV["whole_full"]):.0%} of the way to the target null. From γ = 0.75 on, it also spills past the selectivity gate onto other ops. The hinge arm gets {to_null("anchor-hinge", PV["whole_full"]):.0%} of the way and spills more.

At the embedding slice the two arms differ. There the solid and dashed traces coincide, because a token embedding is the same whatever the op. Each anchored arm has put a few syntax embeddings partway onto e₁:

- on the whole-line arm, `?` ({WHOLE[0][Q_Q]:.2f}), `,` ({WHOLE[0][COMMA]:.2f}), and the line break;
- on the hinge arm, `=` ({HINGE["anchored"][0][Q_EQ]:.2f}) and `,`.

On the earlier grammar a leak like this came through the tied readout (the embedding table reused as the readout table), and untying it brought the leak down ([ex-2.2.7](/docs/m2/ex-2.2.7/report.py)). The readout here is already untied. So this leak seems to come from the pull itself: the pull acts on the embedding slice of every position it pulls, and some of those are tokens every context shares.

On the whole-line arm the bump at `,` sits one position after the answers. That makes the alignment look as though it moves back a token with depth, but the two are separate effects.

On the hinge arm the `=` embedding explains why the query `=` holds alignment on every op. It starts at {HINGE["anchored"][0][Q_EQ]:.2f} for every op and keeps about that much to the last block, while the example `=` positions fade to {min(HINGE["anchored"][-1][i] for i in EX_EQ):.2f} to {max(HINGE["anchored"][-1][i] for i in EX_EQ):.2f}. An edit at every position moves these syntax states in every context, which may be part of why it spills onto other ops.

The preview changes four things in this plan:

- Three new label variants move the pull toward the query `=`, as references rather than candidates for round 3. `prompt` leaves the query answer out of the pull, to see where the pooled term settles without it. `query-eq` pulls the query `=` alone: a position oracle,[^oracle] which shows what an anchor there would allow. `every-eq` pulls every `=`, the positions whose next token depends on the op, to see whether a pull spread over the examples settles at the query `=` too.
- The saturation half of ex-2.2.16's hinge rule goes, since the query `?` did not saturate, though the hinge arm stays a candidate.
- The site and operator rules become an exploratory analysis that chooses no arm. The suppression pass also edits the example answers, and the edits are scored at the query `=` as well as at every position.
- E1 reports the syntax embeddings beside the per-position alignment, and the `no-emb` arm, which leaves the embedding slice out of the pull, shows whether they stay clean without it.

[^oracle]: The term from [ex-2.1.8](/docs/m2/ex-2.1.8/report.py): a pull at a position chosen by hand, which no labeller could give. It is a reference for what the right position would allow, and not a strict ceiling.

## Glossary

Terms follow [ex-2.2.16](/docs/m2/ex-2.2.16/report.py#glossary); these are the ones the rules here use.

<dl>
<dt>Alignment</dt>
<dd>The cosine between a state and e₁, the anchored axis, per position and slice. Cosine similarity compares direction only: 1 when two vectors point the same way, 0 when they are perpendicular.</dd>
<dt>Op margin</dt>
<dd>How far the contexts of the anchored op sit along e₁ beyond the rest. At each slice, take the mean alignment over <code>{ex.ANCHORED_OP}</code> contexts less the mean over all contexts, <code>{ex.ANCHORED_OP}</code> included, at the role where that gap is largest; then average over slices. The anchor term optimizes this quantity, so it checks that the pull landed.</dd>
<dt>Expected exact match (EEM)</dt>
<dd>The probability mass the model puts on the colors the query answer can be under the true op, weighted by how often the op gives each, averaged over held-out contexts. The task measurement.</dd>
<dt>Bayes ceiling and floor</dt>
<dd>The expected exact match of a predictor that holds the posterior over ops given the examples, and of one that ignores the examples.</dd>
<dt>Target null</dt>
<dd>The answer a model that has lost <code>{ex.ANCHORED_OP}</code> and nothing else would give: the posterior-weighted answer distribution with <code>{ex.ANCHORED_OP}</code> removed and the other ops renormalized.</dd>
<dt>Seed band</dt>
<dd>The smallest difference between two seed means a comparison resolves: {ex.SEED_BAND_SD:g}σ√(1/n₁ + 1/n₂) for arms at n₁ and n₂ seeds, with σ the seed standard deviation pooled over the two. At the spread of ex-2.2.19's four runs it is about {BAND_55:.3f} for five seeds against five, {BAND_53:.3f} for five against three, and {BAND_33:.3f} for three against three.</dd>
</dl>

## Conditions

Every arm trains at the center condition of ex-2.2.16 (three examples, ρ = {ex.CENTRE[1]:g}) on the seven-op set, under ex-2.2.19's recipe: the newline mask, a peak rate of {ex.PEAK_LR:g}, {ex.EPOCHS} epochs, {ex.MODEL}. The anchored arms add ex-2.2.14's anchor recipe, each weight scheduled as a fraction of training as before, and crop policy `{ex.CROP_POLICY}`.

{arms_table()}

Ex-2.2.16's arms at other corpus conditions, its larger control, and its two newline-mask arms are gone: the condition is settled, the larger control was for a shortfall the recipe has since closed, and the mask is now in every arm.

**The new label variants.** All three keep ex-2.2.16's labeller: a `{ex.ANCHORED_OP}` context draws a label with probability {ex.LABEL_RATE:g}, keyed on the op array beside the corpus. Variant (e), `prompt`, pulls every position of a labelled context up to and including the query `=`, leaving out the query answer and the line break. Variant (f), `query-eq`, pulls the query `=` alone. Variant (g), `every-eq`, pulls the `=` of each example and of the query, the four positions that predict an answer. The first example's `=` comes before any evidence, so its pull asks for the op before the context shows it.

`prompt` keeps the example answers in the pull, because they are part of the evidence. If the pull still settles on them, that is a result about pooling.

**The seeds.** Fresh model seeds from {ex.SEED_OFFSET}. The control and the whole-line arm get {ex.SEEDS_REFERENCE} each, since every comparison runs through one of them; the others get three.

## The recipe holds, and the anchor costs nothing (H1)

**What we expect.** Round 3 starts from this recipe only if both of these criteria hold:

- **(a)** The control reproduces ex-2.2.19: its seed-mean held-out expected exact match is within {ex.REGRESSION_TOL:g} of {REF19_MEAN:.3f}, the mean of ex-2.2.19's four runs at {ex.EPOCHS} epochs.
- **(b)** The whole-line arm falls short of the control by at most {ex.TASK_COST_TOL:g}, a little wider than the seed band of five seeds against five.

The arms differ from ex-2.2.19 only in their seeds and in the anchor, so (a) checks that nothing else moved.

We expect (b) to hold too. Ex-2.2.14 found the anchor costs the task nothing when {ex.LABEL_RATE:.0%} of contexts are labelled, and the stored runs of ex-2.2.16 agree: the whole-line arm scored {EEM16["anchor-whole"]:.3f} against {EEM16["control-k3-r0.3"]:.3f} for the control, and {EEM16["anchor-mask"]:.3f} against {EEM16["control-mask"]:.3f} with the newline mask.

A miss on (b) would mean the anchor and the task compete on this recipe, and would leave E2 with no candidate, since it asks the same of each.

This replaces corpus rule (a) of ex-2.2.16, which asked the control to come within 0.03 of the calibrated ceiling. The seven-op control misses that: it sits {REF19_CEIL - REF19_MEAN:.3f} below the ceiling at {ex.EPOCHS} epochs, and still misses at 400. Ex-2.2.17 found the remaining gap on contexts whose examples settle the op, where the model keeps mass on the answers of a similar op.

We take the control as it stands. Its skill score and calibration KL[^kl] are reported beside every comparison, but no criterion depends on them. In ex-2.2.19 the KL was about {REF19_KL:.2f} nats; ex-2.2.16 called a model calibrated below 0.05.

[^kl]: A KL divergence: a non-negative measure, in nats, of how far one probability distribution sits from another, 0 when they match. Here it compares the model's answer distribution with the Bayes predictor's.

/// admonition | TODO
A figure of held-out expected exact match per seed for the control and the whole-line arm, seed means as bars, with ex-2.2.19's four runs as a reference column; the ceiling and floor as rules and the band of (a) shaded. Beside it, the calibration KL for the same runs. A table with the seed means, the seed band, and the verdict on each criterion.
///

## The label rule (S1)

**The rule.** Ex-2.2.16's rule (b), unchanged. The whole-line label stays the primary unless a candidate variant (below) beats the whole-line arm on seed-mean held-out expected exact match by more than the seed band, while keeping an op margin of at least {ex.MARGIN_KEEP:.0%} of the whole-line margin. If several qualify, the one with the higher op margin goes forward.

The candidates are `latter`, `prefix`, and `no-emb` (the whole-line label with the embedding slice left out of the pull). Variant (d), `sampled`, trains the grading that round 3 will measure, so it is reported and not promoted.

**What we expect.** No variant clears the gate, for the reasons ex-2.2.16 gave: the label share barely moved the margin in ex-2.2.14, and the variants change less than that.

/// admonition | TODO
A figure with one column per label arm: held-out expected exact match (seeds and seed mean, the whole-line seed band shaded) above, the op margin as a share of the whole-line margin below, with the {ex.MARGIN_KEEP:.0%} line. A table of the same numbers and the verdict.
///

## Where the anchor sits (E1)

Where an edit can work depends on where along the context each anchored arm puts its alignment. E1 measures, per anchored arm, the seed-mean alignment at each position and slice of a held-out context, on `{ex.ANCHORED_OP}` contexts and on the other ops.

/// admonition | TODO
One stacked figure per anchored arm, as in the preview above: positions across, slices up, the alignment on `{ex.ANCHORED_OP}` contexts and the mean over the other ops as two traces, and the other arms as faint traces. A table of the alignment at the query `?`, the query `=`, the query answer, and the example answers, at the last block, and of the syntax embeddings (`?`, `=`, `,`, and the line break) at the embedding slice.
///

## Editing the op out (E2)

E2 asks whether `{ex.ANCHORED_OP}` can be edited out of an anchored model with a dial: an edit whose effect on `{ex.ANCHORED_OP}` grows with its dose while the other ops stay as they were. It chooses no arm, so round 3 is designed after this report, and it replaces hinge rule (c) and operator rule (e) of ex-2.2.16.

The candidates (`{"`, `".join(ex.SITE_ARMS)}`) are arms round 3 could adopt as they stand. The references (`{"`, `".join(ex.SITE_REFERENCES)}`) put the pull where no labeller could.

**The edits.** The suppression pass of ex-2.2.16 runs, scoring only, on the candidates, the references, and the control. It has three operators, each applied at every slice at once:

- the projection, which removes a share γ of the component along e₁, for γ in {{{", ".join(f"{g:g}" for g in ex.DOSE_GAMMAS)}}};
- the repulsion, which moves states aligned above {ex.REPULSION_THRESHOLD:g} down to an alignment of {" and then ".join(f"{b:g}" for b in ex.REPULSION_LANDINGS)};
- the reflection, which flips the component, at one dose.

Each edit is applied at one of four sites: the query `?`, the query `=`, the example answers, or every position. Every drop listed below is a net drop: the seed-mean fall in held-out expected exact match, minus the fall the control shows under the same edit, so only the part the anchor caused counts.

**The two criteria.** Two sites are scored against the criteria below: the query `=`, and every position (the edit that would carry over most readily to another task). At the other two sites the drops are reported without criteria. An operator at a scored site meets the criteria on an arm when both of these hold:

- **It grades.** The net drop on `{ex.ANCHORED_OP}` contexts rises with the dose, allowing a dip between adjacent doses of at most {ex.GRADE_DIP:g}, and at full dose it covers at least {ex.GRADING_MIN_DAMAGE:.0%} of the distance from the clean score to the target null. The reflection has one dose, so it cannot grade.
- **It is selective.** On each of the other six ops, the net drop is at most {ex.SELECTIVITY_GATE:g} at every dose. This is the threshold [ex-2.2.11](/docs/m2/ex-2.2.11/report.py) set for a change in expected exact match too small to matter for the task. It sits well above the seed band of three seeds against three, because the worst of six ops is the largest of six noisy numbers.

A flat threshold treats every other op alike, but some are easier to mistake for `{ex.ANCHORED_OP}` than others. So beside it, E2 shows the net drop on each other op against how plausible `{ex.ANCHORED_OP}` is in that op's contexts: the mean posterior on `{ex.ANCHORED_OP}` given the examples. If the drop rises with that posterior, the edit takes out the op as the model infers it, and the cost on other ops follows the confusions the contexts allow.

**Where the op comes from.** The bypass test in the design for round 3 edits at different positions and slices, looking for a route by which the op reaches the answer around the edit. E2 is a first look at it, varying the position only. If the edit at every position works and the one at the query `=` does not, the answer must take the op from somewhere else, and the edit at the example answers shows whether that is the examples.

**What we expect.** Neither candidate meets both criteria. On the whole-line arm the projection at every position levels off about a third of the way to the null, as in the preview, and on the hinge arm the edit at every position spills onto other ops.

On `query-eq` we expect the projection at the query `=` to meet both, which would say that an anchor there can be edited with a dial, if a label could put it there. On `every-eq` we expect the same, more weakly, since a quarter of its pull lands on the query `=`. We are unsure about `prompt`: its pull may settle on the example answers, which E1 will show.

/// admonition | TODO
One figure per arm, as the preview figure of the suppression pass: the net drop on `{ex.ANCHORED_OP}` and the worst other op against the edit, with the selectivity gate and the halfway level, one panel per site. A table of each operator at the two scored sites on each arm against the two criteria. A figure of the net drop on each other op at full dose against the mean posterior on `{ex.ANCHORED_OP}` in its contexts, one point per op, one panel per arm.
///

## The verification rule (S2)

**The rule.** Ex-2.2.16's rule (d), unchanged. Verification lines stay in the corpus from round 3 on if they leave completion unchanged: the seed-mean held-out expected exact match on completion contexts is within the seed band of the arm without them, on both pairs (`control-verify` against `{ex.CONTROL}`, `anchor-verify` against `{ex.PRIMARY}`). The verification accuracy and the op margin of `anchor-verify` are reported with no gate.

**What we expect.** Completion is unchanged on both pairs. The verification arms see {1 - ex.VERIFY_RATE:.0%} of the completion lines the others do. On the old recipe that cost the control nothing measurable; at {ex.EPOCHS} epochs each line is seen more often, so the cost should be smaller still.

/// admonition | TODO
A figure of held-out expected exact match per seed for the two pairs, with the seed band of each shaded, and verification accuracy beside it. A table of the same and the verdict.
///

## The leans stay with the control (H2)

**What we expect.** On the whole-line arm, the first-operand lean and the trailing-fragment lean stay within the seed band of the control, as ex-2.2.16 defined them. The newline mask now keeps a context from reading the one before it, which should make a cut-off fragment look less like a whole context than it did in ex-2.2.15.

/// admonition | TODO
A figure of both leans per seed for the control and every anchored arm, with the seed band of the control shaded.
///

## Discussion

/// admonition | TODO
After the run. What the pilot settles for round 3: the recipe (H1), the label (S1), and the verification lines (S2). What E2 and its references suggest about the site and the operator for round 3, and where an anchor would need to sit to be edited.
///

## Method

### The corpus and the posterior

The corpus is ex-2.2.18's seven-op corpus at the center condition, which ex-2.2.19 also trained on: one context per line, 300,000 contexts, three examples and ρ = {ex.CENTRE[1]:g}, cube noise at κ = {ex.CUBE_RATE:g}, and stochastic rounding. The held-out and probe sets are ex-2.2.18's. The verification arms train on the same generator with {ex.VERIFY_RATE:.0%} of contexts written as verification lines.

Dropping four ops leaves the posterior on `{ex.ANCHORED_OP}` much as it was: on the seven-op table {BANDS7[1]:.0%} of `{ex.ANCHORED_OP}` contexts sit in the middle band, against {BANDS11[1]:.0%} on eleven ops, with {BANDS7[0]:.0%} below it and {BANDS7[2]:.0%} above. On these sampled contexts the Bayes ceiling is {C7["ceiling"]:.3f} and the floor {C7["floor"]:.3f}; on the held-out set of ex-2.2.19, the ceiling is {REF19_CEIL:.3f}.

But the posterior takes few distinct values: {STEP7:.0%} of the contexts in the middle band sit in one step between 0.85 and 0.95, so the stimulus has about three levels. That matters for the graded stimulus of round 3, but not for the rules here.

{posterior_figure()}

<details markdown="1"><summary>The neighboring conditions</summary>

{grid_table()}

</details>

### The measurements

Ex-2.2.16's measurements, scored on its held-out sets at the end of training, with the mask on in every arm: expected exact match against the ceiling and floor, the calibration KL, the alignment by position and slice, the alignment against the posterior, the op margin, the two leans, the suppression pass, and verification accuracy. The op margin and the leans are also recorded through training on a fixed probe set. Comparisons are between seed means, with the seed band as the resolution; the seed standard deviation is pooled over the two arms compared.

### The new variants

`prompt`, `query-eq`, and `every-eq` are masks, by role within the context, on the positions a labelled context pulls (roles as listed under [Conditions](#conditions)). They run through the same pooled term as every other variant, which gives each labelled context the same total pull however many positions share it.

On `query-eq` the pool has one position, so the whole pull lands on the query `=`: the variant changes how hard that position is pulled as well as where. On `every-eq` the pool has four positions, so the query `=` gets a quarter of that.

E2 scores `query-eq` and `every-eq` without separating the two changes; E1 shows the alignment it reaches at the query `=` beside the other arms.

### Budget

{ex.N_RUNS} runs at about \${ex.cost_per_run():.2f} each on an L4, from ex-2.2.19's unanchored runs at {ex.EPOCHS} epochs; we expect the anchored step to cost about the same and will check on the first runs. The suppression pass is scoring only, on {sum(ex.arm(a).seeds for a in ex.SUPPRESSION_ARMS)} checkpoints. Under \$10 in all.
"""
