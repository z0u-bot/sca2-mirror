# title: Ex 2.2.18: dropping ops with similar answers, a scout

# The design constants and the refs come from `experiment.py` beside this script; the answer table and the posterior
# come from ex-2.2.16 through it, and the seed yardstick from ex-2.2.17's published evaluation.
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import AxesRow, figure_html, light_dark, themed

X = ex.ex2216
ALL_OPS: tuple[str, ...] = tuple(X.OP_NAMES)
SETS: tuple[str, ...] = tuple(s.name for s in ex.OP_SETS)
SINGLES: tuple[str, ...] = tuple(f"no-{op}" for op in ex.PARTNER)
YARDSTICK = tuple(f"sweep-{ex.PEAK_LR:g}-s{s}" for s in range(3))
# A context is confident when the posterior on its true op is above this, as in ex-2.2.17.
CONFIDENT = 0.99


# --- Helpers -------------------------------------------------------------------------------------------------


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str, *, text_cols: int = 1) -> str:
    """An authored result table in the shared report style; the first *text_cols* columns are text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>"
        + "".join(f"<td{' class=num' if i >= text_cols else ''}>{cell_html(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def rule_color() -> str:
    return light_dark("#333", "#ccc")


# --- Loading ------------------------------------------------------------------------------------------------


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists: one `get_refs` and one `get_many`."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict:
    assert path is not None, "not published yet"
    return json.loads(path.read_text())


with tempfile.TemporaryDirectory() as _tmp:
    _refs = [ex.EVAL_REF, ex.TRAJ_REF, ex.ex2217.EVAL_REF, *(ex.EVAL_ARRAYS_REF.format(label=s) for s in SETS)]
    _files = fetch(_refs, Path(_tmp))
    EVAL = read_json(_files[ex.EVAL_REF])
    TRAJ = read_json(_files[ex.TRAJ_REF])
    RUNS = {r["label"]: r for r in EVAL["runs"]}
    STATS = EVAL["op_sets"]
    PRIOR = {r["label"]: r for r in read_json(_files[ex.ex2217.EVAL_REF])["runs"] if r["label"] in YARDSTICK}
    ARRAYS: dict[str, dict[str, np.ndarray]] = {}
    for _s in SETS:
        with np.load(cast(Path, _files[ex.EVAL_ARRAYS_REF.format(label=_s)])) as _z:
            ARRAYS[_s] = {k: _z[k] for k in _z.files}


def ops_of(s: str) -> tuple[str, ...]:
    return tuple(RUNS[s]["ops"])


def score(s: str, key: str = "eem", op: str | None = None, runs: dict | None = None) -> float:
    r = (runs or RUNS)[s]["task"][key]
    return r["all"] if op is None else r["per_op"][list((runs or RUNS)[s].get("ops", ALL_OPS)).index(op)]


def skill(s: str, op: str | None = None) -> float:
    """The share of the way from the floor to the ceiling of its own op set that a run got."""
    e, c, f = (score(s, k, op) for k in ("eem", "ceiling", "floor"))
    return (e - f) / (c - f)


def gap(s: str, op: str | None = None, runs: dict | None = None) -> float:
    return score(s, "ceiling", op, runs) - score(s, "eem", op, runs)


YARD_EEM = [score(s, runs=PRIOR) for s in YARDSTICK]
YARD_GAP = [gap(s, runs=PRIOR) for s in YARDSTICK]
YARD_SPREAD = max(YARD_EEM) - min(YARD_EEM)
# The seed range of each op's gap in ex-2.2.17's three runs of this recipe: the per-op yardstick.
YARD_OP_SPREAD = {
    op: max(gap(s, op, PRIOR) for s in YARDSTICK) - min(gap(s, op, PRIOR) for s in YARDSTICK) for op in ALL_OPS
}


# --- The leak onto the dropped op ------------------------------------------------------------------------------


@memo
def answer_table():
    """The answers of all eleven ops on every pair, so the answers of a dropped op stay defined."""
    return X._get_posterior().build_table(X.TABLE)


@memo
def leak(p16: np.ndarray, op_ids: np.ndarray, post: np.ndarray, pair: np.ndarray, ops: tuple[str, ...]) -> dict:
    """On the confident contexts of each partner op in *ops*, the mass the run puts on colors its dropped op can
    give and the partner cannot; and the same for the Bayes predictive of the full op set, which is near zero on
    these contexts. Answers come from the full table, so they are defined whether or not the op was trained on.
    """
    table = answer_table()
    p = p16.astype(float)
    rows = np.arange(len(p))
    out = {}
    for dropped, partner in ex.PARTNER.items():
        if partner not in ops:
            continue
        sel = (op_ids == ops.index(partner)) & (post[rows, op_ids] > CONFIDENT)
        a, b = ALL_OPS.index(partner), ALL_OPS.index(dropped)
        gives = np.zeros((2, sel.sum(), p.shape[1]), dtype=bool)
        for i, o in enumerate((a, b)):
            idx, ok = table.idx[o, pair[sel]], table.prob[o, pair[sel]] > 0
            for j in range(idx.shape[1]):
                gives[i, np.flatnonzero(ok[:, j]), idx[ok[:, j], j]] = True
        beyond = gives[1] & ~gives[0]
        out[dropped] = {
            "leak": float((p[sel] * beyond).sum(1).mean()),
            "partner_mass": float((p[sel] * gives[0]).sum(1).mean()),
            "n": int(sel.sum()),
        }
    return out


LEAK = {
    s: leak(ARRAYS[s]["p"], ARRAYS[s]["op_ids"], ARRAYS[s]["posterior"], ARRAYS[s]["query_pair"], ops_of(s))
    for s in SETS
}

# --- Figures ------------------------------------------------------------------------------------------------


@memo
def scores_draw(alt_text: str, caption: str) -> str:
    @themed(name="scores", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained")
        axes = cast(AxesRow, axes)
        x = np.arange(len(SETS))
        ax = axes[0]
        ax.plot(x, [score(s, "ceiling") for s in SETS], "_", ms=22, mew=2, color=rule_color(), label="Bayes ceiling")
        ax.plot(x, [score(s, "floor") for s in SETS], "_", ms=22, mew=1, color="0.6", label="floor")
        ax.plot(x, [score(s) for s in SETS], "o", color="C0", label="model")
        ax.axhspan(min(YARD_EEM), max(YARD_EEM), color="C0", alpha=0.15, lw=0, label="ex-2.2.17, three seeds")
        ax.set_ylabel("held-out EEM")
        ax.set_ylim(0, 0.7)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax = axes[1]
        ax.bar(x, [gap(s) for s in SETS], color="C0", width=0.6)
        ax.axhspan(min(YARD_GAP), max(YARD_GAP), color="C0", alpha=0.15, lw=0)
        ax.set_ylabel("gap to own ceiling")
        for a in axes:
            a.set_xticks(x, SETS, rotation=30, ha="right", fontsize=8)
        return fig

    return _plot()


@memo
def gaps_draw(alt_text: str, caption: str) -> str:
    @themed(name="gaps", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        m = np.full((len(ALL_OPS), len(SETS)), np.nan)
        for j, s in enumerate(SETS):
            for op in ops_of(s):
                m[ALL_OPS.index(op), j] = gap(s, op)
        fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
        lim = np.nanmax(np.abs(m))
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=lim, aspect="auto")
        for (i, j), v in np.ndenumerate(m):
            if np.isnan(v):
                ax.text(j, i, "dropped", ha="center", va="center", fontsize=6, color="0.5")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w" if v < lim * 0.6 else "k")
        ax.set_xticks(range(len(SETS)), SETS, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(ALL_OPS)), ALL_OPS, fontsize=8)
        fig.colorbar(im, ax=ax, label="gap to own ceiling (EEM)", shrink=0.8)
        return fig

    return _plot()


@memo
def traj_draw(alt_text: str, caption: str) -> str:
    @themed(name="trajectories", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
        for i, s in enumerate(SETS):
            t = TRAJ[s]
            st = STATS[s]
            y = (np.array(t["eem"]) - st["floor"]) / (st["ceiling"] - st["floor"])
            ax.plot(np.array(t["step"]) / 1e3, y, color=f"C{i}", lw=1.3 if s != "full" else 2, label=s)
        ax.axhline(1, ls="--", color=rule_color(), lw=1)
        ax.set_xlabel("step (thousands)")
        ax.set_ylabel("skill on the probe set")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, frameon=False, ncols=2)
        return fig

    return _plot()


# --- Derived measurements --------------------------------------------------------------------------------------

HSV_CHANNEL = ("hue-hsv", "sat-hsv", "value-hsv")
BEST_PRIOR = max(YARD_EEM)
FULL, FOUR = "full", "no-four"
# The trajectory points are evenly spaced, so index LAST_FIFTH is 80% of the way through training.
LAST_FIFTH = (len(TRAJ[FULL]["step"]) - 1) * 4 // 5


def probe_skill(s: str) -> np.ndarray:
    st = STATS[s]
    return (np.array(TRAJ[s]["eem"]) - st["floor"]) / (st["ceiling"] - st["floor"])


def steps_to(s: str, share: float) -> int:
    """The first logged step at which the probe skill reached *share* of its final value."""
    y = probe_skill(s)
    return int(TRAJ[s]["step"][int(np.argmax(y >= share * y[-1]))])


def late_gain(s: str) -> float:
    """Probe EEM gained over the last fifth of training."""
    return TRAJ[s]["eem"][-1] - TRAJ[s]["eem"][LAST_FIFTH]


def hsv_at(s: str, i: int) -> float:
    """Probe EEM on the three HSV-channel ops, averaged, at trajectory point *i*."""
    e = np.array(TRAJ[s]["eem_per_op"])[i]
    return float(np.mean([e[ops_of(s).index(op)] for op in HSV_CHANNEL]))


STEPS = TRAJ[FULL]["step"][-1]
EARLY = 10

rf"""
# Ex 2.2.18: dropping ops with similar answers, a scout

/// tip |
<!-- tl;dr -->
Dropping `screen`, `multiply`, `hsvmix`, and `exclusion` together raised the Bayes ceiling of the in-context grammar from {score(FULL, "ceiling"):.2f} to {score(FOUR, "ceiling"):.2f}, and the model came closer to it than any run so far, at {score(FOUR):.3f}. Dropping one op at a time mostly moved the ceiling and the model together, with one seed each.
///

Ex-2.2.17 found that its center control, where the examples settle the op, keeps part of its mass on the answers of the op most like the true one: `lighten` onto `screen`, `darken` onto `multiply`, `mix` and `hsvmix` onto each other, and `difference` onto `exclusion`. This scout drops one op of each of those pairs, one at a time and all four together, and trains the ex-2.2.17 recipe on each smaller op set, at one seed per op set.

Dropping ops changes the task, so every op set has its own Bayes ceiling and floor, and we score each run against its own.

## Observations

Each line is a measurement on the runs of this scout, with no gate.

- **E1** [Scores against each ceiling](#scores-against-each-ceiling-e1): every drop raised the ceiling, and the gap to it stayed within about 0.025 of the full set except without `multiply`. Without all four ops, the model came within {gap(FOUR):.3f} of its ceiling, closer than any run so far.
- **E2** [The HSV-channel ops](#the-hsv-channel-ops-e2): the runs without `multiply` and without `exclusion` fell short on the three HSV-channel ops, and the runs without `hsvmix` learned those ops earlier than the others did.
- **E3** [The leak onto a dropped op](#the-leak-onto-a-dropped-op-e3): on contexts of a partner op, the mass the model put on the answers of the dropped op mostly went away with it.
- **E4** [Training time](#training-time-e4): every run reached 95% of its final skill by step {max(steps_to(s, 0.95) for s in SETS):,} of {STEPS:,.0f}, and the last fifth of training added at most {max(late_gain(s) for s in SETS):.3f}.

## Scope

This is a scout, with no preregistration and no gate. Each op set has one run, all from the same model seed, so a difference between two runs is only a hint. For scale, ex-2.2.17 trained this recipe on the full op set at three seeds, and their gaps to the ceiling spanned {min(YARD_GAP):.3f} to {max(YARD_GAP):.3f}.

## Why

The in-context grammar asks the model to infer the op from three examples, and some pairs of ops give the same answer on many operand pairs. An example that fits `lighten` often fits `screen` too, so those examples say less about the op, and the model has two nearly interchangeable answers to choose between. If the pairs are part of why the control falls short of its ceiling, a smaller op set might make a better grammar for the anchoring experiments: more of the ceiling reachable, for the same training.

## The runs

Every run uses the recipe that ex-2.2.17 settled on: the unanchored d64-L4 model with an untied readout and the newline mask, a cosine schedule at a peak learning rate of {ex.PEAK_LR:g} after a warm-up of {ex.WARMUP_EPOCHS:g} epochs, for {ex.EPOCHS} epochs ({STEPS:,.0f} steps). The corpus condition is `k3-r0.3`, three examples and a replacement rate of 0.3, with {ex.ex2216.N_LINES:,} contexts. Each op set gets its own corpus, holdout, and probe set, drawn from its own ops.

"""

table_html(
    ["op set", "ops dropped", "partner kept", "ops"],
    [
        [
            f"`{s}`",
            ", ".join(f"`{o}`" for o in RUNS[s]["dropped"]) or "none",
            ", ".join(f"`{ex.PARTNER[o]}`" for o in RUNS[s]["dropped"]) or "",
            str(len(ops_of(s))),
        ]
        for s in SETS
    ],
    "**The op sets.** Each dropped op has a partner, the op ex-2.2.17 found its answers most often coincide with.",
    text_cols=3,
)

r"""

## The measurements

Held-out *expected exact match* (EEM) is the probability that an answer drawn from the distribution of the model at the query `=` is a correct answer of the true op. The *Bayes ceiling* is the same score for an ideal predictor, which weighs each op by how well it explains the examples and answers with the resulting mixture. The *floor* is the score of a predictor that ignores the examples. Both are computed for each op set on its own ops, as in ex-2.2.16.

The *gap* is the ceiling minus EEM, and *skill* is how far a run got from the floor to the ceiling, as a share of that distance. Gaps compare runs whose ceilings differ, which EEM alone cannot do.

## Scores against each ceiling (E1)

The figure below puts each run beside the ceiling and floor of its op set.

"""

scores_draw(
    f"""
        Two charts over six op sets. Left: held-out expected exact match, with a ceiling mark and a floor mark per op
        set. The ceilings run from {score(FULL, "ceiling"):.2f} for the full set to {score(FOUR, "ceiling"):.2f} for
        no-four, and the model dots sit a little under each ceiling, except no-multiply, which sits further below.
        The full-set dot at {score(FULL):.3f} falls inside the shaded band of ex-2.2.17's three seeds. Right: the gap
        to the ceiling as bars, between {gap(FOUR):.3f} (no-four) and {gap("no-multiply"):.3f} (no-multiply), with the
        ex-2.2.17 band from {min(YARD_GAP):.3f} to {max(YARD_GAP):.3f}.
    """,
    """
        **Scores against each ceiling.** Left: held-out expected exact match of each run (dots), with the Bayes ceiling
        (dark) and floor (light) of its op set. Right: the gap between them. The shaded band on both is the range of
        ex-2.2.17's three seeds on the full op set.
    """,
)

rf"""

The full-set run scored {score(FULL):.3f}, inside the range of ex-2.2.17's seeds ({min(YARD_EEM):.3f} to {max(YARD_EEM):.3f}), so the recipe reproduced. Each single drop raised the ceiling by 0.02 to 0.03, and in three of the four the model rose with it, leaving gaps between {min(gap(s) for s in ("no-screen", "no-hsvmix")):.3f} and {gap("no-exclusion"):.3f}. The run without `multiply` scored lower than the full set, {gap("no-multiply"):.3f} below its ceiling; E2 shows where.

Dropping all four raised the ceiling to {score(FOUR, "ceiling"):.3f}, and the model reached {score(FOUR):.3f}, a skill of {skill(FOUR):.2f} against {skill(FULL):.2f} for the full set. Most of the rise in the ceiling is because the four dropped ops round stochastically, so even a predictor told the op cannot always name their answer, as it can for their partners `lighten`, `darken`, and `difference`: that predictor scores {STATS[FULL]["told_op"]:.3f} on the full set and {STATS[FOUR]["told_op"]:.3f} without the four. The rest comes from examples that point more clearly to one op.

## The HSV-channel ops (E2)

The heatmap below breaks each gap down by op.

"""

gaps_draw(
    f"""
        A heatmap with the eleven ops as rows and the six op sets as columns, each square the gap between the Bayes
        ceiling and the model on that op, with dropped ops marked. Most squares sit between 0.03 and 0.15. The
        no-multiply column is bright on hue-hsv, sat-hsv, and value-hsv ({gap("no-multiply", "hue-hsv"):.2f},
        {gap("no-multiply", "sat-hsv"):.2f}, {gap("no-multiply", "value-hsv"):.2f}), and the no-exclusion column on
        sat-hsv and value-hsv ({gap("no-exclusion", "sat-hsv"):.2f}, {gap("no-exclusion", "value-hsv"):.2f}). The
        no-four column is the darkest, with every op at or below {max(gap(FOUR, o) for o in ops_of(FOUR)):.2f}.
    """,
    """
        **The gap by op.** Each square is the Bayes ceiling minus held-out expected exact match, for one op in one run.
        Brighter is further from the ceiling.
    """,
)

rf"""

Two columns stand out. Without `multiply`, the model got {gap("no-multiply", "sat-hsv"):.2f} and {gap("no-multiply", "value-hsv"):.2f} short of the ceiling on `sat-hsv` and `value-hsv`, where the full-set run was {gap(FULL, "sat-hsv"):.2f} and {gap(FULL, "value-hsv"):.2f} short; without `exclusion`, the same two ops fell short by about twice as much as in the full set. The three HSV-channel ops are blend modes that take one of hue, saturation, or value from one operand and the other two from the other. Neither `multiply` nor `exclusion` gives answers like theirs, so similarity doesn't explain this.

The learning curves suggest timing. Ex-2.2.17 saw these ops rise steeply partway through training, and in this scout the rise came at different times in different runs. At step {TRAJ[FULL]["step"][EARLY]:,.0f}, the mean probe EEM on the three ops was {hsv_at(FULL, EARLY):.2f} in the full-set run, {hsv_at("no-hsvmix", EARLY):.2f} and {hsv_at(FOUR, EARLY):.2f} in the two runs without `hsvmix`, and {hsv_at("no-multiply", EARLY):.2f} without `multiply`. The run without `exclusion` caught up late, and the run without `multiply` had not caught up when the schedule ended. With one seed per op set, we can't tell whether dropping `multiply` makes those ops harder to learn or this run was a slow one. The runs without `hsvmix` starting early is a clearer pattern, since two runs share it.

Away from the HSV-channel ops, the no-four column is lower than the full set on nearly every op. The partners of the dropped ops gained most: `lighten` went from {gap(FULL, "lighten"):.2f} to {gap(FOUR, "lighten"):.2f}, `darken` from {gap(FULL, "darken"):.2f} to {gap(FOUR, "darken"):.2f}, and `difference` from {gap(FULL, "difference"):.2f} to {gap(FOUR, "difference"):.2f}.

## The leak onto a dropped op (E3)

Ex-2.2.17 found the model keeping mass on the answers of a similar op even where the examples settle the op. For each partner op, we took the contexts whose posterior on it is above {CONFIDENT:g} and measured the mass the model puts on colors the dropped op can give and the partner cannot. The Bayes predictive puts almost none there.

"""

table_html(
    ["partner", "dropped op", *(f"`{s}`" for s in (FULL, *SINGLES, FOUR))],
    [
        [f"`{ex.PARTNER[d]}`", f"`{d}`", *(f"{LEAK[s][d]['leak']:.3f}" for s in (FULL, *SINGLES, FOUR))]
        for d in ex.PARTNER
    ],
    f"""
        **The leak onto the dropped op.** Mean mass on colors the dropped op gives and its partner does not, on
        confident contexts of the partner, in each run. The number of contexts per entry runs from
        {min(LEAK[s][d]["n"] for s in SETS for d in LEAK[s])} to {max(LEAK[s][d]["n"] for s in SETS for d in LEAK[s])}.
    """,
    text_cols=2,
)

rf"""

In the full-set run, confident `lighten` contexts put {LEAK[FULL]["screen"]["leak"]:.2f} of their mass on colors only `screen` gives, and the other three pairs {min(LEAK[FULL][d]["leak"] for d in ("multiply", "hsvmix", "exclusion")):.2f} to {max(LEAK[FULL][d]["leak"] for d in ("multiply", "hsvmix", "exclusion")):.2f}. Dropping an op took most of its leak with it: `screen` fell to {LEAK["no-screen"]["screen"]["leak"]:.3f} in the run without it, and all four were at or below {max(LEAK[FOUR][d]["leak"] for d in ex.PARTNER):.3f} in the run without any of them. That is the expected result, since a model never trained on `screen` has no reason to give its answers. It confirms that most of this leak was the pairing, and it also means more contexts count as confident: without `screen`, {LEAK["no-screen"]["screen"]["n"]} held-out `lighten` contexts are confident, against {LEAK[FULL]["screen"]["n"]} in the full set.

## Training time (E4)

Later rounds will look for a cheaper recipe, so we logged how skill on the probe set grew through training. The figure below shows each run.

"""

traj_draw(
    f"""
        A line chart of skill on the probe set against training step, one line per op set, with a dashed line at 1
        for the ceiling. Every line rises quickly at first and levels off in the last fifth of training. No-four ends
        highest, near {probe_skill(FOUR)[-1]:.2f}, and no-multiply lowest, near {probe_skill("no-multiply")[-1]:.2f}.
    """,
    """
        **Skill through training.** Skill is how far a run got from the floor to the ceiling of its op set, measured
        on a probe set of 200 contexts per op every 2% of training.
    """,
)

r"""

The table below gives the step at which each run passed 90% and 95% of its final skill.

"""

table_html(
    ["op set", "final probe skill", "step at 90%", "step at 95%", "EEM gained in the last fifth"],
    [
        [
            f"`{s}`",
            f"{probe_skill(s)[-1]:.2f}",
            f"{steps_to(s, 0.9):,}",
            f"{steps_to(s, 0.95):,}",
            f"{late_gain(s):.3f}",
        ]
        for s in SETS
    ],
    f"""
        **Training time.** The first logged step at which probe skill reached 90% and 95% of its final value, out of
        {STEPS:,.0f}, and the probe expected exact match gained over the last fifth of training.
    """,
)

rf"""

Every run reached 90% of its final skill between step {min(steps_to(s, 0.9) for s in SETS):,} and {max(steps_to(s, 0.9) for s in SETS):,}, and the last fifth added between {min(late_gain(s) for s in SETS):.3f} and {max(late_gain(s) for s in SETS):.3f} of EEM. These are on the cosine schedule, where the rate falls to 1% of its peak by the end, so a shorter run would anneal sooner too and might not lose even that much. Ex-2.2.17 saw the same shape.

## What we make of it

The op set without all four ops is easier in two ways: its ceiling is higher, and the model gets closer to it, at a skill of {skill(FOUR):.2f} against {skill(FULL):.2f} for the full set. Much of the higher ceiling comes from dropping ops that round stochastically, and part of the smaller gap probably comes from the leak onto a similar op going away. Both seem like good properties for the grammar the anchoring experiments train on, with seven ops that are easier to tell apart. It would also mean a new ceiling for every D2.2 comparison so far.

The single drops say less. Dropping one op moved the ceiling and the model by similar amounts, except for `multiply` and `exclusion`, whose runs fell short on the HSV-channel ops; with one seed, that may be timing. The more consistent sign is that both runs without `hsvmix` learned the HSV-channel ops early, which fits ex-2.2.17 finding `hsvmix` the hardest op to compute.

## Method

**Corpora.** Each op set has its own corpus of 300,000 contexts, its own holdout of {ex.ex2216.HOLDOUT_CONTEXTS:,} contexts per op, and its own probe set of 200 per op, drawn with the sampler of ex-2.2.16 from the answer table restricted to its ops. The posterior, ceiling, and floor are ex-2.2.16's, computed on the same restricted table.

**Leak.** The answers a dropped op gives come from the full eleven-op answer table, so they are defined in runs that never trained on it. Confident contexts are chosen on the posterior of the op set of the run, so the set of contexts differs between runs.

**Cost.** The scout cost about \$1.79 on Modal, \$1.70 of it L4 time: about \$0.28 per training run of about 40 minutes.
"""
