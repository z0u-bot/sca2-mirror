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
# Each dropped op and the op whose answers it most often shares, from either side of a pair.
PAIR_OF: dict[str, str] = ex.PARTNER | ex.COUNTERPART
SINGLES: tuple[str, ...] = tuple(f"no-{op}" for op in PAIR_OF)
FOURS: tuple[str, ...] = ("no-four", "no-four-ld")
SETS: tuple[str, ...] = ("full", *SINGLES, *FOURS)
assert set(SETS) == {s.name for s in ex.OP_SETS}
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
    for dropped, partner in PAIR_OF.items():
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
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), layout="constrained")
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
        fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
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
            ax.plot(np.array(t["step"]) / 1e3, y, color=f"C{i}", lw=2 if s == "full" or s in FOURS else 1.1, label=s)
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


def four_of(dropped: str) -> str:
    """The four-op set that drops *dropped*."""
    return next(s for s in FOURS if dropped in RUNS[s]["dropped"])


STEPS = TRAJ[FULL]["step"][-1]
EARLY = 10

LD = "no-four-ld"


def hsv_gap(s: str) -> float:
    """The mean gap over the three HSV-channel ops."""
    return float(np.mean([gap(s, op) for op in HSV_CHANNEL]))


# A run falls short on the HSV-channel ops when their mean gap is more than 0.1, against 0.06 in the full-set run.
HSV_SHORT = tuple(s for s in SETS if hsv_gap(s) > 0.1)
NO_HSVMIX = tuple(s for s in SETS if "hsvmix" not in ops_of(s))
WITH_HSVMIX = tuple(s for s in SETS if "hsvmix" in ops_of(s))


WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def listed(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f", and {items[-1]}"


def names(sets: Sequence[str]) -> str:
    return listed(f"`{s}`" for s in sets)


rf"""
# Ex 2.2.18: dropping ops with similar answers, a scout

/// tip |
<!-- tl;dr -->
Dropping `screen`, `multiply`, `hsvmix`, and `exclusion` together made the in-context grammar easier to solve, and the model got closer to what is solvable than in any run so far. Most of that came from dropping ops whose answers round at random. Dropping `lighten` and `darken` in place of `screen` and `multiply` breaks the same pairs, and it left the ceiling where it was and narrowed the gap by less. Each op set has one seed.
///

Ex-2.2.17 found that its center control, where the examples settle the op, keeps part of its mass on the answers of the op most like the true one: `lighten` onto `screen`, `darken` onto `multiply`, `mix` and `hsvmix` onto each other, and `difference` onto `exclusion`. This scout drops one op of each of those pairs, one at a time and all four together. A second round drops the other side of the two brightening pairs, `lighten` and `darken`, alone and in place of `screen` and `multiply` in the four-op drop. Each op set trains the ex-2.2.17 recipe once.

Dropping ops changes the task, so every op set has its own Bayes ceiling and floor, and we score each run against its own.

## Observations

Each line is a measurement on the runs of this scout, with no gate.

- **E1** [Scores against each ceiling](#scores-against-each-ceiling-e1): dropping `lighten` or `darken` lowered the ceiling, and dropping any of the other four raised it. Without `screen`, `multiply`, `hsvmix`, and `exclusion`, the model came within {gap(FOUR):.3f} of its ceiling, closer than any run so far. The second four-op drop came within {gap(LD):.3f} of a ceiling near that of the full set.
- **E2** [The HSV-channel ops](#the-hsv-channel-ops-e2): {WORDS[len(HSV_SHORT)]} of the {WORDS[len(SINGLES)]} single drops fell short on the three HSV-channel ops, and the {WORDS[len(NO_HSVMIX)]} runs without `hsvmix` learned those ops earlier than the others did.
- **E3** [The leak onto a dropped op](#the-leak-onto-a-dropped-op-e3): on contexts of a partner op, the mass the model put on the answers of the dropped op mostly went away with it.
- **E4** [Training time](#training-time-e4): every run reached 95% of its final skill by step {max(steps_to(s, 0.95) for s in SETS):,} of {STEPS:,.0f}, and the last fifth of training added at most {max(late_gain(s) for s in SETS):.3f}.

## Scope

This is a scout, with no preregistration and no gate. Each op set has one run, all from the same model seed, so a difference between two runs is only a hint. For scale, ex-2.2.17 trained this recipe on the full op set at three seeds, and their gaps to the ceiling spanned {min(YARD_GAP):.3f} to {max(YARD_GAP):.3f}.

## Why

The in-context grammar asks the model to infer the op from three examples, and some pairs of ops give the same answer on many operand pairs. An example that fits `lighten` often fits `screen` too, so those examples say less about the op, and the model has two nearly interchangeable answers to choose between. If the pairs are part of why the control falls short of its ceiling, a smaller op set might make a better grammar for the anchoring experiments: more of the ceiling reachable, for the same training.

Which side of a pair to drop matters too. `lighten`, `darken`, and `difference` give one answer for each pair of operands, while their partners round each channel at random between grid levels, so their answers are spread over a few colors. Dropping the spread-out side raises the ceiling for a reason that has nothing to do with similarity, and dropping the other side separates the two.

## The runs

Every run uses the recipe that ex-2.2.17 settled on: the unanchored d64-L4 model with an untied readout and the newline mask, a cosine schedule at a peak learning rate of {ex.PEAK_LR:g} after a warm-up of {ex.WARMUP_EPOCHS:g} epochs, for {ex.EPOCHS} epochs ({STEPS:,.0f} steps). The corpus condition is `k3-r0.3`, three examples and a replacement rate of 0.3, with {ex.ex2216.N_LINES:,} contexts. Each op set gets its own corpus, holdout, and probe set, drawn from its own ops.

"""

table_html(
    ["op set", "ops dropped", "partner kept", "ops"],
    [
        [
            f"`{s}`",
            ", ".join(f"`{o}`" for o in RUNS[s]["dropped"]) or "none",
            ", ".join(f"`{PAIR_OF[o]}`" for o in RUNS[s]["dropped"]) or "",
            str(len(ops_of(s))),
        ]
        for s in SETS
    ],
    """
        **The op sets.** Each dropped op has a partner, the op ex-2.2.17 found its answers most often coincide with.
        The `no-lighten`, `no-darken`, and `no-four-ld` sets were a second round.
    """,
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
        Two charts over nine op sets. Left: held-out expected exact match, with a ceiling mark and a floor mark per op
        set. The ceilings run from {min(score(s, "ceiling") for s in SETS):.2f} (no-darken) to
        {score(FOUR, "ceiling"):.2f} (no-four), and the model dots sit a little under each ceiling, further below for
        no-multiply and no-darken. The full-set dot at {score(FULL):.3f} falls inside the shaded band of ex-2.2.17's
        three seeds. Right: the gap to the ceiling as bars, between {gap(FOUR):.3f} (no-four) and
        {gap("no-multiply"):.3f} (no-multiply), with the ex-2.2.17 band from {min(YARD_GAP):.3f} to
        {max(YARD_GAP):.3f}; no-four and no-four-ld are the two shortest bars.
    """,
    """
        **Scores against each ceiling.** Left: held-out expected exact match of each run (dots), with the Bayes ceiling
        (dark) and floor (light) of its op set. Right: the gap between them. The shaded band on both is the range of
        ex-2.2.17's three seeds on the full op set.
    """,
)

rf"""

The full-set run scored {score(FULL):.3f}, inside the range of ex-2.2.17's seeds ({min(YARD_EEM):.3f} to {max(YARD_EEM):.3f}), so the recipe reproduced.

Dropping one of `screen`, `multiply`, `hsvmix`, or `exclusion` raised the ceiling by 0.02 to 0.03, and dropping `lighten` or `darken` lowered it by about 0.015. That follows from which side rounds at random. A predictor told the op, with no inference to do, scores {STATS[FULL]["told_op"]:.3f} on the full set: it can't always name the answer of an op that rounds at random, and it always can for `lighten` or `darken`. Without `screen` it scores {STATS["no-screen"]["told_op"]:.3f}, and without `lighten` {STATS["no-lighten"]["told_op"]:.3f}.

The gaps of the single drops mostly stayed near the gap of the full set, {gap(FULL):.3f}. The exceptions were `no-multiply` at {gap("no-multiply"):.3f} and `no-darken` at {gap("no-darken"):.3f}, the two sides of the darkening pair, and `no-exclusion` at {gap("no-exclusion"):.3f}; E2 shows where those fell short.

The two four-op drops break the same four pairs. Without `screen`, `multiply`, `hsvmix`, and `exclusion`, the ceiling rose to {score(FOUR, "ceiling"):.3f} and the model reached {score(FOUR):.3f}, a gap of {gap(FOUR):.3f}. Without `lighten`, `darken`, `hsvmix`, and `exclusion`, the ceiling stayed near that of the full set, at {score(LD, "ceiling"):.3f}, and the model reached {score(LD):.3f}, a gap of {gap(LD):.3f}. So most of the rise in EEM from `no-four` came from its higher ceiling. Both four-op drops narrowed the gap, by {gap(FULL) - gap(FOUR):.3f} and {gap(FULL) - gap(LD):.3f}; with one run each, and ex-2.2.17 seeds spanning {max(YARD_GAP) - min(YARD_GAP):.3f}, that part is a hint.

## The HSV-channel ops (E2)

The heatmap below breaks each gap down by op.

"""

gaps_draw(
    f"""
        A heatmap with the eleven ops as rows and the nine op sets as columns, each square the gap between the Bayes
        ceiling and the model on that op, with dropped ops marked. Most squares sit between 0.03 and 0.15. The
        no-multiply, no-darken, and no-exclusion columns are bright on hue-hsv, sat-hsv, and value-hsv, up to
        {max(gap(s, op) for s in HSV_SHORT for op in HSV_CHANNEL):.2f}. The no-four and no-four-ld columns are the
        darkest.
    """,
    """
        **The gap by op.** Each square is the Bayes ceiling minus held-out expected exact match, for one op in one run.
        Brighter is further from the ceiling.
    """,
)

rf"""

Three single drops fell short on the three HSV-channel ops: {names(HSV_SHORT)}, with a mean gap on those ops of {listed(f"{hsv_gap(s):.2f}" for s in HSV_SHORT)}, against {hsv_gap(FULL):.2f} in the full-set run. The HSV-channel ops are blend modes that take one of hue, saturation, or value from one operand and the other two from the other. None of the dropped ops gives answers like theirs, so similarity doesn't explain this.

The learning curves suggest timing. Ex-2.2.17 saw these ops rise steeply partway through training, and in this scout the rise came at different times in different runs. At step {TRAJ[FULL]["step"][EARLY]:,.0f}, the mean probe EEM on the three ops was {listed(f"{hsv_at(s, EARLY):.2f}" for s in NO_HSVMIX)} in the {WORDS[len(NO_HSVMIX)]} runs without `hsvmix`, and between {min(hsv_at(s, EARLY) for s in WITH_HSVMIX):.2f} and {max(hsv_at(s, EARLY) for s in WITH_HSVMIX):.2f} in the others. The runs that fell short had not caught up when the schedule ended.

With one seed per op set, we can't tell whether dropping `multiply`, `darken`, or `exclusion` makes those ops harder to learn, or these runs were slow ones. Dropping an op also changes the corpus, since the other ops share its contexts, so each gets a little more training. The early start without `hsvmix` is firmer: three runs share it, though two of them also drop three other ops.

Away from the HSV-channel ops, both four-op columns are lower than the full set on nearly every op. `difference` went from {gap(FULL, "difference"):.2f} to {gap(FOUR, "difference"):.2f} and {gap(LD, "difference"):.2f}, and `mix` from {gap(FULL, "mix"):.2f} to {gap(FOUR, "mix"):.2f} and {gap(LD, "mix"):.2f}. In `no-four`, `lighten` and `darken` went from {gap(FULL, "lighten"):.2f} and {gap(FULL, "darken"):.2f} to {gap(FOUR, "lighten"):.2f} and {gap(FOUR, "darken"):.2f}.

## The leak onto a dropped op (E3)

Ex-2.2.17 found the model keeping mass on the answers of a similar op even where the examples settle the op. For each dropped op, we took the contexts whose posterior on its partner is above {CONFIDENT:g} and measured the mass the model puts on colors the dropped op can give and the partner cannot. The Bayes predictive puts almost none there.

"""

table_html(
    ["contexts of", "dropped op", "full set", "dropped alone", "dropped in its four-op set"],
    [
        [
            f"`{PAIR_OF[d]}`",
            f"`{d}`",
            *(f"{LEAK[s][d]['leak']:.3f} ({LEAK[s][d]['n']})" for s in (FULL, f"no-{d}", four_of(d))),
        ]
        for d in PAIR_OF
    ],
    """
        **The leak onto the dropped op.** Mean mass on colors the dropped op gives and its partner does not, on
        confident contexts of the partner, with the number of those contexts in brackets.
    """,
    text_cols=2,
)

rf"""

In the full-set run, confident `lighten` contexts put {LEAK[FULL]["screen"]["leak"]:.2f} of their mass on colors only `screen` gives, and the other three pairs {min(LEAK[FULL][d]["leak"] for d in ("multiply", "hsvmix", "exclusion")):.2f} to {max(LEAK[FULL][d]["leak"] for d in ("multiply", "hsvmix", "exclusion")):.2f}. The leak the other way is smaller: confident `screen` contexts put {LEAK[FULL]["lighten"]["leak"]:.3f} on colors only `lighten` gives. Part of that is in how the measurement counts. `lighten` gives one color, often one of the colors `screen` gives, so there are fewer colors for it to leak onto.

Dropping an op took most of its leak with it: every entry of the last two columns is at or below {max(LEAK[s][d]["leak"] for d in PAIR_OF for s in (f"no-{d}", four_of(d))):.3f}. That is the expected result, since a model never trained on `screen` has no reason to give its answers. It confirms that most of this leak was the pairing, and it also means more contexts count as confident: without `screen`, {LEAK["no-screen"]["screen"]["n"]} held-out `lighten` contexts are confident, against {LEAK[FULL]["screen"]["n"]} in the full set.

## Training time (E4)

To see whether a shorter schedule could suffice, we logged how skill on the probe set grew through training. The figure below shows each run.

"""

traj_draw(
    f"""
        A line chart of skill on the probe set against training step, one line per op set, with a dashed line at 1
        for the ceiling. Every line rises quickly at first and levels off in the last fifth of training. No-four ends
        highest, near {probe_skill(FOUR)[-1]:.2f}, and no-multiply lowest, near {probe_skill("no-multiply")[-1]:.2f}.
    """,
    """
        **Skill through training.** Skill is how far a run got from the floor to the ceiling of its op set, measured
        on a probe set of 200 contexts per op every 2% of training. The full set and the two four-op sets are drawn
        heavier.
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

The op set without `screen`, `multiply`, `hsvmix`, and `exclusion` is easier in two ways: its ceiling is higher, and the model gets closer to it, at a skill of {skill(FOUR):.2f} against {skill(FULL):.2f} for the full set. The second four-op drop suggests the two ways have different causes. The higher ceiling comes from dropping ops that round at random. The smaller gap came with both four-op drops, so breaking up the similar pairs may account for it, though one run each is not enough to be sure. Dropping `screen` and `multiply` rather than `lighten` and `darken` gets both, and leaves seven ops that are easier to tell apart. It would also mean a new ceiling for every D2.2 comparison so far.

The single drops say less. Dropping one op mostly moved the ceiling and the model together, except in the three runs that fell short on the HSV-channel ops; with one seed, that may be timing. The more consistent sign is that all three runs without `hsvmix` learned the HSV-channel ops early, which fits ex-2.2.17 finding `hsvmix` the hardest op to compute.

## Method

**Corpora.** Each op set has its own corpus of 300,000 contexts, its own holdout of {ex.ex2216.HOLDOUT_CONTEXTS:,} contexts per op, and its own probe set of 200 per op, drawn with the sampler of ex-2.2.16 from the answer table restricted to its ops. The posterior, ceiling, and floor are ex-2.2.16's, computed on the same restricted table.

**Leak.** The answers a dropped op gives come from the full eleven-op answer table, so they are defined in runs that never trained on it. Confident contexts are chosen on the posterior of the op set of the run, so the set of contexts differs between runs.

**Cost.** The scout cost about \$2.42 on Modal, \$2.30 of it L4 time: about \$0.26 per training run. Each training run took 14 to 17 minutes on one L4, about 7,000 steps a minute.
"""
