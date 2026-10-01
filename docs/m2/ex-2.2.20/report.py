# title: Ex 2.2.20: a high-rate head start before the recipe schedule

# The design constants come from `experiment.py` beside this script. The runs this draft builds on are published
# already: the 200- and 400-epoch `no-four` runs of ex-2.2.19, the 400-epoch `no-four` run of ex-2.2.18 (the
# 400-epoch run at seed 600), and the three ex-2.2.17 seeds of the recipe on the full op set, which set the per-op
# tolerances.
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, themed

X19 = ex.ex2219
FOUR = ex.OP_SET.name
OPS = ex.OP_SET.ops
HSV_CHANNEL = ("hue-hsv", "sat-hsv", "value-hsv")
REF_E = ex.REFERENCE_EPOCHS
E = ex.EPOCHS
HS = ex.HEAD_START_EPOCHS


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


def span(values: Sequence[float], fmt: str = ".2f") -> str:
    """The range of *values*, or the one value when they all round the same."""
    lo, hi = format(min(values), fmt), format(max(values), fmt)
    return lo if lo == hi else f"{lo} and {hi}"


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
    _refs = [X19.EVAL_REF, X19.TRAJ_REF, ex.ex2218.EVAL_REF, ex.ex2218.TRAJ_REF, ex.ex2217.EVAL_REF]
    _files = fetch(_refs, Path(_tmp))
    EVAL_19 = read_json(_files[X19.EVAL_REF])
    _eval18 = read_json(_files[ex.ex2218.EVAL_REF])
    _ref18 = next(r for r in _eval18["runs"] if r["label"] == FOUR)
    PRIOR = {r["label"]: r for r in read_json(_files[ex.ex2217.EVAL_REF])["runs"] if r["label"] in ex.YARDSTICK}
    # The ex-2.2.18 `no-four` run is the 400-epoch run at seed 600, so it joins the runs of ex-2.2.19.
    REF0 = X19.label_of(REF_E, ex.LO_LR, 0)
    RUNS_19: dict[str, dict] = {r["label"]: r for r in EVAL_19["runs"]} | {REF0: _ref18}
    TRAJ_19: dict[str, dict] = read_json(_files[X19.TRAJ_REF]) | {REF0: read_json(_files[ex.ex2218.TRAJ_REF])[FOUR]}

STATS = EVAL_19["op_set"]
CEILING = STATS["ceiling"]
OP_TOL = ex.op_tolerances([PRIOR[s] for s in ex.YARDSTICK], OPS)


def plain(epochs: int, seed: int) -> str:
    """The label of the ex-2.2.19 run at *epochs* on the recipe schedule."""
    return X19.label_of(epochs, ex.LO_LR, seed)


# The 50-epoch scout run of ex-2.2.19 at the higher peak rate, which the first cycle repeats.
HI50 = X19.label_of(HS, ex.HI_LR, 0)


def eem(label: str, op: str | None = None) -> float:
    r = RUNS_19[label]["task"]["eem"]
    return r["all"] if op is None else r["per_op"][list(RUNS_19[label].get("ops", OPS)).index(op)]


def plain_shortfall(seed: int) -> float:
    return eem(plain(REF_E, seed)) - eem(plain(E, seed))


PLAIN_SHORT = [plain_shortfall(s) for s in ex.GATE_SEEDS]


def traj_skill(label: str) -> np.ndarray:
    """Skill on the probe set along a run: the row for all ops, then one per op; shape (op, point)."""
    t = TRAJ_19[label]
    e = np.array(t["eem_per_op"]).T
    f, c = np.array(STATS["floor_per_op"])[:, None], np.array(STATS["ceiling_per_op"])[:, None]
    total = (np.array(t["eem"]) - STATS["floor"]) / (CEILING - STATS["floor"])
    return np.vstack([total, (e - f) / (c - f)])


def skill_near(label: str, epoch: float) -> float:
    """Skill over all ops at the logged point nearest *epoch*."""
    ep = np.array(TRAJ_19[label]["epoch"])
    return float(traj_skill(label)[0, int(np.argmin(abs(ep - epoch)))])


HSV_HALF = 0.5


def hsv_rise(label: str) -> tuple[float, float]:
    """The first logged epoch at which the mean HSV-channel skill passes *HSV_HALF*, and the learning rate there."""
    curve = traj_skill(label)[[1 + OPS.index(op) for op in HSV_CHANNEL]].mean(0)
    i = int(np.argmax(curve >= HSV_HALF))
    assert curve[i] >= HSV_HALF, f"{label} never passes {HSV_HALF}"
    return float(TRAJ_19[label]["epoch"][i]), float(TRAJ_19[label]["lr"][i])


RISE_200 = {s: hsv_rise(plain(E, s)) for s in ex.SEEDS}
RISE_400 = {s: hsv_rise(plain(REF_E, s)) for s in ex.SEEDS}
RISE_HI200 = hsv_rise(X19.label_of(E, ex.HI_LR, 0))
assert min(r for _, r in RISE_400.values()) > max(r for _, r in RISE_200.values()), "H3 says the rates do not overlap"


def hsv_best(label: str) -> float:
    return float(traj_skill(label)[[1 + OPS.index(op) for op in HSV_CHANNEL]].mean(0).max())


# The short scout runs of ex-2.2.19 annealed to their floor without the HSV-channel ops passing HSV_HALF.
SHORT_BEST = max(hsv_best(X19.label_of(e, lr, 0)) for e in (HS, 2 * HS) for lr in (ex.LO_LR, ex.HI_LR))
assert SHORT_BEST < HSV_HALF
LONGER_AT_50 = [skill_near(plain(e, s), HS) for e in (E, REF_E) for s in ex.SEEDS]
assert traj_skill(HI50)[0, -1] > max(LONGER_AT_50), "the Why section says the head start is ahead at epoch 50"

# --- The schedules ------------------------------------------------------------------------------------------

_t = TRAJ_19[plain(E, 0)]
EPOCH_LENGTH = round(_t["step"][-1] / _t["epoch"][-1])
CHECK = ex.schedule_check(EPOCH_LENGTH)
assert all(ex.schedule_check(EPOCH_LENGTH, p)["second"] < 1e-5 for p in ex.SECOND_PEAKS), (
    "the second cycle is the recipe schedule for its length and peak"
)
BAND = ex.BAND_LR


def schedule_curve(epochs: int, peak: float, sheet: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Epochs and learning rate along a run of *epochs*, from the training code itself."""
    from sca.config import SchedulerConfig
    from sca.training.scheduler import configure_schedule

    config = SchedulerConfig(
        epochs=epochs, warmup_epochs=ex.WARMUP_EPOCHS, min_lr_factor=ex.MIN_LR_FACTOR, lr_sheet=sheet
    )
    steps = np.arange(0, epochs * EPOCH_LENGTH + 1, EPOCH_LENGTH // 8)
    return steps / EPOCH_LENGTH, np.asarray(configure_schedule(config, peak, EPOCH_LENGTH)(steps))


# Where the first cosine falls below every rate at which a plain run learned the HSV-channel ops.
_ep, _lr = schedule_curve(E, ex.HI_LR, ex.lr_sheet())
_falling = (_ep > ex.WARMUP_EPOCHS) & (_ep < HS)
FIRST_PASS = float(
    _ep[_falling][np.argmax(_lr[_falling] <= min(r for _, r in [*RISE_200.values(), *RISE_400.values()]))]
)


@memo
def schedule_draw(alt_text: str, caption: str) -> str:
    @themed(name="schedule", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 3.0), layout="constrained")
        for (epochs, peak, sheet, name), color in zip(
            [
                (REF_E, ex.LO_LR, None, f"{REF_E} epochs"),
                (E, ex.LO_LR, None, f"{E} epochs"),
                (E, ex.HI_LR, ex.lr_sheet(), "head start"),
                (E, ex.HI_LR, ex.lr_sheet(BAND), f"head start, second peak {BAND:g}"),
            ],
            ("C7", "C0", "C1", "C3"),
            strict=True,
        ):
            x, y = schedule_curve(epochs, peak, sheet)
            ax.plot(x, y, color=color, lw=1.4, label=name)
        for rises, color in ((RISE_400, "C7"), (RISE_200, "C0")):
            ax.scatter(*zip(*rises.values(), strict=True), s=14, color=color, zorder=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.set_xlim(0, REF_E)
        fig.legend(loc="outside upper center", ncols=2, frameon=False, fontsize=8)
        return fig

    return _plot()


COST = len(ex.SECOND_PEAKS) * len(ex.SEEDS) * ex.cost_per_run(E)

rf"""

# Ex 2.2.20: a high-rate head start before the recipe schedule

/// tip |
<!-- tl;dr -->
We train the seven-op set for 200 epochs on a schedule of two cycles: a short one at a high learning rate, then the recipe schedule for the rest of the run, or the same with a lower second peak. We ask whether that keeps more of the skill of the 400-epoch recipe than a plain 200-epoch run does.
///

## Findings

- [The head start keeps most of the skill (H1)](#the-head-start-keeps-most-of-the-skill-h1) —
- [The head start beats the plain schedule (H2)](#the-head-start-beats-the-plain-schedule-h2) —
- [The HSV-channel ops come sooner (H3)](#the-hsv-channel-ops-come-sooner-h3) —
- [Where the first cycle leaves off (E1)](#where-the-first-cycle-leaves-off-e1) —
- [Calibration (E2)](#calibration-e2) —
- [Op confusion (E3)](#op-confusion-e3) —
- [A lower second peak (E4)](#a-lower-second-peak-e4) —

/// admonition | How to read this draft
This is a preregistration: the hypotheses, their gates, and the adoption rule were frozen at commit `2ea0b76`, before any run of this experiment. Each section opens with what we expect, and a `TODO` marks where its evidence will go.
///

## Why

Ex-2.2.19 adopted 200 epochs for the seven-op set (`no-four`), half the length of the recipe, as a partial pass. On three fresh seeds the 200-epoch runs fell short of the 400-epoch runs by {np.mean(PLAIN_SHORT):.4f} on average, against a gate of {ex.SHORTFALL_TOL}, and nearly all of the shortfall was in the three HSV-channel ops.

A short run at a high rate gets a long way quickly. The 50-epoch run of ex-2.2.19 at a peak learning rate of {ex.HI_LR:g} ended at a skill of {traj_skill(HI50)[0, -1]:.2f}, where the 200- and 400-epoch runs had reached between {min(LONGER_AT_50):.2f} and {max(LONGER_AT_50):.2f} by epoch {HS}. But it ends before the HSV-channel ops are learned, which in the 200-epoch runs happened between epochs {span([r[0] for r in RISE_200.values()], ".0f")}.

So we try both in one run: {HS} epochs on the schedule of that short run, then {E - HS} epochs on the recipe schedule from where it leaves off. If the second cycle builds on the first, the run should end closer to the 400-epoch runs than a plain 200-epoch run does, at the same cost.

## The runs

Each run follows the {E}-epoch recipe of ex-2.2.19 with a different learning-rate schedule. The first cycle warms up over {ex.WARMUP_EPOCHS:g} epochs to {ex.HI_LR:g} and follows a cosine down to 1% of {ex.LO_LR:g} at epoch {HS}. The second warms up from there over {ex.WARMUP_EPOCHS:g} epochs to {ex.LO_LR:g}, and follows a cosine down to 1% of that at epoch {E}.

The second cycle is the recipe schedule of a {E - HS}-epoch run, step for step.[^sheet] The first cycle is the schedule of the ex-2.2.19 run at {ex.HI_LR:g}, except that it ends where the second warmup starts. The optimizer state carries over between cycles, though resetting it should make no measurable difference.[^adam]

The schedule differs from the plain one in three ways at once: a higher peak for the first {HS} epochs, a restart, and a final anneal {E - HS} epochs long instead of {E}. A result would say whether the schedule helps, and leave open which of the three did it.

A second head-start condition is the same, except that its second cycle peaks at {BAND:g}, about the highest rate at which a plain {E}-epoch run of ex-2.2.19 learned the HSV-channel ops (H3). So it spends longer near those rates, and less time above them, than the first head start. From about epoch 80 on, its rate is close to that of the plain {E}-epoch run, so the two differ mostly in the first 80 epochs.

[^sheet]: The schedule is a dopesheet: keyframes at the start and peak of each warmup and at the end of each cosine, joined by straight lines going up and half cosines going down. At {EPOCH_LENGTH} steps an epoch, the second cycle matches the recipe schedule of a {E - HS}-epoch run to within {CHECK["second"]:.0e} of its peak.

[^adam]: Adam keeps running averages of the gradient and of its square. These decay with time constants of about 10 and 20 steps. The second warmup lasts {ex.WARMUP_EPOCHS * EPOCH_LENGTH:,g} steps, so a reset at the end of the first cycle would be forgotten early in it, while the rate is still near its floor.

"""

schedule_draw(
    f"""
        A line chart of learning rate against epoch from 0 to {REF_E}. The {REF_E}-epoch recipe warms up to
        {ex.LO_LR:g} and falls along a cosine to near zero at {REF_E}. The {E}-epoch recipe does the same by {E}. The
        head-start schedule rises to {ex.HI_LR:g}, falls to near zero by epoch {HS}, rises again to {ex.LO_LR:g} by
        epoch {HS + ex.WARMUP_EPOCHS:g}, and falls to near zero by {E}. The second head-start schedule is the same, with a
        second peak of {BAND:g}. A dot on each recipe curve marks, for each
        seed, where the HSV-channel ops were learned: on the {E}-epoch curve between epochs
        {span([r[0] for r in RISE_200.values()], ".0f")}, and on the {REF_E}-epoch curve between epochs
        {span([r[0] for r in RISE_400.values()], ".0f")}, where its rate is higher.
    """,
    f"""
        **The four schedules.** Learning rate against epoch, as the training code computes it. Each dot is the point
        at which the HSV-channel skill of a plain run of ex-2.2.19 first passed {HSV_HALF:g}, one per seed; seeds
        that passed at the same epoch overlap.
    """,
)

# %%
table_html(
    ["condition", "epochs", "schedule", "model seeds", "runs"],
    [
        [
            "head start (new)",
            f"{E}",
            f"{HS} at {ex.HI_LR:g}, then {E - HS} at {ex.LO_LR:g}",
            "600–603",
            f"{len(ex.SEEDS)}",
        ],
        [
            f"head start, second peak {BAND:g} (new)",
            f"{E}",
            f"{HS} at {ex.HI_LR:g}, then {E - HS} at {BAND:g}",
            "600–603",
            f"{len(ex.SEEDS)}",
        ],
        [f"plain {E} (ex-2.2.19)", f"{E}", f"{ex.LO_LR:g}", "600–603", "reused"],
        [f"plain {REF_E} (ex-2.2.18, ex-2.2.19)", f"{REF_E}", f"{ex.LO_LR:g}", "600–603", "reused"],
    ],
    """
        **The runs.** All train `no-four` from the same four initializations, so each head-start run pairs with a plain
        run of each length.
    """,
    text_cols=3,
)

rf"""

Held-out *expected exact match* (EEM) is the probability that an answer drawn from the model at the query `=` is a correct answer of the true op. The *Bayes ceiling* is the same score for an ideal predictor, and *skill* is EEM rescaled so that the floor (a uniform guess) is 0 and the ceiling is 1.

A run *falls short* of another by how much lower its EEM is, compared between runs that share a model seed and averaged over seeds. The hypotheses are scored on seeds {ex.SEED_OFFSET + min(ex.GATE_SEEDS)}–{ex.SEED_OFFSET + max(ex.GATE_SEEDS)}, as H1 of ex-2.2.19 was. Seed {ex.SEED_OFFSET} is shown beside them: the observation behind this experiment came from its runs, so it could flatter the head start.

## The head start keeps most of the skill (H1)

**What we expect.** The head-start runs fall short of the {REF_E}-epoch runs by at most {ex.SHORTFALL_TOL} on average, and no op falls short by more than its tolerance on average. That is a pass. A shortfall between {ex.SHORTFALL_TOL} and {ex.PARTIAL_TOL} is a partial pass, and one beyond {ex.PARTIAL_TOL}, or an op beyond its tolerance, is a miss. This is the H1 rule of ex-2.2.19, with the same tolerances.

/// admonition | TODO
The shortfall of the head-start and plain {E}-epoch runs from the {REF_E}-epoch runs, one dot per seed joined by seed, with a bar for the mean, the tolerance, and the partial band; and a table of the mean shortfall per op against each tolerance.
///

## The head start beats the plain schedule (H2)

**What we expect.** The head-start run scores a higher EEM than the plain {E}-epoch run at each of the three seeds, and so on average. That is a pass. A higher mean with a lower EEM at one or two seeds is a partial pass, and a mean gain at or below zero is a miss.

A miss would mean the second cycle kept nothing of the first, or lost it in the second warmup. H2 checks only the direction of the gain: the whole gap to the {REF_E}-epoch runs is small, so H1 covers its size.

**Which schedule we adopt.** H1 and H2 score the head start with the recipe peak, and E4 scores the one with a second peak of {BAND:g} by the same rules. If both hypotheses pass for either schedule, the {E}-epoch runs that follow will use that schedule. If both schedules pass, we take the one with the higher mean EEM over seeds {ex.SEED_OFFSET + min(ex.GATE_SEEDS)}–{ex.SEED_OFFSET + max(ex.GATE_SEEDS)}. Otherwise we keep the plain schedule.

/// admonition | TODO
The gain of the head-start run over the plain {E}-epoch run, overall and per op, one dot per seed with a bar for the mean.
///

## The HSV-channel ops come sooner (H3)

**What we expect.** In the plain {E}-epoch runs, the mean skill of the HSV-channel ops first passed {HSV_HALF:g} between epochs {span([r[0] for r in RISE_200.values()], ".0f")}. We expect the head-start runs to pass it about 25 epochs sooner, paired by seed (a gap estimated from the skill curves of ex-2.2.19).

The runs of ex-2.2.19 already suggest these ops do not wait for the learning rate to fall to some level. The {REF_E}-epoch runs passed {HSV_HALF:g} between epochs {span([r[0] for r in RISE_400.values()], ".0f")}, close to the {E}-epoch runs of the same seeds, but at rates between {span([r[1] for r in RISE_400.values()], ".4f")}, higher than any rate at which a {E}-epoch run passed it.

The {HS}- and {2 * HS}-epoch runs fell through those rates to their floor, and their HSV-channel skill stayed at or below {SHORT_BEST:.2f}. That includes the {HS}-epoch run at {ex.HI_LR:g}, the first cycle of the head-start schedule, which falls through all of those rates by about epoch {FIRST_PASS:.0f}.

So this schedule can separate elapsed time from learning progress. A rise at about the same epoch as in the plain run would say these ops wait for a number of epochs, whatever the model has learned by then. A sooner rise, as we expect, would say the progress of the first cycle carries over to them.

A later rise is possible too: the {E}-epoch run of ex-2.2.19 at {ex.HI_LR:g} passed {HSV_HALF:g} only at epoch {RISE_HI200[0]:.0f}, so a high rate early may hold these ops back. The outcomes lie further apart than the {E // ex.ex2218.N_TRAJ_POINTS}-epoch logging interval. There is no gate, since no decision hangs on it.

/// admonition | TODO
Skill against epoch for the head-start and plain {E}-epoch runs, one panel for all ops and one for each op, with the epoch at which the HSV-channel skill first passes {HSV_HALF:g} marked on each run; and a table of that epoch per seed.
///

## Where the first cycle leaves off (E1)

Exploratory, with no prediction: how much of a head start the second cycle receives, and how much of it remains after the second warmup. We show the skill of the head-start runs at epoch {HS}, overall and per op, beside the plain runs at the same epoch and the {HS}-epoch ex-2.2.19 run at {ex.HI_LR:g} (seed {ex.SEED_OFFSET} only).

/// admonition | TODO
Skill at epoch {HS} and at the end of the second warmup, per op, for each schedule.
///

## Calibration (E2)

Exploratory, with no prediction. We show the calibration KL of each run beside its EEM: the KL divergence[^kl] from the Bayes answer distribution to the answer distribution of the model. In ex-2.2.19 the {E}- and {REF_E}-epoch runs were equally well calibrated, so a head start that buys EEM at some cost to calibration would show here.

[^kl]: A measure of how far one probability distribution is from another; it is zero when they match.

/// admonition | TODO
Calibration KL against EEM, one dot per run, for the head-start, plain {E}-epoch, and plain {REF_E}-epoch runs.
///

## Op confusion (E3)

Exploratory, with no prediction. We show the op confusion matrix of each schedule, as in ex-2.2.19. For confident contexts of each true op, it gives the probability mass the model puts on the answers of each op, averaged over seeds {ex.SEED_OFFSET + min(ex.GATE_SEEDS)}–{ex.SEED_OFFSET + max(ex.GATE_SEEDS)}.

In ex-2.2.19 the {E}-epoch runs put more mass off the diagonal than the {REF_E}-epoch runs, mostly in the HSV-channel rows. This shows whether the head start moves that mass back.

/// admonition | TODO
The op confusion matrices of the head-start, plain {E}-epoch, and plain {REF_E}-epoch runs, side by side, on one color scale.
///

## A lower second peak (E4)

Exploratory, with no prediction. We score the head start with a second peak of {BAND:g} by the rules of H1 and H2. As in H3, we also show the epoch at which its HSV-channel skill first passes {HSV_HALF:g}, beside the same epoch for the other head start.

/// admonition | TODO
The shortfall from the {REF_E}-epoch runs and the gain over the plain {E}-epoch run for both head-start schedules, one dot per seed; the H1 and H2 verdicts for this schedule; and its skill curves added to the H3 figure.
///

## What it means for what follows

TODO after the results.

## Method

**Recipe.** The unanchored d64-L4 model with an untied readout and the newline mask, as in ex-2.2.19, on the `no-four` corpus condition that ex-2.2.18 built (`k3-r0.3`). The runs differ from the plain {E}-epoch runs of ex-2.2.19 in the learning-rate schedule alone.

**Measurements.** EEM, ceiling, floor, and calibration KL on the {ex.ex2216.HOLDOUT_CONTEXTS:,} held-out contexts per op, with the evaluation of ex-2.2.18. Skill curves are logged at about {ex.ex2218.N_TRAJ_POINTS} points per run on {ex.ex2218.N_TRAJ_EEM_PER_OP} held-out contexts per op, every {E // ex.ex2218.N_TRAJ_POINTS} epochs at {E} epochs, so an epoch of first passing is known to within that.

**Cost.** The new runs cost about \${COST:.2f} in all, scaled from the cost of an ex-2.2.18 run.

**Per-op tolerances.** As in ex-2.2.19: the seed range of the gap of each op over the three ex-2.2.17 runs, or {ex.PARTIAL_TOL}, whichever is larger.

"""

table_html(
    ["op", "tolerance"],
    [[f"`{op}`", f"{OP_TOL[op]:.3f}"] for op in OPS],
    "**Per-op tolerances.**",
)
