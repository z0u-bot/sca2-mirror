# ruff: noqa: B018
# title: Flat anti weight

# An exploratory report: the recipe of record, `τ_ℓ fixed`, trained at seeds 720 to 727 under five anti-subspace
# weights (the falloff over depth, two holds that are the same at every slice, and two weights held level from the
# start), beside a control at each seed. `experiment.py` beside this script trains and scores the runs.

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed

# --- The stored results ---------------------------------------------------------------------------------------


def fetch_json(ref: str) -> Any:
    store = project_store()
    art = store.get_refs([ref]).get(ref)
    assert art is not None, f"not published: {ref}"
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(Path(store.get(art, Path(tmp) / "results.json")).read_text())


rb, lr = ex.rb, ex.lr
AXES_BODY = fetch_json(ex.AXES_REF)
DESIGN = AXES_BODY["design"]
OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
SEEDS: list[int] = DESIGN["seeds"]
CRITERION = 0.02  # the selectivity criterion of the D2.2 design
TASK_MEAN_GATE, TASK_OP_GATE = -0.02, -0.05  # the task gate of the D2.2 design
LOST_BELOW = 0.5  # a finished run has lost the anchor when the full edit removes less than this
AS_WELL = 0.05  # the removal margin set in the plan for "about as well"
CROSS = 0.4  # α at the example answers that a run has settled past
E1 = 0  # the anchored axis, in code
FULL = len(rb.DOSE_GAMMAS) - 1

CONDITIONS = [c.name for c in ex.CONDITIONS]
RECIPE = "recipe"
SHOWN = {c: c.replace("-", " ") for c in CONDITIONS}
COND = {c.name: c for c in ex.CONDITIONS}


def num(v: float, spec: str = ".3f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


def fmt(values, spec: str = ".2f") -> str:
    v = np.asarray(list(values), float)
    return f"{num(float(v.mean()), spec)} ({num(float(v.min()), spec)} to {num(float(v.max()), spec)})"


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    def code(c: str) -> str:
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", c)

    head, rows = [code(h) for h in head], [[code(c) for c in r] for r in rows]
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def label(c: str, s: int) -> str:
    return ex.label_of(COND[c], s)


# --- At the end: the e₁ edit net of the median over the other 63 axes of the run (the D2.2 rule) ---------------


def net_of(drop) -> np.ndarray:
    """The drop under the e₁ edit less the median drop along the other axes: *drop* is (axes, ...), e₁ first."""
    drop = np.asarray(drop)
    return drop[E1] - np.median(drop[1:], axis=0)


_AX = {r["label"]: r for r in AXES_BODY["runs"]}
CONTROL_CLEAN = np.mean([r["clean"] for r in AXES_BODY["runs"] if r["kind"] == "control"], axis=0)
_RESTRICTED = {r["label"]: r for r in fetch_json(ex.RESTRICTED_REF)["runs"]}
_MEASURED = {r["label"]: r for r in fetch_json(ex.RESULTS_REF)["runs"]}
_TRAJ = fetch_json(ex.TRAJ_REF)


def scores(c: str, s: int) -> dict[str, Any]:
    r = _AX[label(c, s)]
    net = net_of(r["drop"])  # (doses, ops)
    task = np.asarray(r["clean"]) - CONTROL_CLEAN
    t = _TRAJ[label(c, s)]
    return {
        "removal": float(net[FULL, D] / (r["clean"][D] - r["null"][D])),
        "spill": float(net[:, OTHER].max()),
        "block1": float(net_of(_RESTRICTED[label(c, s)]["drop"]["block 1"])[OTHER].max()),
        "task": task,
        "lean": float(_MEASURED[label(c, s)]["r_light_emb"]),
        "window": float(t["window_mean"]),
        "call": t["call"],
    }


END = {(c, s): scores(c, s) for c in CONDITIONS for s in SEEDS}


def by_seed(c: str, key: str) -> np.ndarray:
    return np.array([END[(c, s)][key] for s in SEEDS])


REMOVAL = {c: by_seed(c, "removal") for c in CONDITIONS}
SPILL = {c: by_seed(c, "spill") for c in CONDITIONS}
BLOCK1 = {c: by_seed(c, "block1") for c in CONDITIONS}
TASK = {c: by_seed(c, "task") for c in CONDITIONS}  # (seeds, ops)
LEAN = {c: by_seed(c, "lean") for c in CONDITIONS}
LOST = {c: int((REMOVAL[c] < LOST_BELOW).sum()) for c in CONDITIONS}
OVER = {c: int((SPILL[c] > CRITERION).sum()) for c in CONDITIONS}
BLOCK1_OVER = {c: int((BLOCK1[c] > CRITERION).sum()) for c in CONDITIONS}
TASK_MISS = {c: int(((TASK[c].mean(axis=1) < TASK_MEAN_GATE) | (TASK[c].min(axis=1) < TASK_OP_GATE)).sum()) for c in CONDITIONS}  # fmt: skip
KEPT_CALLS = {c: sum(END[(c, s)]["call"] == "kept" for s in SEEDS) for c in CONDITIONS}
N_CALLED = sum(KEPT_CALLS.values())


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """The difference of means *a* − *b* and its 95% Welch t interval."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    q = stats.t.ppf(0.975, df)
    d = float(a.mean() - b.mean())
    return d, d - q * se, d + q * se


GAP = {c: welch(REMOVAL[c], REMOVAL[RECIPE]) for c in CONDITIONS if c != RECIPE}
GAP_LEVEL = {lv: welch(REMOVAL[lv], REMOVAL[fl]) for lv, fl in (("level-0.115", "flat-0.115"), ("level-0.2", "flat-0.2"))}  # fmt: skip
LOW = {c: [s for s in SEEDS if END[(c, s)]["removal"] < REMOVAL[RECIPE].min()] for c in CONDITIONS}


def fisher(k: int, n: int = len(SEEDS)) -> float:
    """Fisher's exact test (two-sided) of *k* runs of *n* against none of the recipe."""
    return float(stats.fisher_exact([[k, n - k], [0, n]])[1])


# --- Along training -------------------------------------------------------------------------------------------

_ALONG = fetch_json(ex.ALONG_REF)["runs"]
ALONG_EPOCHS = sorted({r["epoch"] for r in _ALONG})
_ALONG_BY = {(r["label"], r["epoch"]): r for r in _ALONG}


def spill_along(c: str, s: int) -> np.ndarray:
    return np.array([net_of(_ALONG_BY[(label(c, s), e)]["drop"])[OTHER].max() for e in ALONG_EPOCHS])


SPILL_ALONG = {(c, s): spill_along(c, s) for c in CONDITIONS for s in SEEDS}
PEAK_ALONG = {c: np.array([SPILL_ALONG[(c, s)].max() for s in SEEDS]) for c in CONDITIONS}
OVER_ALONG = {c: int((PEAK_ALONG[c] > CRITERION).sum()) for c in CONDITIONS}


def removal_along(c: str, s: int) -> np.ndarray:
    """Removal under the full edit at each kept checkpoint and the end, as a share of the way to the target null of
    the end record of the run.
    """
    null = _AX[label(c, s)]["null"][D]
    rs = [_ALONG_BY[(label(c, s), e)] for e in ALONG_EPOCHS]
    return np.array([net_of(r["drop"])[D] / (r["clean"][D] - null) for r in rs])


REMOVAL_ALONG = {(c, s): removal_along(c, s) for c in CONDITIONS for s in SEEDS}
SLIDE = {c: np.array([REMOVAL_ALONG[(c, s)].max() - REMOVAL_ALONG[(c, s)][-1] for s in SEEDS]) for c in CONDITIONS}
SLID = 0.1  # a run has slid when its end removal is this far under its best
N_SLID = {c: int((SLIDE[c] > SLID).sum()) for c in CONDITIONS}
AT_64 = {c: np.array([REMOVAL_ALONG[(c, s)][ALONG_EPOCHS.index(64)] for s in SEEDS]) for c in CONDITIONS}


def alpha_of(c: str, s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Epochs, α at the example answers at slice 3 of the probe contexts, and the anti weight by slice."""
    t = _TRAJ[label(c, s)]["traj"]
    ep = np.asarray(t["epoch"])
    a = np.asarray(t["alpha_anchored"])[:, lr.TRACK_SLICE][:, list(lr.lean.ANSWERS)].mean(axis=1)
    return ep, a, np.asarray(t["anti_weight"])


ALPHA = {(c, s): alpha_of(c, s) for c in CONDITIONS for s in SEEDS}


def crossing(c: str, s: int) -> float:
    ep, a, _ = ALPHA[(c, s)]
    hit = np.nonzero(a >= CROSS)[0]
    return float(ep[hit[0]]) if len(hit) else float("nan")


CROSSING = {c: np.array([crossing(c, s) for s in SEEDS]) for c in CONDITIONS}

# --- Figures ---------------------------------------------------------------------------------------------------

# Inks are (light, dark) pairs, resolved with `light_dark` where a figure draws: it picks by the theme active when
# it is called, so a value resolved at import would give the dark render the light ink.
RULE = ("#333", "#ddd")
ZERO = ("#aaa", "#555")
GHOST = ("#bbb", "#555")
INK = {
    "recipe": ("#08306b", "#c6dbef"),
    "flat-0.115": ("#a6611a", "#dfc27d"),
    "level-0.115": ("#d95f02", "#fdb863"),
    "flat-0.2": ("#1b7837", "#a6dba0"),
    "level-0.2": ("#7570b3", "#bcbddc"),
}
MARKER = {"recipe": "s", "flat-0.115": "v", "level-0.115": "^", "flat-0.2": "o", "level-0.2": "D"}
SLICE_INK = [("#000", "#fff"), ("#444", "#ccc"), ("#777", "#999"), ("#aaa", "#666"), ("#ccc", "#444")]


def hatch(ax, lo: float, hi: float) -> None:
    """Hatch the band from *lo* to *hi* as the failing side of a line (style-fig)."""
    ax.axhspan(lo, hi, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="//", lw=0, alpha=0.12, zorder=0)  # fmt: skip


def _dots(ax, x: float, v, color, m: str, jit) -> None:
    v = np.asarray(v, float)
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + jit, v, "o", ms=2.2, color=color, alpha=0.5, zorder=3, mew=0)
    ax.plot(x, v.mean(), m, ms=5.5, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6)


def _weights(ax, ep, w) -> None:
    """The anti weight by slice along training, one grey line per slice (embedding darkest)."""
    for i in range(w.shape[1]):
        ax.plot(ep, w[:, i], color=light_dark(*SLICE_INK[i]), lw=0.8, zorder=2)
    ax.set_ylim(0, 0.27)
    ax.tick_params(labelsize=6)


def end_figure() -> str:
    data = {
        "names": CONDITIONS,
        "shown": [SHOWN[c] for c in CONDITIONS],
        "removal": {c: REMOVAL[c].tolist() for c in CONDITIONS},
        "spill": {c: SPILL[c].tolist() for c in CONDITIONS},
        "block1": {c: BLOCK1[c].tolist() for c in CONDITIONS},
        "criterion": CRITERION,
        "lost": LOST_BELOW,
    }
    alt = f"""
        Removal is near 0.88 on every run of the recipe and of level 0.115. Flat 0.2 has three runs well below the rest and
        level 0.2 two, the lowest at {REMOVAL["flat-0.2"].min():.2f} and {REMOVAL["level-0.2"].min():.2f}, both under
        the lost line, and the lowest run of flat 0.115 is at {REMOVAL["flat-0.115"].min():.2f}. Spill stays under the
        criterion on every run at 0.2 and on the recipe, whose highest run is just under it at
        {SPILL[RECIPE].max():.3f}, and goes over on one run of each 0.115 condition. Under the edit at block 1 alone, level 0.115 has
        {BLOCK1_OVER["level-0.115"]} runs over and the conditions at 0.2 none.
    """
    return end_draw(data, alt)


@memo
def end_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="flat-anti-end",
        alt_text=alt_text,
        caption="""
            **Removal and spill at epoch 400, by condition.** The seeds are small dots over a thin bar for their
            range, and the seed mean is the large mark. **Left:** removal at the full dose; a run below the dashed line
            has lost the anchor. **Middle:** spill, the largest net drop on another op at any dose. **Right:** spill
            under the edit at block 1 alone. In both spill panels the dashed line is the criterion, with the side
            over it hatched. All are net of the median over the other 63 axes of the run.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.7), layout="constrained")
        rng = np.random.default_rng(0)
        jit = {n: rng.uniform(-0.12, 0.12, len(data["removal"][n])) for n in names}
        for ax, key in zip(axes, ("removal", "spill", "block1"), strict=True):
            for i, n in enumerate(names):
                _dots(ax, i, data[key][n], light_dark(*INK[n]), MARKER[n], jit[n])
            ax.set_xticks(range(len(names)), data["shown"], rotation=40, ha="right", fontsize=6.5)
            ax.set_xlim(-0.6, len(names) - 0.4)
            ax.tick_params(labelsize=7)
        axes[0].set_title("removal ↑", fontsize=8)
        axes[1].set_title("spill ↓", fontsize=8)
        axes[2].set_title("spill, block 1 ↓", fontsize=8)
        axes[0].axhline(data["lost"], color=light_dark(*RULE), lw=0.7, ls="--", zorder=1)
        lo, _ = axes[0].get_ylim()
        hatch(axes[0], min(lo, data["lost"] - 0.02), data["lost"])
        axes[0].set_ylim(min(lo, data["lost"] - 0.02), 1.0)
        his = [ax.get_ylim()[1] for ax in axes[1:]]
        los = [ax.get_ylim()[0] for ax in axes[1:]]
        for ax in axes[1:]:
            ax.axhline(0, color=light_dark(*ZERO), lw=0.5, zorder=0)
            ax.axhline(data["criterion"], color=light_dark(*RULE), lw=0.7, ls="--", zorder=1)
            hatch(ax, data["criterion"], max(his))
            ax.set_ylim(min(los), max(his))
        return fig

    return _plot()


def end_table() -> str:
    rows = []
    for c in CONDITIONS:
        t = TASK[c]
        rows.append([
            f"`{SHOWN[c]}`",
            fmt(REMOVAL[c]),
            f"{LOST[c]}",
            f"{OVER[c]}",
            f"{BLOCK1_OVER[c]}",
            fmt(t.mean(axis=1), ".3f"),
            f"{TASK_MISS[c]}",
            fmt(LEAN[c]),
        ])  # fmt: skip
    return table_html(
        [
            "condition",
            "removal ↑",
            "lost ↓",
            "over ↓",
            "over, block 1 ↓",
            "task change ↑",
            "task gate missed ↓",
            "embedding lean",
        ],  # fmt: skip
        rows,
        """
        **Removal, spill, task and lean at epoch 400**, seed mean and range over eight runs, or a count of runs of
        eight. "Lost" counts runs with removal under 0.5. "Over" counts runs over the spill criterion under the full
        edit, and "over, block 1" under the edit at block 1 alone. The task change is the mean over ops of the clean
        score less the mean of the eight controls. The embedding lean is the correlation of the e₁ component of the
        color embeddings with their lightness.
        """,
    )


def gap_table() -> str:
    rows = [[f"`{SHOWN[c]}`", "`recipe`", *[num(v, "+.3f") for v in GAP[c]]] for c in GAP]
    rows += [[f"`{SHOWN[lv]}`", f"`{SHOWN[fl]}`", *[num(v, "+.3f") for v in GAP_LEVEL[lv]]] for lv, fl in (("level-0.115", "flat-0.115"), ("level-0.2", "flat-0.2"))]  # fmt: skip
    return table_html(
        ["condition", "against", "removal gap", "95% interval, low", "95% interval, high"],
        rows,
        """
        **Removal against the recipe, and each level condition against the flat hold of the same value**: the
        difference of seed means and its 95% Welch t interval, eight runs a side.
        """,
    )


def along_figure() -> str:
    data = {
        "names": CONDITIONS,
        "shown": [SHOWN[c] for c in CONDITIONS],
        "epochs": list(ALONG_EPOCHS),
        "spill": {c: [SPILL_ALONG[(c, s)].tolist() for s in SEEDS] for c in CONDITIONS},
        "removal": {c: [REMOVAL_ALONG[(c, s)].tolist() for s in SEEDS] for c in CONDITIONS},
        "alpha_epochs": ALPHA[(RECIPE, SEEDS[0])][0].tolist(),
        "weights": {c: ALPHA[(c, SEEDS[0])][2].tolist() for c in CONDITIONS},
        "criterion": CRITERION,
    }
    s0, p0 = SEEDS[int(PEAK_ALONG["level-0.115"].argmax())], PEAK_ALONG["level-0.115"].max()
    alt = f"""
        Five columns, one per condition, each with removal on top, spill in the middle, and the anti weight by slice
        underneath. Removal reaches about 0.8 by epoch 64 on most runs of every condition. Under the recipe and the
        conditions at 0.115 it then holds or climbs; under the two conditions at 0.2, a few runs slide or stay near
        0.5 for the rest of training. Spill stays near zero on most runs. Under level 0.115 one run (seed {s0}) rises to
        {p0:.2f} in the first half of training and comes back down, and several others cross the criterion for a while.
        One run of flat 0.115 spikes as high at epoch 96, and one run of the recipe stays over for much of the middle
        of training. Under flat 0.2 spill stays under the criterion throughout, and under level 0.2 one run crosses
        it at one checkpoint. The weight strip shows the recipe
        fanning out by slice as it anneals, the flat holds annealing as one line, and the level conditions flat.
    """
    return along_draw(data, alt)


@memo
def along_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="flat-anti-along",
        alt_text=alt_text,
        caption="""
            **Along training, by condition.** One column per condition, one line per run, at the checkpoints kept at
            epochs 16, 32 and 64 and then every 32 epochs, and at the end. **Top:** removal under the full edit.
            **Middle:** spill under the full edit; the dashed line is the criterion, with the side over it hatched.
            Both are net of the median over the other 63 axes of the run, and spill here is at the full dose only.
            **Bottom:** the anti weight by slice, from the embedding (darkest) to slice 4 (lightest).
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        fig, axes = plt.subplots(
            3, len(names), figsize=(6.8, 4.4), layout="constrained", sharex=True, sharey="row",
            gridspec_kw={"height_ratios": [1.0, 1.2, 0.6]},
        )  # fmt: skip
        ep, ea = np.asarray(data["epochs"]), np.asarray(data["alpha_epochs"])
        for j, n in enumerate(names):
            ink = light_dark(*INK[n])
            for v in data["removal"][n]:
                axes[0, j].plot(ep, v, color=ink, marker=MARKER[n], ms=1.6, lw=0.7, alpha=0.8, zorder=3)
            for v in data["spill"][n]:
                axes[1, j].plot(ep, v, color=ink, marker=MARKER[n], ms=1.6, lw=0.7, alpha=0.8, zorder=3)
            _weights(axes[2, j], ea, np.asarray(data["weights"][n]))
            axes[0, j].set_title(data["shown"][j], fontsize=7)
            axes[1, j].axhline(0, color=light_dark(*ZERO), lw=0.5, zorder=0)
            axes[1, j].axhline(data["criterion"], color=light_dark(*RULE), lw=0.7, ls="--", zorder=1)
            axes[2, j].set_xlabel("epoch", fontsize=6.5)
            for ax in axes[:, j]:
                ax.tick_params(labelsize=6)
        lo, hi = axes[1, 0].get_ylim()
        for ax in axes[1]:
            hatch(ax, data["criterion"], hi)
        axes[1, 0].set_ylim(lo, hi)
        axes[0, 0].set_ylabel("removal ↑", fontsize=6.5)
        axes[0, 0].set_ylim(0, 1)
        axes[1, 0].set_ylabel("spill ↓", fontsize=6.5)
        axes[2, 0].set_ylabel("anti\nweight", fontsize=6.5)
        return fig

    return _plot()


def settle_table() -> str:
    rows = [
        [
            f"`{SHOWN[c]}`",
            f"{KEPT_CALLS[c]} of {len(SEEDS)}",
            fmt([END[(c, s)]["window"] for s in SEEDS]),
            fmt(CROSSING[c], ".0f"),
            fmt(AT_64[c]),
            f"{N_SLID[c]}",
            f"{OVER_ALONG[c]}",
        ]  # fmt: skip
        for c in CONDITIONS
    ]
    return table_html(
        [
            "condition",
            "called kept ↑",
            "α over epochs 44 to 64 ↑",
            "epoch α reaches 0.4 ↓",
            "removal at epoch 64 ↑",
            "slid ↓",
            "over along training ↓",
        ],  # fmt: skip
        rows,
        """
        **How the anchor settles, and spill along training.** The call and its window mean of α at the example
        answers at slice 3; the first record at which that α reaches 0.4; removal at epoch 64 (seed mean and range);
        the runs whose removal at the end is more than 0.1 under their best; and the runs whose spill is over the
        criterion at any kept checkpoint or at the end.
        """,
    )


# %%

r"""
# Flat anti weight

/// tip |
<!-- lede -->
The recipe of record sets the anti-subspace weight high on the early slices and low on the late ones. We tried the same weight at every slice instead. A high weight everywhere kept spill out, but on a few seeds the edit took out much less of the op, and one run in eight lost the anchor. A low weight everywhere kept removal, and one run in eight ended over the spill criterion, a difference from the recipe that eight runs cannot confirm. No run without the high start of the weight lost the anchor early.
///

The recipe of record anneals the anti-subspace weight from 0.25 on every slice to a hold that falls off with depth, from 0.2 at the embedding to 0.03 at slice 4. The falloff came from the [split-anti-seeds](/docs/m2/split-anti-seeds/report.py) experiment, where both holds that were the same at every slice did worse. But that experiment used an older pull: a step on slices 2 to 4, with a sharp pooling temperature from the start and no pool over slices.

Each flat hold there lost the anchor on one run, but the hold can't have caused those losses. The lost runs parted from the kept ones within the first sixty epochs ([loss-rate](/docs/m2/loss-rate/report.py)), while the anti weight was still near its start of 0.25 in every condition. The falloff also lost one run of forty this way, later on, and a soft start for the pooling temperature has since stopped such losses.

Two other results stand against the flat holds: the high hold removed less on the seeds where the pull settled slowly, and the low one spilled. A sharp pool over slices 1 to 3 has since taken away the spill from block 1 ([soft-pool-scout](/docs/m2/soft-pool-scout/report.py)), so a flat hold may now do as well, which would make the recipe simpler to describe.
<!-- REVIEW: the plan first said the flat holds failed by losing the anchor. The assumptions review found the losses parted from the kept runs by about epoch 60, when the anneal had moved the weight by under 0.008 in any condition, so the hold cannot have caused them. Verify: the loss-rate report on when lost runs diverge, and AntiSpec at epochs 20 and 64. -->

This report trains the recipe at eight seeds under five anti weights, with a control at each seed: 48 runs. None of the four other weights meets the margin we set before the runs for doing about as well as the falloff. The two at 0.2 miss it on removal. The two at 0.115 miss it by one run over the spill criterion each, which is too few to tell them from the recipe.
"""

# %%

r"""
## Observations

- [How the anchor settles (E1)](#how-the-anchor-settles-e1): every run of every condition is called kept at epoch 64, with or without the high start, and α at the example answers settles about as fast in all of them.
- [Removal and spill at the end (E2)](#removal-and-spill-at-the-end-e2): at 0.2 everywhere, no run spills and the color embeddings stop leaning onto e₁, but two or three runs in eight remove much less, and one in each condition ends under the lost line. At 0.115 everywhere, removal stays near the recipe, and one run in each condition ends over the spill criterion. With no high start, block 1 spills on more runs.
<!-- REVIEW: was "three runs in eight remove much less, and one in each condition falls to about the lost line". `level 0.2` has two runs far under the recipe (0.49, 0.67) and one at 0.79, and both lowest runs are under 0.5, which the table counts as lost. Verify: REMOVAL and LOST. -->
- [Along training (E3)](#along-training-e3): at 0.2, three of the low runs remove about as much as the rest at epoch 64 and then slide, and two are low from the start and stay low, where the slow runs of the recipe catch up. Spill along training is rarest at 0.2 and most common at `level 0.115`, and the recipe crosses the criterion on more runs than `flat 0.115`.
<!-- REVIEW: was "the low runs at 0.2 remove about as much as the rest early on, then slide or stall". Seed 726 is under 0.6 at epochs 32 and 64 in both conditions at 0.2 and never recovers; seeds 720 and 725 (`flat 0.2`) and 725 (`level 0.2`) are at par at epoch 64 and slide. Added the recipe count (3 runs against 1). Verify: REMOVAL_ALONG and OVER_ALONG. -->

## Scope and method

This is an exploratory study, with no preregistration and no gate of its own. Every run is the recipe of record, `τ_ℓ fixed` as the [fixed-confirm](/docs/m2/fixed-confirm/report.py) experiment trained it, with one change: the anti weight.

Conditions {.manual}
:   `recipe` is `τ_ℓ fixed` itself, with the falloff over depth, retrained here so that every condition shares the card and the arithmetic. `flat 0.115` anneals from 0.25 to a hold of 0.115 at every slice. That is the mean of the falloff over the five slices, so the term has the same total weight, spread evenly. `flat 0.2` anneals to 0.2 at every slice. That is the falloff value at the embedding, which kept the color embeddings from standing in for the op ([embedding-lean](/docs/m2/embedding-lean/report.py)). `level 0.115` and `level 0.2` hold their value at every slice from the first epoch, with no high start; `level 0.2` is close to `flat 0.2`, which only starts at 0.25.

Every condition keeps the end anneal it shares with the anchor, over the last 40 epochs. The anneal from 0.25 is slow to start, so the high start lasts: at epoch 64 the weight is still within a hundredth of 0.25 on every slice of every annealed condition, and at epoch 200 it is a little over halfway to the hold.

The two flat holds match the falloff on its mean and on its highest value. `flat 0.115` is lower than the falloff at slice 1, the same at slice 2, and higher at slices 3 and 4, and `flat 0.2` is higher at every slice but the embedding. No flat hold sits below the falloff on the slices the pull acts on, since 0.03 at every slice let the color embeddings lean onto e₁ in the [late-pull](/docs/m2/late-pull/report.py) experiment.
<!-- REVIEW: was "The two flat holds bracket the falloff on the slices the pull acts on". On slices 1 to 3 the falloff holds are 0.1575, 0.115 and 0.0725, so 0.115 is their middle and 0.2 is above them all; neither is below. Verify: FALLOFF_HOLDS. -->

`level 0.115` asks a separate question, and its comparison is with `flat 0.115`, which differs only in the start. In the [ex-2.1.11](/docs/m2/ex-2.1.11/report.py) experiment, an anti weight held at its low value from the start let the pull latch onto a syntax token. That low value was 0.3 times the anchor weight, where 0.115 is a little more than the anchor weight (0.1), so `level 0.115` is a milder case than that one. The soft start of the pooling temperature now spreads the early pull over positions, and may do that job instead.
<!-- REVIEW: added the scale of the ex-2.1.11 arm (`anti-hold`, ratio 0.30 against a peak ratio of 2.5), since "no high start" could be taken as a repeat of it. Verify: `lam` 0.1 and `hold_ratio` 0.3 in the anti schedule. -->

Seeds 720 to 727 have no finished run of the recipe; the loss-rate counts stopped them at epoch 64. We compare conditions as two samples of eight runs, with a 95% Welch t interval on the difference of seed means,[^welch] and Fisher's exact test for counts of runs (lost, or over the spill criterion).

At eight runs a side, removal gaps under about 0.05 are hard to see. But the seed range of the recipe (about 0.08 in fixed-confirm) suggests an interval of about ±0.03 on removal, so the experiment can bound what a flat hold costs.

We take a flat hold to do about as well as the falloff when all of these hold: its removal is within 0.05 of the recipe, it loses no anchor, and no run is over the spill criterion.

Every run trained and was scored on L40S cards. The earlier runs of the recipe (704 to 719) trained on L4 cards with the determinism flags on, so they are left out.

[^welch]: Welch t interval: a confidence interval for a difference of two means that does not assume the two samples have equal variance.

## The measurements

Every edit below is the e₁ edit. Removal and spill are each a drop under that edit, less a baseline: the median drop when the same edit is made along each of the other 63 coordinate axes of the run (the rule of the D2.2 design).

Removal
:   The drop on `difference` under the full edit at every position and slice, as a share of the way from the clean score to the target null. Higher is better, and a run below 0.5 has lost the anchor.

Spill
:   The largest drop on any other op, at the dose where it is largest. Lower is better, and a run above 0.02 is over the criterion. Along training (E3) it is at the full dose only.
    <!-- REVIEW: was "under the full edit". The criterion of the D2.2 design is at any dose, the axis scoring here stores every dose, and the checkpoints along training are scored at the full dose alone, as in fixed-confirm. Verify: `rb.axes_one` and `ac.kept_axes_one`. -->

Block-1 spill
:   Spill under the full edit at slice 1 alone, where the spill of earlier recipes entered.

Task change
:   The clean score per op, less the mean of the controls. The gate of the D2.2 design asks for a mean change over ops of at least −0.02, and at least −0.05 on any one op.

Call
:   The early verdict on whether a run kept the anchor, from mean α at the example answers of the `difference` probe contexts at slice 3, over epochs 44 to 64. A run below 0.15 is called lost, and one at 0.28 or above is called kept. The cuts were set on runs whose anti weight was near 0.25 over the window, so for the level conditions the call is descriptive and removal at the end decides.

Embedding lean
:   The correlation of the e₁ component of the color embeddings with their lightness. Near zero, no color embedding stands in for the op.
"""

# %%

rf"""
## How the anchor settles (E1)

Does every condition settle on the example answers, and as quickly? Up to the call at epoch 64 the three annealed conditions are the same recipe, so any losses there test the level conditions alone. A level run that aligns a syntax token or the query answer would show up here.

A run of any condition can still lose the anchor after epoch 64. In the loss-rate experiment, the slow runs of the older recipe stalled until the anti weight at slice 3 had fallen to about half. A hold of 0.2 never falls that far, so the speed of settling may separate `flat 0.2` from the rest.

{settle_table()}

All {N_CALLED} runs are called kept, by a wide margin, and α at the example answers reaches 0.4 by epoch 28 on every run. The level conditions settle as quickly as the rest, so the early pull finds the example answers without the high start; nothing latched onto a syntax token or the query answer. The window mean of α is a little higher at `level 0.115` than elsewhere, which fits a weaker push from e₁ over the window.

The time to settle does not separate the conditions either, so whatever sets `flat 0.2` apart comes later (E3). With every run called kept, the low-removal runs of E2 are runs whose α settled on time, and whose removal fell behind or never caught up. Two of them end under 0.5 and so count as lost at the end; the call missed them.

"""

# %%

rf"""
## Removal and spill at the end (E2)

Does a flat hold remove as much, and spill as little? Block-1 spill and the embedding lean say where any spill comes from.

{end_figure()}

{end_table()}

A weight of 0.2 everywhere is the cleanest: no run over the criterion, under the full edit or at block 1, and the color embeddings barely lean onto e₁. But removal sags.

Three runs of `flat 0.2` and two of `level 0.2` end well below the lowest run of the recipe, and the lowest in each ({num(REMOVAL["flat-0.2"].min(), ".2f")} and {num(REMOVAL["level-0.2"].min(), ".2f")}) is under the line where we call the anchor lost, the second only just. That is the same pattern the high hold showed under the older pull, so it does not seem to depend on how the pull is pooled.

A weight of 0.115 everywhere keeps removal close to the recipe. `level 0.115` matches it, and `flat 0.115` is a little lower: four of its runs end under the lowest of the recipe, two of them by less than 0.01, and the lowest at {num(REMOVAL["flat-0.115"].min(), ".2f")}.

Each condition at 0.115 ends with one run over the spill criterion, on an HSV op. At `flat 0.115` that run is seed {SEEDS[int(SPILL["flat-0.115"].argmax())]}, also the run of the recipe nearest the criterion (at {SPILL[RECIPE].max():.3f}), so on spill at the end the two conditions differ by about 0.01 on one seed. The embedding lean stays about where the recipe has it, and with no high start block 1 spills on more runs.
<!-- REVIEW: (1) was "Three runs of each condition at 0.2 end well below ... at about the line where we call the anchor lost"; `level 0.2` has two runs far under and one at 0.79, and 0.44 and 0.49 are both under 0.5 (lost, as the table counts). (2) was "`flat 0.115` ... with two runs a little under the lowest of the recipe"; four are under (0.735, 0.796, 0.820, 0.821 against 0.823). (3) added that the run over at `flat 0.115` and the highest run of the recipe are both seed 720 (0.029 and 0.019). Verify: REMOVAL, LOW, SPILL by seed. -->

The recipe has removal as high as any condition, and no run over the criterion under the full edit at the end, though its run at seed 720 ends just under it. Its embedding lean is about as strong as at 0.115, though it holds 0.2 at the embedding, so the lean seems to follow the weight past the embedding more than the weight on it. Under the sharp pool, the lean costs the recipe no selectivity at the end.

{gap_table()}

On removal the intervals are wide, since the low runs at 0.2 spread the seed range. `level 0.115` sits within a few hundredths of the recipe either way. The counts of runs over the criterion (one against none) are too small for Fisher's exact test to tell apart (p = {fisher(1):.1f}), and the gaps in mean spill have intervals that span zero, so the runs at the end do not separate the low weights from the recipe on spill.

Every condition but the recipe misses the task gate on one run, each time on an HSV op.
<!-- REVIEW: was "so what separates the low weights from the recipe on spill is suggestive". One run against none has p = 1.0, and the Welch intervals on mean spill against the recipe are −0.010 to +0.013 (`flat 0.115`) and −0.005 to +0.020 (`level 0.115`). Verify: welch(SPILL[c], SPILL[RECIPE]). -->
"""

# %%

rf"""
## Along training (E3)

The late spill of earlier recipes rose as the anti weight annealed ([late-spill-scout](/docs/m2/late-spill-scout/report.py)), and a flat hold changes what the weight anneals to. Does spill rise late under any condition, and when do the low runs at 0.2 fall behind?

{along_figure()}

Up to epoch 64 the three annealed conditions have about the same weight, so any difference in how far each run has got by then is run-to-run variation. A few runs of the recipe and of `flat 0.115` are still climbing, and none of `flat 0.2` is.

After epoch 64 the conditions part. Under the recipe and at 0.115, every run climbs to near 0.9 or holds there as the weight on the late slices comes down. At 0.2, a few runs slide back, or sit near 0.5 from early on, while the weight on every slice stays high ({N_SLID["flat-0.2"]} and {N_SLID["level-0.2"]} runs end more than 0.1 under their best, against {N_SLID[RECIPE]} on the recipe).

So a high weight seems to work against holding the op on e₁, on the seeds where the pull holds it less firmly. These runs can't say whether the weight on the late slices matters or the total: the conditions at 0.2 are higher on every slice past the embedding, and mean removal also orders by the total weight over training.

Spill along training is rarest at 0.2: no run of `flat 0.2` crosses the criterion, and one run of `level 0.2` does, at one checkpoint. `level 0.115` crosses it on the most runs ({OVER_ALONG["level-0.115"]}), once up to {PEAK_ALONG["level-0.115"].max():.2f} in the first half of training, and that run comes back down by the end. `flat 0.115` spikes as high once, at epoch 96 while its weight is still near the start on every slice, and that run ends over.

The recipe crosses on {OVER_ALONG[RECIPE]} runs, one of them for much of the middle of training, and all are back under by the end. So spill at the end understates how often spill gets through along the way, for the recipe as much as for `level 0.115`. The counts do not order by the weight: the recipe crosses on more runs than `flat 0.115`, and about as many as `level 0.115`.
<!-- REVIEW: was "Spill along training follows the weight the other way ... So spill at the end understates how often the low weights let spill through along the way; the recipe sits between the two." Runs over at any checkpoint: recipe 3, `flat 0.115` 1, `flat 0.2` 0, `level 0.115` 4, `level 0.2` 1, so the recipe is above `flat 0.115`, and at `flat 0.115` the count at the end (1) equals the count along training. The `flat 0.115` spike is at epoch 96, when its weight is 0.234 on every slice (recipe: 0.244 to 0.223). Verify: OVER_ALONG, SPILL_ALONG, and `anti_weight` in the trajectories. -->
"""

# %%

r"""
## Discussion

A weight of 0.2 on every slice is too much. It keeps spill out, the color embeddings included, but on some seeds removal sags, and one run in each condition at 0.2 ends with the anchor lost. That fits what the split-anti-seeds experiment found under the older pull, though at eight runs a side the gap in mean removal is not settled by these runs alone. Whether the cost comes from the weight on the late slices or from the total is not separated here.

A weight of 0.115 on every slice, the mean of the falloff, does about as well as the recipe on what we can measure at eight runs. `level 0.115` matches it on removal, and `flat 0.115` is a little lower, within the margin on its mean but not clearly. Each ends with one run over the spill criterion where the recipe has none, and the recipe has its own runs that cross the criterion along training.

So these runs do not show that the falloff is needed. They show that a flat weight has to be low enough, and that 0.115 is not clearly worse. Only two flat values were tried, so where between 0.115 and 0.2 the cost starts is open.
<!-- REVIEW: the first draft said the falloff "does two jobs at once" that a flat weight cannot, and "gets both". The claims review found the comparison that isolates placement (recipe against `flat 0.115`, same start and total) gives a removal gap with an interval across zero and one run over against none on a seed where the recipe sits at 0.019; `level 0.115` matches the recipe on removal. Narrowed to what the runs support. Verify: GAP, SPILL at seed 720, fisher(1). -->

No run without the high start lost the anchor early (none of sixteen), and α settled as fast as with it. That bounds the early loss rate only loosely, and both level weights are at or above the anchor weight, a milder case than the one in D2.1. The soft start of the pooling temperature may be what keeps the pull off the syntax tokens now, though no condition here goes without it.
"""

# %%

r"""
## References

[split-anti-seeds](/docs/m2/split-anti-seeds/report.py)
:   Under a step pull on slices 2 to 4, an anti hold that falls off with depth kept the anchor on all eight runs, while holds of 0.2 and 0.1 at every slice each lost one run to the query answer, and the high one removed less.

[late-pull](/docs/m2/late-pull/report.py)
:   With the pull moved off the embedding, an anti hold of 0.03 at every slice let the color embeddings lean onto e₁ and block 1 spill, and higher holds reduced both at some cost to removal.

[embedding-lean](/docs/m2/embedding-lean/report.py)
:   The spill of the older recipe came from a color embedding standing in for the op: the e₁ component of the color embeddings followed their lightness.

[loss-rate](/docs/m2/loss-rate/report.py)
:   With a soft start for the pooling temperature, no run of forty lost the anchor, against one of forty without it.

[soft-pool-scout](/docs/m2/soft-pool-scout/report.py)
:   Keeping the pool over slices 1 to 3 at a sharp temperature removed the early spill, which entered through block 1.

[fixed-confirm](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed` kept the anchor on all eight runs at fresh seeds with removal near 0.9 and no run over the spill criterion, and became the recipe of record.

[late-spill-scout](/docs/m2/late-spill-scout/report.py)
:   Late spill on the recipe rose as the anti weight annealed, and holding the anti weight high lowered it at a cost to removal.

[ex-2.1.11](/docs/m2/ex-2.1.11/report.py)
:   In D2.1, an anti weight held at its low value from the start let the pull latch onto a syntax token, and one held at its high value throughout left the response less graded.
"""
