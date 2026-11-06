# ruff: noqa: B018
# title: When paired runs stop being alike

# An exploratory re-analysis, with no new runs: the correlation across seeds between two conditions trained at the
# same seeds, along training, from the stored trajectories and kept-checkpoint edits of the fixed and anneal
# confirmations, the loss-rate screen, and the late spill scout.

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from mini.lit import memo
from mini.store import project_store
from mini.vis import AxesRow, figure_html, light_dark, themed


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as the ex-2.2 experiments do."""
    path = Path(__file__).resolve().parent.parent / name / "experiment.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


def fetch_json(ref: str) -> Any:
    store = project_store()
    art = store.get_refs([ref]).get(ref)
    assert art is not None, f"not published: {ref}"
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(Path(store.get(art, Path(tmp) / "results.json")).read_text())


fc = _load_sibling("fixed-confirm", "paired_runs_fc")
lss = _load_sibling("late-spill-scout", "paired_runs_lss")
ac, sps, lr, rb = fc.ac, fc.sps, fc.lr, fc.rb

OPS: tuple[str, ...] = tuple(fetch_json(fc.AXES_REF)["design"]["ops"])
D = OPS.index(fc.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
ANSWERS = list(lr.lean.ANSWERS)  # the example-answer roles of the `difference` probe contexts
TRACK = lr.TRACK_SLICE  # slice 3, where the loss-rate call reads α
E1 = 0  # the anchored axis, in code


def ranks(x) -> np.ndarray:
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def r_of(x, y) -> float:
    """The rank correlation (Spearman) of two scores across seeds. Ranks, because spill is near zero on most runs
    and a single run that spills under both conditions would set a Pearson correlation on its own.
    """
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


# The standard error of Fisher's z for a rank correlation over n pairs (the Fieller, Hartley and Pearson
# approximation, sqrt(1.06 / (n − 3))).
def _z_se(n: int) -> float:
    return float(np.sqrt(1.06 / (n - 3)))


def chance(n: int) -> float:
    """The largest |r| that n pairs of unrelated scores give in about one draw of twenty (two-sided 95%)."""
    return float(np.tanh(1.96 * _z_se(n)))


def interval(r: float, n: int) -> tuple[float, float]:
    """A 95% interval for a rank correlation of *r* over *n* pairs, by Fisher's z."""
    z, h = np.arctanh(r), 1.96 * _z_se(n)
    return float(np.tanh(z - h)), float(np.tanh(z + h))


def num(v: float, spec: str = ".2f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


def span(lo: float, hi: float) -> str:
    return f"{num(lo)} to {num(hi)}"


# --- Trajectories: the task score and α, recorded every four epochs during training ---------------------------


def series(traj: dict) -> dict[str, np.ndarray]:
    """The task score (expected exact match on the held-out probe contexts, mean over ops) and α at the example
    answers at slice 3, at each recorded epoch once (a run stopped at epoch 64 records it twice).
    """
    ep = np.round(np.asarray(traj["epoch"]), 3)
    _, first = np.unique(ep, return_index=True)
    alpha = np.asarray(traj["alpha_anchored"])[:, TRACK][:, ANSWERS].mean(axis=1)
    task = np.asarray(traj["eem_per_op"]).mean(axis=1)
    return {"epoch": ep[first], "alpha": alpha[first], "task": task[first]}


# From one initialization, to epoch 64: four recipes at the 32 seeds of the loss-rate screen (712 to 743).
SCREEN_SEEDS = list(lr.SEEDS)
_FC_SCREEN = fetch_json(fc.SCREEN_TRAJ_REF)
_AC_SCREEN = fetch_json(ac.SCREEN_TRAJ_REF)
_LR_SCREEN = {(v["arm"], v["model_seed"]): v["traj"] for v in fetch_json(lr.TRAJ_REF).values()}
FIXED, ANNEAL, STEP, FALLOFF, CONTROL = "fixed", "anneal", "step-soft", "falloff", "control"
SHOWN = {FIXED: "τ_ℓ fixed", ANNEAL: "τ_ℓ anneal", STEP: "step, soft τ", FALLOFF: "falloff", CONTROL: "control"}
SCREEN = {
    FIXED: {s: series(_FC_SCREEN[f"fixed-s{s}-e64"]["traj"]) for s in SCREEN_SEEDS},
    ANNEAL: {s: series(_AC_SCREEN[f"anneal-s{s}-e64"]["traj"]) for s in SCREEN_SEEDS},
    STEP: {s: series(_LR_SCREEN[(STEP, s)]) for s in SCREEN_SEEDS},
    FALLOFF: {s: series(_LR_SCREEN[(FALLOFF, s)]) for s in SCREEN_SEEDS},
}
SCREEN_EPOCHS = SCREEN[FIXED][SCREEN_SEEDS[0]]["epoch"]
assert all(np.array_equal(SCREEN[n][s]["epoch"], SCREEN_EPOCHS) for n in SCREEN for s in SCREEN_SEEDS)
PAIRS = [(FIXED, ANNEAL), (FIXED, STEP), (FIXED, FALLOFF), (STEP, FALLOFF)]


def r_along(runs: dict, a: str, b: str, key: str, seeds: list[int]) -> np.ndarray:
    """The correlation across *seeds* between conditions *a* and *b*, at each recorded epoch."""
    x = np.array([runs[a][s][key] for s in seeds])
    y = np.array([runs[b][s][key] for s in seeds])
    return np.array([r_of(x[:, j], y[:, j]) for j in range(x.shape[1])])


def r_next(runs: dict, n: str, key: str, seeds: list[int]) -> np.ndarray:
    """The correlation across seeds between each run of *n* and the same run four epochs later (from the second
    record on): how much one record of a run predicts its next one.
    """
    x = np.array([runs[n][s][key] for s in seeds])
    return np.array([r_of(x[:, j - 1], x[:, j]) for j in range(1, x.shape[1])])


R_SCREEN = {(a, b, k): r_along(SCREEN, a, b, k, SCREEN_SEEDS) for a, b in PAIRS for k in ("task", "alpha")}
NEXT_SCREEN = {k: np.mean([r_next(SCREEN, n, k, SCREEN_SEEDS) for n in SCREEN], axis=0) for k in ("task", "alpha")}
LATE = SCREEN_EPOCHS >= 44  # the window of the loss-rate call, epochs 44 to 64
LATE_R = {key: float(np.mean(v[LATE])) for key, v in R_SCREEN.items()}
LATE_NEXT = {k: float(np.mean(v[LATE[1:]])) for k, v in NEXT_SCREEN.items()}
FIRST_R = {key: float(v[0]) for key, v in R_SCREEN.items()}

# --- Early edits at forty seeds: removal and spill at epochs 16, 32 and 64, `τ_ℓ fixed` against `τ_ℓ anneal` ---


def net_of(drop) -> np.ndarray:
    """The drop under the e₁ edit less the median drop along the other axes: *drop* is (axes, ...), e₁ first."""
    drop = np.asarray(drop)
    return drop[E1] - np.median(drop[1:], axis=0)


_EARLY = {(r["kind"], r["model_seed"], r["epoch"]): net_of(r["drop"]) for r in fetch_json(fc.EARLY_REF)["runs"]}
EARLY_SEEDS = sorted({s for _, s, _ in _EARLY})
EARLY_EPOCHS = sorted({e for *_, e in _EARLY})
EARLY_R = {
    (key, e): r_of(*([f(_EARLY[(n, s, e)]) for s in EARLY_SEEDS] for n in (FIXED, ANNEAL)))
    for key, f in (("removal", lambda v: v[D]), ("spill", lambda v: v[OTHER].max()))
    for e in EARLY_EPOCHS
}

# --- The whole run, at eight seeds: `τ_ℓ fixed`, `τ_ℓ anneal` and their control at 712 to 719 -----------------

SEEDS8 = list(range(712, 720))
_FC_FIN = fetch_json(fc.FINISHED_TRAJ_REF)
_AC_FIN = fetch_json(ac.FINISHED_TRAJ_REF)
FULL_RUNS = {
    FIXED: {s: series(_FC_FIN[f"fixed-s{s}"]["traj"]) for s in SEEDS8},
    ANNEAL: {s: series(_AC_FIN[f"anneal-s{s}"]["traj"]) for s in SEEDS8},
    CONTROL: {s: series(_AC_FIN[f"control-s{s}"]["traj"]) for s in SEEDS8},
}
FULL_EPOCHS = FULL_RUNS[FIXED][SEEDS8[0]]["epoch"]
R_FULL = {(a, b): r_along(FULL_RUNS, a, b, "task", SEEDS8) for a, b in ((FIXED, ANNEAL), (FIXED, CONTROL), (ANNEAL, CONTROL))}  # fmt: skip

_ALONG = {(n, r["model_seed"], r["epoch"]): net_of(r["drop"]) for n, ref in ((FIXED, fc.ALONG_REF), (ANNEAL, ac.ALONG_REF)) for r in fetch_json(ref)["runs"]}  # fmt: skip
KEPT_EPOCHS = sorted({e for (n, _, e) in _ALONG if n == FIXED})
KEPT_R = {
    key: np.array([r_of(*([f(_ALONG[(n, s, e)]) for s in SEEDS8] for n in (FIXED, ANNEAL))) for e in KEPT_EPOCHS])
    for key, f in (("removal", lambda v: v[D]), ("spill", lambda v: v[OTHER].max()))
}
KEPT_LATE = np.asarray(KEPT_EPOCHS) >= 160

# At the end: the task change of the D2.2 design nets each run against the control at its seed. How alike is a run
# to that control? Five recipe sets, each with the control trained at its seeds.
_AX: dict[str, dict[Any, dict]] = {
    "fc": {r["label"]: r for r in fetch_json(fc.AXES_REF)["runs"]},
    "ac": {r["label"]: r for r in fetch_json(ac.AXES_REF)["runs"]},
    "rb": {r["label"]: r for r in fetch_json(rb.AXES_REF)["runs"]},
    "sps": {(r["condition"], r["model_seed"]): r for r in fetch_json(sps.AXES_REF)["runs"]},
}
OLD8 = list(range(704, 712))
END_SETS = [
    (FIXED, SEEDS8, lambda s: _AX["fc"][fc.label_of(s, None)], lambda s: _AX["ac"][ac.label_of(s, None, "control")]),
    (ANNEAL, SEEDS8, lambda s: _AX["ac"][ac.label_of(s, None)], lambda s: _AX["ac"][ac.label_of(s, None, "control")]),
    (FIXED, OLD8, lambda s: _AX["sps"][(fc.BASE, s)], lambda s: _AX["rb"][f"control-s{s}"]),
    (STEP, OLD8, lambda s: _AX["rb"][rb.SOFT_POOL + f"step-soft-s{s}"], lambda s: _AX["rb"][f"control-s{s}"]),
    (FALLOFF, OLD8, lambda s: _AX["rb"][f"falloff-s{s}"], lambda s: _AX["rb"][f"control-s{s}"]),
]


def end_task(run_of, control_of, seeds) -> dict[str, Any]:
    x = np.array([np.mean(run_of(s)["clean"]) for s in seeds])
    c = np.array([np.mean(control_of(s)["clean"]) for s in seeds])
    return {
        "r": r_of(x, c),
        "sd_run": float(x.std(ddof=1)),
        "sd_control": float(c.std(ddof=1)),
        # The spread of the task change netted per seed, and netted against the seed mean of the control.
        "sd_paired": float((x - c).std(ddof=1)),
        "sd_unpaired": float(np.sqrt(x.var(ddof=1) + c.var(ddof=1) / len(seeds))),
    }


END_TASK: list[dict[str, Any]] = [
    end_task(run, ctl, seeds) | {"name": n, "seeds": seeds} for n, seeds, run, ctl in END_SETS
]
END_R = np.array([t["r"] for t in END_TASK])

# --- From one stored state: the late spill scout, which carries each run on from its epoch-64 state ------------

_LSS_TRAJ = {(v["condition"], v["arm"], v["model_seed"]): v["traj"] for v in fetch_json(lss.TRAJ_REF).values()}
_SPS_FIN = fetch_json(sps.FINISHED_TRAJ_REF)
BASE, LT_LATE, ANTI_LATE, ANTI_HELD, GPU = "base", "lt-late", "anti-late", "anti-held", "L4"
ARMS = [LT_LATE, ANTI_LATE, ANTI_HELD]
SHOWN |= {LT_LATE: "τ_ℓ late", ANTI_LATE: "anti late", ANTI_HELD: "anti held", GPU: "base on L4"}
BRANCH = {a: {s: series(_LSS_TRAJ[(lss.CONDITION, a, s)]) for s in OLD8} for a in (BASE, *ARMS)} | {
    GPU: {s: series(_SPS_FIN[sps.label_of(lss.CONDITION, s, None)]["traj"]) for s in OLD8}
}


def after_branch(run: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keep = run["epoch"] > lss.STOP_EPOCH
    return {k: v[keep] for k, v in run.items()}


BRANCH = {a: {s: after_branch(v) for s, v in runs.items()} for a, runs in BRANCH.items()}
BRANCH_EPOCHS = BRANCH[BASE][OLD8[0]]["epoch"]
assert all(np.array_equal(BRANCH[a][s]["epoch"], BRANCH_EPOCHS) for a in BRANCH for s in OLD8)
R_BRANCH = {(a, k): r_along(BRANCH, a, BASE, k, OLD8) for a in (*ARMS, GPU) for k in ("task", "alpha")}

_LSS_ALONG = {
    (r["arm"], r["model_seed"], r["epoch"]): np.median(np.asarray(r["es_sham"]), axis=0) - np.asarray(r["es_edit"])
    for r in fetch_json(lss.ALONG_REF)["runs"]
    if r["condition"] == lss.CONDITION
}  # the net drop per op under the full edit, against the median of eight reference edits (spill-timing)
LSS_EPOCHS = sorted({e for *_, e in _LSS_ALONG})
LSS_R = {
    (a, key): np.array(
        [
            r_of([f(_LSS_ALONG[(a, s, e)]) for s in OLD8], [f(_LSS_ALONG[(BASE, s, e)]) for s in OLD8])
            for e in LSS_EPOCHS
        ]
    )  # fmt: skip
    for a in ARMS
    for key, f in (("removal", lambda v: v[D]), ("spill", lambda v: v[OTHER].max()))
}
# Check: `anti late` and `anti held` have one schedule to epoch 200, so they must agree exactly to epoch 192.
_SAME = [e for e in LSS_EPOCHS if e <= 192]
assert all(np.array_equal(_LSS_ALONG[(ANTI_LATE, s, e)], _LSS_ALONG[(ANTI_HELD, s, e)]) for s in OLD8 for e in _SAME)

# At the end, `base` (A10G after epoch 64) against the finished run of the soft pool scout (L4 throughout).
_LSS_AX = {(r["condition"], r["arm"], r["model_seed"]): r for r in fetch_json(lss.AXES_REF)["runs"]}


def removal_spill(r: dict) -> tuple[float, float]:
    net = net_of(r["drop"])[len(rb.DOSE_GAMMAS) - 1]
    return float(net[D]), float(net[OTHER].max())


GPU_END = {
    key: r_of(
        [removal_spill(_LSS_AX[(lss.CONDITION, BASE, s)])[i] for s in OLD8],
        [removal_spill(_AX["sps"][(lss.CONDITION, s)])[i] for s in OLD8],
    )
    for i, key in enumerate(("removal", "spill"))
}
BRANCH_LATE = BRANCH_EPOCHS > 288
LSS_LATE = np.asarray(LSS_EPOCHS) >= 320


def binned(epochs, r, start: float, width: float = 32) -> tuple[np.ndarray, np.ndarray]:
    """The mean of a correlation over the records of each stretch of *width* epochs after *start*, placed at the end
    of the stretch: at eight seeds one record is too noisy to draw on its own.
    """
    epochs, r = np.asarray(epochs), np.asarray(r)
    ends = np.arange(start + width, epochs.max() + 1e-6, width)
    ends = np.append(ends, epochs.max()) if ends[-1] < epochs.max() else ends
    return ends, np.array([r[(epochs > e - width) & (epochs <= e)].mean() for e in ends])


R_FULL_BINNED = {k: binned(FULL_EPOCHS, v, 0) for k, v in R_FULL.items()}
R_BRANCH_BINNED = {k: binned(BRANCH_EPOCHS, v, lss.STOP_EPOCH) for k, v in R_BRANCH.items()}

# --- Figures ---------------------------------------------------------------------------------------------------

# Inks are (light, dark) pairs, resolved with `light_dark` where a figure draws.
RULE = ("#333", "#ddd")
BAND = ("#e4e4e4", "#2a2a2a")
GHOST = ("#888", "#999")
INK = {
    (FIXED, ANNEAL): ("#08306b", "#c6dbef"),
    (FIXED, STEP): ("#1b7837", "#a6dba0"),
    (FIXED, FALLOFF): ("#a6611a", "#dfc27d"),
    (STEP, FALLOFF): ("#7570b3", "#bcbddc"),
    (FIXED, CONTROL): ("#a6611a", "#dfc27d"),
    (ANNEAL, CONTROL): ("#7570b3", "#bcbddc"),
    LT_LATE: ("#08306b", "#c6dbef"),
    ANTI_LATE: ("#1b7837", "#a6dba0"),
    ANTI_HELD: ("#a6611a", "#dfc27d"),
    GPU: ("#7570b3", "#bcbddc"),
}
MARKER = {
    (FIXED, ANNEAL): "o",
    (FIXED, STEP): "^",
    (FIXED, FALLOFF): "v",
    (STEP, FALLOFF): "D",
    (FIXED, CONTROL): "v",
    (ANNEAL, CONTROL): "D",
    LT_LATE: "o",
    ANTI_LATE: "^",
    ANTI_HELD: "v",
    GPU: "D",
}


def tex(shown: str) -> str:
    """A recipe name for a figure label, with its symbols as mathtext."""
    return shown.replace("τ_ℓ", "$\\tau_\\ell$").replace("τ", "$\\tau$")


def pair_label(key) -> str:
    if isinstance(key, tuple):
        return f"{tex(SHOWN[key[0]])} ~ {tex(SHOWN[key[1]])}"
    return f"{tex(SHOWN[key])} ~ base"


def panel_data(title: str, lines: dict, band: float, next_line=None) -> dict:
    """One panel: *lines* maps a pairing to (epochs, r); *band* is the chance bound for its seed count."""
    return {
        "title": title,
        "lines": [
            {
                "label": pair_label(k),
                "ink": INK[k],
                "marker": MARKER[k],
                "x": np.asarray(x).tolist(),
                "y": np.asarray(y).tolist(),
            }  # fmt: skip
            for k, (x, y) in lines.items()
        ],
        "band": band,
        "next": None if next_line is None else [np.asarray(v).tolist() for v in next_line],
    }


@memo
def r_draw(name: str, panels: list[dict], size: tuple[float, float], xlim: tuple[float, float], alt_text: str, caption: str) -> str:  # fmt: skip
    @themed(name=name, alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(panels), figsize=size, layout="constrained", sharey=True)
        axes = cast(AxesRow, np.atleast_1d(axes))
        seen: dict[str, Any] = {}
        for ax, p in zip(axes, panels, strict=True):
            ax.axhspan(-p["band"], p["band"], color=light_dark(*BAND), lw=0, zorder=0)
            ax.axhline(0, color=light_dark(*GHOST), lw=0.5, zorder=1)
            if p["next"] is not None:
                (h,) = ax.plot(*p["next"], color=light_dark(*GHOST), lw=1.0, ls="--", zorder=2)
                seen.setdefault("the same run, four epochs on", h)
            for line in p["lines"]:
                (h,) = ax.plot(line["x"], line["y"], color=light_dark(*line["ink"]), marker=line["marker"], ms=2.6, lw=0.9, zorder=3)  # fmt: skip
                seen.setdefault(line["label"], h)
            ax.set_title(p["title"], fontsize=8)
            ax.set_xlim(*xlim)
            ax.set_ylim(-1.0, 1.02)
            ax.set_xlabel("epoch", fontsize=7)
            ax.tick_params(labelsize=6.5)
        axes[0].set_ylabel("rank correlation across seeds", fontsize=7)
        fig.legend(list(seen.values()), list(seen), loc="outside upper center", ncols=3, frameon=False, fontsize=6.5)
        return fig

    return _plot()


def screen_figure() -> str:
    ep = SCREEN_EPOCHS
    lines = {k: {(a, b): (ep, R_SCREEN[(a, b, k)]) for a, b in PAIRS} for k in ("task", "alpha")}
    early = {(FIXED, ANNEAL): (EARLY_EPOCHS, [EARLY_R[("removal", e)] for e in EARLY_EPOCHS])}
    early_spill = {(FIXED, ANNEAL): (EARLY_EPOCHS, [EARLY_R[("spill", e)] for e in EARLY_EPOCHS])}
    panels = [
        panel_data("task score", lines["task"], chance(len(SCREEN_SEEDS)), (ep[1:], NEXT_SCREEN["task"])),
        panel_data(
            "α at the example answers", lines["alpha"], chance(len(SCREEN_SEEDS)), (ep[1:], NEXT_SCREEN["alpha"])
        ),  # fmt: skip
        panel_data("removal (40 seeds)", early, chance(len(EARLY_SEEDS))),
        panel_data("spill (40 seeds)", early_spill, chance(len(EARLY_SEEDS))),
    ]
    alt = f"""
        Four panels of rank correlation against epoch, 4 to 64. Task score: all four pairings start near 0.9 at epoch 4
        and fall to about 0.5 to 0.6 by epochs 44 to 64, below the dashed line for a run against itself four epochs
        on (about {LATE_NEXT["task"]:.1f}). Alpha: the pairings start between 0.6 and 0.75 and settle at about 0.3 to
        0.5, near the dashed line, which is itself low. Removal at epochs 16, 32 and 64 sits between 0.3 and 0.4, and
        spill between 0.05 and 0.25, inside or at the edge of the grey band of chance.
    """
    return r_draw(
        "paired-runs-screen",
        panels,
        (6.8, 2.4),
        (0, 66),
        alt,
        f"""
        **From one initialization: how alike the runs of a seed stay under two recipes.** Each line is one pairing of
        recipes, and each point the rank correlation of a score across the {len(SCREEN_SEEDS)} seeds of the loss-rate
        screen (712 to 743), at one record. **Left two:** the task score and α at the example answers, recorded every
        four epochs during training; the dashed line is each run against itself at its previous record, averaged over
        the four recipes. **Right two:** removal and spill under the full e₁ edit at the three checkpoints kept to
        epoch 64, for the one pairing that scored them there, at forty seeds (704 to 743). The grey band is where a
        correlation falls about nineteen times in twenty when the two scores are unrelated.
        """,
    )


def full_figure() -> str:
    kept = {(FIXED, ANNEAL): (KEPT_EPOCHS, KEPT_R["removal"])}
    kept_spill = {(FIXED, ANNEAL): (KEPT_EPOCHS, KEPT_R["spill"])}
    task = R_FULL_BINNED
    band = chance(len(SEEDS8))
    panels = [
        panel_data("removal", kept, band),
        panel_data("spill", kept_spill, band),
        panel_data("task score", task, band),
    ]
    alt = """
        Three panels of rank correlation against epoch, 0 to 400, at eight seeds. Removal and spill under τ_ℓ fixed
        against τ_ℓ anneal wander between about −0.7 and 0.6 from epoch 32 on, inside the wide grey band of chance
        (±0.72), with removal at 0.88 at epoch 16 the one point outside it. The task score of each pairing, including
        each recipe against the control, wanders between about 0 and 0.8, mostly inside the band.
    """
    return r_draw(
        "paired-runs-full",
        panels,
        (6.8, 2.5),
        (0, 405),
        alt,
        f"""
        **Over the whole run, at eight seeds.** The rank correlation across seeds 712 to 719. **Left two:** removal and
        spill under the full e₁ edit, `τ_ℓ fixed` against `τ_ℓ anneal`, at the checkpoints kept at epochs 16, 32 and 64
        and then every 32 epochs, and at the end. **Right:** the task score, for the two recipes against each other and each
        against the control at the same seeds; each point is the mean correlation over the records (every four epochs)
        of the 32 epochs up to it, or the 16 up to the end. The grey band is where a
        correlation over eight seeds falls about nineteen times in twenty when the two scores are unrelated
        (±{band:.2f}).
        """,
    )


def branch_figure() -> str:
    removal = {a: (LSS_EPOCHS, LSS_R[(a, "removal")]) for a in ARMS}
    spill = {a: (LSS_EPOCHS, LSS_R[(a, "spill")]) for a in ARMS}
    task = {a: R_BRANCH_BINNED[(a, "task")] for a in (*ARMS, GPU)}
    band = chance(len(OLD8))
    panels = [
        panel_data("removal", removal, band),
        panel_data("spill", spill, band),
        panel_data("task score", task, band),
    ]
    alt = """
        Three panels of rank correlation against epoch, 64 to 400, at eight seeds. Removal stays between about 0.5 and
        0.95 for all three arms, highest under τ_ℓ late; under the two anti arms it falls to about 0.5 after epoch 300.
        Spill stays between about 0.7 and 0.98 for every arm. The task score starts near 1 for every arm and for the
        same schedule on L4, stays above about 0.9 to epoch 250, and then falls to between about 0.45 and 0.9, lowest
        for the change of GPU.
    """
    return r_draw(
        "paired-runs-branch",
        panels,
        (6.8, 2.5),
        (60, 405),
        alt,
        f"""
        **From one stored state: the late spill scout.** The rank correlation across seeds 704 to 711 between each arm
        and `base`, all carried on from the stored epoch-64 state of `τ_ℓ 0.1, τ_ℓ fixed` on an A10G. **Left two:**
        removal and spill under the full e₁ edit at the checkpoints kept every 32 epochs from epoch 96, and at the
        end, net of eight reference edits. **Right:** the task score, each point the mean correlation over the records
        of the 32 epochs up to it (16 for the last), with the soft pool scout's own run of the same schedule, trained on L4
        throughout, as one more pairing (`base` on L4). The grey
        band is the chance band for eight seeds (±{band:.2f}).
        """,
    )


def end_table() -> str:
    head = ["recipe", "seeds", "rank correlation", "95% interval", "spread, netted per seed ↓", "spread, netted against the control mean ↓"]  # fmt: skip
    rows = []
    for t in END_TASK:
        lo, hi = interval(t["r"], len(t["seeds"]))
        rows.append([
            f"<code>{SHOWN[t['name']]}</code>",
            f"{t['seeds'][0]}–{t['seeds'][-1]}",
            num(t["r"]),
            span(lo, hi),
            num(t["sd_paired"], ".4f"),
            num(t["sd_unpaired"], ".4f"),
        ])  # fmt: skip
    ths = "".join(f"<th{' class=num' if i > 1 else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f"<td{' class=num' if i > 1 else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)  # fmt: skip
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(
        table,
        caption="""
            **The task score at epoch 400, against the control at the same seeds.** The rank correlation across eight
            seeds between the clean score of each run (mean over ops) and that of the control, with its interval. The
            two spreads are standard deviations over seeds of the task change: netted against the control at the same
            seed, as the D2.2 design nets it, and netted against the seed mean of the control, with the uncertainty of
            that mean included.
        """,
        class_="report-figure",
    )


# --- Values the prose quotes -----------------------------------------------------------------------------------

TASK_FIRST = [FIRST_R[(a, b, "task")] for a, b in PAIRS]
TASK_LATE = [LATE_R[(a, b, "task")] for a, b in PAIRS]
ALPHA_FIRST = [FIRST_R[(a, b, "alpha")] for a, b in PAIRS]
ALPHA_LATE = [LATE_R[(a, b, "alpha")] for a, b in PAIRS]
CHANCE = {n: chance(n) for n in (len(SCREEN_SEEDS), len(EARLY_SEEDS), len(SEEDS8))}
EARLY_REMOVAL = [EARLY_R[("removal", e)] for e in EARLY_EPOCHS if e > 16]
EARLY_SPILL = [EARLY_R[("spill", e)] for e in EARLY_EPOCHS]
KEPT_AFTER = {k: v[1:] for k, v in KEPT_R.items()}  # from epoch 32, once the anchor has formed
FC_400 = {k: float(v[-1]) for k, v in KEPT_R.items()}
FC_400_INTERVAL = interval(FC_400["removal"], len(SEEDS8))
END_PAIRED_LOWER = sum(t["sd_paired"] < t["sd_unpaired"] - 5e-4 for t in END_TASK)
END_PAIRED_HIGHER = sum(t["sd_paired"] > t["sd_unpaired"] + 5e-4 for t in END_TASK)
END_INTERVALS_HAVE_ZERO = all(interval(t["r"], len(t["seeds"]))[0] < 0 for t in END_TASK)
assert END_INTERVALS_HAVE_ZERO
LSS_SPILL_MIN = min(float(LSS_R[(a, "spill")].min()) for a in ARMS)
LSS_REMOVAL_LT = LSS_R[(LT_LATE, "removal")]
LSS_REMOVAL_ANTI_LATE = [float(LSS_R[(a, "removal")][LSS_LATE].mean()) for a in (ANTI_LATE, ANTI_HELD)]
BRANCH_TASK_LATE = {a: float(R_BRANCH[(a, "task")][BRANCH_LATE].mean()) for a in (*ARMS, GPU)}
WORD = "no one two three four five six seven eight nine ten".split()

# %%

rf"""
# When paired runs stop being alike

/// tip |
<!-- lede -->
Two runs at one seed share their initialization, batches and labels. Started together under two recipes, they stay partly alike in the task score through the first 64 epochs, but in removal and spill they are little more alike than runs at different seeds, whether the recipes differ a little or a lot. Carried on from one stored state, they stay alike in all three for hundreds of epochs.
///

We compare conditions as two samples, and pair runs by seed only when they branch from one stored state and are measured soon after ([eng/determinism.md](/eng/determinism.md#seeds-are-draws)). "Soon after" has no number yet. The evidence for the rule is that removal under `τ_ℓ fixed` and `τ_ℓ anneal`, which share their seeds, correlated about 0.2 across eight seeds at epoch 400 (the fixed confirmation), and the low-runs report found that twin runs, the same in all but the GPU arithmetic, are as far apart in their weights as unrelated runs by epoch 16, while their task loss agrees to a hundredth.

So the weights part early, and some scores do not. This report asks how alike two runs at one seed stay, score by score, as training goes on. It runs nothing new: it reads the trajectories and kept checkpoints that four earlier experiments stored, and for each pair of conditions trained at the same seeds, it takes the correlation of a score across seeds at each record.

## Observations

- [From one initialization (E1)](#from-one-initialization-e1): over the first 64 epochs, the task score of the two runs of a seed falls from nearly the same order across seeds to about half of it, under every pairing of recipes. Removal correlates by about a third, at the edge of chance, and spill by no more than chance. The alignment with e₁ moves from record to record so much that one record of a run predicts its next record about as poorly as it predicts the paired run.
- [Over the whole run (E2)](#over-the-whole-run-e2): at eight seeds, no correlation of removal or spill after the anchor forms can be told from zero, the correlation of 0.2 at epoch 400 included. The task score of a run is only loosely tied to that of the control at its seed, and netting the task change against that control seed by seed leaves its spread about where it was.
- [From one stored state (E3)](#from-one-stored-state-e3): runs carried on from one epoch-64 state stay alike in spill to the end, and in removal and the task score for at least a hundred and fifty epochs, under each change of schedule and under a change of GPU. The largest change loosens removal after epoch 300.

## Scope and method

This is an exploratory re-analysis of stored results, with no preregistration and no gate. It takes three kinds of pairing, in rising order of what the two runs of a seed share.

From one initialization
:   Four recipes, each at the {len(SCREEN_SEEDS)} seeds of the loss-rate screen (712 to 743), stopped at epoch 64: `τ_ℓ fixed` and `τ_ℓ anneal` from the fixed and anneal confirmations, and `step, soft τ` and `falloff` from the loss-rate screen. At a seed the four share their initialization, batches, and label draws, and differ by more or less. `τ_ℓ anneal` is `τ_ℓ fixed` with the slice temperature sharpening, by under 1% before epoch 64. `step, soft τ` pulls at a step over slices 2 to 4 in place of the pool over slices 1 to 3, and `falloff` is that step with the position temperature held fixed. The fixed confirmation also scored the edit on both of its recipes at epochs 16, 32, and 64, at forty seeds (704 to 743).

Over the whole run
:   `τ_ℓ fixed` and `τ_ℓ anneal` at seeds 712 to 719, trained to epoch 400, and the control at the same seeds (no pull and no anti term). At the end, three more recipes at seeds 704 to 711 with their own control, as the reference-baseline experiment and the soft pool scout scored them.

From one stored state
:   The four arms of the late spill scout at seeds 704 to 711 (`base`, `τ_ℓ late`, `anti late`, and `anti held`), each carried on from the stored epoch-64 state of `τ_ℓ 0.1, τ_ℓ fixed` at its seed, on an A10G. Each arm changes one schedule from epoch 64 on. The soft pool scout trained the same schedule as `base` from the same state on an L4, which gives one more pairing: the same model, on another GPU.

Within each pairing both runs trained on the same kind of GPU (L4 for the first two kinds, A10G for the third), except for that last one. As a check that the code pairs the right runs, `anti late` and `anti held`, which share a schedule up to epoch 200, agree exactly at every kept checkpoint to epoch 192.

A correlation over a few seeds is noisy. A correlation between unrelated scores lands within ±{CHANCE[len(SCREEN_SEEDS)]:.2f} of zero about nineteen times in twenty at {len(SCREEN_SEEDS)} seeds, within ±{CHANCE[len(EARLY_SEEDS)]:.2f} at forty, and within ±{CHANCE[len(SEEDS8)]:.2f} at eight. Each figure shades that band. Neighboring records of one pairing are not independent draws, so a correlation that stays above the band over many records says more than one point does.

## The measurements

Rank correlation across seeds
:   For two conditions trained at the same seeds, how well the order of the seeds on a score under one condition predicts their order under the other, at one epoch (Spearman's correlation). It is 1 when the seeds come in the same order under both, and near 0 when the order under one says nothing about the other. Near 1, a difference taken seed by seed cancels most of the variation between seeds; near 0, it cancels none of it. Ranks rather than values, because spill sits near zero on most runs, and one run that spills under both conditions would set the correlation of the values on its own.

Task score
:   Expected exact match on the held-out probe contexts, averaged over ops, as recorded during training every four epochs. Higher is better. At the end (E2), the clean score of the scoring suite, averaged over ops.

α at the example answers
:   The mean alignment with e₁ of the states at the example answers of `difference` contexts, at slice 3, recorded every four epochs: how far the pull has brought the anchored op onto e₁. The loss-rate call averages it over epochs 44 to 64.

Removal
:   The drop on `difference` under the full e₁ edit at every position and slice, net of the median drop under the same edit along the other 63 coordinate axes of the run. In E3, net of the median of eight reference edits, as spill-timing scored it. It is the net drop, not a share of the way to the target null, since a stopped run has no end record to scale it by.

Spill
:   The largest net drop on any other op, under the same edit.
"""

# %%

rf"""
## From one initialization (E1)

Started from the same initialization under two recipes, how long do the runs of a seed stay more alike than runs at different seeds? If the seed sets most of what a run does, the correlation should stay high whatever the change; if small differences grow quickly, it should fall early, and fall faster the larger the change.

{screen_figure()}

At epoch 4, the two runs of a seed come in nearly the same order on the task score under every pairing (a rank correlation of {min(TASK_FIRST):.2f} to {max(TASK_FIRST):.2f}). The correlation falls slowly from there, and over epochs 44 to 64 it averages about a half ({min(TASK_LATE):.2f} to {max(TASK_LATE):.2f}). From epoch 16 on, a run against itself four epochs later stays near {LATE_NEXT["task"]:.1f}, so the fall is the two runs parting, not the score jumping about from one record to the next.

The size of the change makes no clear difference. `τ_ℓ fixed` and `falloff`, which pull at different slices with different temperatures, stay about as alike in the task score as `τ_ℓ fixed` and `τ_ℓ anneal`, which differ by under 1% in one temperature. That fits what the low-runs report found in the weights: a small difference grows as quickly as a large one.

The alignment α behaves differently. The runs of a seed start alike in it ({min(ALPHA_FIRST):.2f} to {max(ALPHA_FIRST):.2f} at epoch 4), and by epoch 44 the correlation is about a third ({min(ALPHA_LATE):.2f} to {max(ALPHA_LATE):.2f}). But a run correlates with itself four epochs later about as weakly ({LATE_NEXT["alpha"]:.2f}). So α at one record moves a lot from record to record, and the paired run at the same epoch is about as good a guide to it as the run itself four epochs before. The two runs see the same batches in the same order, which may keep some of that movement in step, though nothing here tests it.

Removal under the edit correlates by about a third at epochs 32 and 64 ({span(min(EARLY_REMOVAL), max(EARLY_REMOVAL))} over forty seeds), at the edge of chance. Spill stays inside the band at all three checkpoints. At epoch 16 the anchor has barely formed, and removal is close to zero on every run.

So by epoch 64, the seed still says something about the task score, a little about removal, and nothing we can detect about spill. Pairing by seed from one initialization would cancel about half of the variation between seeds in the task score, and little of it in removal or spill.
"""

# %%

rf"""
## Over the whole run (E2)

Does what is left of the pairing at epoch 64 last to the end of training? And how alike is a run to the control at its seed, which the task change of the D2.2 design nets against?

{full_figure()}

At eight seeds the band of chance is wide, and from epoch 32 on every correlation of removal and spill between the two recipes falls inside it, wandering from about {num(min(KEPT_AFTER["spill"].min(), KEPT_AFTER["removal"].min()), ".1f")} to {num(max(KEPT_AFTER["spill"].max(), KEPT_AFTER["removal"].max()), ".1f")} between neighboring checkpoints. So the correlation of about 0.2 at epoch 400 is consistent with anything from no pairing at all to a strong one: as a rank correlation it is {num(FC_400["removal"])}, with an interval of {span(*FC_400_INTERVAL)}. The forty seeds of E1 pin it down better at epoch 64, to about a third.

The task score is paired as unsteadily. The two recipes correlate by about a third to a half over the run, and each correlates with the control about as well, though the control has neither the pull nor the anti term. So for the task score, what the two runs of a seed share seems to matter more than how alike their recipes are.

{end_table()}

At epoch 400, the task score of a run correlates with that of the control at its seed by {min(END_R):.2f} to {max(END_R):.2f} across the five sets, and every interval reaches below zero. Netting the task change against the control seed by seed gives a spread close to that of netting against the seed mean of the control: a little smaller on {WORD[END_PAIRED_LOWER]} sets and larger on {WORD[END_PAIRED_HIGHER]}. The control spreads about half as much as the anchored runs, which is why the choice matters little either way.

So at the end of training, eight seeds cannot show that pairing by seed helps on any of these scores, and the per-seed netting of the task change neither helps nor hurts it much.
"""

# %%

rf"""
## From one stored state (E3)

Runs carried on from one stored state share everything up to it: by epoch 64 the seeds have already parted, and both runs of a seed start from where that seed got to. How long do they stay alike after the branch, and does it depend on the change?

{branch_figure()}

They stay alike for a long time. In spill, every arm keeps a rank correlation with `base` of at least {LSS_SPILL_MIN:.2f} at every checkpoint to the end. In removal, every arm stays mostly between 0.8 and 0.95 to epoch 288, with a dip to about a half at single checkpoints. After that `τ_ℓ late` stays near {LSS_REMOVAL_LT[LSS_LATE].mean():.2f} on average, while the two anti arms fall to about a half ({num(min(LSS_REMOVAL_ANTI_LATE))} and {num(max(LSS_REMOVAL_ANTI_LATE))}), with their anti weight still above that of `base`. The late spill scout found the anti arms lowering removal by the end, by about a quarter on one seed, so the runs of a seed parting here fits a change that by then has a real effect.

The task score starts at a correlation near 1 for every arm and for the change of GPU, and stays at about 0.9 or above to about epoch 250. After that it falls, least for the anti arms (about {min(BRANCH_TASK_LATE[ANTI_LATE], BRANCH_TASK_LATE[ANTI_HELD]):.2f} on average after epoch 288), more for `τ_ℓ late` ({BRANCH_TASK_LATE[LT_LATE]:.2f}), and most for the change of GPU ({BRANCH_TASK_LATE[GPU]:.2f}). At the end, `base` on the A10G and on the L4 also correlate in removal and spill ({GPU_END["removal"]:.2f} and {GPU_END["spill"]:.2f}), though at eight seeds only the second clears the band.

So from a stored state, pairing by seed pays on all three scores for at least a hundred and fifty epochs, under each change here and under a change of GPU. The branch carries the seed-to-seed differences of the first 64 epochs into both runs, and the changes after it add less than those differences, until late in the run.
"""

# %%

r"""
## Discussion

There is no one number for "soon after". From one initialization, the two runs of a seed are little more alike in removal and spill than runs at different seeds by the first checkpoints at which the anchor exists, and they keep about half of their pairing in the task score through epoch 64, and less after. From a stored state at epoch 64, they stay paired in all three for hundreds of epochs. So what the runs share when they part seems to matter more than how long ago they parted. In E1 the size of the change made no clear difference either.

That fits the design rule, and it puts the weight on the stored state rather than on the time since. If it holds beyond these experiments, arms that branch from a stored state after the anchor forms can be paired for most of a run, while conditions that share only an initialization gain little from pairing on removal or spill at any checkpoint we have.

The evidence for the long pairing rests on one stored state at one epoch and on eight seeds, so how far it reaches is not settled. A stored state taken earlier, before the seeds have parted, may behave more like a shared initialization.

The task change of the D2.2 design nets each run against the control at its seed. At epoch 400 that neither narrows nor widens its spread much (E2). If the gate were set against the seed mean of the control instead, it would see about the same spread.

α at the example answers moves so much from record to record that a single record is a poor measure of a run (E1). The loss-rate call averages it over six records, which this supports.
"""

# %%

r"""
## References

[fixed confirmation](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed` at the fresh seeds of the anneal confirmation: removal under the two recipes correlated about 0.2 across eight seeds at epoch 400, and the two runs of a seed were up to 0.43 apart in removal at epoch 32.

[anneal confirmation](/docs/m2/anneal-confirm/report.py)
:   The pool over slices 1 to 3 with the slice temperature annealing from 0.1 to 0.025, at seeds 712 to 719 to the end and at forty seeds to epoch 64.

[loss-rate screen](/docs/m2/loss-rate/report.py)
:   `falloff` and `step, soft τ` at the same 32 fresh seeds to epoch 64, called kept or lost from α at the example answers over epochs 44 to 64.

[late spill scout](/docs/m2/late-spill-scout/report.py)
:   Four ways of carrying `τ_ℓ fixed` on from its stored epoch-64 states; holding the anti weight high lowered removal, by about a quarter on one seed.

[soft pool scout](/docs/m2/soft-pool-scout/report.py)
:   The pool over slices 1 to 3 at seeds 704 to 711 on L4, whose epoch-64 states the late spill scout carried on from.

[reference-baseline experiment](/docs/m2/reference-baseline/report.py)
:   Removal and spill net of the median drop along the other coordinate axes of the run itself, scored on the stored runs of `falloff`, `step, soft τ`, and the control at seeds 704 to 711.

[low-runs report](/docs/m2/low-runs/report.py)
:   Twin runs, the same in all but the GPU arithmetic, are as far apart in their weights as unrelated runs by epoch 16, while their task loss agrees to a hundredth for the whole run.

[spill-timing](/docs/m2/spill-timing/report.py)
:   Spill along training, net of the median of eight reference edits.
"""
