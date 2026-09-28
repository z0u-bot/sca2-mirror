# title: Ex 2.2.17: the center control plateau, a scout

# The design constants and the refs come from `experiment.py` beside this script (the script's directory is
# on sys.path while it runs); the answer table and the posterior come from ex-2.2.16 through it.
import colorsys
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed
from sca.data.ops import colors
from sca.vis import plot_rgb_cube

X = ex.ex2216
OPS: tuple[str, ...] = tuple(X.OP_NAMES)
GROUPS: tuple[tuple[str, tuple[str, ...] | None], ...] = (
    ("all ops", None),
    ("mix, screen, multiply", ("mix", "screen", "multiply")),
    ("lighten, darken", ("lighten", "darken")),
    ("difference, exclusion", ("difference", "exclusion")),
    ("hsvmix", ("hsvmix",)),
    ("HSV channels", ("hue-hsv", "sat-hsv", "value-hsv")),
)
PEAK = ex.SWEEP_LRS[2]
# Bands of the posterior on the true op, and the confident band the answer analysis reads.
BANDS = (0.0, 0.5, 0.9, 0.99, 1.01)
BAND_LABELS = ("< 0.5", "0.5–0.9", "0.9–0.99", "> 0.99")
CONFIDENT = 0.99
# A grid color is in the support of the Bayes predictive when q gives it at least this much.
SUPPORT = 0.02
HUE_GAPS = (0, 30, 90, 150, 180.1)
CUBE_RUN = f"sweep-{PEAK:g}-s1"
RGB = np.array(colors(), dtype=float) / 15


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


def gate(ax: Axes, level: float) -> None:
    """The pass line as a dashed rule with the failing side hatched, drawn last so the limits hold."""
    lo, hi = ax.get_ylim()
    ax.axhline(level, ls="--", color=rule_color(), lw=1)
    ax.axhspan(lo, level, facecolor="none", edgecolor=rule_color(), hatch="//", lw=0, alpha=0.1, zorder=0)
    ax.set_ylim(lo, hi)


# --- Loading ------------------------------------------------------------------------------------------------


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists: one `get_refs` and one `get_many`."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict:
    assert path is not None, "ex-2.2.17 has not published this yet"
    return json.loads(path.read_text())


with tempfile.TemporaryDirectory() as _tmp:
    _files = fetch([ex.EVAL_REF, ex.TRAJ_REF, ex.FINDER_REF], Path(_tmp))
    EVAL = read_json(_files[ex.EVAL_REF])
    TRAJ = read_json(_files[ex.TRAJ_REF])
    FINDER = read_json(_files[ex.FINDER_REF])
    RUNS = {r["label"]: r for r in EVAL["runs"]}
    SCORED = tuple(r["label"] for r in EVAL["runs"] if ex.scored(r))
    _refs = [ex.DETAIL_REF, *(ex.SCORE_REF.format(label=lbl) for lbl in SCORED)]
    _arrays = fetch(_refs, Path(_tmp))
    assert all(_arrays.values()), "ex-2.2.17 has not published its answer scoring yet"
    with np.load(cast(Path, _arrays[ex.DETAIL_REF])) as _z:
        DETAIL = {k: _z[k] for k in _z.files}
    ANSWERS = {
        lbl: np.load(cast(Path, _arrays[ex.SCORE_REF.format(label=lbl)]))["p"]
        for lbl in SCORED  # float16, (contexts, 216)
    }


def eem(label: str) -> float:
    return RUNS[label]["task"]["eem"]["all"]


def kl(label: str) -> float:
    return RUNS[label]["task"]["kl"]["all"]


CEILING = RUNS[SCORED[0]]["task"]["ceiling"]["all"]
PASS = CEILING - X.CEILING_MARGIN
CEILING_PER_OP = np.array(RUNS[SCORED[0]]["task"]["ceiling"]["per_op"])


def labels_of(arm: str, seeds: range = range(3)) -> list[str]:
    return [f"{arm}-s{s}" for s in seeds if f"{arm}-s{s}" in RUNS]


def seed_stats(labels: Sequence[str], f=eem) -> tuple[float, float, float]:
    v = [f(lbl) for lbl in labels]
    return float(np.mean(v)), min(v), max(v)


# --- The answer scoring --------------------------------------------------------------------------------------


@memo
def answer_table():
    return X._get_posterior().build_table(X.TABLE)


@memo
def context_info(op: np.ndarray, pair: np.ndarray, post: np.ndarray, source: np.ndarray) -> dict:
    """Per held-out context: its posterior band, its count of noisy examples, the hue gap of its operands, and
    which grid colors are an answer of some op on its query pair (`any_op`, contexts × 216).
    """
    table = answer_table()
    n = len(op)
    rows = np.arange(n)
    any_op = np.zeros((n, len(RGB)), dtype=bool)
    for o in range(X.N_OPS):
        idx, ok = table.idx[o, pair], table.prob[o, pair] > 0
        for j in range(idx.shape[1]):
            any_op[rows[ok[:, j]], idx[ok[:, j], j]] = True
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in RGB])
    a, b = pair // len(RGB), pair % len(RGB)
    gap = np.abs(hsv[a, 0] - hsv[b, 0]) * 360
    gap = np.minimum(gap, 360 - gap)
    return {
        "post_true": post[rows, op],
        "band": np.digitize(post[rows, op], BANDS) - 1,
        "noise": (source == 1).sum(axis=1),
        "gray": (hsv[a, 1] == 0) | (hsv[b, 1] == 0),
        "hue_gap": gap,
        "any_op": any_op,
        "match_idx": np.maximum(table.idx[op, pair], 0),
        "match_prob": table.prob[op, pair],
    }


INFO = context_info(DETAIL["op_ids"], DETAIL["query_pair"], DETAIL["posterior"], DETAIL["source"])
PREDICTIVE = DETAIL["predictive"].astype(float)
OP_IDS = DETAIL["op_ids"]
HO_CEILING = DETAIL["ceiling"]


def matched(p: np.ndarray, info: dict) -> np.ndarray:
    """Expected exact match per context, as `ex2216._match` computes it."""
    return (np.take_along_axis(p, info["match_idx"], axis=1) * info["match_prob"]).sum(axis=1)


@memo
def run_summary(p16: np.ndarray, q: np.ndarray, op: np.ndarray, info: dict) -> dict:
    """One run's expected exact match by posterior band, by noise count, and (hsvmix) by hue gap; and, on the
    confident contexts of each op, the mass it puts outside the Bayes support, split by whether some op gives
    that answer.
    """
    p = p16.astype(float)
    e = matched(p, info)
    off = q < SUPPORT
    conf = info["post_true"] > CONFIDENT
    hsvmix = (op == OPS.index("hsvmix")) & (info["post_true"] > 0.9) & ~info["gray"]
    gap_bin = np.digitize(info["hue_gap"], HUE_GAPS) - 1
    return {
        "eem": float(e.mean()),
        "by_band": [float(e[info["band"] == i].mean()) for i in range(len(BANDS) - 1)],
        "by_noise": [float(e[info["noise"] == k].mean()) for k in range(ex.CENTRE[0] + 1)],
        "by_gap": [float(e[hsvmix & (gap_bin == i)].mean()) for i in range(len(HUE_GAPS) - 1)],
        "off_other": [float((p * off * info["any_op"])[conf & (op == o)].sum(1).mean()) for o in range(X.N_OPS)],
        "off_none": [float((p * off * ~info["any_op"])[conf & (op == o)].sum(1).mean()) for o in range(X.N_OPS)],
    }


SUMMARY = {lbl: run_summary(ANSWERS[lbl], PREDICTIVE, OP_IDS, INFO) for lbl in SCORED}


def ceiling_by(key: str, n: int) -> list[float]:
    return [float(HO_CEILING[INFO[key] == i].mean()) for i in range(n)]


CEIL_BAND = ceiling_by("band", len(BANDS) - 1)
CEIL_NOISE = ceiling_by("noise", ex.CENTRE[0] + 1)
N_BAND = [int((INFO["band"] == i).sum()) for i in range(len(BANDS) - 1)]
N_NOISE = [int((INFO["noise"] == k).sum()) for k in range(ex.CENTRE[0] + 1)]
_hsv_sel = (OP_IDS == OPS.index("hsvmix")) & (INFO["post_true"] > 0.9) & ~INFO["gray"]
_gap_bin = np.digitize(INFO["hue_gap"], HUE_GAPS) - 1
CEIL_GAP = [float(HO_CEILING[_hsv_sel & (_gap_bin == i)].mean()) for i in range(len(HUE_GAPS) - 1)]


def mean_over_runs(key: str) -> np.ndarray:
    return np.mean([SUMMARY[lbl][key] for lbl in SCORED], axis=0)


BAND_MEAN = mean_over_runs("by_band")
NOISE_MEAN = mean_over_runs("by_noise")
GAP_MEAN = mean_over_runs("by_gap")
OFF_OTHER = mean_over_runs("off_other")
OFF_NONE = mean_over_runs("off_none")
_conf = INFO["post_true"] > CONFIDENT
_off = PREDICTIVE < SUPPORT
UNIFORM_SHARE = np.array(
    [
        ((_off & INFO["any_op"])[_conf & (OP_IDS == o)].sum(1) / _off[_conf & (OP_IDS == o)].sum(1)).mean()
        for o in range(X.N_OPS)
    ]
)
OTHER_SHARE = OFF_OTHER / (OFF_OTHER + OFF_NONE)

# The leak on other ops' answers against the final score, across the scored runs.
_leak = np.array([np.mean(SUMMARY[lbl]["off_other"]) for lbl in SCORED])
_final = np.array([SUMMARY[lbl]["eem"] for lbl in SCORED])
LEAK_R = float(np.corrcoef(_leak, _final)[0, 1])


# --- Figures ------------------------------------------------------------------------------------------------


def group_lines_draw(ax_top: np.ndarray, ax_lr: Axes, lines: list[tuple], k: int = 3) -> None:
    def smooth(y: np.ndarray) -> np.ndarray:
        pad = np.pad(y, (k // 2, k // 2), mode="edge")
        return np.convolve(pad, np.ones(k) / k, mode="valid")

    for ax, (name, group) in zip(ax_top, GROUPS, strict=True):
        idx = list(range(len(OPS))) if group is None else [OPS.index(o) for o in group]
        ax.axhline(CEILING if group is None else CEILING_PER_OP[idx].mean(), ls="--", color=rule_color(), lw=1)
        ax.set_title(name, fontsize=9)
        for label, color, ls, lw, _ in lines:
            t = TRAJ[label]
            y = np.array(t["eem"]) if group is None else np.array(t["eem_per_op"])[:, idx].mean(1)
            ax.plot(np.array(t["step"]) / 1e3, smooth(y), color=color, ls=ls, lw=lw)
    for label, color, ls, lw, legend in lines:
        t = TRAJ[label]
        ax_lr.plot(np.array(t["step"]) / 1e3, t["lr"], color=color, ls=ls, lw=lw, label=legend)


@memo
def groups_draw(lines: list[tuple], name: str, caption: str, alt_text: str) -> str:
    @themed(name=name, alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig = plt.figure(figsize=(8.4, 6.8), layout="constrained")
        grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.75])
        top = [fig.add_subplot(grid[i // 3, i % 3]) for i in range(len(GROUPS))]
        for ax in top[1:]:
            ax.sharex(top[0])
            ax.sharey(top[0])
        lr = fig.add_subplot(grid[2, :], sharex=top[0])
        group_lines_draw(np.array(top), lr, lines)
        top[0].set_ylim(0, 0.85)
        for ax in top[::3]:
            ax.set_ylabel("held-out EEM")
        for ax in top[3:]:
            ax.set_xlabel("step (thousands)", fontsize=8)
        lr.set_yscale("log")
        lr.set_ylim(1e-5, 1.5e-2)
        lr.set_ylabel("learning rate")
        lr.set_xlabel("step (thousands)")
        handles, labels = lr.get_legend_handles_labels()
        keep = [(h, lbl) for h, lbl in zip(handles, labels, strict=True) if not lbl.startswith("_")]
        fig.legend(
            *zip(*keep, strict=True), loc="outside upper center", ncols=min(len(keep), 4), frameon=False, fontsize=7
        )
        return fig

    return _plot()


@memo
def conditions_draw(
    conditions: list[tuple[str, list[str]]], values: dict, ceiling: float, passing: float, alt_text: str
) -> str:
    @themed(
        name="conditions",
        alt_text=alt_text,
        caption=f"""
            **The final score of every masked condition at peak {PEAK:g}.** Each column is one condition; faded dots
            are its seeds, the bar their range, and the diamond their mean. Left, held-out expected exact match, with
            the Bayes ceiling as a solid rule and the pass line as a dashed rule over the hatched failing side. Right,
            the calibration KL (lower is better calibrated).
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), layout="constrained")
        rng = np.random.default_rng(0)
        for ax, key in zip(axes, ("eem", "kl"), strict=True):
            for i, (_, labels) in enumerate(conditions):
                v = np.array([values[lbl][key] for lbl in labels])
                c = light_dark("#1f5fa8", "#7fb2ff")
                ax.plot([i, i], [v.min(), v.max()], color=c, lw=6, alpha=0.25, solid_capstyle="butt", zorder=1)
                ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=c, alpha=0.5, lw=0, zorder=2)
                ax.scatter([i], [v.mean()], s=40, marker="D", color=c, zorder=3)
            ax.set_xticks(range(len(conditions)), [n for n, _ in conditions], rotation=40, ha="right", fontsize=7)
            ax.set_xlim(-0.6, len(conditions) - 0.4)
        axes[0].set_ylim(0.40, 0.53)
        axes[0].axhline(ceiling, color=rule_color(), lw=1)
        gate(axes[0], passing)
        axes[0].set_ylabel("held-out EEM")
        axes[1].set_ylim(0, 1.0)
        axes[1].set_ylabel("calibration KL (nats)")
        return fig

    return _plot()


@memo
def gap_draw(summary: dict, ceil_band: list, ceil_noise: list, alt_text: str) -> str:
    @themed(
        name="gap",
        alt_text=alt_text,
        caption=f"""
            **Where the model falls short of the ceiling.** Held-out expected exact match of the {len(summary)} scored
            runs (thin lines) and their mean (thick), with the Bayes ceiling dashed. Left, contexts grouped by the
            posterior on the true op given their examples; right, by how many of their {ex.CENTRE[0]} examples carry
            replacement op noise.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), layout="constrained", sharey=True)
        c = light_dark("#1f5fa8", "#7fb2ff")
        for ax, key, ceil, ticks in (
            (axes[0], "by_band", ceil_band, BAND_LABELS),
            (axes[1], "by_noise", ceil_noise, [str(k) for k in range(len(ceil_noise))]),
        ):
            x = np.arange(len(ceil))
            for s in summary.values():
                ax.plot(x, s[key], color=c, lw=0.6, alpha=0.35)
            ax.plot(x, np.mean([s[key] for s in summary.values()], axis=0), color=c, lw=2, marker="o", ms=4)
            ax.plot(x, ceil, ls="--", color=rule_color(), lw=1.2, marker="_", ms=10)
            ax.set_xticks(x, ticks)
        axes[0].set_xlabel("posterior on the true op")
        axes[1].set_xlabel("noisy examples in the context")
        axes[0].set_ylabel("held-out EEM")
        axes[0].set_ylim(0, 0.8)
        return fig

    return _plot()


def cube_data(label: str, n: int = 90, seed: int = 0) -> dict:
    """On the confident contexts of each op, a fixed sample: the expected answer color under the model and under the
    Bayes predictive (cube units, 0 to 1). And the same for hsvmix, by hue gap of the operands.
    """
    rng = np.random.default_rng(seed)
    p = ANSWERS[label].astype(float)
    em = (p / p.sum(1, keepdims=True)) @ RGB
    eq = PREDICTIVE @ RGB
    per_op = {}
    for o, name in enumerate(OPS):
        idx = np.flatnonzero((OP_IDS == o) & (INFO["post_true"] > CONFIDENT))
        idx = rng.choice(idx, min(n, len(idx)), replace=False)
        per_op[name] = (em[idx], eq[idx])
    per_gap = []
    for i in range(len(HUE_GAPS) - 1):
        idx = np.flatnonzero(_hsv_sel & (_gap_bin == i))
        idx = rng.choice(idx, min(n, len(idx)), replace=False)
        per_gap.append((em[idx], eq[idx]))
    return {"per_op": per_op, "per_gap": per_gap}


@memo
def cubes_draw(per_op: dict, alt_text: str) -> str:
    @themed(
        name="answer-cubes",
        alt_text=alt_text,
        caption=f"""
            **Model answers against Bayes answers, per op.** Each panel is the color cube seen down its gray
            diagonal (white at the center). Each dot is one confident held-out context (posterior on the true op
            above {CONFIDENT:g}) of `{CUBE_RUN}`, at the expected answer color under the model, and its ring is the
            expected answer color under the Bayes predictive, joined by a stub. Up to 90 contexts per op.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(3, 4, figsize=(8.4, 6.6), layout="constrained")
        flat = cast(np.ndarray, axes).ravel()
        for ax, (name, (em, eq)) in zip(flat, per_op.items(), strict=False):
            plot_rgb_cube(ax, em, truth=eq, s=7, view="wheel")
            ax.set_title(name, fontsize=9)
        for ax in flat[len(per_op) :]:
            ax.set_axis_off()
        return fig

    return _plot()


@memo
def hue_gap_draw(per_gap: list, eem_gap: list, ceil_gap: list, alt_text: str) -> str:
    @themed(
        name="hsvmix-hue-gap",
        alt_text=alt_text,
        caption="""
            **hsvmix by the hue gap between its operands.** As in the figure above, for hsvmix contexts with a
            posterior on the true op above 0.9 and two chromatic operands, grouped by how far apart their hues are.
            Under each panel: the expected exact match, mean over the scored runs, against the Bayes ceiling.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.6), layout="constrained")
        for i, ax in enumerate(axes):
            em, eq = per_gap[i]
            plot_rgb_cube(ax, em, truth=eq, s=7, view="wheel")
            ax.set_title(f"{HUE_GAPS[i]:.0f}°–{min(HUE_GAPS[i + 1], 180):.0f}°", fontsize=9)
            ax.text(0, -1.3, f"{eem_gap[i]:.2f} of {ceil_gap[i]:.2f}", ha="center", fontsize=8)
        return fig

    return _plot()


# --- Numbers the prose quotes ----------------------------------------------------------------------------------

R1 = {arm: seed_stats(labels_of(arm)) for arm in ("low", "long", "long-low")}
R1_KL = {arm: seed_stats(labels_of(arm), kl) for arm in ("low", "long", "long-low")}
SWEEP = sorted((r for r in EVAL["runs"] if r["round"] == 2 and r["mask"]), key=lambda r: r["peak_lr"])
BEST_SWEEP = max(SWEEP, key=lambda r: r["task"]["eem"]["all"])
NOMASK = RUNS[f"sweep-nomask-{PEAK:g}-s0"]
MASKED = RUNS[f"sweep-{PEAK:g}-s0"]
FINDER_STEEP = {f["label"]: f["history"][0]["steepest_lr"] for f in FINDER["runs"]}
STEPS_8X = TRAJ[f"sweep-{PEAK:g}-s0"]["step"][-1]
STEPS_16X = TRAJ[f"sweep16x-{PEAK:g}-s0"]["step"][-1]

CONDITIONS: list[tuple[str, list[str]]] = [
    ("cosine 8×", labels_of(f"sweep-{PEAK:g}")),
    ("WSD 8×", labels_of(f"wsd-{PEAK:g}")),
    ("staircase 8×", labels_of(f"stairs-{PEAK:g}")),
    ("cosine 16×", labels_of(f"sweep16x-{PEAK:g}")),
    ("WSD 16×", labels_of(f"wsd16x-{PEAK:g}")),
    ("staircase 16×", labels_of(f"stairs16x-{PEAK:g}")),
    ("d128-L4 8×", labels_of(f"d128-sweep-{PEAK:g}")),
    ("d64-L6 8×", labels_of(f"d64L6-sweep-{PEAK:g}")),
]
COND_EEM = {name: seed_stats(labels) for name, labels in CONDITIONS}
COND_KL = {name: seed_stats(labels, kl) for name, labels in CONDITIONS}
BEST = max(SCORED, key=eem)
SPREAD_8X = [eem(lbl) for name, labels in CONDITIONS[:3] for lbl in labels]
HSVMIX = OPS.index("hsvmix")
BAND_GAP = [c - e for c, e in zip(CEIL_BAND, BAND_MEAN, strict=True)]
TOTAL_GAP = CEILING - float(np.mean([SUMMARY[lbl]["eem"] for lbl in SCORED]))
# How much of the whole gap each band carries: its share of contexts times its gap.
BAND_SHARE = [n * g / sum(N_BAND) / TOTAL_GAP for n, g in zip(N_BAND, BAND_GAP, strict=True)]

rf"""
This scout set out to lift ex-2.2.16's center control (the unanchored d64-L4 model on the corpus condition `k3-r0.3`) to its Bayes ceiling. Ex-2.2.16 trained it for 50 epochs at a peak learning rate of 0.01 and it reached 0.27 held-out expected exact match (EEM), against a ceiling of {CEILING:.3f}. The pass line that experiment set for its controls is {X.CEILING_MARGIN:g} below the ceiling, at {PASS:.3f}.

<details><summary>The measurements</summary>

Held-out *expected exact match* is the probability that an answer drawn from the model's distribution at the query `=` is a correct answer of the true op, averaged over {len(OP_IDS):,} held-out contexts, {len(OP_IDS) // X.N_OPS:,} per op. The *Bayes ceiling* is the same score for the ideal predictor, which weighs the ops by how well each explains the examples of a context and answers with the resulting mixture. It is below 1 because some ops round stochastically and because noisy examples leave the op uncertain. The *calibration KL* is the KL divergence from that ideal answer distribution to the model's, in nats: how much more the model loses on the answer than the ideal predictor does. Zero means the model holds the same distribution the examples support.

</details>

Seven rounds of training moved the control from 0.27 to about 0.45. The rest of the gap did not move. Past the first two rounds, neither the shape of the learning-rate schedule, nor twice the training steps, nor a wider or deeper model changed the final score by more than the spread between seeds. The best run, `{BEST}`, reached {eem(BEST):.3f}, still {PASS - eem(BEST):.3f} short of the pass line.

Scoring the answer distributions against the Bayes predictive shows where the gap sits. The model matches the ceiling where the examples say little about the op, and falls short where they point clearly to one op. Even when the examples settle the op, it keeps some mass on answers that other ops would give, at several times the rate chance would put there. So part of the gap is in how firmly the model commits to the op the examples support. The op hsvmix adds a second, separate shortfall that grows with the distance between the operand hues.

## Rounds 1 and 2: more steps and a lower rate

Round 1 tested two ideas at three seeds each: that the peak learning rate of 0.01 was too high, or that the model needed more steps. The `low` arm trained for ex-2.2.16's length at 0.003; `long` trained for three times the length at 0.01; and `long-low` did both. Neither change alone did much, and the two together did the most.

"""

table_html(
    ["arm", "length", "peak LR", "EEM, seed mean (range)", "KL, seed mean"],
    [
        [
            f"`{arm}`",
            f"{mult}×",
            f"{lr:g}",
            f"{R1[arm][0]:.3f} ({R1[arm][1]:.3f}–{R1[arm][2]:.3f})",
            f"{R1_KL[arm][0]:.2f}",
        ]
        for arm, mult, lr in (("low", 1, 0.003), ("long", 3, 0.01), ("long-low", 3, 0.003))
    ],
    "**Round 1.** Three seeds per arm, unmasked, cosine schedule. Ex-2.2.16's control (1×, 0.01) scored 0.27.",
)

rf"""

Every round-1 curve was still rising when the cosine schedule wound down. So round 2 trained for eight times the length ({STEPS_8X:,.0f} steps), with ex-2.2.16's newline mask, which stops each position attending past the start of its own line. A learning-rate finder on the same configuration, which trains briefly at a rising rate and looks for where the loss falls fastest, put the steepest descent near {FINDER_STEEP["finder-mask"]:.4f}. Round 2 then swept the peak rate from {SWEEP[-1]["peak_lr"]:g} down to {SWEEP[0]["peak_lr"]:g} at one seed each.

"""

groups_draw(
    [
        (r["label"], plt.get_cmap("viridis")(i / (len(SWEEP) - 1)), "-", 1.3, f"{r['peak_lr']:g}")
        for i, r in enumerate(SWEEP)
    ]
    + [(NOMASK["label"], "0.5", "--", 1.6, f"{PEAK:g}, no mask")],
    "round2",
    f"""
        **Round 2: the peak learning rate at eight times the length.** Each panel is one group of ops; the y-axis is
        held-out expected exact match on a probe set during training (a running mean over three points), with the
        Bayes ceiling of the group dashed. One seed per peak rate, shaded by rate. The gray dashed curve is the
        unmasked run at {PEAK:g}. The bottom panel is the learning rate each run followed.
    """,
    f"""
        Six line charts of held-out expected exact match against training step, one per op group, and a learning-rate
        chart below them. The runs with peak rates from 0.00178 to 0.01 all end between 0.41 and 0.45 overall; the
        lower rates end lower. The HSV-channel panel shows sharp rises partway through training, earlier for
        mid-range rates. The unmasked run ends lowest of the mid-range rates, at {NOMASK["task"]["eem"]["all"]:.2f}.
    """,
)

rf"""

Four peak rates from 0.00178 to 0.01 ended within 0.04 of each other, and the best was {BEST_SWEEP["peak_lr"]:g} at {BEST_SWEEP["task"]["eem"]["all"]:.3f}. The mask made the largest single difference in the scout: at {PEAK:g}, the unmasked run scored {NOMASK["task"]["eem"]["all"]:.3f} and the masked one {MASKED["task"]["eem"]["all"]:.3f}. The HSV-channel ops rose steeply once the decaying rate fell below about 0.004, which is what shaped the next two rounds. The later rounds all train with the mask at a peak rate of {PEAK:g}, the middle of the good range.

## Rounds 3 to 7: schedule, length, and model size

Round 3 tried a warmup-stable-decay (WSD) schedule, which warms up to {PEAK:g}, holds near {ex.HOLD_LR:g} until 85% of the run, and then anneals to the same floor as the cosine. Round 4 tried a staircase: holds at {", ".join(f"{s:g}" for s in ex.STAIRS)}, with short anneals between them, the last hold covering the final 11% of the run. Both ran at three seeds, beside two more seeds of the cosine.

"""

groups_draw(
    [
        (f"{arm}-{PEAK:g}-s{s}", color, "-", 1.2, legend if s == 0 else f"_{legend}")
        for s in range(3)
        for arm, color, legend in (
            ("sweep", "C0", "cosine"),
            ("wsd", "C3", "warmup-stable-decay"),
            ("stairs", "C2", "staircase"),
        )
    ],
    "schedules",
    f"""
        **Rounds 3 and 4: three schedules at eight times the length.** As in the round-2 figure, with three seeds per
        schedule, all masked at a peak rate of {PEAK:g}.
    """,
    """
        Six line charts of held-out expected exact match against training step, one per op group, and the three
        learning-rate schedules below. The warmup-stable-decay and staircase runs are flat through their holds and
        rise at each drop in the rate. By the end, the three schedules sit on top of one another in every group,
        and the seeds differ more than the schedules.
    """,
)

rf"""

The schedules change the path and leave the destination in place. WSD curves stay flat through the hold and rise steeply in the final anneal; the staircase gains most at its first two drops, and almost nothing at the lowest rate. All three end together, and the seeds differ more than the schedules do.

Round 5 trained each schedule for sixteen times the length ({STEPS_16X:,.0f} steps) at one seed. Rounds 6 and 7 trained the cosine at eight times on a wider model (d128-L4) and a deeper one (d64-L6), one seed each, both at seed 0 so that they pair with the other seed-0 runs. The figure below gathers the final scores of all of these.

"""

conditions_draw(
    CONDITIONS,
    {lbl: {"eem": eem(lbl), "kl": kl(lbl)} for _, labels in CONDITIONS for lbl in labels},
    CEILING,
    PASS,
    f"""
        Two dot charts over eight conditions. Left, expected exact match: every condition sits between 0.43 and 0.47,
        below the dashed pass line at {PASS:.3f} and the solid ceiling at {CEILING:.3f}. Right, the calibration KL:
        between 0.36 and 0.56 for every condition except the wider d128-L4 model, at {kl(f"d128-sweep-{PEAK:g}-s0"):.2f}.
    """,
)

rf"""

At eight times the length, the nine runs of the three schedules span {min(SPREAD_8X):.3f} to {max(SPREAD_8X):.3f}. Sixteen times the length adds between 0.002 and 0.006 to the seed-0 run of each schedule. The wider and deeper models gain 0.023 and 0.017 on their seed-0 pair, but seed 0 is the weakest seed under every schedule, and the plain model at seed 1 reaches {eem(f"sweep-{PEAK:g}-s1"):.3f}. With one seed each, I read neither as a change.

Two things about the wider model stand apart from its final score. It learns faster: its HSV channels make their jump near 15k steps, against 25k to 45k for the d64 runs, and it leads through the first half of training. And it is much less well calibrated, at a KL of {kl(f"d128-sweep-{PEAK:g}-s0"):.2f} against about 0.5, with a lower training loss than the d64 runs. So it is about as accurate as the others while being more confident than the examples support, which fits a model that has fitted more of its training corpus.

"""

groups_draw(
    [
        (f"sweep-{PEAK:g}-s1", "C0", ":", 0.9, "d64-L4, seeds 1 and 2"),
        (f"sweep-{PEAK:g}-s2", "C0", ":", 0.9, "_d64-L4, seeds 1 and 2"),
        (f"sweep-{PEAK:g}-s0", "C0", "-", 1.6, "d64-L4"),
        (f"d128-sweep-{PEAK:g}-s0", "C1", "-", 1.6, "d128-L4 (wider)"),
        (f"d64L6-sweep-{PEAK:g}-s0", "C4", "-", 1.6, "d64-L6 (deeper)"),
    ],
    "model-size",
    f"""
        **Rounds 6 and 7: a wider and a deeper model.** As in the round-2 figure, for the cosine at eight times the
        length and a peak rate of {PEAK:g}, seed 0 of each model (solid). The dotted curves are seeds 1 and 2 of the
        d64-L4 model, for the seed spread.
    """,
    """
        Six line charts of held-out expected exact match against training step, one per op group, and the shared
        cosine schedule below. The wider model rises fastest early, most visibly in the HSV-channel panel, and all
        three models end close together, within the range of the d64-L4 seeds.
    """,
)

rf"""

## Where the gap sits

The final evaluation reports one number per op, so the scout added a scoring pass. For every held-out context it records the full answer distribution of the model at the query `=`, for the {len(SCORED)} masked runs at {PEAK:g} (every run of rounds 3 to 7 and round 2's cosine). It then redraws the held-out contexts from their seed, which recovers what ex-2.2.16 did not publish: which examples in each context carry replacement op noise. The redrawn contexts match the published ones token for token.

The first question is whether the gap depends on how much the examples say about the op. The figure below groups the held-out contexts two ways: by the posterior on the true op (the probability the ideal predictor gives the true op after reading the examples), and by the number of noisy examples.

"""

gap_draw(
    SUMMARY,
    CEIL_BAND,
    CEIL_NOISE,
    f"""
        Two line charts of expected exact match, with the Bayes ceiling dashed. Left, against four bands of the
        posterior on the true op: the runs sit on the ceiling in the lowest band ({BAND_MEAN[0]:.2f} against
        {CEIL_BAND[0]:.2f}) and below it in the other three, by {BAND_GAP[1]:.2f}, {BAND_GAP[2]:.2f}, and
        {BAND_GAP[3]:.2f}. Right, against the number of noisy examples: below the ceiling with none or one, and on
        it with two or three. The {len(SCORED)} runs lie close together throughout.
    """,
)

rf"""

Where the examples leave the op uncertain (a posterior below 0.5, or two or more noisy examples out of three), the model scores what the ideal predictor scores. The gap opens once the examples point to one op: {BAND_GAP[1]:.3f} in the band from 0.5 to 0.9 and {BAND_GAP[2]:.3f} from 0.9 to 0.99, and it narrows to {BAND_GAP[3]:.3f} where the posterior is above 0.99. Those two middle bands hold {sum(N_BAND[1:3]) / sum(N_BAND):.0%} of the contexts and {sum(BAND_SHARE[1:3]):.0%} of the gap. All {len(SCORED)} runs share this shape, whatever their schedule, length, or model.

A model that knew the true op would lose nothing to op uncertainty. So in the confident band (posterior above {CONFIDENT:g}), what remains of the gap is about how the model computes each op, or about the model hesitating over an op the examples have settled. The two can be told apart by where the model puts the mass it does not put on a plausible answer. The table below counts the mass outside the support of the Bayes predictive (grid colors to which it gives less than {SUPPORT:g}) and splits it by whether some other op would give that color as its answer on the same query.

"""

table_html(
    [
        "op",
        "Bayes ceiling",
        "mass off support",
        "on other ops' answers",
        "on no op's answer",
        "share on other ops",
        "share if spread evenly",
    ],
    [
        [
            f"`{name}`",
            f"{HO_CEILING[(OP_IDS == o) & _conf].mean():.2f}",
            f"{OFF_OTHER[o] + OFF_NONE[o]:.3f}",
            f"{OFF_OTHER[o]:.3f}",
            f"{OFF_NONE[o]:.3f}",
            f"{OTHER_SHARE[o]:.0%}",
            f"{UNIFORM_SHARE[o]:.0%}",
        ]
        for o, name in enumerate(OPS)
    ],
    f"""
        **The mass each op leaks on confident contexts.** Contexts whose posterior on the true op is above
        {CONFIDENT:g}; mass means over the {len(SCORED)} scored runs. "Share if spread evenly" is the share that would
        land on other ops' answers if the off-support mass were spread evenly over the off-support grid colors.
    """,
)

rf"""

On every op, the model keeps between {min(OFF_OTHER + OFF_NONE):.0%} and {max(OFF_OTHER + OFF_NONE):.0%} of its mass off the support, and about half or more of that sits on answers another op would give, where an even spread would put under a tenth. Lighten and darken are the clearest: the Bayes predictive gives the answer with certainty, the model puts its top answer there on every confident context, and still about {OFF_OTHER[OPS.index("lighten")]:.0%} of its mass goes to other ops' answers. So the model hedges across ops more than its examples justify. The runs that hedge more also score lower: across the {len(SCORED)} runs, the mean leak on other ops' answers correlates with the final score at r = {LEAK_R:.2f}, which covers much of the seed-to-seed spread.

One caution on this reading: other ops' answers are often near the true answer in the cube, so part of this mass may be near misses rather than a choice of op. A comparison against colors at the same distance would separate the two.

### Answers in the color cube

The cubes below put the model answers beside the Bayes answers, one panel per op, on the confident contexts of `{CUBE_RUN}`. Each context is drawn at its expected answer color: the mean color of the answer distribution, which moves toward the middle of the cube as mass spreads.

"""

cubes_draw(
    cube_data(CUBE_RUN)["per_op"],
    """
        Eleven color-cube panels, one per op, each seen down the gray diagonal. Dots (model) sit on or very near
        their rings (Bayes) for every op except hsvmix, whose stubs are visibly longer and scattered in direction,
        with a slight pull toward the center of the cube. The deterministic ops (lighten, darken, difference) show
        almost no displacement.
    """,
)

rf"""

For ten of the eleven ops, the model expected answer sits within a fraction of one grid step of the Bayes one, with no shared direction to the displacement. So the model is not biased toward some region of the cube; the leak above is spread thinly. hsvmix is the exception, and the ops table shows it: it leaks {OFF_OTHER[HSVMIX] + OFF_NONE[HSVMIX]:.0%} of its mass on confident contexts, and {OFF_NONE[HSVMIX]:.0%} of its mass goes to colors no op gives. It is the one op where the computation itself falls short.

hsvmix mixes two colors in hue, saturation, and value, with the hue taken around the color wheel. Grouping its confident contexts by how far apart the operand hues are shows where it struggles.

"""

hue_gap_draw(
    cube_data(CUBE_RUN)["per_gap"],
    list(GAP_MEAN),
    CEIL_GAP,
    f"""
        Four color-cube panels for hsvmix, by hue gap between the operands: 0 to 30, 30 to 90, 90 to 150, and 150 to
        180 degrees. The stubs lengthen as the gap grows. The score falls from {GAP_MEAN[0]:.2f} of a
        {CEIL_GAP[0]:.2f} ceiling at the smallest gap to {GAP_MEAN[3]:.2f} of {CEIL_GAP[3]:.2f} at the largest.
    """,
)

rf"""

With nearby hues the model gets {GAP_MEAN[0] / CEIL_GAP[0]:.0%} of what the ceiling allows; with near-opposite hues it gets {GAP_MEAN[3] / CEIL_GAP[3]:.0%}. Near-opposite hues are where the midpoint around the wheel moves furthest for a small change in either operand, and at exactly opposite hues the direction around the wheel is a tie. So hsvmix asks for a more precise computation than the other ops, and the model has not learned it. Width and depth did not help this either: hsvmix scored {RUNS[f"d128-sweep-{PEAK:g}-s0"]["task"]["eem"]["per_op"][HSVMIX]:.2f} on d128-L4 and {RUNS[f"d64L6-sweep-{PEAK:g}-s0"]["task"]["eem"]["per_op"][HSVMIX]:.2f} on d64-L6, against {RUNS[f"sweep-{PEAK:g}-s0"]["task"]["eem"]["per_op"][HSVMIX]:.2f} on their seed-0 pair.

## What this means for the recipe

The control now sits at about 0.45 on the center condition, with the mask, a peak rate of {PEAK:g}, and eight times ex-2.2.16's steps. That falls short of the pass line, but it is a model that does in-context inference over ops: it tracks the ceiling across the whole range of evidence, and its shortfall is concentrated in how firmly it commits once the op is clear, plus one op it computes poorly. That seems a workable baseline for the anchoring experiments, which compare an anchored model with this control on the same contexts.

Two questions from this scout are on the backlog: whether a curriculum over the replacement rate changes how firmly the model commits ([noise curriculum](/todo/science/noise-curriculum-center-control.md)), and whether the wider model with a tuned rate can reach the same level in half the steps ([a cheaper recipe](/todo/science/cheaper-center-control-recipe.md)). The wider model learned faster here, which is some encouragement for the second.

<details><summary>Cost</summary>

The scout cost about \$15 on Modal, nearly all of it L4 time for training. A step takes the same time on d64-L4 and d128-L4 (about 2,000 to 3,000 steps a minute), since at these sizes the step is bound by latency rather than arithmetic, so the cost follows the step count. An eight-times run is about 40 minutes of training.

</details>
"""
