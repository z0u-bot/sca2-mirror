# title: Ex 2.2.13: does a heavier anchor make the leftover predictable?

# The design constants and the refs come from `experiment.py` beside this script (the script's directory
# is on sys.path while it runs).
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from scipy import stats

from mini.lit import memo, stop
from mini.store import project_store
from mini.vis import AxesRow, figure_html, light_dark, themed
from sca.vis_grading import GradingCloud

import experiment as ex

FROZEN_AT = "c533700"


def ladder_html() -> str:
    """The ladder, one row per condition, with what earlier experiments saw of each."""
    head = (
        "<tr><th>condition</th><th class=num>λ_a</th><th>home of <em>red</em></th>"
        "<th class=num>seeds</th><th>seen before</th></tr>"
    )
    notes = {
        "axis-0.1": "ex-2.2.11's recipe, the reference; 20 seeds there",
        "axis-0.2": "ex-2.2.12's <code>lam-0.2</code>, 5 seeds: loose on the kept share, the tightest line margin and ᾱ in the sweep",
        "plane-0.1": "ex-2.2.12's <code>plane</code>, 5 seeds",
        "plane-0.2": "ex-2.2.12's <code>plane-lam-0.2</code>, 5 seeds: the tight kept share",
    }
    rows = [
        f"<tr><td><code>{c.name}</code></td><td class=num>{c.lam:g}</td><td>{c.subspace}</td>"
        f"<td class=num>{ex.SEEDS}</td><td>{notes.get(c.name, '')}</td></tr>"
        for c in ex.GRID
    ]
    rows += [
        f"<tr><td><code>{ex.CONTROL}</code></td><td class=num>0</td><td>none; scored on the axis</td>"
        f"<td class=num>{ex.CONTROL_SEEDS}</td><td>ex-2.2.11's <code>control</code>, served from the store at seeds 100–104; the task reference</td></tr>",
        f"<tr><td><code>{ex.CONTROL_PLANE}</code></td><td class=num>0</td><td>none; scored on the plane</td>"
        f"<td class=num>—</td><td>the same checkpoints, a scoring pass; the ᾱ baseline for the plane</td></tr>",
    ]
    return f'<table class="report-table dense"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


# --- Loading ------------------------------------------------------------------------------------


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists: one `get_refs` and one `get_many`."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict | None:
    return None if path is None else json.loads(path.read_text())


REFS = [ex.METRICS_REF, ex.EX2211_METRICS_REF, ex.EX2212_METRICS_REF]
with tempfile.TemporaryDirectory() as _tmp:
    files = fetch(REFS, Path(_tmp))
    metrics_loaded = read_json(files[ex.METRICS_REF])
    ref_loaded = read_json(files[ex.EX2211_METRICS_REF])
    sweep_loaded = read_json(files[ex.EX2212_METRICS_REF])

CONDITIONS = [c.name for c in ex.GRID]
REFERENCE = ex.REFERENCE
CONTROL = ex.CONTROL
PLANE_CONTROL = ex.CONTROL_PLANE
OTHER_OPS = tuple(op for op in ex.OP_NAMES if op != ex.MISSED_OP)
MARGIN_GATE = ex.MARGIN_RATIO * ex.REF_M_LINE
R2_GATE = ex.GRADE_R2_RATIO * ex.REF_R2_SIM
OLD_LEVEL = ex.KEPT_GATE - ex.OLD_BAND

EARLIER = {
    "axis-0.1": ("ex-2.2.11", "handover"),
    "axis-0.2": ("ex-2.2.12", "lam-0.2"),
    "plane-0.1": ("ex-2.2.12", "plane"),
    "plane-0.2": ("ex-2.2.12", "plane-lam-0.2"),
}
# Which earlier experiment saw each rung, and under what name. The ladder retrains all four at fresh
# seeds, so these are a replication column and not data the gates are scored on.

COSTS = ("task", "m_line", "lead_emb", "contrast", "r2_sim", "nonred_deficit")
COST_LABEL = dict(zip(COSTS, ex.COST_STATISTICS, strict=True))


@dataclass(frozen=True)
class Results:
    """Every published result the report reads: this experiment's metrics, ex-2.2.11's (whose `control` is
    the task reference and the ᾱ baseline for the axis), and ex-2.2.12's (the sweep, for the replication
    column of the H1 table).
    """

    metrics: dict
    ref: dict
    sweep: dict

    def __memo_key__(self) -> str:
        return f"{len(self.metrics['runs'])}:{len(self.metrics['scores'])}:{len(self.ref['runs'])}"

    def _source(self, cond: str) -> dict:
        if cond == CONTROL:
            return self.ref
        return self.metrics

    def runs(self, cond: str, source: dict | None = None) -> list[dict]:
        """The eval records of a condition, in seed order."""
        src = source or self._source(cond)
        return sorted((r for r in src["runs"] if r["condition"] == cond), key=lambda r: r["seed"])

    def scored(self, cond: str, source: dict | None = None) -> list[dict]:
        src = source or self._source(cond)
        return sorted((r for r in src["scores"] if r["condition"] == cond), key=lambda r: r["seed"])

    def n(self, cond: str) -> int:
        return len(self.runs(cond))

    def stat(self, cond: str, key: str, source: dict | None = None) -> np.ndarray:
        """One value per seed of a placement statistic on the `mix` lines."""
        return np.array([r[key] for r in self.runs(cond, source)], float)

    def eem(self, cond: str, op: str) -> np.ndarray:
        return np.array([r["holdout_eem"][op] for r in self.runs(cond)], float)

    def score(self, cond, op, operator, key, group, source: dict | None = None) -> np.ndarray:
        out = []
        for r in self.scored(cond, source):
            s = r["ops"][op]
            out.append((s["clean"][key] if operator is None else s["operators"][operator][key])[group])
        return np.array(out, float)

    def kept(self, cond: str, op: str, source: dict | None = None) -> np.ndarray:
        """The share of clean expected exact match kept under `projection` on an op's removal lines."""
        return self.score(cond, op, "projection", "kept", "removal", source)

    def kept_earlier(self, cond: str) -> np.ndarray:
        """The same measurement as the earlier experiment that saw this rung took it."""
        exp, name = EARLIER[cond]
        return self.kept(name, ex.MISSED_OP, self.ref if exp == "ex-2.2.11" else self.sweep)

    def kept_worst(self, cond: str) -> tuple[str, float]:
        """The op other than the missed one with the highest seed-mean kept share, and that share."""
        means = {op: float(self.kept(cond, op).mean()) for op in OTHER_OPS}
        op = max(means, key=lambda o: means[o])
        return op, means[op]

    def axes(self, cond: str) -> tuple[int, ...]:
        return tuple(self.runs(cond)[0].get("axes", [ex.ANCHOR_AXIS]))

    def baseline(self, cond: str) -> str:
        """The un-anchored condition scored on the same subspace."""
        return PLANE_CONTROL if len(self.axes(cond)) > 1 else CONTROL

    def alpha_ratio(self, cond: str) -> float:
        """ᾱ at op1 as a ratio to the un-anchored baseline scored the same way."""
        return float(self.stat(cond, "alpha_op1").mean() / self.stat(self.baseline(cond), "alpha_op1").mean())

    def alpha_excess(self, cond: str) -> float:
        """ᾱ at op1 as an excess over the un-anchored baseline scored the same way: the ᾱ clause as intended."""
        return float(self.stat(cond, "alpha_op1").mean() - self.stat(self.baseline(cond), "alpha_op1").mean())

    def cost(self, cond: str, key: str) -> np.ndarray:
        """One value per seed of a cost statistic. `task` is the gap from the control on the worst op."""
        match key:
            case "task":
                return np.array([self.eem(cond, op) - self.eem(CONTROL, op).mean() for op in ex.OP_NAMES]).min(0)
            case "nonred_deficit":
                return self.score(cond, ex.PRIMARY_OP, "projection", "deficit", "nonred")
            case _:
                return self.stat(cond, key)

    def task_worst(self, cond: str) -> tuple[str, float]:
        gaps = {op: float(self.eem(cond, op).mean() - self.eem(CONTROL, op).mean()) for op in ex.OP_NAMES}
        op = min(gaps, key=lambda o: gaps[o])
        return op, gaps[op]

    def alpha_by_color(self, cond: str) -> np.ndarray:
        """(seeds, colors): ᾱ at op1 per grid color, averaged over slices."""
        return np.array([r["alpha_op1_by_color"] for r in self.runs(cond)], float)

    # -- the exploratory passes -----------------------------------------------------------

    def missed_lines(self, cond: str) -> dict[str, np.ndarray]:
        """(seeds, lines) of the per-line kept share on the missed op's removal lines, with the shared
        line index and the palette color of each line's second operand.
        """
        rows = [r["missed_lines"] for r in self.scored(cond)]
        return {
            "kept": np.array([r["kept"] for r in rows], float),
            "index": np.array(rows[0]["index"], int),
            "op2_color": np.array(rows[0]["op2_color"], int),
        }

    def achromatic(self, cond: str, pass_: str, group: str = "removal") -> np.ndarray:
        return np.array([r["achromatic"]["kept"][pass_][group] for r in self.scored(cond)], float)


# --- Statistics ---------------------------------------------------------------------------------


def sd(v: np.ndarray) -> float:
    return float(np.std(np.asarray(v, float), ddof=1))


def ucb(v: np.ndarray) -> float:
    """The one-sided upper confidence bound on a seed mean, Student's t at n − 1 degrees of freedom."""
    v = np.asarray(v, float)
    n = len(v)
    return float(v.mean() + stats.t.ppf(ex.UCB_LEVEL, n - 1) * sd(v) / np.sqrt(n))


def paired(v: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    """The paired mean difference of *v* from *ref* and its two-sided 95% interval."""
    d = np.asarray(v, float) - np.asarray(ref, float)
    half = stats.t.ppf(0.975, len(d) - 1) * sd(d) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - half), float(d.mean() + half)


def rungs(subspace: str) -> list[str]:
    return [c.name for c in ex.GRID if c.subspace == subspace]


def trend(res: Results, subspace: str) -> dict:
    """The trend contrast: each run's absolute deviation from its own condition's median, regressed on
    log λ_a across the four rungs of one subspace. H1 wants a negative slope, so the p is one-sided.
    """
    xs, ys = [], []
    for c in ex.GRID:
        if c.subspace != subspace:
            continue
        v = res.kept(c.name, ex.MISSED_OP)
        xs += [np.log(c.lam)] * len(v)
        ys += list(np.abs(v - np.median(v)))
    fit = stats.linregress(np.array(xs), np.array(ys))
    p_one = fit.pvalue / 2 if fit.slope < 0 else 1 - fit.pvalue / 2
    return {"slope": float(fit.slope), "p_one_sided": float(p_one), "r": float(fit.rvalue), "n": len(xs)}


def spread(res: Results, subspace: str) -> dict:
    """H1's two numbers for one subspace: the standard-deviation ratio top rung over bottom, and Levene."""
    names = rungs(subspace)
    v = [res.kept(c, ex.MISSED_OP) for c in names]
    lev = stats.levene(*v, center="median")
    return {
        "sds": [sd(x) for x in v],
        "ratio": sd(v[-1]) / sd(v[0]),
        "levene": float(lev.statistic),
        "levene_p": float(lev.pvalue),
        "trend": trend(res, subspace),
    }


def gate_status(res: Results, cond: str) -> dict[str, bool]:
    """Every cost statistic against its ex-2.2.11 gate, on the seed mean."""
    m = {k: float(res.cost(cond, k).mean()) for k in COSTS}
    return {
        "task": m["task"] >= -ex.TASK_GATE,
        "m_line": m["m_line"] >= MARGIN_GATE,
        "lead_emb": m["lead_emb"] >= ex.LEAD_GATE,
        "contrast": m["contrast"] >= ex.CONTRAST_GATE,
        "r2_sim": m["r2_sim"] >= R2_GATE,
        "nonred_deficit": m["nonred_deficit"] <= ex.NONRED_DEFICIT_GATE,
    }


def adoption(res: Results) -> dict:
    """The adoption rule, clause by clause, plus the verdict under ex-2.2.12's fixed band."""
    ref_ratio = res.alpha_ratio(REFERENCE)
    ref_excess = res.alpha_excess(REFERENCE)
    out: dict[str, dict] = {}
    for c in CONDITIONS:
        g = gate_status(res, c)
        worst_op, _ = res.kept_worst(c)
        others = {op: ucb(res.kept(c, op)) for op in OTHER_OPS}
        worst_ucb_op = max(others, key=lambda o: others[o])
        r = {
            "ucb": ucb(res.kept(c, ex.MISSED_OP)),
            "mean": float(res.kept(c, ex.MISSED_OP).mean()),
            "others_ucb": others[worst_ucb_op],
            "others_ucb_op": worst_ucb_op,
            "worst_op": worst_op,
            "gates": g,
            "failed": [k for k, ok in g.items() if not ok],
            "alpha_ratio": res.alpha_ratio(c),
        }
        r["missed_ok"] = r["ucb"] < ex.KEPT_GATE
        r["others_ok"] = r["others_ucb"] < ex.KEPT_GATE
        # REVIEW: the ᾱ clause is implemented exactly as the frozen rule words it — a ratio to the
        # un-anchored baseline of the condition's own subspace, compared with the reference's ratio to the
        # axis one. The axis baseline turns out to be a signed alignment that averages near zero, so the
        # axis ratios are large and negative and the plane ratios are small and positive, and every plane
        # condition misses this clause on the sign alone. The table prints the raw ᾱ and both baselines so
        # the arithmetic is visible. Verify: ᾱ on the axis is a signed cosine, on the plane a length.
        r["alpha_ok"] = r["alpha_ratio"] <= ref_ratio
        # REVIEW: the review of the first draft read the clause as it was intended rather than to the
        # letter, so the rule is also scored with the ᾱ clause as an excess over the condition's own
        # baseline, which is how ex-2.2.12 compared ᾱ across subspaces. Both verdicts are in the table,
        # and the prose says which one the report acts on. Verify: `plane-0.1` is the only condition
        # whose upper bounds clear the gate, so the two readings differ on it alone.
        r["alpha_excess"] = res.alpha_excess(c)
        r["alpha_ok_intended"] = r["alpha_excess"] <= ref_excess
        r["qualifies"] = r["missed_ok"] and r["others_ok"] and all(g.values()) and r["alpha_ok"]
        r["qualifies_intended"] = r["missed_ok"] and r["others_ok"] and all(g.values()) and r["alpha_ok_intended"]
        r["old_rule"] = r["mean"] <= OLD_LEVEL and all(g.values()) and r["alpha_ok"]
        out[c] = r
    lam = {c.name: c.lam for c in ex.GRID}
    qual = [c for c in CONDITIONS if out[c]["qualifies"]]
    intended = [c for c in CONDITIONS if out[c]["qualifies_intended"]]
    old_qual = [c for c in CONDITIONS if out[c]["old_rule"]]
    adopted = min(qual, key=lambda c: (out[c]["ucb"], lam[c])) if qual else None
    adopted_intended = min(intended, key=lambda c: (out[c]["ucb"], lam[c])) if intended else None
    old_adopted = min(old_qual, key=lambda c: (out[c]["mean"], lam[c])) if old_qual else None
    return {
        "conditions": out,
        "qualifying": qual,
        "adopted": adopted,
        "qualifying_intended": intended,
        "adopted_intended": adopted_intended,
        "old_qualifying": old_qual,
        "old_adopted": old_adopted,
        "ref_ratio": ref_ratio,
        "ref_excess": ref_excess,
    }


# --- Shared drawing and table helpers ---------------------------------------------------------

# λ_a is ordinal, so each subspace is ordered shades of one hue rather than four categorical colors.
SHADES = {
    "axis": ("#f0a030 #d9781a #b0530c #7a3400".split(), "#ffd08a #ffab52 #f08a2e #d06a10".split()),
    "plane": ("#a97fd0 #8a4fb8 #6a2f98 #4a1a70".split(), "#e0c0ff #c79aee #a875d8 #8a55bc".split()),
}
MARKERS = {"axis": "o", "plane": "P"}
CONTROL_INK = ("#6b6b6b", "#b0b0b0")


def ink(cond: str) -> str:
    if cond in (CONTROL, PLANE_CONTROL):
        return light_dark(*CONTROL_INK)
    c = next(c for c in ex.GRID if c.name == cond)
    i = ex.LADDER.index(c.lam)
    return light_dark(SHADES[c.subspace][0][i], SHADES[c.subspace][1][i])


def xs_of(subspace: str) -> np.ndarray:
    """The ladder on a log axis: the four rungs are evenly spaced, which is what a √2 ratio means."""
    off = -0.035 if subspace == "axis" else 0.035
    return np.log(np.array(ex.LADDER)) + off


def dots(ax: Axes, x: float, v: np.ndarray, cond: str, *, rng, ms: float = 6.0, width=0.02, label=None) -> None:
    """One column of per-seed dots, the seed range as a thin bar, a standard-deviation bar, and the mean."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return
    color = ink(cond)
    m = MARKERS[cond.split("-")[0]] if cond in [c.name for c in ex.GRID] else "s"
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=0.9, alpha=0.4, zorder=2, solid_capstyle="butt")
    if len(v) > 1:
        s = sd(v)
        ax.plot([x, x], [v.mean() - s, v.mean() + s], "-", color=color, lw=3.0, alpha=0.5, zorder=3)
    ax.plot(x + rng.uniform(-width, width, len(v)), v, "o", ms=2.2, color=color, alpha=0.45, zorder=4, mew=0)
    ax.plot(x, v.mean(), m, ms=ms, color=color, zorder=5, mec=light_dark("white", "#111"), mew=0.6, label=label)


def column_box(ax: Axes, x: float, v: np.ndarray, color: str, label: str | None = None, *, half=0.03) -> None:
    """A control's seed range as a box under one column, with its seed mean as a tick across it."""
    v = np.asarray(v, float)
    ax.add_patch(
        Rectangle(
            (x - half, v.min()),
            2 * half,
            v.max() - v.min(),
            facecolor=color,
            edgecolor=color,
            alpha=0.2,
            lw=0.6,
            zorder=0,
            label=label,
        )
    )
    ax.plot([x - half, x + half], [v.mean()] * 2, "-", color=color, lw=0.9, alpha=0.8, zorder=1)


def gate_line(ax: Axes, y: float, *, partial: float | None = None, fail: str | None = None) -> None:
    """A dashed gate line, a dotted secondary level when there is one, and the failing side hatched."""
    ax.axhline(y, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=2)
    if partial is not None:
        ax.axhline(partial, color=light_dark("#333", "#ddd"), lw=0.7, ls=":", zorder=2)
    if fail is not None:
        lo, hi = ax.get_ylim()
        span = (lo, y) if fail == "below" else (y, hi)
        ax.axhspan(*span, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="//", lw=0, zorder=0, alpha=0.1)
        ax.set_ylim(lo, hi)


def ladder_axis(ax: Axes) -> None:
    ax.set_xticks(np.log(np.array(ex.LADDER)), [f"{x:g}" for x in ex.LADDER], fontsize=8)
    ax.set_xlim(np.log(ex.LADDER[0]) - 0.16, np.log(ex.LADDER[-1]) + 0.16)
    ax.set_xlabel("λ_a (log scale)", fontsize=8)
    ax.grid(axis="y", alpha=0.2)


def fig_legend(fig: plt.Figure, ax: Axes, ncols: int | None = None, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncols=ncols or max(len(labels), 1),
        frameon=False,
        fontsize=7,
        **kwargs,
    )


def span2(v: np.ndarray, digits: int = 3) -> str:
    """A seed mean with its range, as `0.512 (0.49–0.53)`."""
    v = np.asarray(v, float)
    if v.size == 0 or np.all(np.isnan(v)):
        return "—"
    return f"{np.nanmean(v):.{digits}f} ({np.nanmin(v):.{digits - 1}f}–{np.nanmax(v):.{digits - 1}f})"


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str, *, ref_rows: frozenset[int] = frozenset()) -> str:
    """An authored result table in the shared report style; the first column is text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        f"<tr{' class=ref' if r in ref_rows else ''}>"
        + "".join(f"<td{' class=num' if i else ''}>{cell_html(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for r, row in enumerate(rows)
    )
    table = f'<table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def bold_if(text: str, ok: bool) -> str:
    return f"<b>{text}</b>" if ok else text


def verdict_md(status: str, line: str) -> str:
    """The verdict admonition that closes a hypothesis section; the site hoists its title into the heading."""
    kind = {"pass": "success", "partial": "warning", "miss": "danger", "unresolved": "info"}[status]
    return f"/// admonition | {status.capitalize()}\n    type: {kind}\n{line}\n///"


# --- H1: the spread of the leftover -------------------------------------------------------------


def h1_figure(res: Results) -> str:
    values = {c: res.kept(c, ex.MISSED_OP) for c in CONDITIONS}
    return h1_draw(values, h1_alt(res))


def h1_alt(res: Results) -> str:
    s = {sub: spread(res, sub) for sub in ex.SUBSPACES}
    words = {sub: "narrows" if s[sub]["ratio"] < 1 else "widens" for sub in ex.SUBSPACES}
    return f"""
        Two dot panels side by side, the axis on the left and the plane on the right, with the four anchor
        weights spaced evenly along the bottom on a log scale. Each column is twenty seed dots with a thick
        standard-deviation bar and the seed mean on top. On the axis the spread {words["axis"]} from the
        lowest rung to the top, by a factor of {s["axis"]["ratio"]:.2f}; on the plane it {words["plane"]}, by
        a factor of {s["plane"]["ratio"]:.2f}. A dashed line marks the gate at one fifth, hatched above.
    """


@memo
def h1_draw(values: dict, alt_text: str) -> str:
    @themed(
        name="h1-kept-spread",
        alt_text=alt_text,
        caption=f"""
            **Kept share on the `{ex.MISSED_OP}` removal lines along the ladder.** Left, *red* on the first
            axis; right, *red* on the plane. Each column is one condition: {ex.SEEDS} faint seed dots, a thin
            bar spanning the seed range, a thick bar spanning one standard deviation either side of the mean,
            and the seed mean as the large mark. H1 is about the height of the thick bar rather than where the
            mark sits. The dashed line is the {ex.KEPT_GATE:.0%} gate, hatched above.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), layout="constrained", sharey=True)
        axes = cast(AxesRow, axes)
        rng = np.random.default_rng(7)
        for ax, sub in zip(axes, ex.SUBSPACES, strict=True):
            for x, c in zip(np.log(np.array(ex.LADDER)), rungs(sub), strict=True):
                dots(ax, x, values[c], c, rng=rng)
            ax.set_title(f"red on the {'first axis' if sub == 'axis' else 'plane'}", fontsize=9)
            ladder_axis(ax)
        axes[0].set_ylim(-0.03, max(0.55, max(float(np.max(v)) for v in values.values()) + 0.05))
        for ax in axes:
            gate_line(ax, ex.KEPT_GATE, fail="above")
        axes[0].set_ylabel(f"kept share on {ex.MISSED_OP}")
        return fig

    return _plot()


def h1_table(res: Results) -> str:
    head = ["condition", "seed mean ↓", "SD ↓", "range", "n", "seen before (mean, SD, n)"]
    rows = []
    for c in ex.GRID:
        v = res.kept(c.name, ex.MISSED_OP)
        before = "—"
        if c.name in EARLIER:
            w = res.kept_earlier(c.name)
            before = f"{EARLIER[c.name][0]} {np.mean(w):.3f}, {sd(w):.3f}, {len(w)}"
        rows.append(
            [
                f"`{c.name}`",
                f"{v.mean():.3f}",
                f"{sd(v):.3f}",
                f"{v.min():.2f}–{v.max():.2f}",
                str(len(v)),
                before,
            ]
        )
    sp = {sub: spread(res, sub) for sub in ex.SUBSPACES}
    note = "; ".join(
        f"{sub}: SD ratio top/bottom {sp[sub]['ratio']:.2f} against a gate of {ex.SD_RATIO_GATE:g}, "
        f"Levene W = {sp[sub]['levene']:.2f} (p = {sp[sub]['levene_p']:.3f}), trend slope "
        f"{sp[sub]['trend']['slope']:+.4f} per log unit of λ_a (one-sided p = {sp[sub]['trend']['p_one_sided']:.3f})"
        for sub in ex.SUBSPACES
    )
    return table_html(
        head,
        rows,
        f"Kept share on the `{ex.MISSED_OP}` removal lines per condition, at {ex.SEEDS} seeds. The last column "
        f"is the same measurement as the experiment that saw that rung took it, at its own seed count, as a "
        f"replication rather than as data. {note}.",
        ref_rows=frozenset({CONDITIONS.index(REFERENCE)}),
    )


# --- H2: the level of the leftover ---------------------------------------------------------------


def h2_table(res: Results) -> str:
    head = ["condition", "seed mean ↓", f"paired Δ from `{REFERENCE}`", "95% interval", f"|Δ| ≤ {ex.MEAN_BAND:g}"]
    ref = res.kept(REFERENCE, ex.MISSED_OP)
    rows = []
    for c in CONDITIONS:
        v = res.kept(c, ex.MISSED_OP)
        d, lo, hi = paired(v, ref)
        rows.append(
            [
                f"`{c}`",
                f"{v.mean():.3f}",
                "—" if c == REFERENCE else f"{d:+.3f}",
                "—" if c == REFERENCE else f"{lo:+.3f} to {hi:+.3f}",
                "—" if c == REFERENCE else bold_if("yes" if abs(d) <= ex.MEAN_BAND else "no", abs(d) <= ex.MEAN_BAND),
            ]
        )
    return table_html(
        head,
        rows,
        f"The seed-mean kept share on `{ex.MISSED_OP}` and its paired difference from the reference. Every "
        f"condition trains at the same {ex.SEEDS} seeds, so each difference is paired seed for seed; the "
        f"interval is Student's t at {ex.SEEDS - 1} degrees of freedom. H2's band is {ex.MEAN_BAND:g}.",
        ref_rows=frozenset({CONDITIONS.index(REFERENCE)}),
    )


# --- H3: the worst of the other ops ---------------------------------------------------------------


def h3_figure(res: Results) -> str:
    values = {(c, op): float(res.kept(c, op).mean()) for c in CONDITIONS for op in ex.OP_NAMES}
    worst = {c: res.kept_worst(c) for c in CONDITIONS}
    return h3_draw(values, worst, h3_alt(res))


def h3_alt(res: Results) -> str:
    lo, hi = (res.kept_worst(c)[1] for c in (REFERENCE, rungs("axis")[-1]))
    return f"""
        A strip chart with one row per condition and the kept share along the bottom. Each row carries eleven
        marks, one per operation, coloured by the condition. The ten ops other than {ex.MISSED_OP} sit in a
        cluster at the low end and {ex.MISSED_OP} sits far to the right of them. The worst of the ten is
        ringed and named on each row; it reads {lo:.2f} at the reference and {hi:.2f} at the top of the axis
        ladder. A dashed line marks the gate at one fifth, hatched to the right.
    """


@memo
def h3_draw(values: dict, worst: dict, alt_text: str) -> str:
    @themed(
        name="h3-per-op",
        alt_text=alt_text,
        caption=f"""
            **Per-op seed-mean kept share, per condition.** One row per condition; each small mark is one op's
            kept share on its own removal lines, averaged over the {ex.SEEDS} seeds. The hollow ring is the
            worst of the ten ops other than `{ex.MISSED_OP}`, named beside it; the large mark is
            `{ex.MISSED_OP}` itself. The dashed line is the {ex.KEPT_GATE:.0%} gate, hatched to the right.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.4, 4.0), layout="constrained")
        for y, c in enumerate(CONDITIONS[::-1]):
            color = ink(c)
            for op in OTHER_OPS:
                ax.plot(values[c, op], y, "o", ms=3.4, color=color, alpha=0.6, mew=0, zorder=3)
            ax.plot(
                values[c, ex.MISSED_OP],
                y,
                MARKERS[c.split("-")[0]],
                ms=7,
                color=color,
                zorder=4,
                mec=light_dark("white", "#111"),
                mew=0.6,
            )
            op, v = worst[c]
            ax.plot(v, y, "o", ms=9, mfc="none", mec=color, mew=1.2, zorder=5)
            ax.annotate(op, (v, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7, color=color)
        ax.set_yticks(range(len(CONDITIONS)), CONDITIONS[::-1], fontsize=8)
        ax.set_ylim(-0.7, len(CONDITIONS) - 0.3)
        ax.set_xlim(-0.02, max(values.values()) + 0.06)
        ax.set_xlabel("seed-mean kept share on the op's removal lines", fontsize=8)
        ax.grid(axis="x", alpha=0.2)
        ax.axvline(ex.KEPT_GATE, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=2)
        lo, hi = ax.get_xlim()
        ax.axvspan(
            ex.KEPT_GATE,
            hi,
            facecolor="none",
            edgecolor=light_dark("#000", "#fff"),
            hatch="//",
            lw=0,
            zorder=0,
            alpha=0.1,
        )
        ax.set_xlim(lo, hi)
        return fig

    return _plot()


def h3_table(res: Results) -> str:
    head = ["condition", "worst other op", "its seed mean ↓", f"Δ from `{REFERENCE}`", "ten-op mean ↓"]
    ref_worst = res.kept_worst(REFERENCE)[1]
    rows = []
    for c in CONDITIONS:
        op, v = res.kept_worst(c)
        others = float(np.mean([res.kept(c, o).mean() for o in OTHER_OPS]))
        drop = ref_worst - v
        rows.append(
            [
                f"`{c}`",
                f"`{op}`",
                bold_if(f"{v:.3f}", v <= ex.KEPT_GATE),
                "—" if c == REFERENCE else bold_if(f"{-drop:+.3f}", drop >= ex.WORST_OP_MARGIN),
                f"{others:.3f}",
            ]
        )
    return table_html(
        head,
        rows,
        f"The worst of the ten ops other than `{ex.MISSED_OP}` per condition, and the mean over those ten. "
        f"H3 asks for a fall of at least {ex.WORST_OP_MARGIN:g} from the reference at the top of the ladder; "
        f"bold in the last column marks a fall that large. The worst op is not the same op in every condition.",
        ref_rows=frozenset({CONDITIONS.index(REFERENCE)}),
    )


# --- H4: what the weight spends ------------------------------------------------------------------

PANELS = {
    # key: (y label, gate, failing side)
    "task": ("task gap from control, worst op", -ex.TASK_GATE, "below"),
    "m_line": ("line margin m_line", MARGIN_GATE, "below"),
    "lead_emb": ("lead at the embedding", ex.LEAD_GATE, "below"),
    "contrast": ("contrast, red vs non-red", ex.CONTRAST_GATE, "below"),
    "r2_sim": ("grading r²", R2_GATE, "below"),
    "nonred_deficit": ("non-red deficit, mix", ex.NONRED_DEFICIT_GATE, "above"),
}


def cost_figure(res: Results) -> str:
    values = {(c, k): res.cost(c, k) for c in CONDITIONS for k in COSTS}
    return cost_draw(values, cost_alt(res))


def cost_alt(res: Results) -> str:
    failed = {c: gate_status(res, c) for c in CONDITIONS}
    misses = [f"{c} on {k}" for c, g in failed.items() for k, ok in g.items() if not ok]
    tail = "every column sits on the passing side of its gate" if not misses else "the misses are " + "; ".join(misses)
    return f"""
        Six dot panels in two columns, one per cost statistic, with the four anchor weights along the bottom
        of each on a log scale and the two subspaces as paired columns at each weight. Each column is twenty
        seed dots with a standard-deviation bar and the seed mean. Each panel carries a dashed gate with the
        failing side hatched: {tail}.
    """


@memo
def cost_draw(values: dict, alt_text: str) -> str:
    @themed(
        name="h4-costs",
        alt_text=alt_text,
        caption=f"""
            **The cost side along the ladder.** One panel per statistic H4 names, each against its gate from
            ex-2.2.11 (dashed, failing side hatched): the held-out expected exact match less the control's on
            whichever op the condition is worst on; the line margin on the `{ex.PRIMARY_OP}` lines; the lead at
            the embedding; the contrast between the red and non-red groups; the grading r²; and the non-red
            deficit under the projection. At each weight the left column is the axis and the right the plane;
            each is {ex.SEEDS} seed dots with a standard-deviation bar and the seed mean.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(3, 2, figsize=(8.4, 8.0), layout="constrained")
        flat = cast(AxesRow, axes.reshape(-1))
        rng = np.random.default_rng(8)
        for ax, k in zip(flat, COSTS, strict=True):
            ylabel, gate, fail = PANELS[k]
            for sub in ex.SUBSPACES:
                for x, c in zip(xs_of(sub), rungs(sub), strict=True):
                    dots(ax, x, values[c, k], c, rng=rng, ms=5)
            ax.set_ylabel(ylabel, fontsize=8)
            ladder_axis(ax)
            lo, hi = ax.get_ylim()
            pad = 0.04 * max(hi - lo, 1e-6)
            ax.set_ylim(min(lo, gate - pad), max(hi, gate + pad))
            gate_line(ax, gate, fail=fail)
        return fig

    return _plot()


def check_figure(res: Results) -> str:
    values = {(c, k): res.stat(c, k) for c in [*CONDITIONS, CONTROL, PLANE_CONTROL] for k in ("m_line", "alpha_op1")}
    return check_draw(values, check_alt(res))


def check_alt(res: Results) -> str:
    m = {sub: [float(res.stat(c, "m_line").mean()) for c in rungs(sub)] for sub in ex.SUBSPACES}
    d = "rises" if m["axis"][-1] > m["axis"][0] else "falls"
    return f"""
        Two dot panels. Left, the line margin against the anchor weight on a log scale: on the axis it
        {d} from {m["axis"][0]:.2f} at the lowest rung to {m["axis"][-1]:.2f} at the top, with the plane
        below it, and a tinted box under each column marks the un-anchored control. Right, ᾱ at op1 on the
        same axis, with the control box under each axis column scored on the axis and the one under each
        plane column scored on the plane, the plane boxes sitting higher.
    """


@memo
def check_draw(values: dict, alt_text: str) -> str:
    @themed(
        name="h4-checks",
        alt_text=alt_text,
        caption=f"""
            **The two quantities the treatment acts on.** Left, the line margin, which the anchor term
            optimizes: the manipulation check, with a heavier pull expected to raise it. Right, ᾱ at op1, which
            the anti-subspace term acts on, with no direction attached to it. The tinted box under each
            column is the un-anchored control's seed range, scored on that column's subspace: on the axis for
            the axis conditions and on the plane (`{ex.CONTROL_PLANE}`) for the plane ones. An unsigned
            two-dimensional alignment sits higher than a signed one-dimensional one for every state, so the
            two subspaces compare against different boxes.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), layout="constrained")
        axes = cast(AxesRow, axes)
        rng = np.random.default_rng(9)
        for ax, k, ylabel in zip(axes, ("m_line", "alpha_op1"), ("line margin m_line", "ᾱ at op1"), strict=True):
            for sub, ctrl in (("axis", CONTROL), ("plane", PLANE_CONTROL)):
                for i, (x, c) in enumerate(zip(xs_of(sub), rungs(sub), strict=True)):
                    lab = f"control, scored on the {sub}" if k == "m_line" and i == 0 else None
                    column_box(ax, x, values[ctrl, k], ink(rungs(sub)[0]), label=lab)
                    dots(ax, x, values[c, k], c, rng=rng, label=c if k == "m_line" else None)
            ax.set_ylabel(ylabel, fontsize=8)
            ladder_axis(ax)
        gate_line(axes[0], MARGIN_GATE, fail="below")
        fig_legend(fig, axes[0], ncols=5)
        return fig

    return _plot()


def cost_table(res: Results) -> str:
    head = ["condition", *(f"{COST_LABEL[k]}" for k in COSTS)]
    rows = []
    for c in CONDITIONS:
        g = gate_status(res, c)
        cells = []
        for k in COSTS:
            v = res.cost(c, k)
            text = f"{v.mean():+.3f}" if k == "task" else f"{v.mean():.3f}"
            if k == "task":
                text += f" (`{res.task_worst(c)[0]}`)"
            cells.append(bold_if(text, g[k]))
        rows.append([f"`{c}`", *cells])
    gates = ", ".join(
        [
            f"task ≥ −{ex.TASK_GATE:g}",
            f"margin ≥ {MARGIN_GATE:.3f}",
            f"lead ≥ {ex.LEAD_GATE:g}",
            f"contrast ≥ {ex.CONTRAST_GATE:g}",
            f"r² ≥ {R2_GATE:.3f}",
            f"deficit ≤ {ex.NONRED_DEFICIT_GATE:g}",
        ]
    )
    return table_html(
        head,
        rows,
        f"Every cost statistic per condition, on the seed mean, with a value inside its ex-2.2.11 gate in "
        f"bold. The gates: {gates}. The task column names the op the condition is worst on.",
        ref_rows=frozenset({CONDITIONS.index(REFERENCE)}),
    )


# --- The adoption rule ---------------------------------------------------------------------------


def adoption_table(res: Results, a: dict) -> str:
    head = [
        "condition",
        f"upper bound, `{ex.MISSED_OP}` ↓",
        "upper bound, worst other op ↓",
        "cost gates",
        "ᾱ at op1",
        "ᾱ ratio ↓",
        "qualifies",
        f"under ex-2.2.12's band ({OLD_LEVEL:.2f})",
        "ᾱ excess ↓",
        "qualifies, as intended",
    ]
    rows = []
    for c in CONDITIONS:
        r = a["conditions"][c]
        rows.append(
            [
                f"`{c}`",
                bold_if(f"{r['ucb']:.3f}", r["missed_ok"]),
                bold_if(f"{r['others_ucb']:.3f} (`{r['others_ucb_op']}`)", r["others_ok"]),
                "all pass" if not r["failed"] else "misses " + ", ".join(r["failed"]),
                f"{res.stat(c, 'alpha_op1').mean():.3f}",
                bold_if(f"{r['alpha_ratio']:.2f}", r["alpha_ok"]),
                bold_if("yes" if r["qualifies"] else "no", r["qualifies"]),
                bold_if("yes" if r["old_rule"] else "no", r["old_rule"]),
                bold_if(f"{r['alpha_excess']:.2f}", r["alpha_ok_intended"]),
                bold_if("yes" if r["qualifies_intended"] else "no", r["qualifies_intended"]),
            ]
        )
    return table_html(
        head,
        rows,
        f"The adoption rule clause by clause. The upper bound is the one-sided {ex.UCB_LEVEL:.0%} bound on the "
        f"seed mean, and the gate it is read against is {ex.KEPT_GATE:g}. The ᾱ ratio is ᾱ at op1 over the "
        f"un-anchored baseline for the condition's own subspace, and it passes when it is no higher than the "
        f"reference's {a['ref_ratio']:.2f}. Those two baselines are {res.stat(CONTROL, 'alpha_op1').mean():+.3f} on "
        f"the axis, where the alignment is signed and the control's five seeds straddle zero, and "
        f"{res.stat(PLANE_CONTROL, 'alpha_op1').mean():+.3f} on the plane, where it is unsigned. The last column "
        f"applies ex-2.2.12's fixed band to the seed mean "
        f"instead of the upper bound, with the same cost and ᾱ clauses. The last two columns read the ᾱ "
        f"clause as intended, as an excess over the condition's own baseline, passing when it is no higher "
        f"than the reference's {a['ref_excess']:.2f}; that reading was adopted at review, after the data.",
        ref_rows=frozenset({CONDITIONS.index(REFERENCE)}),
    )


# --- Exploratory: the spread of everything else ---------------------------------------------------

SPREAD_KEYS = ("kept", *COSTS)
SPREAD_LABEL = {"kept": f"kept share, {ex.MISSED_OP}"} | COST_LABEL


def spread_values(res: Results, cond: str, key: str) -> np.ndarray:
    return res.kept(cond, ex.MISSED_OP) if key == "kept" else res.cost(cond, key)


def spread_table(res: Results) -> str:
    """The seed standard deviation of every statistic, per condition, as a ratio to the reference's."""
    head = ["condition", *(f"{SPREAD_LABEL[k]}" for k in SPREAD_KEYS)]
    ref = {k: sd(spread_values(res, REFERENCE, k)) for k in SPREAD_KEYS}
    rows = []
    for c in CONDITIONS:
        cells = []
        for k in SPREAD_KEYS:
            s = sd(spread_values(res, c, k))
            ratio = s / ref[k] if ref[k] > 0 else float("nan")
            cells.append(f"{s:.4f} <span class=range>×{ratio:.2f}</span>")
        rows.append([f"`{c}`", *cells])
    return table_html(
        head,
        rows,
        "Seed standard deviation of every statistic, per condition, with its ratio to the reference's beside "
        "it. H1 scores the first column only; the rest are here to say whether a heavier anchor settles "
        "training as a whole or only the leftover. No gate is read off this table.",
        ref_rows=frozenset({CONDITIONS.index(REFERENCE)}),
    )


# --- Exploratory: which lines are left -------------------------------------------------------------


def line_consistency(res: Results, cond: str) -> dict:
    """How much the surviving lines agree from seed to seed, within one condition."""
    k = res.missed_lines(cond)["kept"]
    ok = ~np.isnan(k).any(axis=0)
    v = k[:, ok]
    live = v > 0.5
    per_line = live.mean(axis=0)
    n_seeds = len(v)
    rs = [
        float(np.corrcoef(v[i], v[j])[0, 1])
        for i in range(n_seeds)
        for j in range(i + 1, n_seeds)
        if np.std(v[i]) > 0 and np.std(v[j]) > 0
    ]
    return {
        "n_lines": int(ok.sum()),
        "r": float(np.mean(rs)) if rs else float("nan"),
        "always": float((per_line == 1).mean()),
        "never": float((per_line == 0).mean()),
        "per_line": per_line,
        "op2_color": res.missed_lines(cond)["op2_color"][ok],
    }


def lines_figure(res: Results) -> str:
    data = {c: line_consistency(res, c) for c in CONDITIONS}
    values = {c: (d["per_line"], d["op2_color"]) for c, d in data.items()}
    return lines_draw(values, lines_alt(data))


def lines_alt(data: dict) -> str:
    r = data[REFERENCE]
    return f"""
        Eight small histograms, one per condition, of how often each of the {r["n_lines"]} removal lines of
        {ex.MISSED_OP} survives the projection, as a share of the twenty seeds. Mass piles up at both ends: at
        the reference {r["never"]:.0%} of the lines survive in no seed and {r["always"]:.0%} survive in every
        seed, with the rest spread between. The mean correlation between two seeds' per-line vectors is
        {r["r"]:.2f} at the reference.
    """


@memo
def lines_draw(values: dict, alt_text: str) -> str:
    @themed(
        name="exp-line-survival",
        alt_text=alt_text,
        caption=f"""
            **How often each `{ex.MISSED_OP}` removal line survives, per condition.** One histogram per
            condition over the removal lines: a line at 0 is answered differently in every seed after the
            projection, and a line at 1 keeps its answer in all {ex.SEEDS}. A leftover made of the same lines
            each time piles up at the two ends; a leftover of a shifting set fills the middle. Post hoc, no
            gate.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 4, figsize=(8.4, 4.2), layout="constrained", sharex=True, sharey=True)
        edges = [i / 10 for i in range(11)]
        for ax, c in zip(cast(AxesRow, axes.reshape(-1)), CONDITIONS, strict=True):
            per_line, _ = values[c]
            ax.hist(per_line, bins=edges, color=ink(c), alpha=0.8)
            ax.set_title(c, fontsize=8)
            ax.grid(axis="y", alpha=0.2)
        for ax in axes[-1]:
            ax.set_xlabel("share of seeds the line survives", fontsize=7)
        for ax in axes[:, 0]:
            ax.set_ylabel("lines", fontsize=8)
        return fig

    return _plot()


def lines_table(res: Results) -> str:
    head = [
        "condition",
        "lines scored",
        "never survives",
        "survives in every seed",
        "mean seed-to-seed r",
        "share from the two reddest op2 colors",
    ]
    reddest = np.argsort(ex.REDNESS)[-2:]
    rows = []
    for c in CONDITIONS:
        d = line_consistency(res, c)
        w = d["per_line"]
        from_red = float(w[np.isin(d["op2_color"], reddest)].sum() / w.sum()) if w.sum() > 0 else float("nan")
        rows.append(
            [
                f"`{c}`",
                str(d["n_lines"]),
                f"{d['never']:.0%}",
                f"{d['always']:.0%}",
                f"{d['r']:.2f}",
                f"{from_red:.0%}",
            ]
        )
    return table_html(
        head,
        rows,
        f"The `{ex.MISSED_OP}` leftover line by line. A line counts as surviving in a seed when it keeps more "
        f"than half of its clean expected exact match. The correlation is the mean over every pair of seeds of "
        f"the correlation between their per-line kept vectors: near one means the same lines survive every "
        f"time. The last column is the share of all surviving weight carried by lines whose second operand is "
        f"one of the two reddest palette colors, which is where ex-2.2.12 found the survival. Post hoc.",
        ref_rows=frozenset({CONDITIONS.index(REFERENCE)}),
    )


# --- Exploratory: the achromatic candidate ---------------------------------------------------------

ACHROMATIC_PASSES = ("blocks", "gray", "gray-blocks")
ACHROMATIC_LABEL = {
    "blocks": "projection at the blocks",
    "gray": "op2 grayed",
    "gray-blocks": "both",
}


def achromatic_figure(res: Results) -> str:
    values = {(c, p): res.achromatic(c, p) for c in CONDITIONS for p in ACHROMATIC_PASSES}
    return achromatic_draw(values, achromatic_alt(res))


def achromatic_alt(res: Results) -> str:
    b, g = (float(res.achromatic(REFERENCE, p).mean()) for p in ("blocks", "gray-blocks"))
    d = "above" if g > b else "below"
    return f"""
        Three dot panels, one per edit, with the four anchor weights along the bottom on a log scale and the
        two subspaces as paired columns. Left, the projection at the blocks alone; middle, the second operand
        replaced by a gray of the same value; right, the two together. At the reference the two together read
        {g:.2f}, {d} the {b:.2f} of the projection alone.
    """


@memo
def achromatic_draw(values: dict, alt_text: str) -> str:
    @themed(
        name="exp-achromatic",
        alt_text=alt_text,
        caption=f"""
            **The achromatic edit on the `{ex.MISSED_OP}` removal lines.** Each panel is a kept share against
            the unedited line's own answer: the projection at the blocks, the gray substitution on its own,
            and the two together. If the surviving answers are read off how achromatic the second operand
            looks, graying it leaves them in place; if they need that operand's own hue, it does not. Post
            hoc, no gate.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2), layout="constrained", sharey=True)
        axes = cast(AxesRow, axes)
        rng = np.random.default_rng(11)
        for ax, p in zip(axes, ACHROMATIC_PASSES, strict=True):
            for sub in ex.SUBSPACES:
                for x, c in zip(xs_of(sub), rungs(sub), strict=True):
                    dots(ax, x, values[c, p], c, rng=rng, ms=5)
            ax.set_title(ACHROMATIC_LABEL[p], fontsize=9)
            ladder_axis(ax)
        axes[0].set_ylabel(f"kept share on {ex.MISSED_OP} removal lines", fontsize=8)
        return fig

    return _plot()


def achromatic_table(res: Results) -> str:
    head = [
        "condition",
        *(ACHROMATIC_LABEL[p] for p in ACHROMATIC_PASSES),
        "both − blocks",
        "both − grayed",
    ]
    rows = []
    for c in CONDITIONS:
        v = {p: float(res.achromatic(c, p).mean()) for p in ACHROMATIC_PASSES}
        rows.append(
            [
                f"`{c}`",
                *(f"{v[p]:.3f}" for p in ACHROMATIC_PASSES),
                f"{v['gray-blocks'] - v['blocks']:+.3f}",
                f"{v['gray-blocks'] - v['gray']:+.3f}",
            ]
        )
    return table_html(
        head,
        rows,
        "Seed-mean kept share under each edit, over the missed op's removal lines, all read against the "
        "unedited line's own answer. Graying the second operand moves about three answers in ten on its own, "
        "so the two difference columns say different things: the first is what graying adds to the "
        "projection, and the second is what the projection still costs once the operand is already gray. "
        "Post hoc.",
        ref_rows=frozenset({CONDITIONS.index(REFERENCE)}),
    )


# --- Exploratory: what the response looks like ------------------------------------------------------


def alpha_figure(res: Results) -> str:
    values = {c: res.alpha_by_color(c) for c in CONDITIONS}
    return alpha_draw(values, alpha_alt(res))


def alpha_alt(res: Results) -> str:
    lo, hi = (float(res.alpha_by_color(c).mean()) for c in (rungs("axis")[0], rungs("axis")[-1]))
    return f"""
        Eight grading clouds in two rows, the axis conditions above and the plane conditions below, one column
        per anchor weight. Within each cloud redness runs left to right and the height is ᾱ at the first
        operand, with the twenty seeds lofted together so the cloud's thickness is the seed spread. The
        response rises with redness in every panel. Its mean over all colors goes from {lo:.2f} at the lowest
        rung of the axis ladder to {hi:.2f} at the top.
    """


@memo
def alpha_draw(values: dict, alt_text: str) -> str:
    @themed(
        name="exp-alpha-clouds",
        alt_text=alt_text,
        caption=f"""
            **The α response at the first operand, per rung.** One cloud per condition, drawn in the colors of
            the grid: within each panel redness runs left to right, and the height is the alignment of that
            color's state with the home of *red*, averaged over slices. The {ex.SEEDS} seeds are lofted into
            one cloud, so its thickness at a given redness is the spread across seeds rather than a mean. Post
            hoc, no gate.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 4, figsize=(8.4, 4.4), layout="constrained", sharex=True, sharey=True)
        lo = min(float(np.min(v)) for v in values.values())
        hi = max(float(np.max(v)) for v in values.values())
        pad = 0.06 * (hi - lo)
        ylim = (lo - pad, hi + pad)
        for row, sub in zip(axes, ex.SUBSPACES, strict=True):
            for ax, c in zip(row, rungs(sub), strict=True):
                GradingCloud(ax, values[c], ylim=ylim, k=26)
                ax.set_ylim(*ylim)
                ax.set_title(c, fontsize=8)
                ax.set_xticks([])
                ax.grid(axis="y", alpha=0.2)
            row[0].set_ylabel("ᾱ at op1", fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("redness →", fontsize=7)
        return fig

    return _plot()


# --- The report --------------------------------------------------------------------------------

if metrics_loaded is None or ref_loaded is None or sweep_loaded is None:
    stop("The run has not published its results yet.")

res = Results(metrics=metrics_loaded, ref=ref_loaded, sweep=sweep_loaded)
verdicts = adoption(res)
ladder_levels = ", ".join(f"{x:g}" for x in ex.LADDER)

# Numbers the verdict prose quotes. Everything here is computed; nothing is typed in.
_sp = {s: spread(res, s) for s in ex.SUBSPACES}
_kept = {c: res.kept(c, ex.MISSED_OP) for c in CONDITIONS}
_top = {s: rungs(s)[-1] for s in ex.SUBSPACES}
_pair = {c: paired(_kept[c], _kept[REFERENCE]) for c in CONDITIONS if c != REFERENCE}
_worst = {c: res.kept_worst(c) for c in CONDITIONS}
_margin = {c: float(res.cost(c, "m_line").mean()) for c in CONDITIONS}
_alpha = {c: float(res.stat(c, "alpha_op1").mean()) for c in CONDITIONS}
_base = {s: float(res.stat(res.baseline(rungs(s)[0]), "alpha_op1").mean()) for s in ex.SUBSPACES}
_excess = {c: _alpha[c] - _base[s] for s in ex.SUBSPACES for c in rungs(s)}
_v = verdicts["conditions"]
V = SimpleNamespace(
    ratio={s: _sp[s]["ratio"] for s in ex.SUBSPACES},
    slope={s: _sp[s]["trend"]["slope"] for s in ex.SUBSPACES},
    p={s: _sp[s]["trend"]["p_one_sided"] for s in ex.SUBSPACES},
    sd={c: sd(_kept[c]) for c in CONDITIONS},
    mean={c: float(_kept[c].mean()) for c in CONDITIONS},
    ref_earlier=(float(res.kept_earlier(REFERENCE).mean()), sd(res.kept_earlier(REFERENCE))),
    p02_earlier=(float(res.kept_earlier("plane-0.2").mean()), sd(res.kept_earlier("plane-0.2"))),
    pair=_pair,
    worst_ref=_worst[REFERENCE],
    worst_top={s: (_worst[_top[s]][0], _worst[REFERENCE][1] - _worst[_top[s]][1]) for s in ex.SUBSPACES},
    margin={s: (_margin[rungs(s)[0]], _margin[_top[s]]) for s in ex.SUBSPACES},
    alpha={s: (_alpha[rungs(s)[0]], _alpha[_top[s]]) for s in ex.SUBSPACES},
    base=_base,
    ref_ratio=verdicts["ref_ratio"],
    p01=_v["plane-0.1"],
    intended=verdicts["qualifying_intended"],
    excess={"plane-0.1": _excess["plane-0.1"], REFERENCE: _excess[REFERENCE]},
    n_outside=sum(abs(d[0]) > ex.MEAN_BAND for d in _pair.values()),
)
_a, _pl = "axis", "plane"


rf"""
# Ex 2.2.13: does a heavier anchor make the leftover predictable?

/// tip |
<!-- tl;dr -->
The recipe from ex-2.2.11 leaves some *red* behind on one op of eleven, and how much swings widely from seed to seed. We climb a ladder of anchor weights, on an axis and on a plane, and ask whether a heavier anchor gives us a leftover we can predict. It does not.
///

Making the anchor 2.8× heavier raises the line margin, the quantity the anchor term optimizes, by about three percent. It does not narrow the seed-to-seed spread of the leftover on either subspace; the tight condition ex-2.2.12 saw was five lucky seeds. What does move the leftover is where *red* lives. On the plane the leftover is smaller at every rung, and the plane at the weight used by the recipe is the lowest in the experiment. Read as the adoption rule intended, that condition qualifies. We keep the recipe at `{ex.REFERENCE}` all the same: the gain is small and it costs a second coordinate. The plane is now a validated alternative for a concept that needs more room. The anchored-op experiments inherit the leftover as a bounded confound rather than a fixed one.

## Findings

- [The leftover gets more predictable (H1)](#the-leftover-gets-more-predictable-h1) — did not hold. On the axis the spread falls a little as the weight rises, to {V.ratio[_a]:.2f} of the spread at the reference, against a gate of {ex.SD_RATIO_GATE:g}. On the plane it widens. The tight plane condition from ex-2.2.12 does not replicate at twenty seeds.
- [The leftover does not get smaller (H2)](#the-leftover-does-not-get-smaller-h2) — held on the axis. The plane lowers the mean at every rung, `plane-{ex.LADDER[0]:g}` by {-V.pair["plane-0.1"][0]:.2f}, with an interval clear of zero. The weight moves the mean in no consistent direction.
- [The worst op improves (H3)](#the-worst-op-improves-h3) — held on the axis, where the worst other op falls by {V.worst_top[_a][1]:.3f} at the top rung. It missed on the plane, where the worst other op turns back up at the top rung.
- [What the weight spends (H4)](#what-the-weight-spends-h4) — held. Every cost statistic is inside its gate at every condition. The line margin rises by about three percent across the ladder, so the treatment does reach the model, and it is close to saturated at the weight used by the recipe.

[The adoption rule](#the-adoption-rule): `plane-{ex.LADDER[0]:g}` clears every clause about the leftover and the costs, and fails the ᾱ clause only on that clause's arithmetic, a fault in the rule as written. Read as intended, it is the one condition that qualifies. The recipe stays at `{ex.REFERENCE}` by a decision made after the data: the gain is about {-V.pair["plane-0.1"][0]:.2f} on the leftover's mean, and it costs a second coordinate of the stream.

## How to read this draft

The ladder, the four predictions, and the adoption rule were fixed before any run, at commit `{FROZEN_AT}`. Everything after that commit is either results filled into their sections or exploratory work, marked as post hoc.

Every anchored run is new. The seeds that produced the observation this experiment follows up are not reused, so the reference is trained again here beside the ladder. The un-anchored control is borrowed from ex-2.2.11. The adoption rule summarizes a leftover by an upper confidence bound rather than by the fixed band ex-2.2.12 used; the [adoption rule](#the-adoption-rule) section gives both verdicts.

## Why this experiment

[Ex-2.2.11](../ex-2.2.11/report.py) put *red* on one axis of the residual stream of a small transformer,[^rs] taught it eleven color operations, and projected the axis out. On ten ops the model lost *red*. On `{ex.MISSED_OP}` it kept about a quarter of the answers that need the red operand, which is over the gate.

[^rs]: The *residual stream* is the running vector of activations that each layer of a transformer reads from and writes back to.

[Ex-2.2.12](../ex-2.2.12/report.py) asked why and got a clean negative result. The story about which part of a hue an axis can hold was wrong, no change to the recipe brought the leftover under the gate, and the survival turned out to come from two of the seven red colors.

It also left something unexplained. The leftover swings from seed to seed: the reference keeps anywhere from 0.06 to 0.41 of those answers, depending on which seed trained the model. One condition of the sweep, the plane at twice the anchor weight, kept 0.14 to 0.22 across its five seeds instead. A range over five draws is narrower than a range over twenty at the same spread, so the two ranges are not directly comparable; the standard deviations behind them, 0.036 against 0.125, are what H1 is built on. A standard deviation from five seeds is itself a loose estimate, which is why the gate below asks for less than that ratio.

That is the observation this experiment is built on, and it is not the one the sweep was scored on. A leftover we cannot remove is a confound for every anchored-op experiment that follows.

If it is the same size every time, we can measure it once and subtract it. If it varies three-fold across seeds, every experiment that meets it has to measure it again. So how wide the seed spread is counts as a result in its own right. Read more broadly, the spread of the leftover is one face of how reproducible training is under the recipe on this grammar, and the ladder is a test of whether a heavier anchor makes training more reproducible.

A second reason to expect something from the weight: we run at λ_a = 0.1 because [ex-2.1.6](../ex-2.1.6/report.py) chose a rung safely inside the region where the task is unhurt. The survey in [ex-2.1.11](../ex-2.1.11/report.py) later mapped the weight on the six-op grammar, putting the plateau of the margin at λ_a 0.28–0.94, with the task gate biting only from about 0.38. On that map we have been running near the bottom of the useful range the whole time, and nothing has tested the region in between.

## Conditions

One ladder, crossed with the home of *red*, and the un-anchored control beside it.

{ladder_html()}

**The weight.** λ_a takes four levels, {ladder_levels}, each a factor of √2 above the last. We read a weight on a log scale, so a constant ratio puts the levels evenly apart and makes the shape of any trend across them easy to see. The lowest rung is the recipe from ex-2.2.11, and {ex.LADDER[2]:g} is the one step ex-2.2.12 took. The top rung is where the margin plateau ex-2.1.11 mapped begins, and it stops short of the level where that survey saw its first task failures (about 0.38, on the six-op grammar at a warmer pooling temperature than ours). Where the useful range ends is left open here; the ladder tests the region between the recipe and that point.

**What else the weight moves.** The anti-subspace term is specified as a ratio to λ_a, peaking at 2.5× λ_a and holding at 0.3× λ_a, so a rung of the ladder raises the repulsion by the same factor as the pull. The ladder is a joint anchor-and-repulsion ladder rather than a pure anchor ladder, and a result along it belongs to the pair. Ex-2.2.12 moved the two separately, one step each (`lam-0.2` and `anti-5`), and neither moved any measurement on its own, so we do not spend runs here separating them; if the ladder moves something, an arm that holds the repulsion fixed at one rung is the follow-up.

**The home of *red*.** *Red* lives either on the first axis e₁ or on the plane spanned by e₁ and e₂. Where the plane is the home, the anchor term, the anti-subspace term, the alignment measurements, and the removal all take the pair of axes instead of the single axis, as ex-2.2.12 defined them.

The recipe has two anchoring terms and no anti-anchor term. The pull is one minus the alignment of a labelled state with the home; on the axis that alignment is the signed cosine, so the pull is toward +e₁ and a state at −e₁ is as far from home as it can be. On the plane it is the unsigned length of the projection, so the pull is toward the plane with no preferred direction within it, and −e₁ is home. The anti-subspace term is the squared alignment averaged over every live position, labelled or not, and it has no sign in either home. An anti-anchor term, the one-sided hinge that kept every state out of the hemisphere opposite the anchor so a fallback could live there, belonged to M1's fallback and last ran in M2 in ex-2.2.2's fallback arm; the recipe line from ex-2.1.6 onward has never carried it, and on the plane there is no hemisphere for it to name.

Ex-2.2.12 settled against the plane's first rationale, that a single axis is too small a home for a hue. Two things keep it here. The tight condition that prompted this experiment was a plane *and* a heavier weight, while the heavier weight on the axis was among the loosest on the kept share in that sweep even as its line margin and ᾱ were the tightest, so a ladder on the axis alone could not say whether the narrowing comes from the weight, from the subspace, or from the two together. And the plane at the recipe's own weight has been seen at five seeds only; at twenty, `plane-{ex.LADDER[0]:g}` against `{ex.REFERENCE}` is a test of the subspace on its own, at a resolution ex-2.2.12 did not have.

**The control.** The un-anchored control is ex-2.2.11's, served from the store at its seeds rather than retrained. It is the reference for the task gate, which asks for a seed mean within a band of the control's, and the ᾱ baseline for the axis conditions. Neither role touches the observation this experiment follows up, which was read off anchored checkpoints, so the borrow spends nothing the design needs; comparisons against it are unpaired across seed sets, which a comparison of seed means absorbs. Every plane condition is compared against the same checkpoints scored on the plane (`{ex.CONTROL_PLANE}`): for every state, anchored or not, an unsigned two-dimensional alignment sits higher than a signed one-dimensional one, so the control has to be scored the same way.

**The seeds.** {ex.SEEDS} per condition, all fresh: condition seed *i* trains at model seed {ex.SEED_OFFSET} + *i*, where ex-2.2.11 and ex-2.2.12 used an offset of 100. Every condition pairs with every other one seed for seed within this experiment, and the reference is retrained rather than borrowed, which makes it a replication of ex-2.2.11's `handover` at seeds it never saw.

The spread question sets the count. A one-sided F-test on the ratio of variances between the top and the bottom of the ladder resolves a halved standard deviation with power 0.90 at twenty seeds a group, 0.81 at fifteen, and 0.63 at ten; a difference in means of the size we care about would be settled by half as many. {ex.N_RUNS} runs in all.

Everything else is unchanged from ex-2.2.11: [table A+](../ex-2.2.4/report.py#the-op-set), the stochastic corpus, the whole-line labeller, the untied readout, the removal lines chosen by hue, τ = 0.1, the shape of the anti-subspace schedule, and 50 epochs at d64-L4.

## Glossary

<dl>
<dt>Kept share</dt>
<dd>How much of its clean accuracy on an op's removal lines a model keeps after the projection. One means the projection did nothing; zero means every one of those answers changed, which is what the gate takes
<em>red</em> being gone to mean.</dd>
<dt>Removal lines</dt>
<dd>The red lines whose answer needs the hue of the red operand: some permutation of the channels of that operand moves the true answer far. The rule from ex-2.2.11, unchanged.</dd>
<dt>Seed spread</dt>
<dd>The standard deviation of a statistic across the seeds of one condition. This is the quantity H1 is about.</dd>
<dt>Upper bound</dt>
<dd>The one-sided {ex.UCB_LEVEL:.0%} upper confidence bound on the seed mean of a condition. One condition can have a low mean and a wide spread, and another a higher mean and a narrow spread, and the two can still have the same upper bound. That is the point of using it.</dd>
<dt>Line margin</dt>
<dd>How far the anchored states on the labelled lines sit above the rest along the home of <em>red</em>. This is the quantity the anchor term optimizes, so it is a check that the treatment landed rather than a result.</dd>
<dt>ᾱ at op1</dt>
<dd>The mean alignment with the anchored subspace over every color at the first operand position. How much the colors that are not red have drifted toward the home of <em>red</em>.</dd>
</dl>

## The leftover gets more predictable (H1)

**What we expect.** Across the ladder, the seed spread of the kept share on the `{ex.MISSED_OP}` removal lines narrows as λ_a rises. The number we score is the ratio of the standard deviation at the top of the ladder to the one at the lowest rung, within a subspace. H1 holds when that ratio falls to {ex.SD_RATIO_GATE:g} or below in both subspaces, and the spread falls monotonically enough that a trend contrast across the four levels has a negative slope at {ex.TREND_ALPHA:g}.[^trend]

It holds in part when one subspace does that and the other does not. On the plane alone, that would say the narrowing needs the plane, and the comparison of `plane-{ex.LADDER[0]:g}` with the reference then says whether the plane narrows the spread by itself or only once the weight rises. A flat or rising spread in both would mean the tight condition in ex-2.2.12 was five lucky seeds, and that nothing on this ladder buys predictability.

One reading of a narrowing has to be ruled out before it counts. The kept share is a proportion over a fixed set of lines, so its spread is bounded below by sampling noise that depends on where the mean sits: a condition whose mean is near zero or near one cannot vary much. H1 is therefore read together with H2. A narrowing that arrives with a mean the ladder also moved is a narrowing we get for free, and the report says so rather than claiming the recipe bought it.

[^trend]: The *trend contrast* regresses each run's absolute distance from the median of its own condition on log λ_a. Working from the median rather than the mean keeps a couple of extreme seeds from deciding it, and asking for a slope rather than for any difference between the levels means a spread that rose and then fell does not count as a pass.

The kept share is a fair thing to score here. The anchor term acts during training on how well a state aligns with the home of *red*; the kept share is taken afterwards, on held-out lines, through a projection the term never sees. Nothing in the treatment is set up to reduce its spread, so a narrowing would be telling us something.

The quantities the two terms do act on — the line margin and ᾱ at op1 — are reported under H4 as manipulation checks rather than with gates on them.
"""

h1_figure(res)

# %%
h1_table(res)

rf"""
**What we saw.** H1 did not hold. On the axis the spread falls a little with the weight: the standard deviation at the top rung is {V.ratio[_a]:.2f} of the reference's, against a gate of {ex.SD_RATIO_GATE:g}, and the trend contrast has a negative slope at one-sided p = {V.p[_a]:.3f}. The trend clause passes and the ratio clause does not, so the narrowing is there and it is small. On the plane the spread widens: the ratio is {V.ratio[_pl]:.2f} and the slope is positive.

The observation this experiment followed up did not replicate. Ex-2.2.12's `plane-{ex.LADDER[2]:g}` had a standard deviation of {V.p02_earlier[1]:.3f} over five seeds; the same condition at twenty fresh seeds has {V.sd["plane-0.2"]:.3f}, in line with every other rung. Its mean did replicate ({V.p02_earlier[0]:.3f} then, {V.mean["plane-0.2"]:.3f} now), and so did the reference's ({V.ref_earlier[0]:.3f} at ex-2.2.11's seeds, {V.mean[ex.REFERENCE]:.3f} here). A standard deviation from five draws has the wide interval the prereg noted, and this is what the low end of that interval looks like when it comes up.

The sampling-noise reading H1 asked us to rule out does not arise: no mean moved close enough to zero or one to bound its spread. The means themselves are H2.

{verdict_md("miss", "The spread narrows a little on the axis and widens on the plane; neither reaches the gate, and the tight condition from ex-2.2.12 does not replicate.")}

## The leftover does not get smaller (H2)

**What we expect.** The seed-mean kept share on `{ex.MISSED_OP}` is flat across the ladder: no level differs from the reference by more than {ex.MEAN_BAND:g}, which is about what twenty paired seeds can resolve at the spread of the reference. H2 is a statement about what we can see at this resolution rather than a claim that the mean is unmoved, and a reviewer reading it as an equivalence test with a wide band is reading it right.

The evidence for it: the one step ex-2.2.12 took moved the mean up on the axis and down on the plane, and its tight plane condition sits almost exactly on the mean of the five reference seeds it pairs with. A level that does move the mean down by more than {ex.MEAN_BAND:g} would be a better result than we expect, and the adoption rule is written to take it.

Together, H1 and H2 give the shape of the claim: climbing the ladder leaves a leftover that is no smaller and that comes out the same size every time.

The means themselves are the large marks of the H1 figure, read along the λ_a axis; the table below pairs them seed for seed with the reference.
"""

h2_table(res)

rf"""
**What we saw.** H2 held on the axis and not on the plane. On the axis, two of the three rungs sit inside the band and `axis-{ex.LADDER[1]:g}` just outside it, with no trend in the weight: the mean goes down, up, and down again along the ladder. On the plane every rung is below the reference, three of the four by more than the band, and `plane-{ex.LADDER[0]:g}` by {-V.pair["plane-0.1"][0]:.3f} with a 95% interval of {-V.pair["plane-0.1"][2]:.3f} to {-V.pair["plane-0.1"][1]:.3f} below it. That is the comparison ex-2.2.12 could not resolve at five seeds, and at twenty it says the subspace lowers the leftover on its own. Adding weight on top of the plane moves the mean back up a little rather than further down.

So the shape of the claim in the prereg, no smaller and the same size every time, came out the other way round: the plane makes the leftover somewhat smaller, and nothing on the ladder makes it more predictable.

{verdict_md("partial", "Flat within the band on the axis; lower than the reference at every rung on the plane.")}

## The worst op improves (H3)

**What we expect.** On the ten ops other than `{ex.MISSED_OP}`, the highest per-op seed-mean kept share falls as λ_a rises. The number we score is that highest value at the top of the ladder against the same quantity at the reference, both at twenty seeds; H3 holds when it falls by at least {ex.WORST_OP_MARGIN:g}.

Any claim that a recipe removes *red* cleanly is limited by whichever op does worst, and that is not the same op in every condition, so a mean over the ten would hide it.

The margin is the {ex.WORST_OP_MARGIN:g} in the statement above, and it is there because a kept share is a proportion over some 330–400 removal lines per op, so on one checkpoint it carries a sampling error of about 0.02 at the values the worst op sits at, and the seed spread of the other ops in ex-2.2.12 was about 0.03. A drop smaller than that is one we could not tell from no drop. In ex-2.2.12 the worst other op was `darken` at the reference, `darken` again one step up the axis, and `lighten` on the plane at twice the weight — the only condition in that sweep with no op over the gate at all — and the spacing between those three values is close to the margin, which is why H3 asks for a margin rather than for a difference. H3 fails if the worst op is flat or rises, which would mean the ladder buys nothing on the ops that already remove *red*.
"""

h3_figure(res)

# %%
h3_table(res)

rf"""
**What we saw.** H3 held on the axis and missed on the plane. The frozen wording does not split the top of the ladder by subspace, so we read it with H1's convention and call it partial. On the axis the worst other op falls at every rung, and by {V.worst_top[_a][1]:.3f} at `{_top[_a]}`, twice the margin. On the plane it falls by the margin or more at the two middle rungs and turns back up at `{_top[_pl]}`, to {V.worst_top[_pl][1]:.3f} under the reference, short of the margin. The worst op is `{V.worst_ref[0]}` at the reference and `{V.worst_top[_a][0]}` everywhere else, as the spacing in ex-2.2.12 suggested, and every condition's worst other op is under the gate. The ten ops that already remove *red* have some room to spare, and a heavier anchor on the axis uses a little of it.

{verdict_md("partial", "The worst other op falls by twice the margin at the top of the axis ladder, and turns back up at the top of the plane's.")}

## What the weight spends (H4)

**What we expect.** Every cost statistic stays inside its gate from ex-2.2.11 on the seed mean through the whole ladder: {ex.COST_LIST}. The ladder stops under the level where the six-op survey saw the task give way, so we expect every cost to hold; the one we name as most likely to leave its gate first is the non-red deficit on the plane conditions, which ex-2.2.12 measured four times higher on the plane than on the axis.

If a statistic leaves its gate at or below λ_a = {ex.LADDER[-1]:g}, the useful range of the recipe on this grammar is narrower than the map in ex-2.1.11 suggested, which is worth knowing on its own. If none does, the range is at least this wide, and where it ends stays open.

We report the line margin here as the manipulation check. It is the quantity the anchor term optimizes, and a heavier pull should raise it along the ladder — one direction, with no argument available for the other. If it stays flat, the weight is not reaching the model, and the other three sections have no treatment to interpret. ᾱ at op1 goes in the same figure without a direction attached to it: it is what the anti-subspace term acts on, and that term is climbing the ladder alongside the pull.
"""

cost_figure(res)

# %%
cost_table(res)

# %%
check_figure(res)

rf"""
**What we saw.** H4 held: every statistic is inside its gate at every condition, and none comes near one. The non-red deficit on the plane is two to eight times the axis's, in the direction predicted, and still a factor of three under its gate; it falls with the weight rather than rising. So the useful range of the recipe on this grammar reaches at least λ_a = {ex.LADDER[-1]:g}, and where it ends is still open.

The manipulation check is the finding of this section. The line margin rises with the weight on both subspaces, but by very little: from {V.margin[_a][0]:.3f} to {V.margin[_a][1]:.3f} on the axis and from {V.margin[_pl][0]:.3f} to {V.margin[_pl][1]:.3f} on the plane, about three percent for an anchor 2.8 times heavier. The weight reaches the model, and the quantity it optimizes is close to saturated at the recipe's weight already. That is the likely reason H1 and H2 had so little to respond to: between {ex.LADDER[0]:g} and {ex.LADDER[-1]:g} the recipe sits on a plateau of the thing it trains. ᾱ at op1 falls with the weight on the axis, from {V.alpha[_a][0]:.3f} to {V.alpha[_a][1]:.3f}, which is the anti-subspace term climbing beside the pull, and barely moves on the plane.

{verdict_md("pass", "Every cost statistic is inside its gate at every rung, and the line margin rises with the weight, by about three percent.")}

## The adoption rule

The rule, frozen before the run:

> {ex.ADOPTION}

What changed from the rule in ex-2.2.12, and why. {ex.OLD_RULE}
"""

adoption_table(res, verdicts)

rf"""
To the letter, no condition qualifies under either rule. Under the fixed band from ex-2.2.12 nothing comes close: the lowest seed mean is `plane-{ex.LADDER[0]:g}` at {V.mean["plane-0.1"]:.3f}, against a band that asks for {ex.KEPT_GATE - ex.OLD_BAND:g}.

Under the rule frozen here, `plane-{ex.LADDER[0]:g}` clears every clause about the leftover and the costs. Its upper bound on `{ex.MISSED_OP}` is {V.p01["ucb"]:.3f} and the upper bound on its worst other op is {V.p01["others_ucb"]:.3f}. Both are under the {ex.KEPT_GATE:g} gate, and every cost statistic passes. The one clause it fails is the ᾱ clause, and it fails on the arithmetic of that clause. The clause takes the ᾱ of each condition as a ratio to the un-anchored baseline of its own subspace, then compares that with the ratio of the reference to the axis baseline. The axis baseline is a signed cosine, and it averages {V.base[_a]:.3f} over the five control seeds, so the ratio for the reference is {V.ref_ratio:.1f}. The plane baseline is an unsigned length, {V.base[_pl]:.3f}, so every plane ratio is a small positive number, and no positive number is below a negative one. In raw terms the ᾱ of the plane conditions runs about {V.alpha[_pl][1]:.2f} to {V.alpha[_pl][0]:.2f}, against {V.alpha[_a][1]:.2f} to {V.alpha[_a][0]:.2f} on the axis. The difference is modest, and the ratio turns it into an impossible one.

The clause was written badly, so we read it as it was meant. Its purpose was to catch a condition that pulls the other colors toward the home of *red* more than the reference does, and the ratio was an attempt in the prereg to put the two subspaces on one scale. The comparison ex-2.2.12 made across subspaces, and the one the clause was reaching for, is the excess over the baseline of each subspace. On that reading `plane-{ex.LADDER[0]:g}` sits {V.excess["plane-0.1"]:.2f} above its own baseline where the reference sits {V.excess[ex.REFERENCE]:.2f} above the axis one, so it passes. The last column of the table applies that reading to every condition, and `plane-{ex.LADDER[0]:g}` is the one condition that qualifies. Reading a frozen rule by its intent is a decision made after the data, so the table keeps the literal column beside it and a reader can weigh the two.

We keep the recipe at `{ex.REFERENCE}` all the same. That is a second decision made after the data, on grounds the rule does not name. The plane lowers the mean of the leftover by about {-V.pair["plane-0.1"][0]:.2f}. In return the concept takes two coordinates of the stream instead of one, the non-red deficit is a few times the one on the axis, and every anchored-op experiment that follows inherits a two-dimensional home for a hue that one dimension holds well enough on ten ops of eleven. That is a small gain at a standing cost. What the ladder has settled is that the plane works, and by how much, so it is a validated alternative for a concept that turns out to need more room than an axis. If it is preregistered, the ᾱ clause should be stated as an excess.

## Exploratory analyses

Four measurements are planned as descriptions rather than tests: they carry no gate, and no finding rests on them. Anything we think of after seeing the data goes here too, marked as post hoc.

**The spread of everything else.** H1 scores the spread of one statistic. The same seed-spread table for every cost statistic, the line margin, and the task, per condition, says whether a heavier anchor makes training as a whole more reproducible on this grammar or only settles the leftover. It carries no gate because we have no prediction for the direction of most of them: a heavier pull could hold the margin to a tighter value across seeds, or could amplify whatever differs between initializations. Ex-2.2.12 gives a hint that the two spreads can move apart: one step up the axis, the line margin and ᾱ at op1 were the tightest in the sweep while the kept share stayed as wide as the reference, and on the plane at the same step it was the other way round.
"""

spread_table(res)

r"""
A heavier anchor does not settle training as a whole. Relative to the reference, the line margin's spread tightens on the plane and not on the axis, the contrast and the grading r² widen at λ_a = 0.2 on both subspaces, and the task's spread roughly doubles at `axis-0.28` and `plane-0.1`. The hint from ex-2.2.12, that the spread of the kept share and the spread of the margin move independently, holds up here.
"""

rf"""
**Which lines are left.** Ex-2.2.12 found that nearly all of the `{ex.MISSED_OP}` survival comes from two of the seven red colors, but it scored conditions rather than individual lines. Scoring per line across the ladder tells us whether a narrow seed spread means the same lines survive every time, which would let us characterize the leftover, or a shifting set of lines that happens to be the same size.
"""

lines_figure(res)

# %%
lines_table(res)

r"""
The leftover is a partly shifting set of lines, and the subspace changes its character. On the axis the per-line survival is bimodal: about a third of the lines survive in no seed, and a second hump survives in most seeds, with seed-to-seed correlations of 0.3 to 0.5. On the plane the histogram is a single hump near zero with correlations of 0.1 to 0.2, which is a thinner leftover spread over more lines. No line survives in all twenty seeds in any condition. Ex-2.2.12's finding that two of the seven red colors carry nearly all of the survival does not hold at this resolution: those two carry between a tenth and a quarter of it.
"""

r"""
**The achromatic candidate.** The exploratory section of ex-2.2.12 proposed that the surviving lines are answered from how gray the second operand looks. The axis does not hold that information, and the blocks can read it off the other channels. To test it, we project at the blocks with the second operand replaced by a gray of the same value. That is a scoring pass on checkpoints this experiment trains anyway.
"""

achromatic_figure(res)

# %%
achromatic_table(res)

r"""
Graying the second operand is a large edit on its own, moving about thirty percent of the answers with no projection at all. Against that, projecting at the blocks costs almost nothing once the operand is gray, where the same projection with the operand intact costs most of the answers. So on these lines the whole effect of the removal runs through the second operand's hue, and once that hue is gone there is nothing left for the projection to take. The achromatic candidate as ex-2.2.12 phrased it, an answer read from how gray the operand looks, is not what this shows; the hue of the second operand is doing the work, through channels the anchored subspace does not hold.
"""

r"""
**What the response looks like.** A grading cloud of the α response at op1 against how red the first operand is, one per level of the ladder. The grading r² in ex-2.2.12 barely moved across its whole sweep, so this is here as a picture of what a near three-fold change in the recipe does to the response. It carries no gate.
"""

alpha_figure(res)

r"""
The curve is a sigmoid in redness at every rung, and the clouds differ in texture. At λ_a = 0.1 the cloud is smooth on both subspaces. At 0.2 it is noisier, with more low-alignment strays through the middle of the redness range. At 0.28 it is stepped, and the middle of the curve separates into bands. The rung used by the recipe, `axis-0.1`, has the cleanest response of the eight. This is the plateau in the line margin seen from the other side: the mean curve barely moves as the weight rises, and what the weight changes is the texture across seeds.
"""

rf"""
## Discussion

The ladder was built on the premise that we had been running near the bottom of the useful range, so a heavier anchor would give the downstream measurements something to respond to. What it shows instead is a plateau. Between λ_a {ex.LADDER[0]:g} and {ex.LADDER[-1]:g} the line margin moves by about three percent, every cost stays well inside its gate, and the leftover on `{ex.MISSED_OP}` neither shrinks nor settles. On this grammar the weight is not the lever we thought it was, so the recipe can stay where it is, with some confidence that the choice does not matter much.

The subspace *is* a lever, but a small one. At the weight used by the recipe, the plane lowers the leftover by about {-V.pair["plane-0.1"][0]:.2f}. It costs a non-red deficit a few times the one on the axis, still far under the gate. It qualifies under the adoption rule read as intended, and the anchored-op experiments inherit `{ex.REFERENCE}` anyway, for the reasons the adoption section gives. The plane will most likely stay on the shelf. What this ladder adds is that it is a validated shelf: if a concept turns up that a single axis cannot hold, we know the plane works and we know what it costs.

For those experiments the leftover is a bounded confound rather than a fixed one. On the axis the kept share is near a quarter, with a standard deviation of about {V.sd[ex.REFERENCE]:.2f} across seeds, and it comes from no fixed set of lines: about a third of the lines never survive, a second group survives in most seeds, and none survives in every seed. Any experiment that meets it will have to measure it at its own seeds, which at twenty seeds resolves a shift of about {ex.MEAN_BAND:g}. Two questions stay open: where the useful range of the weight ends, and whether τ trades against it the way the survey in ex-2.1.11 found. Neither is needed before the anchored-op work goes ahead.

## Method

### Fresh seeds

Ex-2.2.11 and ex-2.2.12 trained at model seeds 100–119. The observation this experiment follows up was read off those checkpoints, and a rule written in advance does not make data we have already seen unseen, so no anchored checkpoint here is served from the store: every anchored condition, the reference included, trains at seeds {ex.SEED_OFFSET}–{ex.SEED_OFFSET + ex.SEEDS - 1}. The four conditions that earlier experiments saw appear in the H1 table beside their earlier values, as a replication rather than as data.

We do not hold seeds back to quote the adopted condition from. The search is eight conditions, so the selection bias on the winner is small, and ex-2.2.12 set the pattern this experiment follows: the ladder proposes, and the anchored-op experiment that inherits the recipe confirms it at seeds of its own. Deciding the rule on half the seeds would halve the precision of the upper bound it rests on, for a correction the next experiment pays anyway.

### The plane

As ex-2.2.12 defined it. The home of *red* is the first two axes of the stream together. Alignment is the length of the projection of a state onto that pair, and the anchor term pulls that length toward one on the labelled lines. The anti-subspace term is the square of that length over every live position, and the removal projects the whole plane out.

So the concept holds two coordinates of sixty-four rather than one, and its share of the variance counts both.

### Budget

{ex.N_RUNS} runs at d64-L4, each as long as a run in ex-2.2.11, at about three minutes a run on an L4. Scoring adds the eleven ops under the operators from ex-2.2.11, the per-line pass, and the achromatic edit. That is about three and a half times the sweep in ex-2.2.12, which cost six dollars on Modal.

### What this experiment does not vary

Depth, the pooling temperature τ, and the anti-subspace ratios stay at the values from ex-2.2.11, which means the anti-subspace weight itself climbs with the ladder. Ex-2.2.12 moved each of them one step and resolved nothing on any measurement, so a third factor here would cost runs without a prediction behind it. The weight above {ex.LADDER[-1]:g} is also left alone: the ladder tests the region ex-2.1.11's map calls useful, and finding its edge on this grammar is a different experiment.

The survey in ex-2.1.11 found that the weight and τ trade against each other, so a λ_a × τ factorial would be the natural follow-up if this ladder finds a level worth having.
"""
