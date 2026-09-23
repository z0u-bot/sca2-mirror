# title: Ex 2.2.14: anchoring an operation

# The design constants and the refs come from `experiment.py` beside this script (the script's directory is
# on sys.path while it runs).
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import experiment as ex
from mini.lit import memo, stop
from mini.store import project_store
from mini.vis import figure_html, light_dark, smooth_step_marks, themed


def conditions_html() -> str:
    """The primary, the sweep, and the served control, one row each."""
    head = "<tr><th>condition</th><th>anchored op</th><th class=num>seeds</th><th>role</th></tr>"
    rows = [
        f"<tr><td><code>{ex.PRIMARY}</code></td><td><code>{ex.ANCHORED_OP}</code></td>"
        f"<td class=num>{ex.SEEDS}</td><td>the primary: the op's lines draw at {ex.LABEL_RATE:g}, the red labeller's share; "
        "H1 and H2 are scored on it</td></tr>",
        f"<tr><td><code>{ex.OPWORD_ARM}</code></td><td><code>{ex.ANCHORED_OP}</code></td>"
        f"<td class=num>{ex.SEEDS}</td><td>arm: the pull covers the op word alone; H3 is scored on it</td></tr>",
        f"<tr><td><code>{ex.FULL_ARM}</code></td><td><code>{ex.ANCHORED_OP}</code></td>"
        f"<td class=num>{ex.SEEDS}</td><td>arm: every line of the op draws</td></tr>",
        f"<tr><td><code>{ex.NOISY_ARM}</code></td><td><code>{ex.ANCHORED_OP}</code></td>"
        f"<td class=num>{ex.SEEDS}</td><td>arm: the primary's labeller with a fifth of its labels moved onto other ops' lines</td></tr>",
        f"<tr><td><code>anchor-&lt;op&gt;</code> ×{len(ex.SWEEP)}</td><td>each other op of table A+</td>"
        f"<td class=num>{ex.SWEEP_SEEDS}</td><td>the sweep: the same reads, reported as a description</td></tr>",
        f"<tr><td><code>{ex.CONTROL}</code></td><td>none</td><td class=num>{ex.CONTROL_SEEDS}</td>"
        "<td>ex-2.2.11's un-anchored control, served from the store; the task reference and the alignment baseline</td></tr>",
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


with tempfile.TemporaryDirectory() as _tmp:
    _files = fetch([ex.METRICS_REF, ex.TRAJ_REF], Path(_tmp))
    metrics_loaded = read_json(_files[ex.METRICS_REF])
    traj_loaded = read_json(_files[ex.TRAJ_REF])

OP = ex.ANCHORED_OP
OTHER_OPS = tuple(o for o in ex.OP_NAMES if o != OP)
SITES = {"op": ex.OP_POSITION, "=": ex.EQUALS_POSITION, "answer": ex.ANSWER_POSITION}
ROLES = ["op1", "op", "op2", "=", "answer", "⏎"]
MARGIN_BAR = ex.MARGIN_RATIO * ex.REF_M_LINE
MARGIN_PARTIAL_BAR = ex.MARGIN_PARTIAL * ex.REF_M_LINE
SWEEP_OF = {c: c.removeprefix("anchor-") for c in ex.SWEEP}


@dataclass(frozen=True)
class Results:
    """Every published result the report reads: the eval records (the served control's among them, under
    `ex.CONTROL`) and the training trajectories of the anchored runs.
    """

    metrics: dict
    traj: dict

    def __memo_key__(self) -> str:
        return f"{len(self.metrics['runs'])}:{len(self.traj)}"

    def runs(self, cond: str) -> list[dict]:
        """The eval records of a condition, in seed order."""
        return sorted((r for r in self.metrics["runs"] if r["condition"] == cond), key=lambda r: r["seed"])

    def op_of(self, cond: str) -> str:
        """The op a condition anchors (the anchored op for the control, which is scored on it)."""
        rs = self.runs(cond)
        return rs[0]["op"] or OP if rs else OP

    def eem(self, cond: str, op: str) -> np.ndarray:
        return np.array([r["holdout_eem"][op] for r in self.runs(cond)], float)

    def task_gap(self, cond: str, op: str) -> float:
        """Seed-mean held-out expected exact match minus the control's."""
        return float(self.eem(cond, op).mean() - self.eem(ex.CONTROL, op).mean())

    def worst_gap(self, cond: str) -> tuple[str, float]:
        """The op furthest from the control, by absolute gap, and its signed gap."""
        gaps = {o: self.task_gap(cond, o) for o in ex.OP_NAMES}
        o = max(gaps, key=lambda k: abs(gaps[k]))
        return o, gaps[o]

    def margin(self, cond: str, op: str | None = None) -> np.ndarray:
        """Per seed, the op margin of *op* (by default the op the condition anchors)."""
        op = op or self.op_of(cond)
        return np.array([r["op_margin"][op] for r in self.runs(cond)], float)

    def stat(self, cond: str, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.runs(cond)], float)

    def retention_ok(self, cond: str) -> bool:
        """Every run whose margin reaches the floor at the anneal start ends at the gate share of it."""
        at, ret = self.stat(cond, "m_line_at_anneal"), self.stat(cond, "retention_anneal")
        return bool(np.all((at < ex.RETENTION_FLOOR) | (ret >= ex.RETENTION_GATE)))

    def cos(self, cond: str, op: str) -> np.ndarray:
        """(seeds, L1, T): the mean cosine with e₁ over an op's probe lines."""
        return np.array([r["cos_mean"][op] for r in self.runs(cond)], float)

    def contrast(self, cond: str, op: str | None = None) -> np.ndarray:
        """(seeds, L1, T): the anchored op's mean cosine minus the mean over the other ops' lines."""
        op = op or self.op_of(cond)
        others = np.mean([self.cos(cond, o) for o in ex.OP_NAMES if o != op], axis=0)
        return self.cos(cond, op) - others

    def containment(self, cond: str, op: str) -> np.ndarray:
        """(seeds, L1): the mean cosine at the op position over one op's lines."""
        return self.cos(cond, op)[:, :, ex.OP_POSITION]

    def r2(self, cond: str) -> np.ndarray:
        """(seeds, L1, T): held-out op-identity R² per site."""
        return np.array([r["probe_r2"] for r in self.runs(cond)], float)

    def trajectories(self, cond: str) -> list[dict]:
        return [self.traj[r["label"]]["traj"] for r in self.runs(cond)]


def sd(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


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
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def bold_if(text: str, ok: bool) -> str:
    return f"<b>{text}</b>" if ok else text


def verdict_md(status: str, line: str) -> str:
    """The verdict admonition that closes a hypothesis section; the site hoists its title into the heading."""
    kind = {"pass": "success", "partial": "warning", "miss": "danger", "unresolved": "info"}[status]
    return f"/// admonition | {status.capitalize()}\n    type: {kind}\n{line}\n///"


INKS = {
    ex.CONTROL: ("#6b6b6b", "#b0b0b0"),
    ex.PRIMARY: ("#c0392b", "#ff8a76"),
    ex.OPWORD_ARM: ("#2b6cb0", "#7fb3ff"),
    ex.FULL_ARM: ("#2e8b57", "#7fd8a4"),
    ex.NOISY_ARM: ("#7b3fa0", "#cfa3ff"),
    "sweep": ("#a08a2e", "#e6d27a"),
}
MARKERS = {ex.CONTROL: "s", ex.PRIMARY: "o", ex.OPWORD_ARM: "^", ex.FULL_ARM: "D", ex.NOISY_ARM: "v", "sweep": "P"}


def ink(cond: str) -> str:
    return light_dark(*INKS.get(cond, INKS["sweep"]))


def dots(
    ax: Axes, x: float, v: np.ndarray, cond: str, *, rng, ms: float = 5.0, width: float = 0.08, label=None
) -> None:
    """One column of per-seed dots, a thin bar over the seed range, and the seed mean in the condition's marker."""
    v = np.asarray(v, float)
    color, m = ink(cond), MARKERS.get(cond, "P")
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + rng.uniform(-width, width, len(v)), v, "o", ms=2.2, color=color, alpha=0.45, zorder=3, mew=0)
    ax.plot(x, v.mean(), m, ms=ms, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6, label=label)


def fig_legend(fig: plt.Figure, ax: Axes, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7, **kwargs)


def gate_line(ax: Axes, y: float, *, partial: float | None = None, fail: str | None = None) -> None:
    """A dashed gate line, a dotted partial level beside it, and the failing side hatched."""
    ax.axhline(y, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=2)
    if partial is not None:
        ax.axhline(partial, color=light_dark("#333", "#ddd"), lw=0.7, ls=":", zorder=2)
    if fail is not None:
        lo, hi = ax.get_ylim()
        span = (lo, y) if fail == "below" else (y, hi)
        ax.axhspan(*span, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="//", lw=0, zorder=0, alpha=0.1)
        ax.set_ylim(lo, hi)


def band_gate(ax: Axes, half: float, partial: float) -> None:
    """A two-sided gate around zero: dashed at ±half, dotted at ±partial, hatched outside the partial band."""
    for s in (-1, 1):
        ax.axhline(s * half, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=2)
        ax.axhline(s * partial, color=light_dark("#333", "#ddd"), lw=0.7, ls=":", zorder=2)
    lo, hi = ax.get_ylim()
    lo, hi = min(lo, -partial * 1.6), max(hi, partial * 1.6)
    for span in ((lo, -half), (half, hi)):
        ax.axhspan(*span, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="//", lw=0, zorder=0, alpha=0.1)
    ax.set_ylim(lo, hi)


def fmt_ratio(num: float, den: float) -> str:
    return f"{num / den:.2f}" if abs(den) > 1e-9 else "—"


# --- Verdicts -----------------------------------------------------------------------------------


def h1_status(res: Results, cond: str = ex.PRIMARY) -> str:
    gaps = np.array([abs(res.task_gap(cond, o)) for o in ex.OP_NAMES])
    return "pass" if np.all(gaps <= ex.TASK_GATE) else "partial" if np.all(gaps <= ex.TASK_PARTIAL) else "miss"


def margin_status(res: Results, cond: str = ex.PRIMARY) -> str:
    m = float(res.margin(cond).mean())
    return "pass" if m >= MARGIN_BAR else "partial" if m >= MARGIN_PARTIAL_BAR else "miss"


def h3_sites(res: Results) -> dict[str, dict]:
    """At the final slice: the arm's and the control's seed-mean contrast per site, and whether each use site
    clears the floor above the control.
    """
    arm = res.contrast(ex.OPWORD_ARM)[:, -1]
    ctl = res.contrast(ex.CONTROL, OP)[:, -1]
    out = {}
    for name, pos in SITES.items():
        a, c = float(arm[:, pos].mean()), float(ctl[:, pos].mean())
        out[name] = {"arm": a, "control": c, "excess": a - c, "ok": a - c >= ex.USE_CONTRAST_FLOOR}
    return out


def h3_status(res: Results) -> str:
    s = h3_sites(res)
    return "pass" if s["="]["ok"] and s["answer"]["ok"] else "miss"


def qualifies(res: Results, cond: str) -> dict:
    """The rule's three gates for one condition: every op inside the task gate, the margin at the bar, and
    every run clearing the retention rule.
    """
    task = all(abs(res.task_gap(cond, o)) <= ex.TASK_GATE for o in ex.OP_NAMES)
    margin = float(res.margin(cond).mean())
    retention = res.retention_ok(cond)
    return {
        "task": task,
        "margin": margin,
        "margin_ok": margin >= MARGIN_BAR,
        "retention": retention,
        "ok": task and margin >= MARGIN_BAR and retention,
    }


def adoption(res: Results) -> dict:
    """The rule for the follow-up, applied: the primary if it passes H1 and H2 in full, otherwise the
    qualifying sweep op with the largest seed-mean margin, otherwise none.
    """
    primary_ok = h1_status(res) == "pass" and margin_status(res) == "pass" and res.retention_ok(ex.PRIMARY)
    sweep = {c: qualifies(res, c) for c in ex.SWEEP if res.runs(c)}
    passing = [c for c, q in sweep.items() if q["ok"]]
    if primary_ok:
        chosen = OP
    elif passing:
        chosen = SWEEP_OF[max(passing, key=lambda c: sweep[c]["margin"])]
    else:
        chosen = None
    return {"primary_ok": primary_ok, "sweep": sweep, "passing": [SWEEP_OF[c] for c in passing], "chosen": chosen}


# --- H1: the task ----------------------------------------------------------------------------------


def h1_figure(res: Results) -> str:
    gaps = {o: res.eem(ex.PRIMARY, o) - res.eem(ex.CONTROL, o).mean() for o in ex.OP_NAMES}
    worst_op, worst = res.worst_gap(ex.PRIMARY)
    alt = f"""
        A dot chart with the eleven ops along the bottom and the gap in held-out expected exact match from
        the control up the side, centred on zero. Each op has a column of {ex.SEEDS} seed dots and the seed
        mean; `{OP}` is marked. Dashed lines at ±{ex.TASK_GATE:g} bound the gate and dotted lines at
        ±{ex.TASK_PARTIAL:g} the partial level, hatched outside. The largest seed-mean gap is on {worst_op},
        at {worst:+.3f}.
    """
    return h1_draw(gaps, alt)


@memo
def h1_draw(gaps: dict, alt_text: str) -> str:
    @themed(
        name="h1-task-gap",
        alt_text=alt_text,
        caption=f"""
            **The task gap per op.** Each column is one op: the primary's {ex.SEEDS} seeds as faint dots, a thin
            bar over their range, and the seed mean as the large mark, each as a difference from the control's
            seed mean on the same op. The anchored op, `{OP}`, is drawn in the primary's ink and the others in
            grey. Dashed lines mark the gate at ±{ex.TASK_GATE:g}, dotted lines the partial level at
            ±{ex.TASK_PARTIAL:g}, hatched outside.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.4, 3.0), layout="constrained")
        rng = np.random.default_rng(7)
        for x, o in enumerate(ex.OP_NAMES):
            dots(ax, x, gaps[o], ex.PRIMARY if o == OP else ex.CONTROL, rng=rng)
        ax.axhline(0, color=light_dark("#999", "#666"), lw=0.6, zorder=1)
        ax.set_xticks(range(len(ex.OP_NAMES)), [ex.short(o) for o in ex.OP_NAMES], rotation=30, ha="right", fontsize=8)
        ax.get_xticklabels()[ex.OP_NAMES.index(OP)].set_fontweight("bold")
        ax.set_ylabel("EEM gap from control")
        band_gate(ax, ex.TASK_GATE, ex.TASK_PARTIAL)
        return fig

    return _plot()


def h1_table(res: Results) -> str:
    head = ["op", "primary EEM ↑", "control EEM", "gap", f"|gap| ≤ {ex.TASK_GATE:g}"]
    rows = []
    for o in ex.OP_NAMES:
        p, c = res.eem(ex.PRIMARY, o).mean(), res.eem(ex.CONTROL, o).mean()
        ok = abs(p - c) <= ex.TASK_GATE
        rows.append([f"`{o}`", f"{p:.3f}", f"{c:.3f}", f"{p - c:+.3f}", bold_if("yes" if ok else "no", ok)])
    return table_html(
        head,
        rows,
        f"Held-out expected exact match per op: the primary's seed mean over {len(res.runs(ex.PRIMARY))} seeds and "
        f"the control's over {len(res.runs(ex.CONTROL))}, their difference, and whether it is inside the gate. "
        f"H1: {h1_status(res)}.",
        ref_rows=frozenset({ex.OP_NAMES.index(OP)}),
    )


# --- H2: the margin and its retention ----------------------------------------------------------------


def h2_figure(res: Results) -> str:
    trajs = [{k: list(t[k]) for k in ("epoch", "m_line")} for t in res.trajectories(ex.PRIMARY)]
    anneal = res.stat(ex.PRIMARY, "anneal_epoch").tolist()
    end = res.margin(ex.PRIMARY)
    ctl = res.margin(ex.CONTROL, OP)
    alt = f"""
        Two panels. Left, a line chart of the op margin over the {ex.EPOCHS} training epochs, one line per
        primary seed, with a vertical band marking where the anchor weight anneals. Right, a dot column of the
        end-of-training op margin for the {len(end)} primary seeds, mean {end.mean():.3f}, beside the control's
        column at {ctl.mean():.3f}. A dashed line marks the bar at {MARGIN_BAR:.3f} and a dotted line the
        partial level at {MARGIN_PARTIAL_BAR:.3f}, hatched below.
    """
    return h2_draw(trajs, anneal, end, ctl, alt)


@memo
def h2_draw(trajs: list, anneal: list, end: np.ndarray, ctl: np.ndarray, alt_text: str) -> str:
    @themed(
        name="h2-op-margin",
        alt_text=alt_text,
        caption=f"""
            **The op margin over training and at the end.** Left: the margin of `{OP}` on the trajectory
            probe lines, recorded every {ex.TRAJ_STRIDE} epochs, one line per primary seed; the shaded band
            runs from the earliest to the latest anneal start. Right: the end-of-training margin on the full
            probe sets, per seed with the seed mean as the large mark, beside the control scored on the same
            op. The dashed line is the bar, {ex.MARGIN_RATIO:.0%} of *red*'s {ex.REF_M_LINE:g}; the dotted
            line is the partial level at {ex.MARGIN_PARTIAL:.0%}; hatched below.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained", sharey=True, width_ratios=(3, 1))
        a.axvspan(
            min(anneal),
            max(anneal) if max(anneal) > min(anneal) else min(anneal) + 0.5,
            color=light_dark("#000", "#fff"),
            alpha=0.08,
            lw=0,
            zorder=0,
        )
        for t in trajs:
            a.plot(t["epoch"], t["m_line"], color=ink(ex.PRIMARY), lw=1.1, alpha=0.8, zorder=3)
        a.set_xlabel("epoch")
        a.set_ylabel(f"op margin ({ex.short(OP)})")
        a.text(min(anneal), 0.02, " anneal", transform=a.get_xaxis_transform(), fontsize=7, va="bottom")
        rng = np.random.default_rng(3)
        dots(b, 0, end, ex.PRIMARY, rng=rng)
        dots(b, 1, ctl, ex.CONTROL, rng=rng)
        b.set_xticks([0, 1], ["primary", "control"], fontsize=8)
        b.set_xlim(-0.6, 1.6)
        lo = min(-0.02, float(np.min(ctl)) - 0.02)
        hi = max(MARGIN_BAR * 1.15, max(float(np.max(end)), max(max(t["m_line"]) for t in trajs)) + 0.05)
        a.set_ylim(lo, hi)
        for ax in (a, b):
            gate_line(ax, MARGIN_BAR, partial=MARGIN_PARTIAL_BAR, fail="below")
        return fig

    return _plot()


def h2_table(res: Results) -> str:
    head = ["seed", "anneal start (epoch)", "margin at anneal", "margin at end", "retention ↑", "end margin (eval) ↑"]
    rows = []
    at, fin, ret = (res.stat(ex.PRIMARY, k) for k in ("m_line_at_anneal", "m_line_final", "retention_anneal"))
    ep, end = res.stat(ex.PRIMARY, "anneal_epoch"), res.margin(ex.PRIMARY)
    for i, r in enumerate(res.runs(ex.PRIMARY)):
        gated = at[i] >= ex.RETENTION_FLOOR
        ok = (not gated) or ret[i] >= ex.RETENTION_GATE
        rows.append(
            [
                str(r["seed"]),
                f"{ep[i]:g}",
                f"{at[i]:.3f}",
                f"{fin[i]:.3f}",
                bold_if(f"{ret[i]:.3f}", ok) + ("" if gated else " (under floor)"),
                bold_if(f"{end[i]:.3f}", end[i] >= MARGIN_BAR),
            ]
        )
    rows.append(
        [
            "mean",
            "",
            f"{at.mean():.3f}",
            f"{fin.mean():.3f}",
            f"{ret.mean():.3f}",
            bold_if(f"{end.mean():.3f}", end.mean() >= MARGIN_BAR),
        ]
    )
    return table_html(
        head,
        rows,
        f"The op margin per primary seed: on the trajectory at the start of the anneal and at the end, their "
        f"ratio (retention, gated at {ex.RETENTION_GATE:g} for runs at or above {ex.RETENTION_FLOOR:g} at the "
        f"anneal start), and the end-of-training margin on the full probe sets, against the bar of "
        f"{MARGIN_BAR:.3f} ({ex.MARGIN_RATIO:.0%} of {ex.REF_M_LINE:g}); bold passes. Seed-mean margin over "
        f"*red*'s: {end.mean() / ex.REF_M_LINE:.2f}. Margin: {margin_status(res)}; retention: "
        f"{'pass' if res.retention_ok(ex.PRIMARY) else 'miss'}.",
        ref_rows=frozenset({len(rows) - 1}),
    )


# --- H3: the use sites ----------------------------------------------------------------------------


def h3_figure(res: Results) -> str:
    prof = {c: res.contrast(c, OP).mean(0) for c in (ex.OPWORD_ARM, ex.PRIMARY)}
    s = h3_sites(res)
    alt = f"""
        Two stacked smooth-step charts, the op-word arm above and the primary below, with the six positions of
        a line along the bottom and the contrast in mean cosine with e₁ up the side. Each has one series per
        slice from the embedding to the final slice, in shades from light to dark. The op, `=` and answer
        positions are marked. On the op-word arm at the final slice the contrast is {s["op"]["arm"]:.3f} at the
        op position, {s["="]["arm"]:.3f} at `=`, and {s["answer"]["arm"]:.3f} at the answer, against
        {s["="]["control"]:.3f} and {s["answer"]["control"]:.3f} for the control at the two use sites.
    """
    return h3_draw(prof, alt)


def slice_shades(n: int) -> list[str]:
    """Ordered shades for the slices, light (embedding) to dark (final), picked per theme."""
    light = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, n))
    dark = plt.get_cmap("Blues")(np.linspace(0.6, 0.15, n))
    return [
        light_dark(matplotlib.colors.to_hex(a), matplotlib.colors.to_hex(b)) for a, b in zip(light, dark, strict=True)
    ]


def site_marks(ax: Axes) -> None:
    for p in SITES.values():
        ax.axvline(p, color=light_dark("#999", "#555"), lw=0.6, ls=":", zorder=0)


@memo
def h3_draw(prof: dict, alt_text: str) -> str:
    @themed(
        name="h3-contrast",
        alt_text=alt_text,
        caption=f"""
            **Contrast over the line, per slice.** The seed-mean contrast: mean cosine with e₁ on the `{OP}`
            probe lines minus the mean over the other ten ops' lines, at each of the six positions. One series
            per residual-stream slice, light for the embedding to dark for the final slice. Top, the op-word
            arm, whose pull covers the op word alone; bottom, the primary, whose pull covers the whole line.
            Dotted verticals mark the op, `=` and answer positions.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.8), layout="constrained", sharex=True, sharey=True)
        x = np.arange(len(ROLES))
        for ax, (c, p) in zip(axes, prof.items(), strict=True):
            shades = slice_shades(p.shape[0])
            for layer in range(p.shape[0]):
                smooth_step_marks(
                    ax,
                    x,
                    p[layer],
                    ramp=0.5,
                    color=shades[layer],
                    lw=1.5,
                    zorder=3,
                    label="embedding" if layer == 0 else f"slice {layer}",
                )
            site_marks(ax)
            ax.axhline(0, color=light_dark("#999", "#666"), lw=0.6, zorder=1)
            ax.set_title(c, fontsize=9)
            ax.set_ylabel("contrast")
        axes[-1].set_xticks(x, ROLES)
        fig_legend(fig, axes[0])
        return fig

    return _plot()


def h3_table(res: Results) -> str:
    head = ["seed", "op position", "`=`", "answer", "`=` / op", "answer / op"]
    arm = res.contrast(ex.OPWORD_ARM)[:, -1]
    rows = []
    for i, r in enumerate(res.runs(ex.OPWORD_ARM)):
        o, e, a = (arm[i, p] for p in SITES.values())
        rows.append([str(r["seed"]), f"{o:.3f}", f"{e:.3f}", f"{a:.3f}", fmt_ratio(e, o), fmt_ratio(a, o)])
    s = h3_sites(res)
    m = [s[k]["arm"] for k in SITES]
    rows.append(
        [
            "mean",
            *(bold_if(f"{v:.3f}", k != "op" and bool(s[k]["ok"])) for k, v in zip(SITES, m, strict=True)),
            fmt_ratio(m[1], m[0]),
            fmt_ratio(m[2], m[0]),
        ]
    )
    ctl = [s[k]["control"] for k in SITES]
    rows.append(["control", *(f"{v:.3f}" for v in ctl), "", ""])
    prim = res.contrast(ex.PRIMARY)[:, -1].mean(0)
    pm = [float(prim[p]) for p in SITES.values()]
    rows.append(
        [f"`{ex.PRIMARY}` (check)", *(f"{v:.3f}" for v in pm), fmt_ratio(pm[1], pm[0]), fmt_ratio(pm[2], pm[0])]
    )
    return table_html(
        head,
        rows,
        f"Contrast at the final slice on the op-word arm, per seed, at the op position and the two use sites, "
        f"with the ratio of each use site to the op position. The mean row is bold where it is at least "
        f"{ex.USE_CONTRAST_FLOOR:g} above the control's seed mean (the row below it, scored on `{OP}`). The last "
        f"row is the primary's seed mean, a manipulation check. H3: {h3_status(res)}.",
        ref_rows=frozenset({len(rows) - 3}),
    )


# --- Containment -------------------------------------------------------------------------------------


def containment_figure(res: Results) -> str:
    per_op = {o: (res.containment(ex.PRIMARY, o).mean(1), res.containment(ex.CONTROL, o).mean(1)) for o in OTHER_OPS}
    per_slice = {c: np.mean([res.containment(c, o) for o in OTHER_OPS], axis=0) for c in (ex.PRIMARY, ex.CONTROL)}
    mean_p = float(np.mean([v[0].mean() for v in per_op.values()]))
    mean_c = float(np.mean([v[1].mean() for v in per_op.values()]))
    alt = f"""
        Two panels. Left, a dot chart with the ten ops that were not anchored along the bottom and the mean
        cosine with e₁ at the op position up the side, averaged over slices: per op, the primary's seeds and
        the control's side by side. A dashed line marks *red*'s ᾱ at op1, {ex.CONTAINMENT_REF:g}. Over the ten
        ops the primary averages {mean_p:.3f} and the control {mean_c:.3f}. Right, the same quantity averaged
        over the ten ops, per slice, for the primary and the control.
    """
    return containment_draw(per_op, per_slice, alt)


@memo
def containment_draw(per_op: dict, per_slice: dict, alt_text: str) -> str:
    @themed(
        name="containment",
        alt_text=alt_text,
        caption=f"""
            **Containment at the op position.** Left: per op other than `{OP}`, the mean cosine with e₁ at the
            op position over its probe lines, averaged over slices; the primary's seeds and the control's side
            by side. The dashed line is *red*'s ᾱ at op1 under the same recipe, {ex.CONTAINMENT_REF:g}, for
            scale; there is no gate. Right: the mean over the ten ops per slice, seed mean with the seed range
            shaded.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained", width_ratios=(3, 1.3), sharey=True)
        rng = np.random.default_rng(11)
        for x, (p, c) in enumerate(per_op.values()):
            dots(a, x - 0.17, p, ex.PRIMARY, rng=rng, width=0.04, label="primary" if x == 0 else None)
            dots(a, x + 0.17, c, ex.CONTROL, rng=rng, width=0.04, label="control" if x == 0 else None)
        a.set_xticks(range(len(per_op)), [ex.short(o) for o in per_op], rotation=30, ha="right", fontsize=8)
        a.set_ylabel("mean cos with e₁ at the op position")
        for ax in (a, b):
            ax.axhline(ex.CONTAINMENT_REF, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=2)
            ax.axhline(0, color=light_dark("#999", "#666"), lw=0.6, zorder=1)
        layers = np.arange(next(iter(per_slice.values())).shape[1])
        for c, v in per_slice.items():
            b.fill_between(layers, v.min(0), v.max(0), color=ink(c), alpha=0.15, lw=0, zorder=1)
            b.plot(layers, v.mean(0), "-", marker=MARKERS[c], ms=4, color=ink(c), lw=1.3, zorder=3)
        b.set_xticks(layers, ["emb", *map(str, layers[1:])])
        b.set_xlabel("slice")
        fig_legend(fig, a)
        return fig

    return _plot()


def containment_table(res: Results) -> str:
    head = ["op", "primary", "control", "difference"]
    rows = []
    for o in OTHER_OPS:
        p, c = res.containment(ex.PRIMARY, o).mean(), res.containment(ex.CONTROL, o).mean()
        rows.append([f"`{o}`", f"{p:.3f}", f"{c:.3f}", f"{p - c:+.3f}"])
    p = float(np.mean([res.containment(ex.PRIMARY, o).mean() for o in OTHER_OPS]))
    c = float(np.mean([res.containment(ex.CONTROL, o).mean() for o in OTHER_OPS]))
    rows.append(["mean of ten", f"{p:.3f}", f"{c:.3f}", f"{p - c:+.3f}"])
    return table_html(
        head,
        rows,
        f"Containment per op: the mean cosine with e₁ at the op position, averaged over slices and seeds, on "
        f"the primary and the control. *Red*'s ᾱ at op1 under the same recipe was {ex.CONTAINMENT_REF:g}.",
        ref_rows=frozenset({len(rows) - 1}),
    )


# --- The rule --------------------------------------------------------------------------------------


def rule_table(res: Results) -> str:
    a = adoption(res)
    head = ["condition", "op", "seeds", "op margin ↑", "worst task gap", "retention", "qualifies"]
    q = qualifies(res, ex.PRIMARY)
    wo, wg = res.worst_gap(ex.PRIMARY)
    rows = [
        [
            f"`{ex.PRIMARY}`",
            f"`{OP}`",
            str(len(res.runs(ex.PRIMARY))),
            bold_if(f"{q['margin']:.3f}", q["margin_ok"]),
            bold_if(f"{wg:+.3f} ({ex.short(wo)})", q["task"]),
            bold_if("yes" if q["retention"] else "no", q["retention"]),
            bold_if(f"H1 {h1_status(res)}, H2 margin {margin_status(res)}", a["primary_ok"]),
        ]
    ]
    for c, q in sorted(a["sweep"].items(), key=lambda kv: ex.OP_NAMES.index(res.op_of(kv[0]))):
        wo, wg = res.worst_gap(c)
        rows.append(
            [
                f"`{c}`",
                f"`{SWEEP_OF[c]}`",
                str(len(res.runs(c))),
                bold_if(f"{q['margin']:.3f}", q["margin_ok"]),
                bold_if(f"{wg:+.3f} ({ex.short(wo)})", q["task"]),
                bold_if("yes" if q["retention"] else "no", q["retention"]),
                bold_if("yes" if q["ok"] else "no", q["ok"]),
            ]
        )
    chosen = f"`{a['chosen']}`" if a["chosen"] else "none"
    return table_html(
        head,
        rows,
        f"The rule applied. The primary qualifies on H1 and H2 in full (a partial counts as a miss); a sweep op "
        f"qualifies when every op is within {ex.TASK_GATE:g} of the control, its seed-mean margin is at or above "
        f"{MARGIN_BAR:.3f}, and every run clears the retention rule. Bold passes. The worst task gap is the op "
        f"furthest from the control. The rule picks: {chosen}.",
        ref_rows=frozenset({0}),
    )


# --- Exploratory -----------------------------------------------------------------------------------


def sweep_figure(res: Results) -> str:
    conds = [ex.PRIMARY, *[c for c in ex.SWEEP if res.runs(c)]]
    margin = {c: res.margin(c) for c in conds}
    gap = {
        c: np.array(
            [
                np.max(np.abs([r["holdout_eem"][o] - res.eem(ex.CONTROL, o).mean() for o in ex.OP_NAMES]))
                for r in res.runs(c)
            ]
        )
        for c in conds
    }
    order = sorted(conds, key=lambda c: ex.OP_NAMES.index(res.op_of(c)))
    names = {c: ex.short(res.op_of(c)) for c in conds}
    top = max(conds, key=lambda c: margin[c].mean())
    alt = f"""
        Two panels sharing the ops along the bottom, in the order of table A+. Top, the op margin per
        seed with the seed mean, the anchored op of the primary at {ex.SEEDS} seeds and each sweep op at
        {ex.SWEEP_SEEDS}; dashed and dotted lines mark the H2 bar and its partial level. The highest is
        {names[top]} at {margin[top].mean():.3f}. Bottom, the worst absolute task gap per seed, with the gate
        at {ex.TASK_GATE:g}.
    """
    return sweep_draw(order, margin, gap, names, alt)


@memo
def sweep_draw(order: list, margin: dict, gap: dict, names: dict, alt_text: str) -> str:
    @themed(
        name="sweep",
        alt_text=alt_text,
        caption="""
            **The sweep.** Each column is one op anchored on e₁, in the order of table A+; the primary's
            op is in its ink, the sweep ops in gold, and the order-sensitive ops' labels are italic. Top: the op
            margin, with the H2 bar (dashed) and partial level (dotted), hatched below. Bottom: per run, the
            largest absolute gap in held-out expected exact match from the control over the eleven ops, with
            the task gate dashed and the partial level dotted, hatched above.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (a, b) = plt.subplots(2, 1, figsize=(8.4, 4.6), layout="constrained", sharex=True)
        rng = np.random.default_rng(5)
        for x, c in enumerate(order):
            cond = ex.PRIMARY if c == ex.PRIMARY else "sweep"
            dots(a, x, margin[c], cond, rng=rng)
            dots(b, x, gap[c], cond, rng=rng)
        a.set_ylabel("op margin")
        a.set_ylim(min(-0.02, min(float(v.min()) for v in margin.values()) - 0.02), None)
        gate_line(a, MARGIN_BAR, partial=MARGIN_PARTIAL_BAR, fail="below")
        b.set_ylabel("worst |task gap|")
        b.set_ylim(0, max(ex.TASK_PARTIAL * 1.3, max(float(v.max()) for v in gap.values()) * 1.1))
        gate_line(b, ex.TASK_GATE, partial=ex.TASK_PARTIAL, fail="above")
        b.set_xticks(range(len(order)), [names[c] for c in order], rotation=30, ha="right", fontsize=8)
        for t, c in zip(b.get_xticklabels(), order, strict=True):
            if any(ex.short(o) == names[c] for o in ex.ORDER_SENSITIVE):
                t.set_fontstyle("italic")
        return fig

    return _plot()


def sweep_table(res: Results) -> str:
    head = ["op", "seeds", "op margin ↑", "worst task gap", "retention (min)", "`=` contrast", "answer contrast"]
    conds = sorted([ex.PRIMARY, *[c for c in ex.SWEEP if res.runs(c)]], key=lambda c: ex.OP_NAMES.index(res.op_of(c)))
    rows = []
    for c in conds:
        con = res.contrast(c)[:, -1].mean(0)
        wo, wg = res.worst_gap(c)
        op = res.op_of(c)
        tag = " (order-sensitive)" if op in ex.ORDER_SENSITIVE else ""
        rows.append(
            [
                f"`{op}`{tag}",
                str(len(res.runs(c))),
                f"{res.margin(c).mean():.3f}",
                f"{wg:+.3f} ({ex.short(wo)})",
                f"{res.stat(c, 'retention_anneal').min():.2f}",
                f"{con[ex.EQUALS_POSITION]:.3f}",
                f"{con[ex.ANSWER_POSITION]:.3f}",
            ]
        )
    return table_html(
        head,
        rows,
        "Every op anchored the same way, in the order of table A+: the primary's op at its seeds, the rest "
        "at the sweep's. The worst task gap is the seed-mean gap on the op furthest from the control; retention "
        "is the lowest over the op's runs; the contrasts are at the final slice, seed mean (the whole-line pull "
        "covers `=` on every one of these, so they are manipulation checks). Post hoc description; no gate.",
        ref_rows=frozenset({conds.index(ex.PRIMARY)}),
    )


def scan_figure(res: Results) -> str:
    r2 = {c: res.r2(c).mean(0) for c in (ex.PRIMARY, ex.OPWORD_ARM, ex.CONTROL)}
    d = r2[ex.PRIMARY] - r2[ex.CONTROL]
    i = np.unravel_index(np.argmax(np.abs(d)), d.shape)
    alt = f"""
        Six small panels, one per position of the line, each with the slice along the bottom and the held-out
        op-identity R² up the side, for the primary, the op-word arm and the control. The largest seed-mean
        difference between the primary and the control is {d[i]:+.3f}, at slice {i[0]} of position
        {ROLES[i[1]]}.
    """
    return scan_draw(r2, alt)


@memo
def scan_draw(r2: dict, alt_text: str) -> str:
    @themed(
        name="scan",
        alt_text=alt_text,
        caption="""
            **Op-identity R² per site.** One panel per position; within each, the held-out R² of a ridge probe
            from the state to the one-hot op word, per slice (the embedding at the left), seed mean. The
            primary and the op-word arm against the control. Post hoc description; no gate.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(ROLES), figsize=(8.4, 2.6), layout="constrained", sharey=True)
        layers = np.arange(next(iter(r2.values())).shape[0])
        for p, ax in enumerate(axes):
            for c, v in r2.items():
                ax.plot(layers, v[:, p], "-", marker=MARKERS[c], ms=3.5, color=ink(c), lw=1.2, label=c)
            ax.set_title(ROLES[p], fontsize=9)
            ax.set_xticks(layers, ["e", *map(str, layers[1:])], fontsize=7)
            ax.set_ylim(-0.05, 1.05)
        axes[0].set_ylabel("held-out R²")
        fig_legend(fig, axes[0])
        return fig

    return _plot()


def scan_table(res: Results) -> str:
    head = ["site", *[f"`{c}`" for c in (ex.PRIMARY, ex.OPWORD_ARM, ex.CONTROL)], "primary − control"]
    r2 = {c: res.r2(c).mean(0) for c in (ex.PRIMARY, ex.OPWORD_ARM, ex.CONTROL)}
    last = next(iter(r2.values())).shape[0] - 1
    rows = []
    for p, role in enumerate(ROLES):
        for layer in (0, last):
            v = [r2[c][layer, p] for c in r2]
            rows.append(
                [
                    f"{role}, {'embedding' if layer == 0 else f'slice {layer}'}",
                    *(f"{x:.3f}" for x in v),
                    f"{v[0] - v[2]:+.3f}",
                ]
            )
    d = r2[ex.PRIMARY] - r2[ex.CONTROL]
    return table_html(
        head,
        rows,
        f"Op-identity R² at the embedding and the final slice of each position, seed mean. Over every site, the "
        f"primary minus the control spans {d.min():+.3f} to {d.max():+.3f} (mean |difference| "
        f"{np.abs(d).mean():.3f}). Post hoc description; no gate.",
    )


def arms_figure(res: Results) -> str:
    conds = (ex.PRIMARY, ex.FULL_ARM, ex.NOISY_ARM)
    trajs = {}
    for c in conds:
        ts = res.trajectories(c)
        trajs[c] = {"epoch": list(ts[0]["epoch"]), "m": np.array([t["m_line"] for t in ts], float)}
    gaps = {
        c: np.array(
            [max(abs(r["holdout_eem"][o] - res.eem(ex.CONTROL, o).mean()) for o in ex.OP_NAMES) for r in res.runs(c)]
        )
        for c in conds
    }
    end = {c: float(res.margin(c).mean()) for c in conds}
    alt = f"""
        Two panels. Left, the op margin over training for the primary, the every-line arm and the noisy arm,
        seed mean with the seed range shaded; they end at {end[ex.PRIMARY]:.3f}, {end[ex.FULL_ARM]:.3f} and
        {end[ex.NOISY_ARM]:.3f} on the full probe sets. Right, a dot column per condition of the worst absolute
        task gap per run, with the gate.
    """
    return arms_draw(trajs, gaps, alt)


@memo
def arms_draw(trajs: dict, gaps: dict, alt_text: str) -> str:
    @themed(
        name="arms",
        alt_text=alt_text,
        caption=f"""
            **The arms.** Left: the op margin of `{OP}` over training, seed mean with the seed range shaded, for
            the primary, the every-line arm (every line of the op labelled) and the noisy arm (a fifth of the
            primary's labels moved onto other ops' lines); dashed and dotted lines mark the H2 bar and partial
            level. Right: per run, the largest absolute gap in held-out expected exact match from the control
            over the eleven ops, with the task gate. Post hoc description; no gate.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained", width_ratios=(3, 1.3))
        for c, t in trajs.items():
            a.fill_between(t["epoch"], t["m"].min(0), t["m"].max(0), color=ink(c), alpha=0.15, lw=0, zorder=1)
            a.plot(t["epoch"], t["m"].mean(0), "-", color=ink(c), lw=1.4, zorder=3, label=c)
        a.set_xlabel("epoch")
        a.set_ylabel("op margin")
        a.set_ylim(min(-0.02, a.get_ylim()[0]), max(MARGIN_BAR * 1.15, a.get_ylim()[1]))
        gate_line(a, MARGIN_BAR, partial=MARGIN_PARTIAL_BAR, fail="below")
        rng = np.random.default_rng(9)
        for x, (c, g) in enumerate(gaps.items()):
            dots(b, x, g, c, rng=rng)
        b.set_xticks(range(len(gaps)), ["primary", "every line", "noisy"], fontsize=8)
        b.set_ylabel("worst |task gap|")
        b.set_ylim(0, max(ex.TASK_PARTIAL * 1.3, max(float(g.max()) for g in gaps.values()) * 1.1))
        gate_line(b, ex.TASK_GATE, partial=ex.TASK_PARTIAL, fail="above")
        fig_legend(fig, a)
        return fig

    return _plot()


def arms_table(res: Results) -> str:
    head = [
        "condition",
        "label share (lines)",
        "op margin ↑",
        "vs primary",
        "worst task gap",
        "retention (min)",
        "`=` contrast",
        "answer contrast",
    ]
    base = res.margin(ex.PRIMARY).mean()
    rows = []
    for c in (ex.PRIMARY, ex.FULL_ARM, ex.NOISY_ARM, ex.OPWORD_ARM):
        m = res.margin(c).mean()
        con = res.contrast(c)[:, -1].mean(0)
        share = np.mean([r["label_share"]["lines"] for r in res.runs(c)])
        wo, wg = res.worst_gap(c)
        rows.append(
            [
                f"`{c}`",
                f"{share:.4f}",
                f"{m:.3f}",
                f"{m / base:.2f}",
                f"{wg:+.3f} ({ex.short(wo)})",
                f"{res.stat(c, 'retention_anneal').min():.2f}",
                f"{con[ex.EQUALS_POSITION]:.3f}",
                f"{con[ex.ANSWER_POSITION]:.3f}",
            ]
        )
    return table_html(
        head,
        rows,
        f"The arms on the primary's statistics, seed means over {ex.SEEDS} seeds. Label share is the fraction of "
        f"training lines carrying a label in the first epoch (*red*'s was {ex.RED_LABEL_SHARE:g}). The contrasts "
        "are at the final slice. The op-word arm is H3's and is listed for completeness. Post hoc description.",
        ref_rows=frozenset({0}),
    )


def map_figure(res: Results) -> str:
    anchored = res.cos(ex.PRIMARY, OP).mean(0)
    others = np.mean([res.cos(ex.PRIMARY, o) for o in OTHER_OPS], axis=0).mean(0)
    i = np.unravel_index(np.argmax(anchored), anchored.shape)
    alt = f"""
        Two stacked smooth-step charts sharing a scale, the six positions of a line along the bottom and the
        mean cosine with e₁ up the side, one series per slice from light (embedding) to dark (final). Top, the
        primary's `{OP}` lines; bottom, the other ten ops' lines. The highest value on the `{OP}` lines is
        {anchored[i]:.3f}, at {"the embedding" if i[0] == 0 else f"slice {i[0]}"} of position {ROLES[i[1]]}.
    """
    return map_draw({f"{OP} lines": anchored, "other ops' lines": others}, alt)


@memo
def map_draw(prof: dict, alt_text: str) -> str:
    @themed(
        name="alignment-map",
        alt_text=alt_text,
        caption=f"""
            **Where the anchor put the op.** The primary's mean cosine with e₁ at each position, one series per
            slice, light for the embedding to dark for the final slice, seed mean. Top, the `{OP}` probe lines;
            bottom, the other ten ops' probe lines. Dotted verticals mark the op, `=` and answer positions.
            Post hoc description.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.8), layout="constrained", sharex=True, sharey=True)
        x = np.arange(len(ROLES))
        for ax, (name, p) in zip(axes, prof.items(), strict=True):
            shades = slice_shades(p.shape[0])
            for layer in range(p.shape[0]):
                smooth_step_marks(
                    ax,
                    x,
                    p[layer],
                    ramp=0.5,
                    color=shades[layer],
                    lw=1.5,
                    zorder=3,
                    label="embedding" if layer == 0 else f"slice {layer}",
                )
            site_marks(ax)
            ax.axhline(0, color=light_dark("#999", "#666"), lw=0.6, zorder=1)
            ax.set_title(name, fontsize=9)
            ax.set_ylabel("mean cos with e₁")
        axes[-1].set_xticks(x, ROLES)
        fig_legend(fig, axes[0])
        return fig

    return _plot()


# --- The report --------------------------------------------------------------------------------

if metrics_loaded is None or traj_loaded is None:
    stop("The run has not published its results yet.")

res = Results(metrics=metrics_loaded, traj=traj_loaded)


def numbers(res: Results) -> dict:
    """The figures the prose quotes, gathered once so every sentence reads the same data as its table."""
    s = h3_sites(res)
    a = adoption(res)
    wo, wg = res.worst_gap(ex.PRIMARY)
    sweep_m = {SWEEP_OF[c]: q["margin"] for c, q in a["sweep"].items()}
    conds = (ex.PRIMARY, ex.CONTROL, ex.OPWORD_ARM, ex.FULL_ARM)
    return {
        "margin": float(res.margin(ex.PRIMARY).mean()),
        "worst_op": ex.short(wo),
        "worst_gap": wg,
        "diff_gap": res.task_gap(ex.PRIMARY, OP),
        "retention_min": float(res.stat(ex.PRIMARY, "retention_anneal").min()),
        "sites": s,
        "primary_sites": {k: float(res.contrast(ex.PRIMARY)[:, -1].mean(0)[p]) for k, p in SITES.items()},
        "containment": float(np.mean([res.containment(ex.PRIMARY, o).mean() for o in OTHER_OPS])),
        "containment_ctl": float(np.mean([res.containment(ex.CONTROL, o).mean() for o in OTHER_OPS])),
        "sweep_lo": min(sweep_m.values()),
        "sweep_hi": max(sweep_m.values()),
        "sweep_lo_op": min(sweep_m, key=lambda o: sweep_m[o]),
        "n_qualify": len(a["passing"]),
        "chosen": a["chosen"],
        "full_ratio": float(res.margin(ex.FULL_ARM).mean() / res.margin(ex.PRIMARY).mean()),
        "noisy_ratio": float(res.margin(ex.NOISY_ARM).mean() / res.margin(ex.PRIMARY).mean()),
        # Op-identity R² per condition, seed mean: (L1, T).
        "r2": {c: res.r2(c).mean(0) for c in (ex.PRIMARY, ex.OPWORD_ARM, ex.CONTROL)},
        # Mean cosine with e₁ at the first operand at the final slice, over every op's lines: (cond -> float).
        "op1": {c: float(np.mean([res.cos(c, o)[:, -1, 0] for o in ex.OP_NAMES])) for c in conds},
        "op1_peak": float(np.mean([res.cos(ex.PRIMARY, o)[:, :, 0] for o in ex.OP_NAMES], axis=(0, 1)).max()),
    }


N = numbers(res)


rf"""
# Ex 2.2.14: anchoring an operation

/// tip |
<!-- tl;dr -->
Every anchor so far has held a property of one token: how red a color is. Here we anchor an *operation*, `{ex.ANCHORED_OP}`, on the axis *red* used to have. It lands at twice the margin *red* reached, the task does not move, and every other op of the table anchors the same way.
///

There is no *red* anchor beside it. An op is a step up in abstraction from a color: a color is defined by what one token looks like, an op by what it does to a pair of operands. The abstraction made no difference to the anchor, because the op is named by one token and the pull can read that token's embedding.

This was a few-seed smoke test, run before the many-seed equivalence experiment spends its budget, so the equivalence experiment can go ahead. It also asked whether the blocks carry the op forward to where the answer is computed: they carry a little of it to `=` and none to the answer position.

## Findings

- [The task survives (H1)](#the-task-survives-h1) — **{h1_status(res)}.** No op's held-out exact match moves by more than {abs(N["worst_gap"]):.3f}, a quarter of the gate; `{ex.ANCHORED_OP}` itself moves by {abs(N["diff_gap"]):.3f}.
- [The op lands and holds (H2)](#the-op-lands-and-holds-h2) — **{margin_status(res)}.** The op margin is {N["margin"]:.2f}, {N["margin"] / ex.REF_M_LINE:.1f}× what *red* reached, and every seed holds it through the anneal. It saturates by epoch 3: the op word's embedding sits on the axis, and the margin reads that position.
- [The blocks carry it to the use sites (H3)](#the-blocks-carry-it-to-the-use-sites-h3) — **{h3_status(res)}**, on one site of two. With the op word alone pulled, the contrast at `=` clears the floor ({N["sites"]["="]["excess"]:+.3f} over the control) and the contrast at the answer does not ({N["sites"]["answer"]["excess"]:+.3f}).
- [Containment, reported without a gate](#containment-reported-without-a-gate) — the other ten ops sit at {N["containment"]:.2f} at the op position, {N["containment"] - N["containment_ctl"]:+.2f} over the control and far under *red*'s {ex.CONTAINMENT_REF:g}. Post hoc: the first operand leans toward e₁ on the primary alone, a candidate mechanism for that rise.

[The rule for the follow-up](#the-rule-for-the-follow-up): the equivalence experiment anchors `{N["chosen"]}`. All {N["n_qualify"]} ops of the sweep qualify too, with margins from {N["sweep_lo"]:.2f} to {N["sweep_hi"]:.2f}, so the data do not separate them and the analytical choice stands.

## How to read this draft

The op, the three predictions, the containment measure, and the rule for the follow-up were fixed before any run, at commit `2497993`. Everything after that commit is either results filled into their sections or exploratory work, marked as post hoc.

The [D2.2 design](../d2.2/design.md#anchor-one-operation) asks for this smoke test before the equivalence read, scored on the alignment and task gates alone. The probe scan it names runs here as a description.

## Why this experiment

D2.2 asks whether an anchor can hold more than a token's identity at a labelled site. *Red* is a property of the token at a known position. An operation is a property of the computation: named at the op word, and used at `=` and after, deeper in the stack. It is a more abstract concept than a color, and the first rung of a ladder toward concepts in natural language.

If anchoring an op works, the suppression experiment that follows can ask the deliverable's central question: whether removing the op from the axis removes the ability to perform it, selectively and by a bounded amount.

[Ex-2.2.11](../ex-2.2.11/report.py) fixed the grammar and the recipe, and [ex-2.2.13](../ex-2.2.13/report.py) confirmed the recipe at fresh seeds with the weight unchanged. One issue stays open: the projection leaves a quarter of the red-dependent answers of `hue-hsv` in place. That belongs to the *red* anchor, which this model does not have.

The label is new in kind: until now a color decided which lines were labelled, and here the op word decides. The pull covers the whole line, so the label says which lines carry the concept but not where on the line it sits.

On the primary, no alignment measurement can tell an anchored op from an anchored token. The pull puts the op word's embedding on the axis by construction, and asks for the axis at `=` and at the answer too. So these measurements only say whether the pull landed.

One arm pulls the op word alone, so any alignment it shows at the *use sites* (`=` and the answer position, where the op is applied rather than named) is something the pull did not ask for. Whether the anchor captured the operation itself is a question for suppression, and we don't ask it here.

## Conditions

{conditions_html()}

**The op.** `{ex.ANCHORED_OP}` was chosen analytically. On three quarters of its lines no other op in the table gives the same answer (its *op-relevance*),[^op-relevance] the highest of any commutative op. It is commutative, so operand order plays no part in its read.

It is total on the grid, so each line has one answer, and the control learns it to near-perfect held-out accuracy. The ops that round stochastically are capped well below that.

[^op-relevance]: The term is the design's. The sweep reads every other op the same way, so the choice can be revisited on data.

**The labeller.** A line whose op word is the anchored op gets a label at a rate of {ex.LABEL_RATE:g}. The pull covers all six positions of a labelled line, as the handover's whole-line labeller does.

That labels a fifth of a percent of the corpus, the red labeller's share, so each labelled line gets the same pull a red line got, and the op differs from *red* in kind rather than in label share.

The primary uses this sparse rate because it resembles the labels the method will have further up the ladder: for an abstract concept in natural language, labels will be scarce and sometimes wrong. One arm below labels every line of the op instead. We record the label share for each run.

<!-- REVIEW: the rate-0.02 labeller and the every-line labeller swapped places at the human's call, the realistic labeller as the primary. The every-line arm keeps the bracket: its share is fifty times red's, and the anchor term is normalized by the mask's weight, so each of its labelled lines gets a proportionally weaker pull. -->

**The arms.** All three run at the primary's seeds. The op-word arm pulls only the op word, at the primary rate, so the use sites fall outside the mask: any contrast at `=` or the answer at the final slice was carried there by the blocks without the pull asking for it. H3 is scored on this arm.

Pulling one position instead of six also makes each of this arm's pulls about six times stronger, because the anchor term is normalized by the mask. So we report its contrast at the op position beside the primary's.

The every-line arm labels every line of the anchored op, a tenth of the corpus. If its margin agrees with the primary's, label share does not set the margin; if they differ, the ratio of the two margins shows what pulling every line of a categorical concept gains.

The noisy arm moves a fifth of the primary's labels onto lines of the other ops, so the labelled share of the corpus stays the primary's and the total pull with it. It shows what a labeller of that precision costs the margin and the task. Neither the every-line arm nor the noisy arm enters a hypothesis or the rule.

**The seeds.** {ex.SEEDS} for the primary and each arm, and {ex.SWEEP_SEEDS} for each op of the sweep, all fresh. The control is served from the store at its own five seeds; comparisons against it are between seed means, which absorbs the unpaired seed sets.

Everything else is unchanged from ex-2.2.11: [table A+](../ex-2.2.4/report.py#the-op-set), the stochastic corpus, the untied readout, λ_a = 0.1 annealed over the last tenth of training, τ = 0.1, the anti-subspace schedule, and 50 epochs at d64-L4. *Red* is not anchored: e₁ belongs to the op.

## Glossary

<dl>
<dt>Op margin</dt>
<dd>The line margin (m_line) of every experiment since ex-2.1.10, with the anchored op's lines as the labelled group: per slice, the mean alignment with e₁ over the anchored op's lines minus the mean over all eleven ops' probe lines, at the span role where that gap is largest, then averaged over every slice, the embedding included. Taking the largest role per slice is what keeps a clean <code>=</code> embedding from costing anything: at the embedding slice the op word's own role carries the gap. The quantity the anchor term optimizes, so it is a check that the treatment landed.</dd>
<dt>Contrast at a site</dt>
<dd>At one position and slice, the mean cosine with e₁ (how closely the state points along e₁, from −1 to 1, ignoring length) over the anchored op's lines minus the mean over the other ops' lines. At the op position it is the op word's own placement; at <code>=</code> and the answer it is what the blocks carried there, since those tokens are the same on every line.</dd>
<dt>Use sites</dt>
<dd><code>=</code> and the answer position: where the op is applied rather than named.</dd>
<dt>Containment</dt>
<dd>The mean alignment with e₁ at the op position over the other ten ops' lines: how much the ops that were not anchored drifted toward the axis. The categorical analogue of ᾱ at op1.</dd>
<dt>Op-identity R²</dt>
<dd>How linearly readable the op is at one site, anchored against control: the held-out R² (share of variance explained; 1 is perfect) of a ridge probe (a linear regression with a penalty on large weights) from the state to the one-hot of the op word.</dd>
</dl>

## The task survives (H1)

**What we expect.** Anchoring `{ex.ANCHORED_OP}` costs the task nothing we can measure: for each of the eleven ops, the primary's seed-mean held-out expected exact match is within {ex.TASK_GATE:g} of the control's. It holds in part when every op is within {ex.TASK_PARTIAL:g}.

A miss on the anchored op alone would say the pull on its lines competes with producing its answer. The whole-line span makes that possible for the first time on a categorical concept. A miss spread over the table would say the pull disturbs lines it never touches, at a label share of a fifth of a percent.

The design's risk table names this as the first thing an abstract anchor might cost. The red anchor on the same recipe cost nothing on any op at twenty seeds. The metric is the control's own, expected exact match on the grid; an RGB-distance readout beside it is an [open item](/todo/science/rgb-distance-readout-beside-exact-match.md) and is not scored here.

"""

h1_figure(res)

# %%
h1_table(res)

rf"""
{verdict_md(h1_status(res), f"The largest seed-mean gap is {N['worst_gap']:+.3f}, on `{N['worst_op']}`, a quarter of the {ex.TASK_GATE:g} gate, and the anchored op's own gap is {abs(N['diff_gap']):.3f}. The whole-line pull on `{ex.ANCHORED_OP}` lines does not compete with producing their answer, and the other ten ops' lines are untouched.")}

## The op lands and holds (H2)

**What we expect.** This is a manipulation check: the op margin is the quantity the anchor term optimizes, so it says whether the pull landed, not whether the op was captured.

On the seed mean, the op margin on the primary reaches at least {ex.MARGIN_RATIO:.0%} of what *red* reached under the same recipe ({ex.REF_M_LINE:g} on the `mix` lines in ex-2.2.11), and holds in part from {ex.MARGIN_PARTIAL:.0%}. The margin is also retained: every run whose margin reaches {ex.RETENTION_FLOOR:g} when the anchor weight starts to anneal ends training at {ex.RETENTION_GATE:g} of that value.

The two margins are computed the same way (a gap in mean cosine between a labelled group and the pool), so they are comparable in scale. But their groups and baselines differ in ways the method describes, which pull in opposite directions, so the bar is rough; the equivalence experiment sets its own from what this one measures.

A margin far under the bar with the task intact would say the op's lines are harder to pull together than red's: a first sign that a categorical concept spread over eleven contexts wants a different weight.

A margin that falls through the anneal would be the retention failure ex-2.2.9 saw on `handover` alone, and would point to the anneal as the thing to look at before the equivalence experiment.

The learning rate is low over the anneal, so a tighter share than {ex.RETENTION_GATE:g} would be defensible. But one `handover` seed in twenty ended under it in ex-2.2.11, so we keep the bar where a known failure sits and report the per-seed values.

"""

h2_figure(res)

# %%
h2_table(res)

rf"""
{verdict_md(margin_status(res), f"The seed-mean op margin is {N['margin']:.3f}, {N['margin'] / ex.REF_M_LINE:.2f}× *red*'s {ex.REF_M_LINE:g} against a bar of {MARGIN_BAR:.3f}. It reaches about 0.8 by epoch 3 and barely moves after: the op word's embedding sits at a cosine near 1 with e₁ on every slice, and the margin takes the largest role per slice, so the op position sets it. The bar was set for a graded concept spread over a line, and a categorical one named by a single token clears it with room to spare. A margin this saturated tells ops and arms apart poorly, which the sweep and the arms below bear out.")}

{verdict_md("pass" if res.retention_ok(ex.PRIMARY) else "miss", f"Every seed ends training within {1 - N['retention_min']:.1%} of its margin at the anneal start. Nothing gives way as the weight comes down.")}

## The blocks carry it to the use sites (H3)

**What we expect.** Some of the op reaches the use sites. On the op-word arm at the final slice, the contrast at `=` and the contrast at the answer position are each at least {ex.USE_CONTRAST_FLOOR:g} above the control's, on the seed mean. The ratio of each to the contrast at the op position is reported with no bar.

No decision depends on this prediction, the least certain of the three. We have seen the model put the answer color at the use sites; this measures whether it also holds a copy of the op there, or only what the op produced. The floor asks only for a clearly positive contrast.

At slice 0 (the embedding), the contrast at the op position is the op word's embedding projected on e₁, which the pull puts there by construction. At the use sites the token (`=` or the answer) is the same on every line, so the slice-0 contrast is zero and any later contrast came through attention, which on this arm nothing requested.

A pass would mean that once the op word is on the axis, the blocks carry some of the op to where it is used, which an intervention at the op word would rely on. A miss would mean that, at this weight, the anchor marks the op but the blocks do not carry it to the use sites; it would not mean the model computes the op somewhere else.

On the primary the whole-line pull covers the use sites too, so the contrast there is part of what the treatment optimizes and we expect a pass from the method alone. We report it beside the op-word arm as a manipulation check.

<!-- REVIEW: H3 was a manipulation check on the primary, where the whole-line span pulls positions 3 and 4 of labelled lines. It is now scored on the op-word arm, whose mask excludes the use sites, so the contrast there is carried rather than optimized. The primary stays the whole-line labeller because that is the realistic one; the narrower arm is where this hypothesis means something. -->

"""

h3_figure(res)

# %%
h3_table(res)

rf"""
{verdict_md(h3_status(res), f"On the op-word arm at the final slice, the contrast at `=` is {N['sites']['=']['arm']:.3f} against the control's {N['sites']['=']['control']:.3f}, over the floor; at the answer it is {N['sites']['answer']['arm']:.3f} against {N['sites']['answer']['control']:.3f}, under it. Beside the op position's {N['sites']['op']['arm']:.2f} that is a ratio of {N['sites']['=']['arm'] / N['sites']['op']['arm']:.2f} at `=` and about zero at the answer: the blocks carry a twentieth of the op word's alignment to `=` and nothing to the answer. On the primary, where the pull covers the use sites, both sit near {N['primary_sites']['=']:.1f} (`=` {N['primary_sites']['=']:.3f}, answer {N['primary_sites']['answer']:.3f}), so the whole-line pull asks for more at those sites than the blocks bring on their own. This is the reading the design allowed for: at this weight the anchor marks the op where it is named, a little of it reaches `=`, and the answer position holds what the op produced rather than the op.")}

## Containment, reported without a gate

**What we expect.** The other ten ops drift a little toward e₁ at the op position. The backlog item [containment rises under the untied readout](/todo/science/containment-rises-under-the-untied-readout.md) records ᾱ at op1 at {ex.CONTAINMENT_REF:g} on the red anchor under this recipe, up from the 0.1 the earlier gates asked for, with no mechanism named for the rise. We report the categorical analogue per op, beside that number and the control.

Three things could be behind the rise:
(a) the red pull itself (a graded label on a color, at every position of the line),
(b) the two changes that arrived with it (the untied readout and the whole-line span), and
(c) the weights of the recipe.

This experiment keeps (b) and (c) and replaces (a) with a categorical pull on an op, so it can only say whether the red pull was needed. Containment near the control would say it was, alone or together with the shared factors; containment near the red value would say the shared factors produce the rise on their own.

A value between the two would mean both play a part, which seems likely, since the pull and the readout could each contribute. Either way this is one experiment's worth of evidence, and the item stays open.

"""

containment_figure(res)

# %%
containment_table(res)

rf"""
**What we saw.** The other ten ops sit at {N["containment"]:.3f} at the op position on the primary, against {N["containment_ctl"]:.3f} on the control: a rise of about {N["containment"] - N["containment_ctl"]:.2f} on every op, and well under *red*'s {ex.CONTAINMENT_REF:g}. The rise grows over the slices, from nothing at the embedding (the other op words are not pulled) to its plateau by slice 1. On this reading the shared factors (b) and (c) do not produce the rise on their own: the graded red pull was needed for most of it.

One exploratory observation below complicates that reading. The primary's first operand leans toward e₁ on every line, a site the containment statistic here does not read but *red*'s ᾱ at op1 does. So the item stays open, with a note.

## The rule for the follow-up

> {ex.ADOPTION}

The primary passes H1 and H2 in full, so the equivalence experiment anchors `{N["chosen"]}`. Every op of the sweep qualifies as well, with seed-mean margins from {N["sweep_lo"]:.3f} to {N["sweep_hi"]:.3f}: the data do not separate the ops, and the analytical choice stands on its own criteria.

"""

rule_table(res)

rf"""

## Exploratory analyses

Anything we think of after seeing the data goes here, marked as post hoc. Four analyses are planned in advance as descriptions, with no gate, and run whichever way the predictions come out. Each gets a figure as well as a table. When the results come in, each analysis's figure and table go right after the paragraph that defines it, so definition and result read together.

**The sweep.** Every other op of table A+ anchored the same way at {ex.SWEEP_SEEDS} seeds, read on the same task gap, op margin, retention, and use-site contrast. It is the design's "sweep over all ops to see whether they can all be anchored equally well", run cheaply while the machinery is warm, and the rule above falls back on it.

Whether the three order-sensitive ops behave differently is the one pattern worth looking for in advance.

"""

sweep_figure(res)

# %%
sweep_table(res)

rf"""
All ten anchor alike. The margins span {N["sweep_hi"] - N["sweep_lo"]:.2f}, every retention is at or above 0.99, and every seed-mean task gap is inside the gate, with single seeds of `sat` and `screen` touching it. The three order-sensitive ops (italic in the figure) have the three lowest margins, `{N["sweep_lo_op"]}` lowest at {N["sweep_lo"]:.3f}, but the spread is within what three seeds resolve, and their task gaps are the table's. Nothing sets them apart here. The margin's saturation at the op word is why: an anchor that reads the op's own embedding does not care what the op does to its operands.

**The op-identity scan.** The op-identity R² at every site, anchored against control. The design's equivalence claim is that anchoring does not change how readable the op is anywhere the anchor does not reach. The equivalence experiment has to declare a margin for that, and this scan gives it an observed spread. It carries no gate here.

"""

scan_figure(res)

# %%
scan_table(res)

rf"""
Anchoring leaves the op about as readable as the control has it. At the first operand nothing is readable, since the op word comes after it; at the op word the probe is perfect on every condition. At `=` the primary dips at slice 1 ({N["r2"][ex.PRIMARY][1, ex.EQUALS_POSITION]:.2f} against the control's {N["r2"][ex.CONTROL][1, ex.EQUALS_POSITION]:.2f}) and matches it from slice 2. At the answer the primary reads *above* the control at every slice past the embedding ({N["r2"][ex.PRIMARY][-1, ex.ANSWER_POSITION]:.2f} against {N["r2"][ex.CONTROL][-1, ex.ANSWER_POSITION]:.2f} at the last), with the op-word arm a little under it: the whole-line pull keeps more of the op at the answer than the control does. The largest shift anywhere is at the newline at slice 1, {N["r2"][ex.PRIMARY][1, 5] - N["r2"][ex.CONTROL][1, 5]:+.2f}, where the control's own seeds spread by about 0.4. The equivalence experiment can take its margin from this: a band of ±0.1 in R² would hold at every site but that one.

**The arms.** The every-line arm and the noisy-label arm, read on the same statistics as the primary: the task gap, the op margin, retention, and the use-site contrast. The every-line arm's margin against the primary's is the price or gain of the label share; the noisy arm's margin and task gap against the primary's are the cost of a fifth of wrong labels. The op-word arm carries H3 and is read there.

"""

arms_figure(res)

# %%
arms_table(res)

rf"""
Label share barely moves the margin. Fifty times the labels raise it to {N["full_ratio"]:.2f}× the primary's, and a fifth of wrong labels leave it at {N["noisy_ratio"]:.2f}× with the task gap unchanged. Both say the same thing as H2: the margin saturates at the op word whatever the label share, and once that embedding is on the axis there is nothing left for more labels to pull. Where the every-line arm does differ is the newline: its embedding leans toward e₁ at {float(res.cos(ex.FULL_ARM, OP)[:, 0, 5].mean()):.2f} against the primary's {float(res.cos(ex.PRIMARY, OP)[:, 0, 5].mean()):.2f}, a free token the pull can place at no cost to the task. For the ladder this is encouraging: a scarce labeller and an imperfect one land the concept as well as an exhaustive one.

**The alignment map.** The primary's mean cosine with e₁ over position and slice on the anchored op's lines and on the others, as a picture of where the anchor put the op.

"""

map_figure(res)

rf"""
The op word is at a cosine near 1 on the anchored lines at every slice and near 0 on the others: that is the contrast, and it is all at one position. The two panels agree at the first operand, as they must, since under causal attention the state there comes before the op word and cannot know the op. What they agree on is a lean toward e₁ that grows over the slices to {N["op1_peak"]:.2f}.

**Post hoc: the first operand's lean is the primary's alone.** At the final slice the mean cosine at the first operand over every op's lines is {N["op1"][ex.PRIMARY]:.2f} on the primary, {N["op1"][ex.CONTROL]:.2f} on the control, {N["op1"][ex.OPWORD_ARM]:.2f} on the op-word arm and {N["op1"][ex.FULL_ARM]:.2f} on the every-line arm. The whole-line pull asks the first operand of a labelled line to align with e₁, and nothing at that position distinguishes a `{ex.ANCHORED_OP}` line, so the model answers with a lean that every line shares. That the every-line arm does not show it says the per-line strength matters: at fifty times the labels each pull is fifty times weaker, and the arm places the newline instead.

This is a candidate mechanism for the rise the [containment item](/todo/science/containment-rises-under-the-untied-readout.md) records: the whole-line span arrived with the untied readout, and it pulls a position that cannot carry the concept. Under *red* the first operand does carry redness, so the analogy is partial. A red anchor with the first operand excluded from the span would test it. This was seen after the data and is not scored.


## Discussion

An op anchors at least as readily as a color. A color is a graded property spread over three tokens of a line, so *red*'s margin had to be assembled from many partial alignments. An op is named by one token, and the pull puts that token's embedding on the axis in the first epochs. So the margin saturates, the task never feels it, all ten other ops do the same, and label share and label precision hardly matter. The alignment measurements here confirm that the pull landed, which is all the design asked of them.

The use-site contrast says where that leaves the op. With only the op word pulled, `=` picks up a twentieth of its alignment and the answer position none. The blocks carry a trace of the op to where it is applied and the answer position holds the answer. Whether the anchor captured the *operation* is still the question for suppression, and this result sets the expectation: an intervention at the op word will have little to work with downstream on its own, and the whole-line pull is what puts the op at the use sites.

The equivalence experiment inherits `{N["chosen"]}`, a saturated margin that will not separate conditions, a scan whose spread outside the newline sits within ±0.1 in R², and the arms' finding that a scarce, imperfect labeller lands the concept. The first operand's lean is the loose end: a whole-line span pulls positions that cannot carry the concept, and the containment item now has a mechanism to test.

## Method

### The labeller

This experiment adds a labeller keying in which the op word draws the label, against a per-op rate table: {ex.LABEL_RATE:g} for the anchored op and zero otherwise on the primary, one for the anchored op on the every-line arm, and {ex.NOISY_TRUE_RATE:g} for the anchored op with {ex.FALSE_RATE:.4g} for each other op on the noisy arm. The pull covers the whole line, or the op word alone on the op-word arm. It consumes the random stream differently from the color labellers, so the control's batches are not this experiment's; hence the seed-mean comparison, as in ex-2.2.13.

### The measurements

The op margin is ex-2.2.3's `line_margin` with uniform line weights summing to one over the anchored op's lines and zero elsewhere, on the pooled per-op probe sets ex-2.2.9 built. The function is red's m_line, but the comparison is not like for like: red's weights grade with redness (`p_slot`) and its baseline is the `mix` lines alone, where the op's weights are binary and its baseline includes its own lines as one eleventh of the pool. The binary label concentrates the labelled mean and the pooled baseline shrinks the gap by about a tenth, so the two differences pull in opposite directions and the ratio in H2 is a rough bar rather than a matched one.

The probe sets are stored per op, and a per-op baseline would make the op margin zero by construction (the weights are uniform within one op's lines), so the baseline is pooled over all eleven ops' lines at equal counts and the one-eleventh clause stands. Retention is the end-of-training margin over the margin at the start of the anchor weight's anneal, on the trajectory recorded every few epochs, with the anneal start located as ex-2.2.11 located it. Contrast and containment are the glossary's mean cosines, taken over the same probe lines. The op-identity scan fits a ridge probe at ridge strength {ex.PROBE_RIDGE:g} from the state at each site to the one-hot op word, with a held-out split by line.

### Budget

{ex.N_RUNS} runs at d64-L4, each as long as a run in ex-2.2.13, which trained 160 of them for about fourteen dollars on Modal. Eval adds the alignment measurements on eleven probe sets and the scan; no intervention is scored.

### What this experiment does not do

It does not anchor *red* beside the op, does not suppress anything, does not vary the weight, and does not claim equivalence on the scan. Each of those is for a later experiment.
"""
