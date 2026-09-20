# title: Ex 2.2.10: three reads before the handover re-run

import colorsys
import itertools
from collections.abc import Sequence
import json
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# The constants come from `experiment.py` beside this script (the script's directory is on
# sys.path while it runs), which in turn binds ex-2.2.9's grammar and gates.
import experiment as ex
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from mini.lit import stop
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed
from sca.data.colors import redness
from sca.data.ops import CANDIDATE_BY_NAME, OP_BY_NAME, TOP, vocabulary
from sca.vis import CUBE_VIEWS, ViewName, draw_cube_bound, plot_rgb_cube, project_cube

# What a result cell shows while the run has not published yet.
RESULTS_TO_COME = "/// admonition | TODO\n    type: warning\nResults to come.\n///"

OPS = ex.OPS_READ
# The red operand's position and the group name ex-2.2.9 scores it under.
SLOTS = ((0, "op1"), (2, "op2"))
# The two forward passes the scoring stored: as trained, and with the axis removed everywhere.
PASSES = ("clean", "projection")
CONDS = ("control", "handover", "handover-slot", "handover-tied", "handover-narrow")
ALL_OPS = OP_BY_NAME | CANDIDATE_BY_NAME
# The 216 grid colors in palette order, which is the order the stored palette indices use.
GRID = [ex.PALETTE[n] for n in ex.PALETTE]
GRID_UNIT = np.asarray(GRID, float) / TOP
# The seven grid colors with redness at or above the red dose.
RED_WORDS = {n for n in ex.PALETTE if redness(ex.PALETTE[n]) >= ex.RED_DOSE}


def load_json(ref: str) -> dict | None:
    """A published JSON result as a dict, or None before it exists."""
    store = project_store()
    art = store.get_refs([ref])[ref]
    if art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "data.json")])
        return json.loads(path.read_text())


def load_npz(ref: str) -> dict[str, np.ndarray] | None:
    """A published npz as a dict of arrays, or None before it exists."""
    store = project_store()
    art = store.get_refs([ref])[ref]
    if art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "arrays.npz")])
        with np.load(path) as z:
            return {k: z[k] for k in z.files}


def ink(cond: str) -> str:
    """One ink per condition, the same pairs as ex-2.2.9's report."""
    inks = {
        "control": ("#6b6b6b", "#b0b0b0"),
        "handover": ("#c0392b", "#ff8a76"),
        "handover-slot": ("#2b6cb0", "#7fb3ff"),
        "handover-tied": ("#7b3fa0", "#cfa3ff"),
        "handover-narrow": ("#2e8b57", "#7fd8a4"),
        "clean": ("#6b6b6b", "#b0b0b0"),
        "projection": ("#c0392b", "#ff8a76"),
    }
    return light_dark(*inks[cond])


def marker(cond: str) -> str:
    return {
        "control": "s",
        "handover": "o",
        "handover-slot": "^",
        "handover-tied": "D",
        "handover-narrow": "v",
        "clean": "s",
        "projection": "o",
    }[cond]


def dots(ax, x: float, v: np.ndarray, cond: str, *, rng, ms: float = 5.0, width: float = 0.06, label=None) -> None:
    """One column of per-seed dots with the seed mean on top, in the condition's ink and marker; a thin bar
    behind spans the seed range (the `dots` of ex-2.2.9's report).
    """
    v = np.asarray(v, float)
    color, m = ink(cond), marker(cond)
    jit = rng.uniform(-width, width, len(v))
    ax.plot([x, x], [np.nanmin(v), np.nanmax(v)], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + jit, v, "o", ms=2.2, color=color, alpha=0.45, zorder=3, mew=0)
    ax.plot(x, np.nanmean(v), m, ms=ms, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6, label=label)


def fig_legend(fig: plt.Figure, ax: Axes, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7, **kwargs)


def gate_line(ax: Axes, y: float, *, partial: float | None = None, fail: str | None = None) -> None:
    """A dashed gate line with the failing side hatched, as ex-2.2.9's report draws its gates."""
    ax.axhline(y, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=2)
    if partial is not None:
        ax.axhline(partial, color=light_dark("#333", "#ddd"), lw=0.7, ls=":", zorder=2)
    if fail is not None:
        lo, hi = ax.get_ylim()
        span = (lo, y) if fail == "below" else (y, hi)
        ax.axhspan(*span, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="//", lw=0, alpha=0.1, zorder=0)
        ax.set_ylim(lo, hi)


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    """An authored result table in the shared report style; the first column is text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{cell_html(c)}</td>" for i, c in enumerate(row)) + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def span2(v, digits: int = 2) -> str:
    """A seed mean with its range, as `0.51 (0.4–0.6)`."""
    v = np.asarray(v, float)
    if v.size == 0 or np.all(np.isnan(v)):
        return "—"
    return f"{np.nanmean(v):.{digits}f} ({np.nanmin(v):.{digits}f}–{np.nanmax(v):.{digits}f})"


@dataclass
class Results:
    m229: dict
    traj: dict
    probes: dict[str, np.ndarray]
    metrics: dict | None
    arrays: dict[str, np.ndarray] | None

    @property
    def runs(self) -> dict[str, dict]:
        return {r["label"]: r for r in self.m229["runs"]}

    def by_cond(self, cond: str) -> list[dict]:
        return [r for r in self.m229["runs"] if r["condition"] == cond]

    def scores(self, cond: str) -> list[dict]:
        return [s for s in self.m229["scores"] if s["condition"] == cond]

    def kept(self, cond: str, op: str, group: str) -> np.ndarray:
        """Per-seed kept share under the projection, from ex-2.2.9's scores."""
        return np.array([s["ops"][op]["operators"][ex.OPERATOR]["kept"][group] for s in self.scores(cond)], float)

    @property
    def labels(self) -> list[str]:
        return [] if self.metrics is None else [s["label"] for s in self.metrics["scores"]]

    @property
    def width(self) -> int:
        """The residual stream's width, from a stored probe."""
        return (
            int(next(v for k, v in self.arrays.items() if k.endswith("/probe/weights")).shape[-2]) if self.arrays else 0
        )

    def arr(self, label: str, key: str) -> np.ndarray:
        assert self.arrays is not None
        return self.arrays[f"{label}/{key}"]

    def stacked(self, key: str) -> np.ndarray:
        """One per-line array of every scored run, stacked on a leading seed axis."""
        return np.stack([self.arr(lb, key) for lb in self.labels])


def require[T](value: T | None, ref: str) -> T:
    """An ex-2.2.9 input the report cannot do without."""
    if value is None:
        raise FileNotFoundError(f"ex-2.2.9 result {ref!r} is not in the store; this report reads its runs")
    return value


def tok2color() -> np.ndarray:
    """Token id → palette index on ex-2.2.9's vocabulary (the tokenizer sorts its words), −1 at syntax."""
    vocab = sorted({""} | set(vocabulary(ex.TABLE)))
    names = list(ex.PALETTE)
    out = np.full(len(vocab), -1)
    for i, w in enumerate(vocab):
        if w in ex.PALETTE:
            out[i] = names.index(w)
    return out


def red_lines(res: Results, op: str) -> dict[str, np.ndarray]:
    """The red lines of one op in ex-2.2.9's probe set: their operand colors, red slot, and to-zero move."""
    t2c = tok2color()
    tok, r1, r2, move = (res.probes[f"{op}/{k}"] for k in ("tokens", "r1", "r2", "move"))
    red = np.maximum(r1, r2) >= ex.RED_DOSE
    return {
        "rows": np.flatnonzero(red),
        "a": t2c[tok[red, 0]],
        "b": t2c[tok[red, 2]],
        "slot": np.where(r1 >= r2, 0, 2)[red],
        "move": move[red],
    }


def counterfactual_kept(res: Results, op: str) -> dict[int, dict[str, float]]:
    """Per red slot, the share of red lines whose true answer survives each counterfactual on the red operand."""
    rule = ALL_OPS[op]
    f = red_lines(res, op)

    def dist(a, b):
        return float(np.linalg.norm(np.subtract(a, b)) / TOP)

    out: dict[int, dict[str, list[float]]] = {0: {}, 2: {}}
    for a_i, b_i, slot in zip(f["a"], f["b"], f["slot"], strict=True):
        a, b = GRID[a_i], GRID[b_i]
        red = a if slot == 0 else b
        v = max(red)
        subs = {
            "to zero": [(0, red[1], red[2])],
            "change of hue": [p for p in set(itertools.permutations(red)) if p != red],
            "to gray": [(v, v, v)],
        }
        truth = rule(a, b)
        for name, cands in subs.items():
            moves = [dist(rule(*((c, b) if slot == 0 else (a, c))), truth) for c in cands]
            out[slot].setdefault(name, []).append(max(moves) < ex.FAR_MOVE)
    return {slot: {k: float(np.mean(v)) for k, v in d.items()} for slot, d in out.items()}


def removal_mask(res: Results, label: str, op: str, slot: int) -> np.ndarray:
    """The removal lines with the red operand in *slot*, in the stored red-line order."""
    return res.arr(label, f"{op}/removal") & (res.arr(label, f"{op}/red_operand") == slot)


def answer_pairs(res: Results) -> dict[tuple[str, int, str], Counter]:
    """Per op, red slot, and pass: how many (line, seed) answers on the removal lines made each
    (true color, greedy answer) move.
    """
    # The stored greedy answer is a token id over the whole vocabulary; a syntax token is an
    # off-vocab answer, counted under the palette index −1.
    t2c = tok2color()
    out = {}
    for op in OPS:
        for (slot, _g), pas in itertools.product(SLOTS, PASSES):
            pairs: Counter = Counter()
            for lb in res.labels:
                m = removal_mask(res, lb, op, slot)
                truth = res.arr(lb, f"{op}/ans_idx")[m]
                guess = t2c[res.arr(lb, f"{op}/{pas}/guess")[m]]
                pairs.update(zip(truth.tolist(), guess.tolist(), strict=True))
            out[op, slot, pas] = pairs
    return out


def cube_cloud(ax: Axes, mass: np.ndarray, *, view: ViewName = "wheel", n: int = 2500, rng, s: float = 3.0) -> None:
    """A dithered cloud in one view of the cube: *n* dots shared out over the grid colors in proportion to *mass*,
    each jittered within its color's cell and drawn in that color.
    """
    draw_cube_bound(ax, view, labels=True)
    counts = rng.multinomial(n, mass / mass.sum())
    idx = np.repeat(np.arange(len(mass)), counts)
    rgb = GRID_UNIT[idx] + rng.uniform(-0.5, 0.5, (len(idx), 3)) / TOP
    xy = project_cube(rgb, view)
    # Nearer the reader draws last, as plot_rgb_cube does, so the solid view shows its front.
    order = np.argsort(rgb @ CUBE_VIEWS[view].toward, kind="stable")
    ax.scatter(
        xy[order, 0],
        xy[order, 1],
        c=np.clip(GRID_UNIT[idx][order], 0, 1),
        s=s,
        lw=0,
        alpha=0.8,
        zorder=3,
        clip_on=False,
    )


def split_moves(d: np.ndarray, w: np.ndarray, iters: int = 10) -> list[np.ndarray]:
    """Weighted 2-means on displacement vectors: the member mask of each side (one or two).

    Seeded by splitting across the mean direction, so a cell whose answers fan out to either side of it comes back as two moves rather than one that points between them.
    """
    m = (w[:, None] * d).sum(0) / w.sum()
    axis = np.array([-m[1], m[0]]) / np.hypot(*m) if np.hypot(*m) > 1e-9 else np.array([1.0, 0.0])
    lab = (d - m) @ axis > 0
    for _ in range(iters):
        cs = [
            (w[lab == k, None] * d[lab == k]).sum(0) / w[lab == k].sum() if w[lab == k].sum() > 0 else m
            for k in (False, True)
        ]
        new = ((d - cs[1]) ** 2).sum(1) < ((d - cs[0]) ** 2).sum(1)
        if (new == lab).all():
            break
        lab = new
    return [lab == k for k in (False, True) if w[lab == k].sum() > 0]


def flow_arrows(
    ax: Axes,
    truth: np.ndarray,
    guess: np.ndarray,
    n: np.ndarray,
    *,
    view: ViewName = "wheel",
    sigma: float = 0.2,
    step: float = 0.2,
    min_share: float = 0.25,
    min_sep: float = 0.3,
    min_move: float = 0.05,
    spread: bool = True,
) -> None:
    """The (truth → guess) moves as a sampled flow field in one view of the cube.

    A stub per move stacks up illegibly once moves span the wheel, so this smooths them instead: at each point of a hex lattice with a truth mark nearby, the count-weighted mean move of the truths within a Gaussian of width *sigma*, drawn as an arrow in the local mean truth color and sized by the local count. Where the moves split two ways — a cell whose answers go left or right but rarely between — the cell gets an arrow per side, provided each side holds *min_share* of the weight and they differ by *min_sep*. Moves that spread more ways than two average out, so a short arrow can also mean a scatter; a mean move under *min_move* draws nothing, since the marks already show an answer that stays put. With *spread*, each arrow sits in a faint wedge spanning one circular standard deviation of its side's directions (weighted by count and length, so the short moves' noisy directions count for little), at the side's mean length.
    """
    from matplotlib.patches import FancyArrowPatch, Wedge
    from matplotlib.path import Path as MplPath

    t, g = project_cube(truth, view), project_cube(guess, view)
    d = g - t
    length = np.hypot(d[:, 0], d[:, 1])
    angle = np.arctan2(d[:, 1], d[:, 0])
    xs = np.arange(-1, 1.001, step)
    ys = np.arange(-1, 1.001, step * np.sqrt(3) / 2)
    pts = np.array([(x + (step / 2 if j % 2 else 0), y) for j, y in enumerate(ys) for x in xs])
    # Sample only inside the cube (a tail outside it would read as an answer from nowhere), and
    # only where a truth mark lies close enough for the smoothed move to describe it.
    inside = MplPath(project_cube(CUBE_VIEWS[view].rim, view)).contains_points(pts, radius=0.02)
    cells = []
    for p in pts[inside]:
        dist = np.sqrt(((t - p) ** 2).sum(1))
        if dist.min() > step * 0.75:
            continue
        w = n * np.exp(-(dist**2) / (2 * sigma**2))
        total = float(w.sum())
        color = np.clip((w[:, None] * truth).sum(0) / total, 0, 1)
        sides = split_moves(d, w)
        means = [(w[s, None] * d[s]).sum(0) / w[s].sum() for s in sides]
        if len(sides) == 2 and (
            min(w[s].sum() for s in sides) / total < min_share or np.hypot(*(means[0] - means[1])) < min_sep
        ):
            sides = [np.ones(len(d), bool)]
        parts = []
        for s in sides:
            ws = w[s]
            m = (ws[:, None] * d[s]).sum(0) / ws.sum()
            # Circular mean and spread of the direction, weighted by count × length.
            wl = ws * length[s]
            z = (wl * np.exp(1j * angle[s])).sum() / wl.sum() if wl.sum() > 0 else 1.0
            sd = np.sqrt(max(-2 * np.log(max(abs(z), 1e-9)), 0))
            parts.append((m, float(ws.sum()), float(np.angle(z)), float(sd), float((ws * length[s]).sum() / ws.sum())))
        cells.append((p, parts, total, color))
    w_max = max(c[2] for c in cells)
    for p, parts, _, color in cells:
        for m, wk, mean_angle, sd, mean_len in parts:
            if np.hypot(*m) < min_move:
                continue  # the marks already show where an answer stays put
            size = np.sqrt(wk / w_max)
            if spread and sd > 0.05:
                a, half = np.degrees(mean_angle), np.degrees(min(sd, np.pi))
                ax.add_patch(
                    Wedge(p, mean_len, a - half, a + half, facecolor=color, lw=0, alpha=0.12, zorder=1, clip_on=False)
                )
            # The head is sized in points, so cap it against the shaft or a short move is all head.
            head = min(4 + 7 * size, 30 * float(np.hypot(*m)))
            arrow = FancyArrowPatch(
                p,
                p + m,
                arrowstyle="-|>",
                mutation_scale=head,
                lw=0.4 + 1.8 * size,
                color=color,
                alpha=0.9,
                zorder=4,
                clip_on=False,
                shrinkA=0,
                shrinkB=0,
            )
            ax.add_patch(arrow)


def answer_mass(res: Results) -> dict[tuple[str, int, str], np.ndarray]:
    """Per op, red slot, and pass: the mean answer mass over the 216 grid colors on the removal lines."""
    out = {}
    for op in OPS:
        for slot, _ in SLOTS:
            for pas in PASSES:
                acc, k = np.zeros(len(GRID)), 0
                for lb in res.labels:
                    m = removal_mask(res, lb, op, slot)
                    if m.any():
                        acc += res.arr(lb, f"{op}/{pas}/mass")[m].astype(float).sum(0)
                        k += int(m.sum())
                out[op, slot, pas] = acc / max(k, 1)
    return out


def peakedness(res: Results) -> dict[tuple[str, int, str], tuple[float, float]]:
    """How peaked the answer distribution of a removal line is under each pass.

    Returns the mean mass on the top color, and the effective number of colors (the exponential of the
    mean entropy over lines and seeds).
    """
    out = {}
    for op in OPS:
        for (slot, _g), pas in itertools.product(SLOTS, PASSES):
            top, eff, k = 0.0, 0.0, 0
            for lb in res.labels:
                m = removal_mask(res, lb, op, slot)
                if not m.any():
                    continue
                p = res.arr(lb, f"{op}/{pas}/mass")[m].astype(float)
                p /= p.sum(1, keepdims=True)
                top += p.max(1).sum()
                eff += -(p * np.log(np.maximum(p, 1e-12))).sum()
                k += int(m.sum())
            out[op, slot, pas] = (top / k, float(np.exp(eff / k))) if k else (np.nan, np.nan)
    return out


def decoded_colors(res: Results) -> dict:
    """Mean over seeds of the decoded coordinates on the removal lines, per op: the red operand at its own
    position and the answer at `=`, for both passes, per slice.
    """
    out = {}
    for op in OPS:
        dec = {pas: res.stacked(f"{op}/{pas}/decoded").astype(float) for pas in PASSES}  # (S, L1, T, 3, N, 3)
        lb0 = res.labels[0]
        tok = res.arr(lb0, f"{op}/tokens")
        ans = res.arr(lb0, f"{op}/ans_idx")
        red_slot = res.arr(lb0, f"{op}/red_operand")
        removal = res.arr(lb0, f"{op}/removal")
        rows = np.flatnonzero(removal)
        # The red operand at its own position: target index 0 at position 0, or 1 at position 2.
        pos = red_slot[rows]
        tgt = np.where(pos == 0, 0, 1)
        for pas in PASSES:
            d = dec[pas].mean(0)  # (L1, T, 3, N, 3)
            out[op, pas, "operand"] = np.stack(
                [d[:, pos[i], tgt[i], rows[i]] for i in range(len(rows))], axis=1
            )  # (L1, n, 3)
            out[op, pas, "answer"] = d[:, ex.DECODE_POS, 2, rows]  # (L1, n, 3)
        out[op, "operand_truth"] = GRID_UNIT[tok[rows, pos]]
        out[op, "answer_truth"] = GRID_UNIT[ans[rows]]
    return out


def anneal_start(res: Results, label: str) -> float:
    """The epoch at which the anchor weight first drops below its plateau."""
    t = res.traj[label]["traj"]
    w = np.array(t["weight"])
    ep = np.array(t["epoch"])
    below = np.flatnonzero(w < 0.99 * w.max())
    below = below[below > np.argmax(w)]
    return float(ep[below[0]]) if len(below) else float(ep[-1])


def retention_stats(res: Results) -> dict[str, np.ndarray]:
    """Per condition, one row per seed: peak alignment, its epoch, the alignment at the start of the anneal,
    the final alignment, and the epoch the anneal started.
    """
    out = {}
    for cond in ("handover", "handover-slot", "handover-tied"):
        rows = []
        for r in res.by_cond(cond):
            t = res.traj[r["label"]]["traj"]
            ep, ml = np.array(t["epoch"]), np.array(t["m_line"])
            a0 = anneal_start(res, r["label"])
            at = ml[np.searchsorted(ep, a0) - 1]
            rows.append((ml.max(), ep[ml.argmax()], at, ml[-1], a0))
        out[cond] = np.array(rows)
    return out


def containment_rows(res: Results, cond: str) -> dict[str, np.ndarray]:
    """Per seed of one condition: ᾱ at op1, the mean absolute axis component of the non-red color rows, the
    axis component of the `⏎` and `=` embedding rows, and of the `⏎` readout row where there is one.
    """
    out = {"alpha": [], "nonred": [], "eol": [], "eq": [], "eol_readout": []}
    for r in res.by_cond(cond):
        rows, rr = r["rows"], r["rows_readout"]
        out["alpha"].append(r["alpha_op1"])
        out["nonred"].append(np.mean([abs(rows[n]) for n in ex.PALETTE if n not in RED_WORDS and n in rows]))
        out["eol"].append(rows["\n"])
        out["eq"].append(rows["="])
        out["eol_readout"].append(rr["\n"] if rr else np.nan)
    return {k: np.array(v, float) for k, v in out.items()}


r"""
# Ex 2.2.10: three reads before the handover re-run

/// tip |
<!-- tl;dr -->
Ex-2.2.9 moved the anchor onto the larger grammar and left three loose ends. Removal was one-sided on the three HSV ops, one seed in twenty fell under the retention gate, and the non-red alignment at op1 sat above the old reference. This notebook reads each off the stored runs, with one small scoring pass for the cube figures and no new training.

The removal miss comes from the rule we used to pick the lines. The projection acts like a change in the hue of red, and the lines that missed are the ones whose answer takes only the saturation or value of the red operand. Red is the one anchored concept, so these reads cannot say whether another concept would come apart the same way.

The retention drop happens before the anneal begins. The op1 alignment rises under either half of the handover, so it belongs to the new grammar rather than to the readout alone. We propose the re-run.
///

## Observations

None of the lines below is a result; ex-2.2.11 will adopt what it needs from here and score it at fresh seeds.

- [Removal](#removal-on-the-order-sensitive-ops): on the three HSV ops the kept share follows a change of hue on the red operand, and not the to-zero rule we used to pick the removal lines. Where the answer takes only the saturation or value of the red operand (`sat-hsv` and `value-hsv` with red at op2), two thirds of the clean exact match survives the projection. Where the answer takes the hue of red, or the whole color, it goes. One case no counterfactual predicts: `hue-hsv` with red at op1, whose answer needs only the saturation and value of red, and where a third of the clean exact match survives; its projected answers keep the hue of op2 and come out paler. A linear probe on the stream reads the projected red operand as a hue rotated away from red, at a lower value, and the rotation grows block by block.
- [Retention](#retention-and-the-anneal): on `handover` the alignment reaches a noisy plateau by epoch 10, and its peak is just the high point of that noise. By the time the anneal begins at epoch 45 the seeds sit at 0.66, a few of them drifting downward. Through the anneal itself every condition is flat, so the gate is reading the noise in the plateau plus the drift. `handover-slot` and `handover-tied` hold their plateaus, so the drift needs the whole-line labeller and the untied readout together.
- [Containment](#containment-under-the-untied-readout): ᾱ at op1 on the non-red lines is 0.28 on `handover`, 0.18 on `handover-slot`, and 0.16 on `handover-tied`, against 0.02 on the control. Each half of the handover raises it, so the untied readout is not the whole story. The `⏎` embedding row picks up the axis only under the whole-line labeller: 0.17 on `handover` and zero on `handover-slot`.
- [What we make of it](#what-we-make-of-it): score removal on the lines whose answer takes the hue of the red operand, judge retention against the alignment at the start of the anneal (or drop the ratio and report a level), and carry the op1 alignment as a report line rather than a gate.

## How to read this

This is a scouting notebook in the shape of [ex-2.2.4](../ex-2.2.4/report.py): no hypotheses, no gates, no verdicts. Everything here comes from the checkpoints, probe set, metrics, and training trajectories that ex-2.2.9 stored. The cube figures also needed the answer distributions and residual states of the red lines, which ex-2.2.9 did not keep, so a scoring-only experiment ([`experiment.py`](experiment.py)) re-ran its projection read on the twenty `handover` checkpoints and stored those.

The reads use the vocabulary of ex-2.2.9, and the report for ex-2.2.9 has the [full glossary](../ex-2.2.9/report.py). A *red line* has a red operand, meaning redness at or above 0.8; that operand sits in slot op1 or op2. The *projection* removes the anchored axis from the residual stream at every slice and position.[^stream] *Kept* is the share of the clean expected exact match that survives the projection, over a group of lines. A *removal line* is a red line where setting the R channel of the red operand to zero moves the true answer by at least 0.4 in the unit cube; the removal gate of ex-2.2.9 wanted kept to fall under 0.2 on those.

Neither set matches what the training labeller used.[^labeller]

[^labeller]: Under the whole-line labeller, each of the three colors in a line draws a label at a small rate that rises steeply with its redness (4% per visit at pure red), and the line is labeled if any of them draws. So a red line is labeled on some of its visits, and a line with a red answer can earn the label too.

[^stream]: The *residual stream* is the running vector the transformer carries from block to block; each block reads it and adds to it. A *slice* is that vector at one depth, and a *position* is one token in the line.
"""

res = Results(
    m229=require(load_json(ex.EX229_METRICS_REF), ex.EX229_METRICS_REF),
    traj=require(load_json(ex.EX229_TRAJ_REF), ex.EX229_TRAJ_REF),
    probes=require(load_npz(ex.EX229_PROBE_REF), ex.EX229_PROBE_REF),
    metrics=load_json(ex.METRICS_REF),
    arrays=load_npz(ex.ARRAYS_REF),
)

r"""
## Removal on the order-sensitive ops

Ex-2.2.9 scored removal on eleven ops, and the projection cleared the gate on eight. The three that missed are the ones that read operand order: `hue-hsv`, `sat-hsv`, and `value-hsv` each take one HSV attribute from op2 and the other two from op1. On each the miss was one-sided by slot: with red at op1 the projection removed most of the answer on `sat-hsv` and `value-hsv`, and with red at op2 the answer largely survived. `hue-hsv` ran the other way.

Those removal lines were picked by the *to-zero* rule of ex-2.2.4: a red line counts if setting the R channel of the red operand to zero moves the true answer far. That rule says nothing about which attribute of red the answer actually needs, and it has a blind spot on pure red. Zeroing R on (5, 0, 0) gives black, which has no saturation and no value, so every `sat-hsv` and `value-hsv` line that takes only the saturation or value of red counts as a removal line.

### Three counterfactuals

Each is a different reading of what "losing red" should do to the red operand.
*To zero* sets its R channel to zero; this is the rule that picked the lines.
A *change of hue* replaces the operand by a permutation of its channels, keeping its saturation and value while moving its hue; a line survives when every permutation leaves the true answer within 0.4. A permutation reaches only six hues, a sixth of a turn apart, so this is a coarse hue rotation, and a finer one could behave differently between the hues it samples.
*To gray* replaces it by the gray of the same value, which keeps value and removes hue and saturation together.
"""

cf = {op: counterfactual_kept(res, op) for op in OPS}
cf_rows = []
for op in OPS:
    for slot, g in SLOTS:
        n_red = int((red_lines(res, op)["slot"] == slot).sum())
        p = cf[op][slot]
        cf_rows.append(
            [
                f"`{op}`, red at {g}",
                str(n_red),
                f"{p['to zero']:.2f}",
                f"{p['change of hue']:.2f}",
                f"{p['to gray']:.2f}",
                span2(res.kept("handover", op, f"red_{g}")),
                span2(res.kept("handover", op, f"removal_{g}")),
            ]
        )
table_html(
    [
        "Op and slot",
        "Red lines",
        "To zero",
        "Change of hue",
        "To gray",
        "Observed, red lines",
        "Observed, removal lines",
    ],
    cf_rows,
    "**Predicted and observed kept share, by op and red slot.** The three middle columns are the share of red lines whose true answer survives the counterfactual on the red operand (moves by less than 0.4). The last two are what the `handover` seeds kept under the projection, mean with the seed range, on the red lines and on the removal subset. The removal subset is the lines that fail the to-zero counterfactual, so the to-zero share there is zero by construction.",
)

# %%


@themed(
    name="counterfactuals",
    caption="**Kept share against the two counterfactuals that could differ from the rule, by op and red slot.** Each column is one op and slot. Dots are the kept share of the twenty `handover` seeds on the red lines under the projection, with the seed mean in the larger marker. Beside them: the share of those lines whose true answer would survive a change in the hue of the red operand (plus), or its replacement by gray of the same value (cross). To zero is the rule that picked the removal lines, and it predicts near zero everywhere.",
    alt_text="Chart of kept share by op and red slot. Observed kept shares track the change-of-hue prediction in six of eight columns; hue-hsv with red at op1 sits at a third where the hue prediction is one, and mix and value-hsv with red at op1 sit near zero under every prediction.",
)
def plot_cf() -> plt.Figure:
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 2.9), layout="constrained")
    xs, labels = [], []
    cfs = (("change of hue", "P", "#2b6cb0", "#7fb3ff"), ("to gray", "X", "#7b3fa0", "#cfa3ff"))
    for i, (op, (slot, g)) in enumerate(itertools.product(OPS, SLOTS)):
        x = i + (i // 2) * 0.5
        xs.append(x)
        labels.append(f"{op}\nred at {g}")
        dots(
            ax,
            x,
            res.kept("handover", op, f"red_{g}"),
            "handover",
            rng=rng,
            label="observed, red lines" if i == 0 else None,
        )
        for j, (name, m, lt, dk) in enumerate(cfs):
            ax.plot(
                x + 0.22 + 0.16 * j,
                cf[op][slot][name],
                m,
                ms=5,
                color=light_dark(lt, dk),
                mec=light_dark("white", "#111"),
                mew=0.5,
                zorder=4,
                label=name if i == 0 else None,
            )
    ax.set_xticks(xs, labels, fontsize=6.5)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("kept share")
    fig_legend(fig, ax)
    return fig


plot_cf()

# %%
hue = {(op, s): cf[op][s]["change of hue"] for op in OPS for s, _ in SLOTS}
obs = {(op, s): float(res.kept("handover", op, f"red_{g}").mean()) for op in OPS for s, g in SLOTS}
lost = max(obs[k] for k in (("mix", 0), ("mix", 2), ("hue-hsv", 2), ("sat-hsv", 0), ("value-hsv", 0)))
rf"""
The projection behaves like the change-of-hue counterfactual, and nothing like the other two. Where a change of hue leaves the answer alone ({hue[("sat-hsv", 2)]:.0%} of `sat-hsv` red-at-op2 lines, {hue[("value-hsv", 2)]:.0%} of `value-hsv` red-at-op2 lines) the seeds keep {obs[("sat-hsv", 2)]:.0%} and {obs[("value-hsv", 2)]:.0%}. Where it moves the answer (`mix` in either slot, `hue-hsv` with red at op2, `sat-hsv` and `value-hsv` with red at op1), the seeds keep {lost:.0%} or less.

To gray would have preserved `sat-hsv` red-at-op1 ({cf["sat-hsv"][0]["to gray"]:.0%}) and `hue-hsv` red-at-op2 ({cf["hue-hsv"][2]["to gray"]:.0%}). The model keeps neither, so the projection does not turn red into gray.

One slot the hue reading does not predict is `hue-hsv` with red at op1, whose answer is the hue of op2 at the saturation and value of red. A change of hue leaves every one of those answers alone, yet the seeds keep only {obs[("hue-hsv", 0)]:.0%}.

So when the red operand supplies saturation and value and something else supplies the hue, the projection costs the model most of that saturation and value read as well. The same loss shows up more mildly in the two slots that survive: `sat-hsv` and `value-hsv` red-at-op2 keep two thirds where the hue reading says all of it.

One reading is that the removal gate should count the lines whose answer needs the *hue* of the red operand, which is what the to-zero rule was meant to proxy. Under a change-of-hue rule, `sat-hsv` and `value-hsv` red-at-op2 drop out of the gate, and `hue-hsv` red-at-op1 drops out with them. What remains are the cases the projection already clears, and the rule rests on the six hues a permutation can reach.

What stays open is whether the partial loss of saturation and value comes from the projection itself or from this checkpoint.

### Where the answers go

The counterfactual table says which answers are lost. The next three figures show what the model answers instead, on the removal lines of each op and slot, from the twenty `handover` checkpoints. All three draw the RGB cube twice. The top row is the wheel view the probe-cube figures of ex-2.1.1 use: looking down the gray diagonal, so hue runs around the hexagon with red at the top, and lightness collapses onto the center. The row under it is the solid view: the cube turned so red points at the reader, which puts white at the top, black at the bottom, and lightness up the page. A move the wheel view hides, a color getting lighter or darker at the same hue, shows in the solid view; and a move along the red–cyan axis, which the solid view looks along, shows only in the wheel. Where a figure draws the moves themselves, it does so as a smoothed flow: an arrow for the mean move of the answers near it, with a faint wedge for their spread, so a wide wedge is a group of answers that went off in several directions.
"""

# The scoring pass is what the rest of the report reads; the retention and containment reads
# below need only ex-2.2.9's stored runs, and their prose renders with pending marks until the
# scoring has published.
if res.arrays is None or res.metrics is None:
    stop(RESULTS_TO_COME)

guess_pairs = answer_pairs(res)
# The two house views of the cube, one row each: down the gray diagonal, then red toward the reader.
VIEWS: tuple[ViewName, ...] = ("wheel", "solid")


def view_grid[G, C](
    groups: Sequence[G], cols: Sequence[C], *, figsize: tuple[float, float], stacked: bool = False
) -> tuple[plt.Figure, dict[tuple[ViewName, G, C], Axes]]:
    """The figure the removal-line figures share: one sub-figure per group, so the gap between groups is
    wider than the gap within (side by side, or *stacked*), with a row per view and a column per *cols*.
    Only the top row of a group carries titles; the row under it is the same panel seen from the side.
    """
    fig = plt.figure(figsize=figsize, layout="constrained")
    shape = (len(groups), 1) if stacked else (1, len(groups))
    subs = fig.subfigures(*shape, squeeze=False, hspace=0.06, wspace=0.08).ravel()
    axes = {}
    for group, sub in zip(groups, subs, strict=True):
        grid = sub.subplots(len(VIEWS), len(cols))
        for (i, view), (j, col) in itertools.product(enumerate(VIEWS), enumerate(cols)):
            axes[view, group, col] = grid[i, j]
    return fig, axes


def title(ax: Axes, view: ViewName, text: str) -> None:
    if view == VIEWS[0]:
        ax.set_title(text, fontsize=7, pad=8)  # padded clear of the top corner letter


def plot_guess(op: str) -> plt.Figure:
    fig, axes = view_grid([slot for slot, _ in SLOTS], PASSES, figsize=(8.4, 4.6))
    for view, (slot, g), pas in itertools.product(VIEWS, SLOTS, PASSES):
        ax = axes[view, slot, pas]
        pairs = guess_pairs[op, slot, pas]
        if not pairs:
            draw_cube_bound(ax, view, labels=True)
            title(ax, view, f"red at {g}, {pas}: no removal lines")
            continue
        keys = np.array(list(pairs))
        n = np.array([pairs[tuple(k)] for k in keys], float)
        off = int(n[keys[:, 1] < 0].sum())
        on = keys[:, 1] >= 0
        keys, n = keys[on], n[on]
        truth, guess = GRID_UNIT[keys[:, 0]], GRID_UNIT[keys[:, 1]]
        dia = 0.01 + 0.1 * np.sqrt(n / n.max())
        plot_rgb_cube(ax, guess, truth, s=0.01, diameter=dia, view=view, labels=True)
        if pas == "projection":
            # The solid view looks down the red–cyan axis, so a move along it leaves only jitter here.
            flow_arrows(ax, truth, guess, n, view=view, sigma=0.2, step=0.4, min_move=0.05 if view == "wheel" else 0.08)
        note = f", {off} off-vocab" if off else ""
        title(ax, view, f"red at {g}, {pas} ({int(n.sum())}{note})")
    return fig


figure_html(
    "".join(themed(plot_guess, name=f"guess-{op}", caption=f"`{op}`")(op) for op in OPS),
    caption="**Greedy answers on the removal lines, clean and under the projection.** One block per op; each pair of panels is one red slot, clean on the left and projected on the right, with the number of (line, seed) answers in the title. **Top row:** the wheel view of the RGB cube, down the gray diagonal, so hue runs around the hexagon and lightness collapses onto the center. **Bottom row:** the solid view, red toward the reader, so lightness runs up the panel from black (K) to white (W) and red and cyan fall inside. Corner letters name the cube's corners. Each mark is a greedy answer, placed at its own color and colored by the true answer, and sized by how many answers made that move; the clean panels show where the true answers lie, and a projected mark whose color matches its place is an answer that still matches the truth. The projected panels add the moves as a smoothed flow: each arrow is the mean move of the answers whose truth lies near its tail, sized by their count and colored by their mean truth; a cell whose answers go two ways gets an arrow each way, and the faint wedge behind an arrow spans one standard deviation of the directions it averages.",
    aria_label="Cube panels of greedy answers per op and red slot, clean beside projected. Clean, nearly every mark sits on its ring; projected, marks move off their rings wherever the answer needs the hue of red, and stay on them on sat-hsv and value-hsv with red at op2.",
)

r"""
Clean, the greedy answer is the true one on almost every removal line, so the clean panels are where the truth lies. The projected panels sort the ops, roughly, into three kinds of move.

On `mix` the projected answers move toward the center of the cube, and their mean is a gray with a little red left in it. Their hues stay near red: nine in ten sit within a third of a turn of it, which is why the green and blue half of the wheel stays empty. A `mix` answer is the midpoint of its two operands, and a red operand that reads as orange or pink after the projection cannot pull a midpoint to the far side of the wheel. The solid row of `mix` shows only jitter, because a move from red toward gray runs along the red–cyan axis, the one direction that view cannot show.

On `hue-hsv` with red at op1, the case no counterfactual predicted, the answers keep their hue and lose saturation. The true answers are the hue of op2 at the full saturation and value of red, so they sit on the rim of the wheel; projected, each moves straight in toward the center, and in the solid row the arrows run level toward the gray axis, so the answers get paler without getting darker. The model still reads the hue of op2 and returns it as a washed-out color; what the projection has cost it is the saturation read of the red operand.

The other four broken slots are the hue rotation. On `hue-hsv` with red at op2 the true answers are reds at the saturation and value of op1, and projected they split into an orange lobe and a pink lobe, at the same lightness as before; the flow draws two arrows from each cell for this reason, one to each side. `sat-hsv` and `value-hsv` with red at op1 fan out the same way, from red toward orange and pink, with wider wedges: these answers take two attributes of red, and the moves scatter as well as rotate. In the solid row of `value-hsv` the arrows point every way, so those answers seem to change in lightness as well as hue, with no one direction to it; that panel is the least tidy of the set, and the rotation is only the largest part of what it shows. On the two slots the projection leaves alone, `sat-hsv` and `value-hsv` with red at op2, the arrows are short and the marks stay on their clean positions.

The greedy answer is only one token. The whole answer distribution says how confidently the model moved, and whether the mass that left the true answer went to one color or spread out. The next figure draws that distribution as a dithered cloud in the cube.
"""

cloud_mass = answer_mass(res)


def plot_cloud(op: str) -> plt.Figure:
    rng = np.random.default_rng(1)
    fig, axes = view_grid([slot for slot, _ in SLOTS], PASSES, figsize=(8.4, 4.6))
    for view, (slot, g), pas in itertools.product(VIEWS, SLOTS, PASSES):
        ax = axes[view, slot, pas]
        m = cloud_mass[op, slot, pas]
        if m.sum() == 0:
            draw_cube_bound(ax, view, labels=True)
        else:
            cube_cloud(ax, m, view=view, rng=rng)
        title(ax, view, f"red at {g}, {pas}")
    return fig


figure_html(
    "".join(themed(plot_cloud, name=f"cloud-{op}", caption=f"`{op}`")(op) for op in OPS),
    caption="**The answer distribution on the removal lines, clean and under the projection.** One block per op; each pair of panels is one red slot, clean on the left and projected on the right, in the wheel view (top) and the solid view (bottom), as in the previous figure. The dots of each panel are shared out over the 216 grid colors in proportion to the mean answer mass those lines put on each color, so a dense patch is where the model expects the answer to be. The clean panels show where the true answers of those lines lie.",
    aria_label="Dithered cube clouds of answer mass per op and red slot, clean beside projected. Each clean cloud sits where the true answers are; projected, the cloud spreads over the whole wheel wherever red supplies the hue, and stays close to the clean one on sat-hsv and value-hsv with red at op2.",
)

# %%
peak = peakedness(res)
peak_rows = []
for op in OPS:
    for slot, g in SLOTS:
        c, p = peak[op, slot, "clean"], peak[op, slot, "projection"]
        peak_rows.append([f"`{op}`, red at {g}", f"{c[0]:.2f}", f"{p[0]:.2f}", f"{c[1]:.1f}", f"{p[1]:.1f}"])
# The cases the projection breaks, and the two it leaves alone, as (clean, projected) pairs.
broken_cases = (("mix", 0), ("mix", 2), ("hue-hsv", 2), ("sat-hsv", 0), ("value-hsv", 0))
broken = [(peak[op, slot, "clean"], peak[op, slot, "projection"]) for op, slot in broken_cases]
kept_cases = [
    (peak[op, slot, "clean"], peak[op, slot, "projection"]) for op, slot in (("sat-hsv", 2), ("value-hsv", 2))
]
table_html(
    ["Op and slot", "Top mass, clean", "Top mass, proj.", "Colors, clean", "Colors, proj."],
    peak_rows,
    "**How peaked the answer distribution of each line is, clean and under the projection.** Mean over the removal lines and the twenty seeds. *Top mass* is the probability on the most likely grid color. *Colors* is the effective number of colors, the exponential of the mean entropy: 1 for a certain answer, 216 for a uniform one.",
)

rf"""
Per line, the projected answer is unsure among a handful of colors rather than spread over the wheel. On the cases the projection breaks, the mass on the top color falls from about {np.mean([b[0][0] for b in broken]):.1f} clean to about {np.mean([b[1][0] for b in broken]):.1f}, and the effective number of colors rises from one or two to about {np.mean([b[1][1] for b in broken]):.0f}. On the two cases it leaves alone, the top mass stays at {min(k[1][0] for k in kept_cases):.1f} or above.

So the wheel-wide spread of the projected clouds is a spread across lines, each moved to its own neighbourhood. That is what we would expect if the operand reads as a hue rotated one way or the other, which is what the probe finds.

The solid row adds one thing the greedy answers did not show. On `hue-hsv` the projected mass reaches every lightness, from near black to near white, where the greedy answers of the previous figure kept the lightness of the truth. Read with the table above, one reading is that the runners-up a line hesitates among differ from its top answer in lightness as well as hue; the clouds pool the lines, so this is a guess about what is inside each one rather than a measurement.

The structure inside each cloud is the set of answers the op can produce on the grid, carried around the wheel by the rotation. Take `sat-hsv` with red at op1: the clean answers are reds of every saturation, a ray from white at the center out to red at the top, and projected that ray appears at every hue. With red at op2 the answers are fully saturated colors at the hue and value of op1, which in the wheel view are the rim and the spokes running in toward the center, and the projected panel keeps that skeleton.

### What the stream says

The answers show what the model concluded. The residual stream shows what it was working from. For each op we fit a linear probe from the stream at each slice and position to the three colors a line carries: op1, op2, and the answer the rule gives. The probes were fit on the clean stream of the non-red lines, and we then decoded the red lines through them, clean and under the projection.[^probes]

Fitting on the non-red lines keeps the axis out of the probes. A probe fit on lines with red in them would learn the axis as the direction of red, and would then read its removal as a loss of red at the embedding as much as anywhere. A probe blind to the axis reads what the blocks make of its absence, which is the question here. The red lines are outside the fit set, so they are held out without a leave-one-out scheme.

[^probes]: The probe-cube figures of ex-2.1.3 are the precedent. The probes read RGB, so their output lands in the cube with no rotation needed to fit it. A hue probe would be ill-posed, since hue is circular, and saturation and value are piecewise-linear in RGB, so we convert the decoded RGB whenever an HSV number is wanted.
"""

decoded = decoded_colors(res)


READS = ("operand", "answer")


def slice_name(sl: int) -> str:
    """Slice 0 is the embedding; each later slice is the stream after that many blocks."""
    return "emb" if sl == 0 else f"slice {sl}"


def plot_decoded(op: str) -> plt.Figure:
    n_sl = len(ex.SLICES)
    fig, axes = view_grid(READS, ex.SLICES, figsize=(1.75 * n_sl, 7.6), stacked=True)
    for view, what, sl in itertools.product(VIEWS, READS, ex.SLICES):
        ax = axes[view, what, sl]
        truth = decoded[op, f"{what}_truth"]
        d = np.clip(decoded[op, "projection", what][sl], -0.2, 1.2)
        ring = np.clip(decoded[op, "clean", what][sl], -0.2, 1.2)
        if what == "operand":
            # A few dozen lines that all start at red: one stub each is the legible picture. No rings,
            # since a ring at red under a red stub adds nothing.
            plot_rgb_cube(ax, d, truth, truth=ring, rings=False, s=5, view=view, labels=True)
        else:
            # Hundreds of lines all round the wheel: light marks, and the moves as a flow. The weights
            # are flat (one line per move); the lattice is finer than the answer figures' since the
            # moves are shorter.
            plot_rgb_cube(ax, d, truth, s=2, view=view, labels=True)
            flow_arrows(
                ax,
                ring,
                d,
                np.ones(len(d)),
                view=view,
                sigma=0.15,
                step=0.25,
                min_move=0.05 if view == "wheel" else 0.08,
            )
        title(ax, view, f"{'red operand' if what == 'operand' else 'answer at ='}, {slice_name(sl)}")
    return fig


figure_html(
    "".join(themed(plot_decoded, name=f"decoded-{op}", caption=f"`{op}`")(op) for op in OPS),
    caption="**Colors of the removal lines decoded from the residual stream, projected against clean.** One block per op, mean over the twenty seeds. **Top row:** the red operand, read at its own position by the probe fit at that slice. One mark per line, at the RGB decoded under the projection and colored by the true color of the operand, with a stub from the clean decode of the same line, so the stub is what the projection changed. **Bottom row:** the answer from the rule, read at `=` and colored by the true answer; there are too many lines for rings and stubs, so the moves from the clean decodes to the projected ones are drawn as a smoothed flow, on the same terms as the greedy-answer figure. Each read is shown in the wheel view and, under it, the solid view, as in the answer figures. *emb* is the embedding, and slice *n* is the stream after *n* blocks.",
    aria_label="Cube panels of probe-decoded colors across five slices, in wheel and solid views. The operand marks start at red and slide further from it with each slice, toward orange on one side and pink on the other, at nearly constant lightness; the answer flow points away from red for the reddish answers and gently inward elsewhere.",
)

# %%
probe_rows = []
for op in OPS:
    r2 = {
        k: np.array([s["ops"][op]["probe"]["r2_red"][k] for s in res.metrics["scores"]]).mean(0)
        for k in ("op1@op1", "op2@op2", "ans@=")
    }
    for k, v in r2.items():
        probe_rows.append([f"`{op}`, {k}", *(f"{x:.2f}" for x in v)])
table_html(
    ["Op and site", *(slice_name(s) for s in ex.SLICES)],
    probe_rows,
    "**Probe fit on the red lines, R² per slice, mean over seeds.** Each probe is a ridge fit on the clean stream of the non-red lines, and is read here on the clean stream of the red lines. `op1@op1` reads op1 at position 0, `op2@op2` reads op2 at position 2, and `ans@=` reads the raw answer from the rule at `=`. The negative fit of the answer probe at *emb* is expected: at the embedding, `=` has nothing of the line in it.",
)


# %%
def hsv(a: np.ndarray) -> np.ndarray:
    return np.array([colorsys.rgb_to_hsv(*np.clip(c, 0, 1)) for c in a])


# Where the decoded colors land, clean and under the projection, at the first block and the last:
# mean distance from the true color, circular hue offset, saturation, and value, over the removal lines.
hsv_rows = []
last = len(ex.SLICES) - 1
for op in OPS:
    for what, name in (("operand", "red operand"), ("answer", "answer at `=`")):
        truth = decoded[op, f"{what}_truth"]
        ht = hsv(truth)
        for sl in (1, last):
            cells = [f"`{op}`, {name}, {slice_name(sl)}"]
            for pas in PASSES:
                d = decoded[op, pas, what][sl]
                h = hsv(d)
                dh = np.abs(((h[:, 0] - ht[:, 0] + 0.5) % 1) - 0.5).mean()
                cells += [
                    f"{np.linalg.norm(d - truth, axis=1).mean():.2f}",
                    f"{dh:.2f}",
                    f"{h[:, 1].mean():.2f}",
                    f"{h[:, 2].mean():.2f}",
                ]
            hsv_rows.append(cells)
table_html(
    ["Op, site, slice", *(f"{pas} {q}" for pas in ("clean", "proj.") for q in ("move", "|ΔH|", "S", "V"))],
    hsv_rows,
    "**Where the decoded colors land, clean and under the projection.** Mean over the removal lines and the twenty seeds, at the first block and the last. *Move* is the distance from the decoded color to the true one in the unit RGB cube. *|ΔH|* is the hue offset from the true color, in turns, so 0.5 is the opposite hue. *S* and *V* are the decoded saturation and value; the true operands have both near 1.",
)

r"""
At the embedding, the projection changes nothing the probe can see. Clean and projected reads coincide there, on operand and answer alike, and the small offset both show is the shrinkage of the ridge fit. The probes are blind to the axis, so whatever the projection takes at the embedding becomes visible only once the blocks have acted on it.[^emb]

[^emb]: At the embedding the stream at `=` is the same vector on every line, since nothing of the line has reached that position yet. A probe there can only return one color for every answer: the mean of its fit set, near the center of the wheel.

From the first block on, the projected red operand reads as a different color, and the difference grows with depth. The move roughly doubles from slice 1 to slice 4 on every op. In HSV terms it is a rotation of hue away from red, a fifth of a turn by the last slice, at nearly full saturation and with a lower value. In the wheel view that is the stubs sliding along the upper edges of the hexagon, toward orange on one side and pink on the other, rather than toward the center. The solid view says how the value is lost. Red sits at the center of that view, and most stubs run level from it toward yellow or magenta, so those operands trade some red for green or blue and keep their lightness. The stubs colored pure red are the exception: the operands with no green or blue in them to start with move a little up, or on a couple of lines a long way down toward black, keeping the hue of red and losing value the plain way. The pattern is the same on every op.

So this is the change-of-hue counterfactual that the kept shares followed, with a loss of value alongside it. M1 saw a related asymmetry when it deleted the hue subspace of an autoencoder (ex-2.7 in [ex-preppy](https://github.com/z0u/ex-preppy), under its ablation figures): red, green, and blue darkened, and yellow, cyan, and magenta lightened. One reading of both is that the primaries sit nearer black on the gray diagonal than the secondaries do, so what remains of red once a hue direction is gone leans toward black. The few lines that drop toward black here fit that reading; the larger group that trades red for a neighbouring channel is a hue rotation with the value loss as a side effect.

The slots that take the saturation and value of the red operand have only what the stream keeps of it, so that lost value is the partial loss the counterfactual table could not explain.

The answer at `=` follows the operand. Clean, it reaches its ring by the last slice. Projected, it stops short, with a smaller hue offset than the operand and a saturation about two tenths under the clean read. In the answer rows of the figure the flow points away from red on the reddish answers and gently inward on the rest, and on `mix` it is the same straight move down the wheel, invisible in the solid view, that the greedy answers made. The answers that survive on `sat-hsv` and `value-hsv` with red at op2 are the ones whose true color the rotated, dimmer operand still snaps to.

## Retention and the anneal

The retention read of ex-2.2.9 asks whether the alignment survives the anneal of the anchor weight: every run whose line alignment peaks above 0.2 should end at 0.8 of that peak. One `handover` seed in twenty ended at 0.77, and the seed mean was 0.88 against 0.96 on the references.

The natural reading was that the anneal, over the last tenth of training, lets the alignment slip. The trajectories say otherwise.
"""

ret = retention_stats(res)


@themed(
    name="retention",
    caption="**Line alignment over training, and two retention ratios.** Left: line alignment against epoch for the three handover conditions, one faint line per seed with the seed mean drawn over them; the shaded band is the anneal window, where the anchor weight falls from 0.1 to its floor. Middle: the retention ratio ex-2.2.9 gated, the final alignment over its peak, per seed with the seed mean in the larger marker; the dashed rule is the gate and the hatched side misses it. Right: the final alignment over its value at the start of the anneal, the ratio that would measure the cost of the anneal alone, with the same gate level dotted for reference.",
    alt_text="Three panels. Left, alignment trajectories with faint per-seed lines under a bold mean: handover rises to about 0.75 by epoch 20 and drift down to about 0.66 before the anneal band begins at epoch 45, then stay flat; handover-slot and handover-tied peak later and drift less. Middle, end over peak: handover sits around 0.88 with one seed under the 0.8 gate, the other two conditions above 0.9. Right, end over the alignment at the anneal start: all three conditions sit at 1.0.",
)
def plot_ret() -> plt.Figure:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), layout="constrained", width_ratios=[2.2, 1, 1])
    ax = axes[0]
    conds = ("handover", "handover-slot", "handover-tied")
    for cond in conds:
        trajs = [res.traj[r["label"]]["traj"] for r in res.by_cond(cond)]
        ep = np.array(trajs[0]["epoch"])
        ml = np.array([np.interp(ep, t["epoch"], t["m_line"]) for t in trajs])
        for row in ml:
            ax.plot(ep, row, "-", color=ink(cond), lw=0.5, alpha=0.18, zorder=1)
        ax.plot(ep, ml.mean(axis=0), "-", color=ink(cond), lw=1.6, label=cond, zorder=3)
    a0 = float(np.mean([anneal_start(res, r["label"]) for r in res.by_cond("handover")]))
    ax.axvspan(
        a0,
        max(res.traj["handover-s0"]["traj"]["epoch"]),
        facecolor=light_dark("#000", "#fff"),
        alpha=0.06,
        lw=0,
        zorder=0,
    )
    ax.text(a0, 0.02, " anneal", fontsize=6.5, va="bottom", color=light_dark("#333", "#ccc"))
    ax.set_xlabel("epoch")
    ax.set_ylabel("line alignment m_line")
    ax.set_ylim(0, 0.85)
    gate = res.m229["design"]["gates"]["retention"]
    for ax, col, name in ((axes[1], (3, 0), "end ÷ peak"), (axes[2], (3, 2), "end ÷ at anneal start")):
        for i, cond in enumerate(conds):
            v = ret[cond][:, col[0]] / ret[cond][:, col[1]]
            dots(ax, i, v, cond, rng=rng)
        ax.set_xticks(range(len(conds)), [c.replace("handover-", "h.-") for c in conds], fontsize=6.5)
        ax.set_ylim(0.7, 1.02)
        ax.set_title(name, fontsize=8)
    gate_line(axes[1], gate, fail="below")
    axes[2].axhline(gate, color=light_dark("#333", "#ddd"), lw=0.7, ls=":", zorder=2)
    fig_legend(fig, axes[0])
    return fig


plot_ret()

# %%
h, s, t = (ret[c] for c in ("handover", "handover-slot", "handover-tied"))
peak_ep = {c: float(ret[c][:, 1].mean()) for c in ("handover", "handover-slot", "handover-tied")}
end_over_at = {c: float((ret[c][:, 3] / ret[c][:, 2]).mean()) for c in ("handover", "handover-slot", "handover-tied")}
rf"""
The anneal costs nothing. Over the anneal window the `handover` seeds end at {end_over_at["handover"]:.3f} of where they started it, and `handover-slot` and `handover-tied` at {end_over_at["handover-slot"]:.3f} and {end_over_at["handover-tied"]:.3f}. The whole of the drop the gate measured happens earlier.

`handover` climbs to a plateau by epoch 10. Its peak, at {h[:, 0].mean():.2f} and epoch {peak_ep["handover"]:.0f} on average, is just the high point of a noisy series. We cannot tell whether the high epochs are the ones with more red lines in their batches, since the stored trajectories do not record what each epoch drew; the corpus sampler is seeded, so a replay could tell.

By the start of the anneal the seeds sit at {h[:, 2].mean():.2f}, a few of them drifting down to the bottom of the band. The two references peak later (epoch {peak_ep["handover-slot"]:.0f} on `handover-slot`, {peak_ep["handover-tied"]:.0f} on `handover-tied`) because their plateaus are still rising, and they drift less, which is why their ratio comes out higher.

So the ratio of the end to the peak is comparing the last sample of a noisy plateau against the height of that plateau, with a slow drift under a constant anchor weight added on some seeds. That drift needs both the whole-line labeller and the untied readout: either one alone holds its plateau.

Whether the drift matters is a different question from the one the gate asked. The end-of-training alignment is {h[:, 3].mean():.2f} on `handover`, against {s[:, 3].mean():.2f} and {t[:, 3].mean():.2f} on the references. Every read downstream of it (grading, removal, containment) is taken at the end, so that level is what the later experiments inherit. A stepped anneal, which the [backlog item](/todo/science/retention-under-longer-training.md) floated, would act on the window where nothing is lost.

## Containment under the untied readout

Ex-2.2.9 reported ᾱ, the mean alignment of the non-red lines at op1, at 0.28 on `handover`. The reference from ex-2.2.3 read 0.1, and the old gate was set there. The report put the rise down to the untied readout, which is what the pilot in ex-2.2.7 had suggested.

But `handover` changed two things at once relative to the recipe of ex-2.2.3: the whole-line labeller as well as the readout. Ex-2.2.9 ran one arm with each change undone.
"""

cont = {c: containment_rows(res, c) for c in CONDS}


@themed(
    name="containment",
    caption="**Containment by condition.** Per seed, with the seed mean in the larger marker. Left: the mean alignment of the stream on the non-red lines with the anchor axis at op1, with the reference level from ex-2.2.3 dotted. Middle: the mean absolute axis component of the embedding rows for the 209 non-red color words. Right: the axis component of the `⏎` embedding row, with the same row of the untied readout as an open marker beside it where the condition has one.",
    alt_text="Three dot panels by condition. Alpha at op1: control near 0, handover-tied 0.16, handover-slot 0.18, handover 0.28, handover-narrow 0.30. Non-red rows: all conditions between 0.06 and 0.10. The newline row: near zero on control and handover-slot, 0.17 on handover, 0.27 on handover-tied.",
)
def plot_cont() -> plt.Figure:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.7), layout="constrained")
    panels = (
        ("alpha", "ᾱ at op1, non-red lines"),
        ("nonred", "|e₁ component|, non-red color rows"),
        ("eol", "e₁ component, ⏎ embedding row"),
    )
    for ax, (key, title) in zip(axes, panels, strict=True):
        for i, cond in enumerate(CONDS):
            dots(ax, i, cont[cond][key], cond, rng=rng, label=cond if key == "alpha" else None)
        ax.set_xticks(range(len(CONDS)), [c.replace("handover-", "h.-") for c in CONDS], fontsize=6.5)
        ax.set_title(title, fontsize=8)
    ref = res.m229["design"]["gates"]["mean_align_ref"]
    axes[0].axhline(ref, color=light_dark("#333", "#ddd"), lw=0.7, ls=":", zorder=2)
    axes[2].axhline(0, color=light_dark("#333", "#ddd"), lw=0.5, zorder=1)
    for ax in axes[2:]:
        ax.plot(
            [1.35],
            [np.nanmean(cont["handover"]["eol_readout"])],
            marker("handover"),
            ms=5,
            color=ink("handover"),
            mfc="none",
            zorder=4,
        )
        ax.plot(
            [2.35],
            [np.nanmean(cont["handover-slot"]["eol_readout"])],
            marker("handover-slot"),
            ms=5,
            color=ink("handover-slot"),
            mfc="none",
            zorder=4,
        )
    fig_legend(fig, axes[0])
    return fig


plot_cont()

# %%
a = {c: float(cont[c]["alpha"].mean()) for c in CONDS}
e = {c: float(cont[c]["eol"].mean()) for c in CONDS}
er = {c: float(np.nanmean(cont[c]["eol_readout"])) for c in ("handover", "handover-slot")}
nr = {c: float(cont[c]["nonred"].mean()) for c in CONDS}
rf"""
ᾱ at op1 is {a["handover"]:.2f} on `handover`, {a["handover-slot"]:.2f} with the slot labeller back, and {a["handover-tied"]:.2f} with the readout tied again. Undoing either change takes back about half the rise. So the rise belongs to the handover as a whole, and the untied readout is one of two contributors. `handover-narrow`, which has fewer lines per op, sits at {a["handover-narrow"]:.2f}.

Why either half raises it is open. For the labeller half there is a candidate: under the whole-line labeller a line can earn its label from its answer, and the pull then lands on every position of that line, op1 included. So a non-red operand is pulled toward the axis whenever its answer draws. For the readout half we have no candidate yet. The [backlog item](/todo/science/containment-rises-under-the-untied-readout.md) carries both.

Whatever the mechanism, none of it reaches the embedding table: the non-red color rows hold {nr["handover"]:.2f} of the axis on `handover` against {nr["control"]:.2f} on the control.

The `⏎` row is a separate matter, and a cleaner one. Its embedding component is {e["handover"]:.2f} on `handover` and {e["handover-slot"]:.2f} on `handover-slot`, so the whole-line labeller is what puts it there. The pull lands on every position of a red line, `⏎` included, and the slot labeller never touches that position.

The readout row for `⏎` is at {er["handover"]:.2f} on `handover` too. On `handover-tied` the row reads {e["handover-tied"]:.2f}, and so does `=`, which is the shared table doing double duty.

The backlog item on the [`⏎` row](/todo/science/eol-embedding-row-keeps-the-axis.md) asks whether the residual redness of the answer position is part of this. Answering that needs a per-position alignment on the red lines, which ex-2.2.9 did not store, so it stays open here.

## What we make of it

Three changes to the design of the re-run, and one read to carry.

**Removal lines by hue.** Define a removal line as a red line whose true answer moves by at least 0.4 under some channel permutation of the red operand. Those are the lines whose answer needs the *hue* of red. On `mix` and the eight other channel-wise ops this is the to-zero set, or close to it. On the three HSV ops it drops the slots that take only saturation or value (`sat-hsv` and `value-hsv` with red at op2), and `hue-hsv` with red at op1 goes with them.

The gate stays at kept under 0.2, and on the lines that remain every `handover` seed already clears it. The permutation rule reaches six hues; if a finer rotation of the operand in HSV, snapped to the grid, ever disagrees with it, the finer one is the rule to keep.

The saturation- and value-taking slots become a second read, with no gate. The kept share there says how much of the saturation and value of the red operand the projection takes with it, which is a cost of the operator worth reporting.

**Retention against the start of the anneal.** The retention read keeps its name and changes its denominator, to the final alignment over the alignment at the start of the anneal, gated at 0.8 as before. That is the question the read was written to ask.

A high percentile of the plateau, in place of the peak, would soften the ratio and keep the drift inside it; the anneal-start denominator takes the drift out and leaves it to the level line. The early peak and the drift get a line of their own in the report: the level at the end of training, beside the references.

**The op1 alignment as a line rather than a gate.** ᾱ at op1 rises under either half of the handover, and the embedding rows do not move, so the old reference of 0.1 belonged to the old grammar. The re-run reports ᾱ at op1 with `handover-slot` and `handover-tied` as its references, and gates nothing on it. A gate can return once we can name a mechanism.

**The `⏎` row** stays as it is, with its backlog item open. The whole-line labeller is what puts it there, and the slot labeller is not coming back for it. If the edits in M3 ever touch the answer position, this row is the first place to look.

One question this notebook could not settle: whether the partial loss of saturation and value under the projection comes from the operator or from this checkpoint. Answering it would take the same read on `handover-tied` and `handover-slot`, which the scoring pass can run at no design cost, so it is worth adding to the scoring for the re-run.
"""

design = ex.design()
n_scored = len(res.labels)
rf"""
## Method

**What ran.** A scoring-only experiment over the {len(design["seeds"])} `{design["condition"]}` checkpoints of ex-2.2.9 ({n_scored} scored so far), on the ops {", ".join(f"`{o}`" for o in design["ops"])}, using the probe set of ex-2.2.9. For each checkpoint and op, the task ran two forward passes over every probe line: one clean, and one with the `{design["operator"]}` operator applied at every slice and position. For the red lines it stored the answer distribution over the 216 grid colors, the greedy answer, and the expected exact match under both passes. Its per-group kept shares reproduce those of ex-2.2.9 to the third decimal, which is our check that it ran the same read.

**Probes.** One ridge regression (λ = {design["probe_l2"]:g}) per slice, position, and target, from the {res.width}-wide residual state to a color in the unit cube, fit on the clean stream of the non-red lines and read on the red lines. The three targets are op1, op2, and the raw (unrounded) answer the rule gives. The sites read in the figures are each operand at its own position, and the answer at `=` (position {design["decode_pos"]}). We store the decoded coordinates of every red line under both passes for all seeds, and the raw states for the first {len(design["state_seeds"])} seeds.

**Counterfactuals.** For each red line, the red operand is the one with the higher redness. *To zero* sets its R channel to zero. *Change of hue* takes the five other permutations of its channels, and the line survives if all five move the true answer by less than {design["far_move"]}. *To gray* replaces it by the gray of its value. Answers are the snapped answers of the op, and distances are in the unit cube, the same as `to_zero_move` in ex-2.2.9.

**Retention.** From the trajectories stored by ex-2.2.9 (100 points over training). The anneal starts at the first point where the anchor weight is under 0.99 of its plateau, and the alignment at that start is the last point before it.

**Containment.** From the run metrics stored by ex-2.2.9: ᾱ at op1 as defined there, plus the axis component of each embedding row and, where the readout is untied, each readout row.
"""
