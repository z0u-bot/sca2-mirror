import marimo

__generated_with = "0.24.0"
app = marimo.App(
    app_title="Ex 2.2.10: three reads before the handover re-run",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import inspect
    import itertools
    import json
    import tempfile
    from collections import Counter
    from dataclasses import dataclass
    from pathlib import Path

    # The constants come from `experiment.py` beside this notebook (Marimo puts the notebook
    # directory on sys.path), which in turn binds ex-2.2.9's grammar and gates.
    import experiment as ex
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.axes import Axes

    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import figure_html, light_dark, themed
    from sca.data.colors import redness
    from sca.data.ops import CANDIDATE_BY_NAME, OP_BY_NAME, TOP, vocabulary
    from sca.vis import draw_cube_bound, plot_rgb_cube, project_cube

    use_publisher(report_bundle(__file__))

    RESULTS_TO_COME = "/// admonition | TODO\n    type: warning\nResults to come.\n///"
    """What a result cell shows while the run has not published yet."""

    OPS = ex.OPS_READ
    SLOTS = ((0, "op1"), (2, "op2"))
    """The red operand's position and the group name ex-2.2.9 scores it under."""
    CONDS = ("control", "handover", "handover-slot", "handover-tied", "handover-narrow")
    ALL_OPS = OP_BY_NAME | CANDIDATE_BY_NAME
    GRID = [ex.PALETTE[n] for n in ex.PALETTE]
    """The 216 grid colors in palette order, which is the order the stored palette indices use."""
    GRID_UNIT = np.asarray(GRID, float) / TOP
    RED_WORDS = {n for n in ex.PALETTE if redness(ex.PALETTE[n]) >= ex.RED_DOSE}
    """The seven grid colors with redness at or above the red dose."""

    # A cell renders its last expression, and a trailing docstring is one.
    None


@app.function(hide_code=True)
def load_json(ref: str) -> dict | None:
    """A published JSON result as a dict, or None before it exists."""
    store = project_store()
    art = store.get_refs([ref])[ref]
    if art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "data.json")])
        return json.loads(path.read_text())


@app.function(hide_code=True)
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


@app.function(hide_code=True)
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


@app.function(hide_code=True)
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


@app.function(hide_code=True)
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


@app.function(hide_code=True)
def fig_legend(fig: plt.Figure, ax: Axes, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7, **kwargs)


@app.function(hide_code=True)
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


@app.function(hide_code=True)
def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


@app.function(hide_code=True)
def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    """An authored result table in the shared report style; the first column is text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{cell_html(c)}</td>" for i, c in enumerate(row)) + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=mo.md(caption).text, class_="report-figure")


@app.function(hide_code=True)
def span2(v, digits: int = 2) -> str:
    """A seed mean with its range, as `0.51 (0.4–0.6)`."""
    v = np.asarray(v, float)
    if v.size == 0 or np.all(np.isnan(v)):
        return "—"
    return f"{np.nanmean(v):.{digits}f} ({np.nanmin(v):.{digits}f}–{np.nanmax(v):.{digits}f})"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.2.10: three reads before the handover re-run

    /// tip |
    <!-- tl;dr -->
    Ex-2.2.9 moved the anchor onto the larger grammar and left three loose ends. Removal was one-sided on the three HSV ops, one seed in twenty fell under the retention gate, and the non-red alignment at op1 sat above the old reference. This notebook reads each off the stored runs, with one small scoring pass for the cube figures and no new training.

    The removal miss comes from the rule we used to pick the lines: the projection acts like a change of hue, and the lines that missed are the ones whose answer takes only the saturation or value of the red operand. The retention drop happens before the anneal, so the anneal is not the cause. The op1 alignment rises under either half of the handover, so it belongs to the new grammar rather than to the readout alone. We propose the re-run.
    ///

    ## Observations

    Each line below is a read on the stored runs, with the number it rests on. None of them is a result; ex-2.2.11 will adopt what it needs from here and score it at fresh seeds.

    - [Removal](#removal-on-the-order-sensitive-ops): on the three HSV ops the kept share follows a hue-change counterfactual, and not the to-zero rule we used to pick the removal lines. Where the answer takes only the red operand's saturation or value (`sat-hsv` and `value-hsv` with red at op2), two thirds of it survives the projection. Where the answer takes red's hue, or the whole color, it goes. The one slot no counterfactual predicts is `hue-hsv` with red at op1: the answer needs only red's saturation and value, and a third survives. A linear probe on the stream reads the projected red operand as a hue rotated away from red at lower value, and the rotation grows block by block.
    - [Retention](#retention-and-the-anneal): on `handover` the alignment reaches a noisy plateau by epoch 10, and its peak is just the high point of that noise. By the time the anneal begins at epoch 45 the seeds sit at 0.66, a few of them drifting. Through the anneal itself every condition is flat. So the gate is reading the noise in the plateau and the drift. `handover-slot` and `handover-tied` hold their plateaus, which puts the drift down to the whole-line labeller on the untied readout.
    - [Containment](#containment-under-the-untied-readout): ᾱ at op1 on the non-red lines is 0.28 on `handover`, 0.18 on `handover-slot`, and 0.16 on `handover-tied`, against 0.02 on the control. Each half of the handover raises it, so the untied readout is not the whole story. The axis component of the `⏎` embedding row is a whole-line effect: 0.17 on `handover` and zero on `handover-slot`.
    - [What we make of it](#what-we-make-of-it): score removal on the lines whose answer takes the hue of the red operand, judge retention against the alignment at the start of the anneal (or drop the ratio and report a level), and carry the op1 alignment as a report line rather than a gate.

    ## How to read this

    This is a scouting notebook in the shape of [ex-2.2.4](../ex-2.2.4/report.py): no hypotheses, no gates, no verdicts. Everything here comes from the checkpoints, probe set, metrics, and training trajectories that ex-2.2.9 stored. The cube figures also needed the answer distributions and residual states of the red lines, which ex-2.2.9 did not keep, so a scoring-only experiment ([`experiment.py`](experiment.py)) re-ran its projection read on the twenty `handover` checkpoints and stored those. No model was trained.

    The reads use the vocabulary of ex-2.2.9. A *red line* has a red operand, meaning redness at or above 0.8; that operand sits in slot op1 or op2. The *projection* removes the anchored axis from the residual stream at every slice and position.[^stream] *Kept* is the share of the clean expected exact match that survives the projection, over a group of lines. A *removal line* is a red line where setting the R channel of the red operand to zero moves the true answer by at least 0.4 in the unit cube; the removal gate of ex-2.2.9 wanted kept to fall under 0.2 on those. The report for ex-2.2.9 has the [full glossary](../ex-2.2.9/report.py).

    [^stream]: The *residual stream* is the running vector the transformer carries from block to block; each block reads it and adds to it. A *slice* is that vector at one depth, and a *position* is one token in the line.
    """)
    return


@app.class_definition(hide_code=True)
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


@app.function(hide_code=True)
def require[T](value: T | None, ref: str) -> T:
    """An ex-2.2.9 input the notebook cannot do without."""
    if value is None:
        raise FileNotFoundError(f"ex-2.2.9 result {ref!r} is not in the store; this notebook reads its runs")
    return value


@app.cell(hide_code=True)
def _():
    res = Results(
        m229=require(load_json(ex.EX229_METRICS_REF), ex.EX229_METRICS_REF),
        traj=require(load_json(ex.EX229_TRAJ_REF), ex.EX229_TRAJ_REF),
        probes=require(load_npz(ex.EX229_PROBE_REF), ex.EX229_PROBE_REF),
        metrics=load_json(ex.METRICS_REF),
        arrays=load_npz(ex.ARRAYS_REF),
    )
    return (res,)


@app.function(hide_code=True)
def tok2color() -> np.ndarray:
    """Token id → palette index on ex-2.2.9's vocabulary (the tokenizer sorts its words), −1 at syntax."""
    vocab = sorted({""} | set(vocabulary(ex.TABLE)))
    names = list(ex.PALETTE)
    out = np.full(len(vocab), -1)
    for i, w in enumerate(vocab):
        if w in ex.PALETTE:
            out[i] = names.index(w)
    return out


@app.function(hide_code=True)
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Removal on the order-sensitive ops

    Ex-2.2.9 scored removal on eleven ops, and the projection cleared the gate on eight. The three that missed are the ones that read operand order: `hue-hsv`, `sat-hsv`, and `value-hsv` each take one HSV attribute from op2 and the other two from op1. On each the miss was one-sided by slot: with red at op1 the projection removed most of the answer on `sat-hsv` and `value-hsv`, and with red at op2 the answer largely survived. `hue-hsv` ran the other way.

    Those removal lines were picked by the *to-zero* rule of ex-2.2.4: a red line counts if setting the R channel of the red operand to zero moves the true answer far. That rule says nothing about which attribute of red the answer actually needs, and it has a blind spot on pure red. Zeroing R on (5, 0, 0) gives black, which has no saturation and no value, so every `sat-hsv` and `value-hsv` line that takes only the saturation or value of red counts as a removal line.

    ### Three counterfactuals

    Each is a different reading of what "losing red" should do to the red operand.
    *To zero* sets its R channel to zero; this is the rule that picked the lines.
    A *change of hue* replaces the operand by a permutation of its channels, which keeps its saturation and value and moves its hue; a line survives when every permutation leaves the true answer within 0.4.
    *To gray* replaces it by the gray of the same value, which keeps value and removes hue and saturation together.
    """)
    return


@app.function(hide_code=True)
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


@app.cell(hide_code=True)
def _(res):
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
    cf_table = table_html(
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
    return cf, cf_table


@app.cell(hide_code=True)
def _(cf, res):
    @themed
    def _plot():
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

    fig_cf = figure_html(
        _plot(),
        caption="**Kept share against the two counterfactuals that could differ from the rule, by op and red slot.** Each column is one op and slot. Dots are the kept share of the twenty `handover` seeds on the red lines under the projection, with the seed mean in the larger marker. Beside them: the share of those lines whose true answer would survive a change in the hue of the red operand (plus), or its replacement by gray of the same value (cross). To zero is the rule that picked the removal lines, and it predicts near zero everywhere.",
        aria_label="Chart of kept share by op and red slot. Observed kept shares track the change-of-hue prediction in six of eight columns; hue-hsv with red at op1 sits at a third where the hue prediction is one, and mix and value-hsv with red at op1 sit near zero under every prediction.",
    )
    return (fig_cf,)


@app.cell(hide_code=True)
def _(cf, cf_table, fig_cf, res):
    def _text():
        hue = {(op, s): cf[op][s]["change of hue"] for op in OPS for s, _ in SLOTS}
        obs = {(op, s): float(res.kept("handover", op, f"red_{g}").mean()) for op in OPS for s, g in SLOTS}
        lost = max(obs[k] for k in (("mix", 0), ("mix", 2), ("hue-hsv", 2), ("sat-hsv", 0), ("value-hsv", 0)))
        return inspect.cleandoc(f"""
        The projection behaves like the change-of-hue counterfactual, and nothing like the other two. Where a change of hue leaves the answer alone ({hue[("sat-hsv", 2)]:.0%} of `sat-hsv` red-at-op2 lines, {hue[("value-hsv", 2)]:.0%} of `value-hsv` red-at-op2 lines) the seeds keep {obs[("sat-hsv", 2)]:.0%} and {obs[("value-hsv", 2)]:.0%}. Where it moves the answer (`mix` in either slot, `hue-hsv` with red at op2, `sat-hsv` and `value-hsv` with red at op1), the seeds keep {lost:.0%} or less.

        To gray would have preserved `sat-hsv` red-at-op1 ({cf["sat-hsv"][0]["to gray"]:.0%}) and `hue-hsv` red-at-op2 ({cf["hue-hsv"][2]["to gray"]:.0%}). The model keeps neither, so the projection does not turn red into gray.

        One slot the hue reading does not predict is `hue-hsv` with red at op1, whose answer is the hue of op2 at the saturation and value of red. A change of hue leaves every one of those answers alone, yet the seeds keep only {obs[("hue-hsv", 0)]:.0%}.

        So when the red operand supplies saturation and value and something else supplies the hue, the projection costs the model most of that saturation and value read as well. The same loss shows up more mildly in the two slots that survive: `sat-hsv` and `value-hsv` red-at-op2 keep two thirds where the hue reading says all of it.

        That tells us what the removal gate should have counted. A removal line is one whose answer needs the *hue* of the red operand, and the to-zero rule is a proxy that fails on pure red. Under the change-of-hue rule, `sat-hsv` and `value-hsv` red-at-op2 drop out of the gate, and `hue-hsv` red-at-op1 drops out with them. The slots that remain are the ones the projection already clears.

        What stays open is whether the partial loss of saturation and value comes from the projection itself or from this checkpoint.
        """)

    mo.md(cf_table + "\n\n" + fig_cf + "\n\n" + _text())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Where the answers go

    The counterfactual table says which answers are lost. The next three figures say what the model answers instead, on the removal lines of each op and slot, from the twenty `handover` checkpoints. All three use the same view of the color cube, the one the probe-cube figures of ex-2.1.1 use: looking down the gray diagonal, so hue runs around the hexagon with red at the top and lightness collapses onto the center.
    """)
    return


@app.function(hide_code=True)
def removal_mask(res: Results, label: str, op: str, slot: int) -> np.ndarray:
    """The removal lines with the red operand in *slot*, in the stored red-line order."""
    return res.arr(label, f"{op}/removal") & (res.arr(label, f"{op}/red_operand") == slot)


@app.cell(hide_code=True)
def _(res):
    def _panels():
        # The stored greedy answer is a token id over the whole vocabulary; a syntax token is an
        # off-vocab answer, counted under the palette index −1.
        t2c = tok2color()
        out = {}
        for op in OPS:
            for slot, _g in SLOTS:
                pairs: Counter = Counter()
                for lb in res.labels:
                    m = removal_mask(res, lb, op, slot)
                    truth = res.arr(lb, f"{op}/ans_idx")[m]
                    guess = t2c[res.arr(lb, f"{op}/projection/guess")[m]]
                    pairs.update(zip(truth.tolist(), guess.tolist(), strict=True))
                out[op, slot] = pairs
        return out

    guess_pairs = None if res.arrays is None else _panels()
    return (guess_pairs,)


@app.cell(hide_code=True)
def _(guess_pairs):
    def _plot(op: str):
        fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.4), layout="constrained")
        for ax, (slot, g) in zip(axes, SLOTS, strict=True):
            pairs = guess_pairs[op, slot]
            if not pairs:
                draw_cube_bound(ax, "wheel")
                ax.set_title(f"red at {g}: no removal lines", fontsize=7)
                continue
            keys = np.array(list(pairs))
            n = np.array([pairs[tuple(k)] for k in keys], float)
            off = int(n[keys[:, 1] < 0].sum())
            on = keys[:, 1] >= 0
            keys, n = keys[on], n[on]
            truth, guess = GRID_UNIT[keys[:, 0]], GRID_UNIT[keys[:, 1]]
            plot_rgb_cube(ax, guess, truth, truth=truth, diameter=0.04 + 0.16 * np.sqrt(n / n.max()), view="wheel")
            note = f", {off} off-vocab" if off else ""
            ax.set_title(f"red at {g} ({int(n.sum())} answers{note})", fontsize=7)
        return fig

    fig_guess = (
        RESULTS_TO_COME
        if guess_pairs is None
        else figure_html(
            "".join(figure_html(themed(_plot, name=f"guess-{op}")(op), caption=f"`{op}`") for op in OPS),
            caption="**Greedy answers under the projection, on the removal lines.** One row per op, one panel per red slot; wheel view of the cube (hue around the hexagon, red at the top, lightness collapsed). Each mark is a greedy answer placed at its own color in the cube and colored by the true answer, with an open ring at the true answer and a stub between them; marks are sized by the number of (line, seed) pairs that made that move. Marks on their rings are answers the projection left alone.",
            aria_label="Cube panels of greedy answers under the projection, per op and red slot. On mix the marks sit below their rings, toward the gray center. Where red supplies the hue (hue-hsv with red at op2, sat-hsv and value-hsv with red at op1) the marks spread over every hue of the wheel. On hue-hsv with red at op1 the rings sit on the rim and many marks have moved inward from them. On sat-hsv and value-hsv with red at op2 most marks sit on their rings.",
        )
    )
    mo.md(fig_guess)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The greedy answer is only one token. The whole answer distribution says how confidently the model moved, and whether the mass that left the true answer went to one color or spread out. The next figure draws that distribution as a dithered cloud in the cube.
    """)
    return


@app.function(hide_code=True)
def cube_cloud(ax: Axes, mass: np.ndarray, *, n: int = 2500, rng, s: float = 3.0) -> None:
    """A dithered cloud in the wheel view: *n* dots shared out over the grid colors in proportion to *mass*,
    each jittered within its color's cell and drawn in that color.
    """
    draw_cube_bound(ax, "wheel")
    counts = rng.multinomial(n, mass / mass.sum())
    idx = np.repeat(np.arange(len(mass)), counts)
    rgb = GRID_UNIT[idx] + rng.uniform(-0.5, 0.5, (len(idx), 3)) / TOP
    xy = project_cube(rgb, "wheel")
    order = rng.permutation(len(idx))
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


@app.cell(hide_code=True)
def _(res):
    def _mass():
        out = {}
        for op in OPS:
            for slot, _ in SLOTS:
                for pas in ("clean", "projection"):
                    acc, k = np.zeros(len(GRID)), 0
                    for lb in res.labels:
                        m = removal_mask(res, lb, op, slot)
                        if m.any():
                            acc += res.arr(lb, f"{op}/{pas}/mass")[m].astype(float).sum(0)
                            k += int(m.sum())
                    out[op, slot, pas] = acc / max(k, 1)
        return out

    cloud_mass = None if res.arrays is None else _mass()
    return (cloud_mass,)


@app.cell(hide_code=True)
def _(cloud_mass):
    def _plot(op: str):
        rng = np.random.default_rng(1)
        fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.3), layout="constrained")
        for i, ((slot, g), pas) in enumerate(itertools.product(SLOTS, ("clean", "projection"))):
            ax = axes[i]
            m = cloud_mass[op, slot, pas]
            if m.sum() == 0:
                draw_cube_bound(ax, "wheel")
            else:
                cube_cloud(ax, m, rng=rng)
            ax.set_title(f"red at {g}, {pas}", fontsize=7)
        return fig

    fig_cloud = (
        RESULTS_TO_COME
        if cloud_mass is None
        else figure_html(
            "".join(figure_html(themed(_plot, name=f"cloud-{op}")(op), caption=f"`{op}`") for op in OPS),
            caption="**The answer distribution on the removal lines, clean and under the projection.** One row per op; each pair of panels is one red slot, clean on the left and projected on the right. Wheel view of the cube. The dots of each panel are shared out over the 216 grid colors in proportion to the mean answer mass those lines put on each color, so a dense patch is where the model expects the answer to be; the clean panels are where the true answers of those lines lie.",
            aria_label="Dithered cube clouds of answer mass per op and red slot, clean beside projected. Each clean cloud sits where the true answers are: the red wedge at the top where red supplies the hue, the rim on hue-hsv with red at op1, and the whole wheel on sat-hsv and value-hsv with red at op2. Where red supplies the hue the projected cloud spreads over the whole wheel; on hue-hsv with red at op1 it fills the interior from the rim; on sat-hsv and value-hsv with red at op2 it stays close to the clean one.",
        )
    )
    mo.md(fig_cloud)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### What the stream says

    The answers show what the model concluded. The residual stream shows what it was working from. For each op we fit a linear probe from the stream at each slice and position to the three colors a line carries: op1, op2, and the answer the rule gives. The probes were fit on the clean stream of the non-red lines, and we then decoded the red lines through them, clean and under the projection. The probe-cube figures of ex-2.1.3 are the precedent.

    The probes read RGB, so their output lands in the cube with no rotation needed to fit it. A hue probe would be ill-posed, since hue is circular, and saturation and value are piecewise-linear in RGB, so we convert the decoded RGB whenever an HSV number is wanted.
    """)
    return


@app.cell(hide_code=True)
def _(res):
    def _decoded():
        """Mean over seeds of the decoded coordinates on the removal lines, per op: the red operand at its own
        position and the answer at `=`, for both passes, per slice.
        """
        out = {}
        for op in OPS:
            dec = {
                pas: res.stacked(f"{op}/{pas}/decoded").astype(float) for pas in ("clean", "projection")
            }  # (S, L1, T, 3, N, 3)
            lb0 = res.labels[0]
            tok = res.arr(lb0, f"{op}/tokens")
            ans = res.arr(lb0, f"{op}/ans_idx")
            red_slot = res.arr(lb0, f"{op}/red_operand")
            removal = res.arr(lb0, f"{op}/removal")
            rows = np.flatnonzero(removal)
            # The red operand at its own position: target index 0 at position 0, or 1 at position 2.
            pos = red_slot[rows]
            tgt = np.where(pos == 0, 0, 1)
            for pas in ("clean", "projection"):
                d = dec[pas].mean(0)  # (L1, T, 3, N, 3)
                out[op, pas, "operand"] = np.stack(
                    [d[:, pos[i], tgt[i], rows[i]] for i in range(len(rows))], axis=1
                )  # (L1, n, 3)
                out[op, pas, "answer"] = d[:, ex.DECODE_POS, 2, rows]  # (L1, n, 3)
            out[op, "operand_truth"] = GRID_UNIT[tok[rows, pos]]
            out[op, "answer_truth"] = GRID_UNIT[ans[rows]]
        return out

    decoded = None if res.arrays is None else _decoded()
    return (decoded,)


@app.cell(hide_code=True)
def _(decoded):
    def _plot(op: str):
        n_sl = len(ex.SLICES)
        fig, axes = plt.subplots(2, n_sl, figsize=(1.75 * n_sl, 3.9), layout="constrained")
        for r, what in enumerate(("operand", "answer")):
            truth = decoded[op, f"{what}_truth"]
            for s in range(n_sl):
                ax = axes[r, s]
                d = np.clip(decoded[op, "projection", what][s], -0.2, 1.2)
                plot_rgb_cube(ax, d, truth, truth=truth, s=5, view="wheel")
                ax.set_title(f"{'red operand' if what == 'operand' else 'answer at ='}, slice {s}", fontsize=7)
        return fig

    fig_dec = (
        RESULTS_TO_COME
        if decoded is None
        else figure_html(
            "".join(figure_html(themed(_plot, name=f"decoded-{op}")(op), caption=f"`{op}`") for op in OPS),
            caption="**Colors of the removal lines decoded from the residual stream under the projection.** One block per op. Top row: the red operand read at its own position by the probe fit at that slice, one mark per line at the decoded RGB, colored by the true color of the operand, with a ring at that true color and a stub between; mean over the twenty seeds. Bottom row: the answer from the rule read at `=`, colored by the true answer. Wheel view; slice 0 is the embedding and each later slice is the stream after one more block. At slice 0 the `=` position has seen nothing of the line yet, so every answer decodes to the probe's mean at the center.",
            aria_label="Cube panels of probe-decoded colors under the projection, per op, across five slices. Top rows: at slice 0 the marks sit just off their rings at the red corner; from slice 1 they slide along the hexagon's upper edges away from red, toward orange on one side and pink on the other, further with each slice. Bottom rows: at slice 0 every answer decodes to the center; from slice 1 the answers spread toward their rings and stop short of them, inside the hexagon.",
        )
    )
    mo.md(fig_dec)
    return


@app.cell(hide_code=True)
def _(res):
    def _table():
        rows = []
        for op in OPS:
            r2 = {
                k: np.array([s["ops"][op]["probe"]["r2_red"][k] for s in res.metrics["scores"]]).mean(0)
                for k in ("op1@op1", "op2@op2", "ans@=")
            }
            for k, v in r2.items():
                rows.append([f"`{op}`, {k}", *(f"{x:.2f}" for x in v)])
        return table_html(
            ["Op and site", *(f"slice {s}" for s in ex.SLICES)],
            rows,
            "**Probe fit on the red lines, R² per slice, mean over seeds.** Each probe is a ridge fit on the clean stream of the non-red lines, and is read here on the clean stream of the red lines. `op1@op1` reads op1 at position 0, `op2@op2` reads op2 at position 2, and `ans@=` reads the raw answer from the rule at `=`. The negative fit of the answer probe at slice 0 is expected: at the embedding, `=` has nothing of the line in it.",
        )

    probe_table = RESULTS_TO_COME if res.metrics is None else _table()
    mo.md(probe_table)
    return


@app.cell(hide_code=True)
def _(decoded):
    def _hsv_table():
        """Where the decoded colors land, clean and under the projection, at the first block and the last: mean
        distance from the true color, circular hue offset, saturation, and value, over the removal lines.
        """
        import colorsys

        def hsv(a):
            return np.array([colorsys.rgb_to_hsv(*np.clip(c, 0, 1)) for c in a])

        rows = []
        last = len(ex.SLICES) - 1
        for op in OPS:
            for what, name in (("operand", "red operand"), ("answer", "answer at `=`")):
                truth = decoded[op, f"{what}_truth"]
                ht = hsv(truth)
                for sl in (1, last):
                    cells = [f"`{op}`, {name}, slice {sl}"]
                    for pas in ("clean", "projection"):
                        d = decoded[op, pas, what][sl]
                        h = hsv(d)
                        dh = np.abs(((h[:, 0] - ht[:, 0] + 0.5) % 1) - 0.5).mean()
                        cells += [
                            f"{np.linalg.norm(d - truth, axis=1).mean():.2f}",
                            f"{dh:.2f}",
                            f"{h[:, 1].mean():.2f}",
                            f"{h[:, 2].mean():.2f}",
                        ]
                    rows.append(cells)
        head = ["Op, site, slice", *(f"{pas} {q}" for pas in ("clean", "proj.") for q in ("move", "|ΔH|", "S", "V"))]
        return table_html(
            head,
            rows,
            "**Where the decoded colors land, clean and under the projection.** Mean over the removal lines and the twenty seeds, at the first block and the last. *Move* is the distance from the decoded color to the true one in the unit cube; *|ΔH|* is the hue offset from the true color in turns (0.5 is the opposite hue); *S* and *V* are the decoded saturation and value. The true operands have S and V near 1.",
        )

    def _text():
        if decoded is None:
            return RESULTS_TO_COME
        return (
            _hsv_table()
            + "\n\n"
            + inspect.cleandoc("""
    At the embedding, the projection changes nothing the probe can see. Clean and projected reads coincide at slice 0, on operand and answer alike, and the small offset both show is the shrinkage of the ridge fit. The probes were fit on the non-red lines, whose stream has little of the axis in it, so they are blind to its removal. Whatever the projection takes at slice 0 becomes visible only once the blocks have acted on it.

    From the first block on, the projected red operand reads as a different color, and the difference grows with depth: the move roughly doubles from slice 1 to slice 4 on every op. In HSV terms it is a rotation of hue away from red, a fifth of a turn by the last slice, at nearly full saturation and with a lower value. In the figure that is the marks sliding along the upper edges of the hexagon, toward orange on one side and pink on the other, rather than toward the center.

    So this is the change-of-hue counterfactual that the kept shares followed, with a loss of value alongside it. The value the stream keeps of the red operand is what the saturation- and value-taking slots have to work with, so the part of it that goes is the partial loss the counterfactual table could not explain.

    The answer at `=` follows the operand. Clean, it reaches its ring by the last slice. Projected, it stops short, with a smaller hue offset than the operand and a saturation about two tenths under the clean read. The answers that survive on `sat-hsv` and `value-hsv` with red at op2 are the ones whose true color the rotated, dimmer operand still snaps to.
    """)
        )

    mo.md(_text())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Retention and the anneal

    The retention read of ex-2.2.9 asks whether the alignment survives the anneal of the anchor weight: every run whose line alignment peaks above 0.2 should end at 0.8 of that peak. One `handover` seed in twenty ended at 0.77, and the seed mean was 0.88 against 0.96 on the references.

    The natural reading was that the anneal, over the last tenth of training, lets the alignment slip. The trajectories say otherwise.
    """)
    return


@app.function(hide_code=True)
def anneal_start(res: Results, label: str) -> float:
    """The epoch at which the anchor weight first drops below its plateau."""
    t = res.traj[label]["traj"]
    w = np.array(t["weight"])
    ep = np.array(t["epoch"])
    below = np.flatnonzero(w < 0.99 * w.max())
    below = below[below > np.argmax(w)]
    return float(ep[below[0]]) if len(below) else float(ep[-1])


@app.cell(hide_code=True)
def _(res):
    def _stats():
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

    ret = _stats()
    return (ret,)


@app.cell(hide_code=True)
def _(res, ret):
    @themed
    def _plot():
        rng = np.random.default_rng(0)
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), layout="constrained", width_ratios=[2.2, 1, 1])
        ax = axes[0]
        conds = ("handover", "handover-slot", "handover-tied")
        for cond in conds:
            for i, r in enumerate(res.by_cond(cond)):
                t = res.traj[r["label"]]["traj"]
                ax.plot(
                    t["epoch"], t["m_line"], "-", color=ink(cond), lw=0.7, alpha=0.5, label=cond if i == 0 else None
                )
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
        for (
            ax,
            col,
            name,
        ) in ((axes[1], (3, 0), "end ÷ peak"), (axes[2], (3, 2), "end ÷ at anneal start")):
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

    fig_ret = figure_html(
        _plot(),
        caption="**Line alignment over training, and two retention ratios.** Left: every seed's line alignment against epoch, one thin line per seed, for the three handover conditions; the shaded band is the anneal window, where the anchor weight falls from 0.1 to its floor. Middle: the retention ratio ex-2.2.9 gated, the final alignment over its peak, per seed with the seed mean in the larger marker; the dashed rule is the gate and the hatched side misses it. Right: the final alignment over its value at the start of the anneal, the ratio that would measure the cost of the anneal alone, with the same gate level dotted for reference.",
        aria_label="Three panels. Left, alignment trajectories: handover seeds rise to about 0.75 by epoch 20 and drift down to about 0.66 before the anneal band begins at epoch 45, then stay flat; handover-slot and handover-tied peak later and drift less. Middle, end over peak: handover sits around 0.88 with one seed under the 0.8 gate, the other two conditions above 0.9. Right, end over the alignment at the anneal start: all three conditions sit at 1.0.",
    )
    return (fig_ret,)


@app.cell(hide_code=True)
def _(fig_ret, ret):
    def _text():
        h, s, t = (ret[c] for c in ("handover", "handover-slot", "handover-tied"))
        peak_ep = {c: float(ret[c][:, 1].mean()) for c in ("handover", "handover-slot", "handover-tied")}
        end_over_at = {
            c: float((ret[c][:, 3] / ret[c][:, 2]).mean()) for c in ("handover", "handover-slot", "handover-tied")
        }
        return inspect.cleandoc(f"""
        The anneal costs nothing. Over the anneal window the `handover` seeds end at {end_over_at["handover"]:.3f} of where they started it, and `handover-slot` and `handover-tied` at {end_over_at["handover-slot"]:.3f} and {end_over_at["handover-tied"]:.3f}. The whole of the drop the gate measured happens earlier.

        `handover` climbs to a plateau by epoch 10, and its peak, at {h[:, 0].mean():.2f} and epoch {peak_ep["handover"]:.0f} on average, is just the high point of a noisy series. By the start of the anneal the seeds sit at {h[:, 2].mean():.2f}, a few of them drifting to the bottom of the band. The two references peak later (epoch {peak_ep["handover-slot"]:.0f} on `handover-slot`, {peak_ep["handover-tied"]:.0f} on `handover-tied`) because their plateaus are still rising, and they drift less, which is why their ratio comes out higher.

        So the ratio of the end to the peak compares the height of a noisy plateau against its last sample, and adds in a slow drift on some seeds under a constant anchor weight. That drift needs both the whole-line labeller and the untied readout, since either one alone holds its plateau.

        Whether the drift matters is a different question from the one the gate asked. The end-of-training alignment is {h[:, 3].mean():.2f} on `handover` against {s[:, 3].mean():.2f} and {t[:, 3].mean():.2f} on the references. Every read downstream of it (grading, removal, containment) is taken at the end, so that level is what the later experiments inherit. A stepped anneal, which the backlog item floated, would act on the window where nothing is lost.
        """)

    mo.md(fig_ret + "\n\n" + _text())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Containment under the untied readout

    Ex-2.2.9 reported ᾱ, the mean alignment of the non-red lines at op1, at 0.28 on `handover`. The reference from ex-2.2.3 read 0.1, and the old gate was set there. The report put the rise down to the untied readout, which is what the pilot in ex-2.2.7 had suggested.

    But `handover` changed two things at once relative to the recipe of ex-2.2.3: the whole-line labeller as well as the readout. Ex-2.2.9 ran one arm with each change undone.
    """)
    return


@app.cell(hide_code=True)
def _(res):
    def _rows(cond: str):
        out = {"alpha": [], "nonred": [], "eol": [], "eq": [], "eol_readout": []}
        for r in res.by_cond(cond):
            rows, rr = r["rows"], r["rows_readout"]
            out["alpha"].append(r["alpha_op1"])
            out["nonred"].append(np.mean([abs(rows[n]) for n in ex.PALETTE if n not in RED_WORDS and n in rows]))
            out["eol"].append(rows["\n"])
            out["eq"].append(rows["="])
            out["eol_readout"].append(rr["\n"] if rr else np.nan)
        return {k: np.array(v, float) for k, v in out.items()}

    cont = {c: _rows(c) for c in CONDS}
    return (cont,)


@app.cell(hide_code=True)
def _(cont, res):
    @themed
    def _plot():
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

    fig_cont = figure_html(
        _plot(),
        caption="**Containment by condition.** Per seed, with the seed mean in the larger marker. Left: the mean alignment of the stream on the non-red lines with the anchor axis at op1, with the reference level from ex-2.2.3 dotted. Middle: the mean absolute axis component of the embedding rows for the 209 non-red color words. Right: the axis component of the `⏎` embedding row, with the same row of the untied readout as an open marker beside it where the condition has one.",
        aria_label="Three dot panels by condition. Alpha at op1: control near 0, handover-tied 0.16, handover-slot 0.18, handover 0.28, handover-narrow 0.30. Non-red rows: all conditions between 0.06 and 0.10. The newline row: near zero on control and handover-slot, 0.17 on handover, 0.27 on handover-tied.",
    )
    return (fig_cont,)


@app.cell(hide_code=True)
def _(cont, fig_cont):
    def _text():
        a = {c: float(cont[c]["alpha"].mean()) for c in CONDS}
        e = {c: float(cont[c]["eol"].mean()) for c in CONDS}
        er = {c: float(np.nanmean(cont[c]["eol_readout"])) for c in ("handover", "handover-slot")}
        nr = {c: float(cont[c]["nonred"].mean()) for c in CONDS}
        return inspect.cleandoc(f"""
        ᾱ at op1 is {a["handover"]:.2f} on `handover`, {a["handover-slot"]:.2f} with the slot labeller back, and {a["handover-tied"]:.2f} with the readout tied again. Undoing either change takes back about half the rise, and undoing neither leaves it. So the rise belongs to the handover as a whole, and the untied readout is one of two contributors. `handover-narrow`, which has fewer lines per op, sits at {a["handover-narrow"]:.2f}.

        Whatever the mechanism, none of it reaches the embedding table: the non-red color rows hold {nr["handover"]:.2f} of the axis on `handover` against {nr["control"]:.2f} on the control.

        The `⏎` row is a separate matter, and a cleaner one. Its embedding component is {e["handover"]:.2f} on `handover` and {e["handover-slot"]:.2f} on `handover-slot`, so the whole-line labeller is what puts it there. The pull lands on every position of a red line, `⏎` included, and the slot labeller never touches that position.

        The readout row for `⏎` is at {er["handover"]:.2f} on `handover` too. On `handover-tied` the row reads {e["handover-tied"]:.2f}, and so does `=`, which is the shared table doing double duty.

        The backlog item on the [`⏎` row](/todo/science/eol-embedding-row-keeps-the-axis.md) asks whether the residual redness of the answer position is part of this. Answering that needs a per-position alignment on the red lines, which ex-2.2.9 did not store, so it stays open here.
        """)

    mo.md(fig_cont + "\n\n" + _text())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What we make of it

    Three changes to the design of the re-run, and one read to carry.

    **Removal lines by hue.** Define a removal line as a red line whose true answer moves by at least 0.4 under some channel permutation of the red operand, so the lines whose answer needs the *hue* of red. On `mix` and the eight other channel-wise ops this is the to-zero set, or close to it. On the three HSV ops it drops the slots that take only saturation or value (`sat-hsv` and `value-hsv` with red at op2), and `hue-hsv` with red at op1 goes with them. The gate stays at kept under 0.2, and on the lines that remain every `handover` seed already clears it.

    The saturation- and value-taking slots become a second read with no gate: the kept share there says how much of the saturation and value of the red operand the projection takes with it, which is a cost of the operator worth reporting.

    **Retention against the start of the anneal.** The retention read keeps its name and changes its denominator: the final alignment over the alignment at the start of the anneal, gated at 0.8 as before. That is the question the read was written to ask. The early peak and the drift get a line of their own in the report, as the level at the end of training beside the references, since that level is what the later experiments inherit.

    **The op1 alignment as a line rather than a gate.** ᾱ at op1 rises under either half of the handover, and the embedding rows do not move, so the old reference of 0.1 belonged to the old grammar. The re-run reports ᾱ at op1 with `handover-slot` and `handover-tied` as its references, and gates nothing on it. A gate can return once we can name a mechanism.

    **The `⏎` row** stays as it is, with its backlog item open. The whole-line labeller is what puts it there, and the slot labeller is not coming back for it. If the edits in M3 ever touch the answer position, this row is the first place to look.

    One question this notebook could not settle: whether the partial loss of saturation and value under the projection comes from the operator or from this checkpoint. Answering it would take the same read on `handover-tied` and `handover-slot`, which the scoring pass can run at no design cost, so it is worth adding to the scoring for the re-run.
    """)
    return


@app.cell(hide_code=True)
def _(res):
    def _text():
        d = ex.design()
        n = len(res.labels) if res.metrics else 0
        return f"""
        ## Method

        **What ran.** A scoring-only experiment over the {len(d["seeds"])} `{d["condition"]}` checkpoints of ex-2.2.9 ({n} scored so far), on the ops {", ".join(f"`{o}`" for o in d["ops"])}, using the probe set of ex-2.2.9. For each checkpoint and op, the task ran two forward passes over every probe line: one clean, and one with the `{d["operator"]}` operator applied at every slice and position. For the red lines it stored the answer distribution over the 216 grid colors, the greedy answer, and the expected exact match under both passes. Its per-group kept shares reproduce those of ex-2.2.9 to the third decimal, which is our check that it ran the same read.

        **Probes.** One ridge regression (λ = {d["probe_l2"]:g}) per slice, position, and target, from the {res.width}-wide residual state to a color in the unit cube, fit on the clean stream of the non-red lines and read on the red lines. The three targets are op1, op2, and the raw (unrounded) answer the rule gives. The sites read in the figures are each operand at its own position, and the answer at `=` (position {d["decode_pos"]}). We store the decoded coordinates of every red line under both passes for all seeds, and the raw states for the first {len(d["state_seeds"])} seeds.

        **Counterfactuals.** For each red line, the red operand is the one with the higher redness. *To zero* sets its R channel to zero. *Change of hue* takes the five other permutations of its channels, and the line survives if all five move the true answer by less than {d["far_move"]}. *To gray* replaces it by the gray of its value. Answers are the snapped answers of the op, and distances are in the unit cube, the same as `to_zero_move` in ex-2.2.9.

        **Retention.** From the trajectories stored by ex-2.2.9 (100 points over training). The anneal starts at the first point where the anchor weight is under 0.99 of its plateau, and the alignment at that start is the last point before it.

        **Containment.** From the run metrics stored by ex-2.2.9: ᾱ at op1 as defined there, plus the axis component of each embedding row and, where the readout is untied, each readout row.
        """

    mo.md(_text())
    return


if __name__ == "__main__":
    app.run()
