# title: Where the op1 lean sits: a reanalysis of ex-2.2.11's stored runs

r"""
# Where the op1 lean sits: a reanalysis of ex-2.2.11's stored runs

/// tip |
<!-- tl;dr -->
A reanalysis of ex-2.2.11's 54 stored runs, asking where the containment rise (ᾱ at op1) comes from. The lean is not in the embedding table: it grows block by block, and at the last block it sits at the three positions whose next token is a syntax word, and nowhere else. Under an untied readout, the readout vectors of every color move to −e₁ and those of the syntax words to +e₁, so the e₁ coordinate of a state becomes a *syntax word comes next* feature that raises the syntax log-odds at op1 by a few nats. The tied readout cannot move the color rows and gets little of it. The whole-line labeller's half is not the pull landing on op1 through answer-labelled lines: the rise it adds is the same size on every color.
///

## Observations

Each line is a measurement on stored runs, with no gate. None is a result; the [closing section](#what-we-make-of-it) says what a preregistered follow-up would test.

- [Where in the stream](#where-in-the-stream): ᾱ at op1 is near zero at the embedding on every condition and grows with depth. The three anchored conditions separate from the first block on and are furthest apart at the last.
- [Which positions](#which-positions): at the last block the lean sits at op1, op2, and the answer, the positions whose next token is a syntax word (the op word, `=`, `⏎`). At the op word, `=`, and `⏎` it is near zero on every condition, the control's answer position aside.
- [The readout table](#the-readout-table): with a readout of its own, the model puts every color's readout vector at about −0.2 on e₁ and the syntax words' at +0.1 to +0.2, red and non-red colors alike. The tied readout moves only the syntax rows. The embedding rows of the non-red colors are flat on every condition.
- [What e₁ contributes](#what-e-contributes-to-the-next-token): removing the e₁ coordinate from the state's contribution to the logits lowers the syntax-against-color log-odds at op1 by a few nats on `handover`, by less on `handover-slot`, and by a fraction of a nat on `handover-tied` and on the control.
- [Between seeds](#between-seeds): within `handover` and within `handover-slot`, the seeds with a wider readout gap lean more at op1. The two conditions have the same gap and different leans, so the labeller's half is not through the gap.
- [Per color](#per-color): every color leans, graded by redness above an intercept. The rise from the slot labeller to the whole-line one is about the same on every color, and the color's own embedding row predicts its lean better than its exposure to answer-earned labels does.
- [The labeller's half](#the-labellers-half): on lines with two non-red operands and a red answer, the pull lands on op1 at a small share of the line's budget, and the op1 state on those lines leans no more than on lines with a non-red answer. The direct pull through the answer is not the mechanism.

## Scope

This is a reanalysis of stored artifacts, planned in the [backlog item](/todo/science/containment-rises-under-the-untied-readout.md): no training, no gates, no verdicts. ᾱ at op1 is the mean over the 216 grid colors of the cosine between a color's state at the first operand and e₁, the axis *red* is anchored to. It is 0.02 on ex-2.2.11's un-anchored control, 0.28 on `handover`, and about half of that when either of the handover's two changes is undone: 0.18 with the slot labeller back (`handover-slot`), 0.16 with the readout tied again (`handover-tied`). Every experiment since ex-2.1.8 gated the statistic at 0.1 and the handover experiments carry it as a line instead, because nobody has said what produces it.

Two mechanisms have been proposed and each has been checked once. For the readout's half, that the untied readout frees the embedding table from the logit pressure that kept non-red colors off e₁: ex-2.2.10 measured the non-red embedding rows and found them flat on every condition. For the labeller's half, that a line labelled through its answer has its pull land on op1: ex-2.2.12 tried the per-line split on `mix`, where a mean is red only when an operand is, so the answer-labelled group was empty.

The second check cannot be run as a per-line split at op1 on any op. Attention is causal, so the state above the first token is a function of that token alone, and two lines with the same op1 color have the same op1 state, whatever their answers. A split of ᾱ at op1 by how the line earned its label is a split by the op1 color's composition in each group and nothing else. What can be measured is where the pull's *gradient* lands on those lines, which the stored alignment maps give, and whether the colors that are exposed to answer-earned labels lean more than those that are not.

## Why

The containment line is carried into every handover experiment as a prediction without a mechanism, and the review asked twice for the why. A drift that costs nothing on the non-red lines is a nuisance; a drift with a mechanism says which knob would remove it and whether the D2.3 experiments should expect it under their grammars. The stored runs hold the embedding and readout tables, the alignment at every slice and position on every probe line, and the checkpoints, which is enough to say where the lean sits and what the readout does with the axis.

## The runs

Every run is one of ex-2.2.11's final checkpoints, with the eval arrays and the run-table entry the experiment published beside it.
"""

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import experiment as ex
from mini.lit import memo, read_npz, stop
from mini.store import Artifact, project_store
from mini.vis import figure_html, light_dark, themed
from sca.data.ops import colors as grid_colors

SLICES = ("emb", "L1", "L2", "L3", "L4")
LAST = ex.N_SLICES - 1
ORDER: tuple[str, ...] = ("control", "handover-tied", "handover-slot", "handover")
"""The conditions in the order the columns of each figure take: the control, then the arms by their ᾱ at op1."""
ANCHORED = ORDER[1:]
CONDS = {c.cond: c for c in ex.CONDITIONS}


def fetch(refs: Sequence[str], into: Path) -> dict[str, tuple[Artifact, Path] | None]:
    """Each ref's artifact and published file under *into*, or None before it exists: one `get_refs` and one `get_many` for the lot."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | {r: (a, p) for (r, a), p in zip(have.items(), paths, strict=True)}


@dataclass
class Results:
    metrics: dict
    arrays: Mapping[str, np.ndarray]
    sources: tuple[Artifact | None, ...]

    def __memo_key__(self) -> list[str | None]:
        """The figure cache keys a `Results` by the hashes of its sources rather than digesting every array it holds."""
        return [a.sha256 if a is not None else None for a in self.sources]

    def runs(self, cond: str) -> list[dict]:
        return [r for r in self.metrics["runs"] if r["cond"] == cond]

    def stack(self, cond: str, name: str) -> np.ndarray:
        """One array per run of *cond*, stacked along a new first axis."""
        return np.stack([self.arrays[f"{r['label']}/{name}"] for r in self.runs(cond)])

    def alpha_pos(self, cond: str) -> np.ndarray:
        """(run, slice, role): the mean alignment with e₁ over the reference op's probe lines."""
        return np.array([r["alpha_pos"] for r in self.runs(cond)])

    def e1(self, cond: str, table: str, group: str) -> np.ndarray:
        """(run,): the mean e₁ component of one token group's vectors in one table."""
        return np.array([r["e1"][table][group] for r in self.runs(cond)])

    def e1_full(self, cond: str, table: str) -> np.ndarray:
        """(run, vocab): the e₁ column of one table, in vocabulary order."""
        return np.array([r["e1_full"][table] for r in self.runs(cond)])

    def gap(self, cond: str, table: str = "readout") -> np.ndarray:
        """(run,): the syntax words' mean e₁ minus the colors' mean e₁, in one table."""
        syn = np.mean([self.e1(cond, table, g) for g in ("op words", "=", "⏎")], axis=0)
        return syn - self.e1(cond, table, "colors")

    def per_line_mean(self, cond: str, name: str) -> np.ndarray:
        """(run, role): a per-line, per-role array averaged over the probe lines."""
        return self.stack(cond, name).mean(axis=1)

    def op1_alpha(self, cond: str) -> np.ndarray:
        """(run, slice, color): each color's alignment at op1."""
        return self.stack(cond, "op1_alpha")

    def by_color(self, cond: str, table: str) -> np.ndarray:
        """(run, color): the e₁ component of each grid color's vector in one table, in grid order."""
        vocab = {w: i for i, w in enumerate(self.metrics["vocab"])}
        from sca.data.ops import NAMES

        idx = [vocab[NAMES[c]] for c in grid_colors()]
        return self.e1_full(cond, table)[:, idx]

    @property
    def redness(self) -> np.ndarray:
        return np.array(self.metrics["redness"])

    def exposure(self, through: str) -> np.ndarray:
        """(color,): the mean over ops of a color's per-line probability, as op1, of a label earned *through*."""
        return self.arrays[f"grammar/through/{through}"].mean(axis=0)

    @property
    def line_groups(self) -> list[str]:
        return self.metrics["line_groups"]


def load_results() -> Results | None:
    with tempfile.TemporaryDirectory() as tmp:
        got = fetch([ex.METRICS_REF, ex.ARRAYS_REF], Path(tmp))
        gm, ga = got[ex.METRICS_REF], got[ex.ARRAYS_REF]
        arrays = None if ga is None else read_npz(ga[1])
        if gm is None or ga is None or arrays is None:
            return None
        return Results(json.loads(gm[1].read_text()), arrays, (gm[0], ga[0]))


# --- Figure style --------------------------------------------------------------------------------


def ink(cond: str) -> str:
    """One ink per condition, as ex-2.2.9 draws them."""
    inks = {
        "control": ("#6b6b6b", "#b0b0b0"),
        "handover": ("#c0392b", "#ff8a76"),
        "handover-slot": ("#2b6cb0", "#7fb3ff"),
        "handover-tied": ("#7b3fa0", "#cfa3ff"),
    }
    return light_dark(*inks[cond])


def marker(cond: str) -> str:
    return {"control": "s", "handover": "o", "handover-slot": "^", "handover-tied": "D"}[cond]


def dots(ax: Axes, x: float, v: np.ndarray, cond: str, *, rng, ms: float = 5.0, width: float = 0.05, label=None):
    """One column of per-seed dots with the seed mean drawn on top, in the condition's ink and marker."""
    v = np.asarray(v, float)
    color, m = ink(cond), marker(cond)
    jit = rng.uniform(-width, width, len(v))
    ax.plot([x, x], [np.nanmin(v), np.nanmax(v)], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + jit, v, "o", ms=2.2, color=color, alpha=0.45, zorder=3, mew=0)
    ax.plot(x, np.nanmean(v), m, ms=ms, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6, label=label)


def fig_legend(fig: plt.Figure, ax: Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)


def offsets(n: int) -> np.ndarray:
    """The x offsets of *n* condition columns at one tick: 0.15 apart, up to a spread of ±0.3."""
    half = min(0.075 * (n - 1), 0.3)
    return np.linspace(-half, half, n) if n > 1 else np.zeros(1)


def zero_line(ax: Axes) -> None:
    ax.axhline(0, color=light_dark("#333", "#ddd"), lw=0.6, ls=":", zorder=1)


def columns_panel(ax: Axes, res: Results, values, ticks: Sequence[str], *, rng, conds=ORDER, legend: bool = False):
    """Seed-dot columns per condition at each tick. *values(cond)* gives a (run, tick) array."""
    xs = np.arange(len(ticks))
    off = offsets(len(conds))
    for x in xs:
        for o, c in zip(off, conds, strict=True):
            dots(ax, x + o, values(c)[:, x], c, rng=rng, width=0.03, label=c if legend and x == 0 else None)
    ax.set_xticks(xs, ticks)
    ax.grid(axis="y", alpha=0.2)
    zero_line(ax)


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    ths = "".join(f"<th{' class=num' if i else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{cell_html(c)}</td>" for i, c in enumerate(row)) + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def mean_range(v: np.ndarray, digits: int = 2) -> str:
    v = np.asarray(v, float)
    return f"{v.mean():+.{digits}f} ({v.min():+.{digits}f}–{v.max():+.{digits}f})"


def runs_table() -> str:
    rows = [
        [
            f"`{c.cond}`",
            "tied" if c.tied else "untied",
            "whole line" if c.span == 6 else "prompt",
            f"{len(c.seeds)}",
            c.title,
        ]
        for c in ex.CONDITIONS
    ]
    return table_html(
        ["condition", "readout", "span", "seeds", "what it is"],
        rows,
        "**The runs.** Every seed of each of ex-2.2.11's four conditions. The span is the set of positions the anchor term pools over on a labelled line: the four prompt roles under the slot labeller, and the whole line, answer and `⏎` included, under the whole-line labeller. Under the whole-line labeller a line can also earn its label through its answer.",
    )


runs_table()

r"""
## The measurements

Three kinds of stored artifact are read, and the [method](#method) gives each one's definition.

- The **run table** ex-2.2.11 published carries, per run, the mean alignment with e₁ at every slice and position over the `mix` probe lines, and the e₁ component of every token's vector in the embedding table and in the readout table (the same table, when tied).
- The **eval arrays** carry each color's alignment at op1 at every slice, and the alignment of every position of every `hue-hsv` probe line, from which the softmin weights the anchor term would assign are recomputed.
- The **checkpoints** give the logits on the `mix` probe lines. nGPT's logits are $s_z \, h R^\top$, linear in the state $h$, so the part every logit owes to the e₁ coordinate is $s_z \, h_1 R_{\cdot 1}$ and can be taken out and the log-odds recomputed.

Each figure shows a column of seed dots per condition: one small dot per run, the larger mark the seed mean, and the bar the seed range.[^figkey] The conditions keep ex-2.2.9's inks and markers.

[^figkey]: `control` has five seeds, `handover-tied` nine, and the other two twenty, so a control column's range is a rougher measurement of its spread than the others'.
"""

res = load_results()
if res is None:
    stop("_Results are not published yet; the figures render once they are._")


def summary(res: Results) -> dict[str, dict[str, float]]:
    """Seed means of the numbers the prose quotes, per condition."""
    out: dict[str, dict[str, float]] = {}
    for c in ORDER:
        ap = res.alpha_pos(c)
        out[c] = {
            "op1_emb": ap[:, 0, 0].mean(),
            "op1_last": ap[:, LAST, 0].mean(),
            "answer_last": ap[:, LAST, 4].mean(),
            "op2_last": ap[:, LAST, 2].mean(),
            "syntax_last": ap[:, LAST, [1, 3, 5]].mean(),
            "readout_colors": res.e1(c, "readout", "colors").mean(),
            "readout_syntax": np.mean([res.e1(c, "readout", g).mean() for g in ("op words", "=", "⏎")]),
            "readout_gap": res.gap(c).mean(),
            "embedding_nonred": res.e1(c, "embedding", "non-red").mean(),
            "delta_op1": res.per_line_mean(c, "delta")[:, 0].mean(),
            "logodds_op1": res.per_line_mean(c, "logodds")[:, 0].mean(),
        }
    return out


S = summary(res)

"""
## Where in the stream

**What we expected.** If the untied readout released the embedding table, the lean would already be there at the embedding and the blocks would carry it along. If the blocks build it, the embedding is flat and the lean grows with depth.

**What we saw.** The second. At the embedding every condition is within a few hundredths of zero, the control included. The anchored conditions rise from the first block on, and the gap between them opens with depth: they are closest at the first block and furthest apart at the last.
"""


@memo
def stream_figure(res: Results) -> str:
    @themed(
        name="stream",
        alt_text="Dot chart of ᾱ at op1 per residual slice: every condition starts near zero at the embedding, the control stays there, and the three anchored conditions rise block by block, handover highest.",
        caption="**ᾱ at op1 per slice.** Each column is one condition's seeds at one slice: the mean over the 216 grid colors of the cosine between the color's state at the first operand and e₁, on `mix`'s probe lines.",
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 3.2), layout="constrained")
        columns_panel(ax, res, lambda c: res.alpha_pos(c)[:, :, 0], SLICES, rng=np.random.default_rng(0), legend=True)
        ax.set_ylabel("ᾱ at op1")
        fig_legend(fig, ax)
        return fig

    return _plot()


stream_figure(res)

f"""
The control sits at {S["control"]["op1_last"]:+.2f} at the last block and `handover` at {S["handover"]["op1_last"]:+.2f}, with `handover-slot` at {S["handover-slot"]["op1_last"]:+.2f} and `handover-tied` at {S["handover-tied"]["op1_last"]:+.2f}: the same ordering ex-2.2.10 and ex-2.2.11 reported, now seen to be built by the blocks rather than given by the table.

## Which positions

**What we expected.** A pull that lands on every position of a labelled line would leave a lean at every position, largest where the pull is cheapest. A lean that the readout recruits would sit where the readout has a use for it.

**What we saw.** At the last block the lean sits at three positions and is near zero at the other three, on every anchored condition. The three are op1, op2, and the answer, whose next token is a syntax word: the op word after op1, `=` after op2, `⏎` after the answer. At the op word, `=`, and `⏎` (whose next token is a color) the lean is within a few hundredths of zero. The pattern is the same on the three anchored conditions and only its size differs; the whole-line labeller's answer position is the one place the conditions part company, since under the slot labeller the answer is outside the span.

The control is flat at every position but the answer, where it leans the other way. That is the un-anchored model's own use of the axis, and it says the readout of an un-anchored model already reads something at the answer along the coordinate the anchor later takes.
"""


@memo
def roles_figure(res: Results) -> str:
    @themed(
        name="roles",
        alt_text="Dot chart of the last-block alignment with e₁ at each of a line's six positions: the anchored conditions lean at op1, op2 and the answer and sit at zero at the op word, the equals sign and the newline; the control is flat except a negative answer position.",
        caption="**Alignment with e₁ by position, at the last block.** The mean over `mix`'s probe lines of the cosine between the state at each position and e₁. The positions alternate between predicting a syntax word (op1, op2, answer) and predicting a color (op, `=`, `⏎`).",
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 3.2), layout="constrained")
        columns_panel(
            ax, res, lambda c: res.alpha_pos(c)[:, LAST, :], ex.ROLES, rng=np.random.default_rng(1), legend=True
        )
        ax.set_ylabel("alignment with e₁, last block")
        fig_legend(fig, ax)
        return fig

    return _plot()


roles_figure(res)

f"""
On `handover` the answer position leans at {S["handover"]["answer_last"]:+.2f} against {S["handover-slot"]["answer_last"]:+.2f} on `handover-slot`; op2 leans at {S["handover"]["op2_last"]:+.2f}. The three color-predicting positions average {S["handover"]["syntax_last"]:+.3f} on `handover`.

## The readout table

**What we expected.** Ex-2.2.10 found the non-red embedding rows flat on every condition, so if the untied readout is doing something with e₁ it is doing it in the readout table.

**What we saw.** It is. With a readout of its own, the model puts the readout vector of every color at about −0.2 on e₁, red and non-red colors alike, and the readout vectors of the syntax words at +0.1 to +0.2. The tied conditions cannot do the first: in `handover-tied` the readout is the embedding, whose color rows the anchor holds (red at the top, non-red near zero), so only the syntax rows move, and they move by about the same amount as under the untied readout. The control's readout is flat on e₁ for every group, so the split is a response to the anchor rather than a habit of the untied readout. The embedding rows of the non-red colors are flat on every condition, as ex-2.2.10 found.
"""


@memo
def tables_figure(res: Results) -> str:
    groups = ex.TOKEN_GROUPS

    @themed(
        name="tables",
        alt_text="Two dot charts of the e₁ component of token vectors by group, embedding table left and readout table right: in the readout table the untied anchored conditions put the colors below zero and the syntax words above it, the tied condition moves only the syntax words, and the control is flat; in the embedding table only the red colors stand out.",
        caption="**The e₁ component of the tables, by token group.** Left: the embedding table. Right: the readout table, which on `handover-tied` is the embedding table again. *colors* is the mean over all 216 grid colors, *red* and *non-red* over the colors at or over the red dose and at or under the non-red dose.",
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained", sharey=True)
        rng = np.random.default_rng(2)
        for ax, table in zip(axes, ("embedding", "readout"), strict=True):
            columns_panel(
                ax,
                res,
                lambda c, t=table: np.stack([res.e1(c, t, g) for g in groups], axis=1),
                groups,
                rng=rng,
                legend=table == "embedding",
            )
            ax.set_title(f"{table} table", fontsize=9)
        axes[0].set_ylabel("e₁ component")
        fig_legend(fig, axes[0])
        return fig

    return _plot()


tables_figure(res)

f"""
The readout gap, the syntax words' mean e₁ less the colors', is {S["handover"]["readout_gap"]:+.2f} on `handover`, {S["handover-slot"]["readout_gap"]:+.2f} on `handover-slot`, {S["handover-tied"]["readout_gap"]:+.2f} on `handover-tied`, and {S["control"]["readout_gap"]:+.2f} on the control. The non-red embedding rows sit at {S["handover"]["embedding_nonred"]:+.2f} on `handover` and {S["control"]["embedding_nonred"]:+.2f} on the control.

## What e₁ contributes to the next token

**What we expected.** A readout that puts colors at −e₁ and syntax at +e₁ turns the e₁ coordinate of a state into a vote for *a syntax word comes next*. If the model uses that vote, taking the e₁ coordinate out of the logits should lower the syntax-against-color log-odds at the positions that predict syntax, and the more so the wider the gap.

**What we saw.** That is what it does. The logits are linear in the state, so the share of every logit that the e₁ coordinate carries can be removed and the log-odds recomputed from the other 63 coordinates. At op1 the removal lowers the syntax log-odds by a few nats on `handover`, by less on `handover-slot`, and not at all on the control. On `handover-tied` the contribution at op1 is a fraction of a nat, about the control's, and at op2 and the answer it goes the other way: the tied table has the red colors at +e₁ and the rest of the colors near zero, so a state's e₁ coordinate there votes for *red* as much as for syntax, and removing it raises the syntax log-odds a little at those positions.

The log-odds themselves are large at every position, tens of nats, so the model is in no doubt about which kind of token comes next; e₁'s few nats are a part of a margin the rest of the state also supplies. What the measurement says is what the readout has recruited the axis for, not that the model needs it.
"""


@memo
def delta_figure(res: Results) -> str:
    @themed(
        name="delta",
        alt_text="Two dot charts by position: left, the syntax-against-color log-odds of the next token, alternating between large positive and large negative; right, what the e₁ coordinate contributes to it, positive at op1, op2 and the answer on the untied anchored conditions, near zero at op1 and slightly negative at op2 on the tied one, and zero on the control.",
        caption="**The next-token log-odds and e₁'s part of it, by position, at the last block.** Left: log P(syntax word next) − log P(color next), mean over `mix`'s probe lines. Right: the change in that log-odds when the e₁ coordinate's share of every logit is removed, so a positive value means e₁ argued for a syntax word.",
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained")
        rng = np.random.default_rng(3)
        columns_panel(axes[0], res, lambda c: res.per_line_mean(c, "logodds"), ex.ROLES, rng=rng, legend=True)
        columns_panel(axes[1], res, lambda c: res.per_line_mean(c, "delta"), ex.ROLES, rng=rng)
        axes[0].set_ylabel("log-odds syntax : color (nats)")
        axes[1].set_ylabel("e₁'s contribution (nats)")
        fig_legend(fig, axes[0])
        return fig

    return _plot()


delta_figure(res)

f"""
At op1, e₁ contributes {S["handover"]["delta_op1"]:+.2f} nats on `handover`, {S["handover-slot"]["delta_op1"]:+.2f} on `handover-slot`, {S["handover-tied"]["delta_op1"]:+.2f} on `handover-tied`, and {S["control"]["delta_op1"]:+.2f} on the control, to a log-odds of about {S["handover"]["logodds_op1"]:+.0f} nats.

## Between seeds

**What we expected.** If the readout gap is what pulls the non-red colors onto the axis, the seeds of one condition that open a wider gap should lean more at op1.

**What we saw.** They do, within each untied condition. The correlation is moderate rather than tight, it is the same on `handover` and `handover-slot`, and it is weaker on `handover-tied`, whose gap is only in the syntax rows. What the scatter also shows is that the two conditions occupy the same range of gaps and different ranges of lean: at a given gap `handover` leans more. So the readout's half is visible in the gap, and the labeller's half is something else.
"""


@memo
def seeds_figure(res: Results) -> str:
    stats = {c: (res.gap(c), res.alpha_pos(c)[:, LAST, 0]) for c in ANCHORED}
    corr = {c: float(np.corrcoef(g, a)[0, 1]) for c, (g, a) in stats.items()}

    @themed(
        name="seeds",
        alt_text="Scatter of the readout gap against ᾱ at op1 at the last block, one point per seed: handover and handover-slot share the same range of gaps but handover sits higher, and within each the lean rises with the gap; handover-tied has a narrower gap and a lower lean.",
        caption="**The readout gap against ᾱ at op1, per seed.** Each mark is one run: on the x axis the syntax words' mean e₁ in the readout table less the colors', and on the y axis ᾱ at op1 at the last block. The legend gives the Pearson r within each condition.",
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        for c in ANCHORED:
            g, a = stats[c]
            ax.plot(
                g,
                a,
                marker(c),
                ms=5,
                color=ink(c),
                mec=light_dark("white", "#111"),
                mew=0.5,
                ls="none",
                label=f"{c} (r = {corr[c]:+.2f})",
            )
        ax.set_xlabel("readout gap on e₁ (syntax − colors)")
        ax.set_ylabel("ᾱ at op1, last block")
        ax.grid(alpha=0.2)
        fig_legend(fig, ax)
        return fig

    return _plot()


seeds_figure(res)

"""
## Per color

**What we expected.** A lean recruited by the readout as a vote for *syntax next* has no reason to prefer one color over another, so it should be a shift shared by every color, with the anchor's own grading on top of it. A lean that comes from answer-earned labels should be largest on the colors whose lines most often have a red answer.

**What we saw.** Every color leans, and the lean is a shift plus a grade: the non-red colors sit above zero by a nearly constant amount and the grade with redness is the anchor's. The rise from `handover-slot` to `handover` is close to the same on every color, so whatever the whole-line labeller adds, it adds to every color alike. A color's own embedding row on e₁ predicts its lean well. Its exposure to answer-earned labels predicts the rise less well: the colors form a flat band, and the correlation that remains rests on a few colors at the highest exposure, which are also the ones that rose most.
"""


@memo
def colors_figure(res: Results) -> str:
    rgb = np.asarray(grid_colors(), dtype=float) / 15.0
    red = res.redness
    leans = {c: res.op1_alpha(c)[:, LAST, :].mean(axis=0) for c in ORDER}

    @themed(
        name="colors",
        alt_text="Four scatter panels, one per condition, of each grid color's op1 alignment at the last block against its redness, marks in the color itself: the control is flat at zero, and the anchored conditions rise with redness from a positive intercept, highest on handover.",
        caption="**Each color's alignment at op1 against its redness, at the last block.** One mark per grid color in the color itself, the seed mean per condition. Redness is r(1 − g/2 − b/2) on the unit cube, the grading target.",
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(ORDER), figsize=(10.4, 3.0), layout="constrained", sharey=True)
        for ax, c in zip(axes, ORDER, strict=True):
            ax.scatter(red, leans[c], c=rgb, s=14, edgecolors=light_dark("#0004", "#fff4"), linewidths=0.4)
            zero_line(ax)
            ax.set_title(c, fontsize=9, color=ink(c))
            ax.set_xlabel("redness")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("alignment at op1, last block")
        return fig

    return _plot()


colors_figure(res)

# %%


@memo
def color_fit(res: Results) -> dict[str, dict[str, float]]:
    """Per condition: the intercept and slope of the last-block op1 alignment against redness, and the non-red mean."""
    red = res.redness
    out = {}
    for c in ORDER:
        lean = res.op1_alpha(c)[:, LAST, :].mean(axis=0)
        slope, intercept = np.polyfit(red, lean, 1)
        out[c] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "nonred": float(lean[red <= ex.NONRED_DOSE].mean()),
        }
    return out


CF = color_fit(res)

f"""
Against redness, the last-block lean has an intercept of {CF["handover"]["intercept"]:+.2f} on `handover`, {CF["handover-slot"]["intercept"]:+.2f} on `handover-slot`, and {CF["handover-tied"]["intercept"]:+.2f} on `handover-tied`, with slopes of {CF["handover"]["slope"]:.2f}, {CF["handover-slot"]["slope"]:.2f}, and {CF["handover-tied"]["slope"]:.2f}; the control's intercept is {CF["control"]["intercept"]:+.2f}.
"""


@memo
def rise_figure(res: Results) -> str:
    rgb = np.asarray(grid_colors(), dtype=float) / 15.0
    red = res.redness
    nonred = red <= ex.NONRED_DOSE
    lean = res.op1_alpha("handover")[:, LAST, :].mean(axis=0)
    rise = lean - res.op1_alpha("handover-slot")[:, LAST, :].mean(axis=0)
    row = res.by_color("handover", "embedding").mean(axis=0)
    expo = res.exposure("answer")
    r_row = float(np.corrcoef(row[nonred], lean[nonred])[0, 1])
    r_expo = float(np.corrcoef(expo[nonred], rise[nonred])[0, 1])
    rise_sd = float(rise[nonred].std())

    @themed(
        name="rise",
        alt_text="Two scatter panels over the non-red colors, marks in the color itself: left, the color's embedding row on e₁ against its op1 lean on handover, a clear upward trend; right, its exposure to answer-earned labels against the rise from the slot labeller to the whole-line one, a flat band, with the few highest-exposure colors also the highest.",
        caption=f"**What predicts a non-red color's lean, on `handover`.** Left: the color's own embedding row on e₁ against its last-block alignment at op1 (r = {r_row:+.2f}). Right: the color's exposure to labels earned through the answer (its per-line probability as op1, mean over the eleven ops and every partner) against the rise in its lean from `handover-slot` to `handover` (r = {r_expo:+.2f}; the rise has a spread of {rise_sd:.2f} over colors). Non-red colors only, seed means.",
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained")
        edge = light_dark("#0004", "#fff4")
        axes[0].scatter(row[nonred], lean[nonred], c=rgb[nonred], s=14, edgecolors=edge, linewidths=0.4)
        axes[0].set_xlabel("embedding row on e₁")
        axes[0].set_ylabel("alignment at op1, last block")
        axes[1].scatter(expo[nonred] * 1e3, rise[nonred], c=rgb[nonred], s=14, edgecolors=edge, linewidths=0.4)
        axes[1].set_xlabel("exposure to answer-earned labels (×10⁻³ per line)")
        axes[1].set_ylabel("rise, slot → whole line")
        for ax in axes:
            ax.grid(alpha=0.2)
        return fig

    return _plot()


rise_figure(res)

"""
The left panel is the embedding table doing what a table does: a color whose embedding row has a little e₁ arrives at the first block with it, and the blocks amplify it. It is a description of where the lean is carried, not of what sets it, since the row and the state are trained together. The right panel is the test of the labeller mechanism as a per-color prediction, and the prediction is not there: the exposure varies ten-fold across the non-red colors and the rise does not follow it.

## The labeller's half

**What we expected.** Under the whole-line labeller a line with two non-red operands can earn its label through a red answer, and the pull then pools over the whole line, op1 included. If that pull lands on op1, the softmin weight the anchor term assigns op1 on such lines should be a fair share of the line's budget, and the op1 state on those lines should lean more than on lines that earn no label.

**What we saw.** Neither. The lines are `hue-hsv`'s, where the answer takes op2's hue at op1's saturation and value, so two non-red operands can produce a red answer; the two groups have both operands at or under the non-red dose and differ only in the answer, which keeps the color composition at op1 the same. On those lines the share the pull assigns op1 is small at every slice on the whole-line conditions, and the pull goes to the answer and `⏎` instead, where the alignment is cheapest. The op1 alignment on the answer-labelled lines is the same as on the unlabelled ones, to within the seed spread, on every condition. That group is small, nineteen lines, so its seed bands are wide and a small difference would hide in them; what the read can say is that the share landing on op1 is near zero there, and that nothing in the alignment sets those lines apart. The `hue-hsv` lines are the ones the pull would land on through the answer, and it does not land on op1 there.
"""


@memo
def labeller_figure(res: Results) -> str:
    groups = res.line_groups
    n_lines = res.runs("handover")[0]["n_lines"]

    @themed(
        name="labeller",
        alt_text="Two line charts over the five residual slices, solid for answer-labelled lines and dashed for unlabelled ones: left, the softmin share op1 takes of the pull, small on every anchored condition; right, the op1 alignment on the two groups of lines, which lie on top of each other for every condition.",
        caption=f"**Lines that can earn a label only through their answer, against lines that earn none.** `hue-hsv` probe lines with both operands non-red, split by a red answer ({n_lines[groups[0]]} lines) or a non-red one ({n_lines[groups[1]]}). Left: the softmin weight the anchor term assigns op1, the share of the line's pull that lands there, over the condition's span. Right: the op1 alignment on the same lines. Lines are seed means and bands the seed range; solid for the answer-labelled group, dashed for the unlabelled one.",
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained")
        xs = np.arange(ex.N_SLICES)
        for ax, name in zip(axes, ("share", "line_alpha"), strict=True):
            for c in ORDER:
                v = res.stack(c, name)  # (run, group, slice)
                for gi, ls in ((0, "-"), (1, "--")):
                    ax.fill_between(xs, v[:, gi].min(0), v[:, gi].max(0), color=ink(c), alpha=0.12, lw=0, zorder=1)
                    ax.plot(
                        xs,
                        v[:, gi].mean(0),
                        ls,
                        color=ink(c),
                        marker=marker(c),
                        ms=3.5,
                        lw=1.2,
                        label=c if gi == 0 and name == "share" else None,
                    )
            ax.set_xticks(xs, SLICES)
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("softmin share on op1")
        axes[1].set_ylabel("alignment at op1")
        fig_legend(fig, axes[0])
        return fig

    return _plot()


labeller_figure(res)

r"""
So what the whole-line labeller adds is not the pull landing on op1 through the answer. What it does visibly change is the answer position, which under the whole-line span the pull *does* land on, and the readout's `⏎` row, which is the row that position predicts. On `handover` the answer state leans and the `⏎` readout vector sits further up e₁ than on `handover-slot`; on `handover-slot` the answer is outside the span, its state stays near zero, and the `⏎` row barely moves. That is one more position at which the readout is asked to read e₁ as *syntax next*, and it may be the whole of the labeller's half: an axis the readout uses at three syntax-predicting positions rather than two is a stronger feature, and op1, the position with the least context, inherits more of it. This report does not test that; the next section says what would.

## What we make of it

The rise in ᾱ at op1 looks like the readout recruiting the anchor axis. The anchor puts *red* on e₁ at the operand positions. With a readout of its own, the model finds that a component along e₁ separates the colors from the syntax words in the readout table, and since every color state already carries some e₁ on the red lines, it sets the colors' readout vectors to −e₁ and the syntax words' to +e₁. From then on, a state's e₁ coordinate is read as *a syntax word comes next*, and the positions where that is the prediction (op1, op2, the answer) drift up the axis, every color alike, on top of the anchor's grading. The tied readout cannot set the color rows, because the anchor holds them, so it gets a smaller version of the same thing through the syntax rows alone. The whole-line labeller adds the answer position to the set the pull lands on and the `⏎` row to the set the readout moves, and the lean at op1 grows with it.

Three things this report does not say. It does not say the lean costs anything: ex-2.2.11 found the paired non-red deficit at 0.004. It does not say the readout *needs* e₁ for the syntax prediction; the log-odds are tens of nats and e₁ is a few of them. And it does not say the labeller's half is the answer position; that is the one candidate left standing, and it is untested.

**To do.** The readout's half is cheap to test, because it is a table. A pilot on `handover`'s recipe that holds the readout vectors of every non-red color and every syntax word off e₁ (the readout-side analogue of `clean_embedding_rows`, applied on the same schedule) should bring ᾱ at op1 down toward `handover-tied`'s level without touching *red*, and it should take the e₁ contribution to the syntax log-odds with it. The labeller's half wants the span narrowed by one position: a `handover` arm whose whole-line labeller keeps the keying but pools over roles 1 to 5, or over the prompt roles alone, would say whether the answer position is what the whole-line span adds. Both are one condition each on the existing setup, and both read ᾱ at op1 beside the two arms this report has already characterised.

## Method

**Sources.** Ex-2.2.11's four conditions at every stored seed: the checkpoint, the eval arrays, and the run-table entry per run, and the probe lines every run was scored on. The report reads this pass's outputs through `project_store()`.

**Alignment by position.** From the run table: the mean over the `mix` probe lines of the cosine between the state at each position and e₁, at each of the five slices, as ex-2.2.9's eval pass computes it. At op1 the mean over lines is the mean over the 216 colors, since each color is op1 on the same number of lines.

**Tables.** The e₁ component of every token's vector in the embedding table and in the readout table, from the run table, summarized by token group: the 216 colors, those at or over the red dose (0.8) and at or under the non-red dose (0.2), the eleven op words, `=`, and `⏎`. The readout gap is the mean over the three syntax groups less the colors' mean.

**e₁'s contribution.** On the `mix` probe lines, from the checkpoint: the logits $s_z \, h R^\top$ at every position, and the log-odds $\log \sum_{\text{syntax}} p - \log \sum_{\text{colors}} p$. The contribution is that log-odds less the log-odds of $s_z \, h R^\top - s_z \, h_1 R_{\cdot 1}$, the logits with the e₁ coordinate's share removed. Nothing is renormalized between the two, so the difference is exactly what the coordinate carried.

**Per color.** From the eval arrays: each grid color's alignment at op1 at each slice, on `mix`. The fit against redness is ordinary least squares over the 216 colors of the seed mean.

**Exposure.** From the grammar and the labeller's model (each slot draws a label at redness⁸ × 0.04): for each color as op1, the mean over the eleven ops and every partner of the probability that the line is labelled through op1, through op2 when op1 did not draw, and through the answer alone, with the answer's redness taken in expectation over the stochastic rounding.

**The labeller's lines.** `hue-hsv` probe lines at walk 0 with both operands at or under the non-red dose, split by an answer at or over the red dose or at or under the non-red dose. The softmin share is `softmin_weights(1 − α, τ = 0.1)` over the condition's span (the four prompt roles or the whole line), read at op1.

**Where the data is.** The run table is under the `metrics` ref of `reports/m2/op1-lean` and every array under `arrays`, keyed by run label and name, with the exposure model's arrays under `grammar/`.
"""
