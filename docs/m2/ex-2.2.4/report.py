import marimo

__generated_with = "0.24.0"
app = marimo.App(
    app_title="Ex 2.2.4: a scouting round before the anchored-op experiments",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    from collections import Counter
    from dataclasses import dataclass
    from functools import cached_property

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import EllipseCollection

    from mini.reports import report_bundle, use_publisher
    from mini.vis import figure_html, light_dark, themed
    from sca.data.colors import Rgb, redness
    from sca.data.ops import (
        CANDIDATES,
        LEVELS,
        OPS,
        TOP,
        Op,
        colors,
        commutativity,
        lines,
        mix_probe_lines,
        on_grid,
        probe_partners,
        relevance,
        unordered_pairs,
    )
    from sca.vis import CUBE_VIEWS, draw_cube_bound, grid_diameter, project_cube

    use_publisher(report_bundle(__file__))

    ALL_OPS: tuple[Op, ...] = OPS + CANDIDATES
    """The six ops of ex-2.2.3, then the nine candidates, in the order every table and figure uses."""

    RED_DOSE = 0.8
    """Ex-2.2.3's red line: the redder operand has redness ≥ 0.8."""

    N_PROBE, PROBE_SEED = 27, 0
    """Ex-2.2.3's probe draw: 27 partners per color, shared across the non-`mix` ops."""

    STEP = LEVELS[1] - LEVELS[0]
    """One grid level: the drop the dependence rule of E8 applies to the R of the red operand."""

    GRID_RGB = np.asarray(colors(), dtype=float) / TOP
    """The 216 colors as unit-cube RGB, in palette order."""

    TABLES: dict[str, tuple[str, ...]] = {
        "current": ("mix", "add", "screen", "multiply", "lighten", "darken"),
        "A": ("mix", "screen", "multiply", "lighten", "darken", "difference", "exclusion", "hsvmix"),
        "B": ("mix", "add", "screen", "multiply", "lighten", "darken", "hue-hsv", "sat-hsv", "value-hsv"),
    }
    """Three tables the relevance section reads: ex-2.2.3's, a commutative widening (A), and the HSV
    widening the backlog item proposed (B)."""

    None


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.2.4: a scouting round before the anchored-op experiments

    /// tip |
    <!-- tl;dr -->
    This is a scouting pass, so there are no hypotheses. Before we anchor anything on the six-op grammar, we ask whether a more varied set of operations would make the removal tests easier to read. We work from the color grid alone; nothing is trained. The proposed hue, saturation, and brightness ops do spread their answers through the color cube. But they do not make the answer depend on the red operand any more than the current ops do. Three commutative ops do both: `difference`, `exclusion`, and `mix` done in HSV.
    ///

    ## Observations

    Each line below is a read on the grid, together with the statistic it rests on. None of them is a result. The next preregistration will adopt what it needs from here and check it on trained models.

    - [The op set](#the-op-set): every candidate spreads its answers more evenly than the current ops do, and only the commutative candidates are more sensitive to the red operand than `mix` is. We propose table A, which drops `add` and adds `difference`, `exclusion`, and `hsvmix`.
    - The dependence rule from ex-2.2.3 (E8) responds to rounding as much as to the op itself. By that rule `mix` counts as dependent on 61% of its own red probe lines, and about half on the shared draw. Under every op, current or candidate, dropping the red operand's R to zero changes the answer on more than 70% of red lines.

    ## How to read this

    This is the scouting round planned in the [backlog item](/todo/science/scouting-round-before-the-anchored-op-experiments.md): one notebook, the cheapest version of each question, no gates and no verdicts. Each section says what we ran, shows one figure or table, and says whether it changes the design. Sections land as their questions are run, and this one covers the op set. The tied-readout diagnostic, the τ against λ_a trade, and the answer-drawing labeller are still to come.

    Ex-2.2.3 anchored *red* on a grammar of six operations, then measured removal: does the model still answer correctly once the *red* axis is projected out of the residual stream?[^stream] Removal came out partial on four of the six ops. E8 traced that to the ops themselves: on the saturating ops, most red lines have an answer that would be the same even if the red operand were a little less red, so a model that has lost *red* can still answer them.

    The [diverse-op item](/todo/science/diverse-operator-set-hue-saturation-brightness.md) proposes adding ops whose answers spread through the cube and depend on both operands, such as the hue, saturation, and brightness blend modes, alongside some of the saturating ones, so that the removal tests have both kinds to read. Every later D2.2 experiment runs on whatever table comes out of this, so the question comes first.

    [^stream]: The *residual stream* is the running vector of activations that each transformer layer reads from and writes back to; it is where we place the anchor.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The op set

    **What we ran.** We took nine candidate ops, each defined on the same 216-color grid and snapped to it the way the six ops of ex-2.2.3 are, so every line still has an answer in the vocabulary.

    Three of them work channel by channel and are commutative: `difference` and `exclusion`, which are the blend modes of those names, and `hsvmix`, which is `mix` done in HSV (the average hue, saturation, and value). The other six take one attribute from the second operand and the rest from the first: the hue, saturation, and luminosity modes of the W3C compositing spec, as in Photoshop, and the same three in HSV proper, as in Krita.
    """)
    return


@app.function(hide_code=True)
def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    """A result table in the shared classes: numeric cells right-aligned."""
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(row)) + "</tr>"
        for row in rows
    )
    table = f'<div class="report-table-scroll"><table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table></div>'
    return figure_html(table, caption=caption, class_="report-figure")


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class Scout:
    """One op's answers on every ordered pair, and the reads the section takes on them."""

    op: Op

    @cached_property
    def table(self) -> dict[tuple[Rgb, Rgb], Rgb]:
        return {(a, b): self.op(a, b) for a, b in lines()}

    @cached_property
    def counts(self) -> np.ndarray:
        """Unordered pairs whose answer is each grid color, in palette order."""
        index = {c: i for i, c in enumerate(colors())}
        n = np.zeros(len(index))
        for c, k in Counter(self.table[(a, b)] for a, b in unordered_pairs()).items():
            n[index[c]] = k
        return n

    @cached_property
    def spread(self) -> float:
        """Entropy of the answer distribution over the grid, in bits; log2(216) ≈ 7.75 is uniform."""
        p = self.counts / self.counts.sum()
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    @cached_property
    def marginals(self) -> np.ndarray:
        """(channel, level): the share of unordered pairs whose answer has that level in that channel."""
        ans = np.array([self.table[(a, b)] for a, b in unordered_pairs()])
        return np.stack([[np.mean(ans[:, c] == lv) for lv in LEVELS] for c in range(3)])

    @cached_property
    def light_half(self) -> float:
        """The share of unordered pairs whose answer is lighter than mid-gray."""
        return float(np.mean([sum(self.table[(a, b)]) > 1.5 * TOP for a, b in unordered_pairs()]))

    @cached_property
    def commutative(self) -> float:
        return commutativity(self.op)

    @cached_property
    def redder(self) -> int:
        """Lines whose answer is redder than either operand."""
        return sum(redness(ans) > max(redness(a), redness(b)) + 1e-9 for (a, b), ans in self.table.items())

    def dependence(self, pairs: list[tuple[Rgb, Rgb]], drop: int) -> float:
        """The share of *pairs* (ordered) whose answer changes when the redder operand's R falls by *drop*."""
        changed = 0
        for a, b in pairs:
            if redness(a) >= redness(b):
                changed += self.op((max(a[0] - drop, 0), a[1], a[2]), b) != self.table[(a, b)]
            else:
                changed += self.op(a, (max(b[0] - drop, 0), b[1], b[2])) != self.table[(a, b)]
        return changed / len(pairs)

    def sensitivity(self, role: int, seed: int = 0) -> float:
        """The share of lines whose answer changes when the operand in *role* (0 or 1) is replaced by a
        random other color: does the op read that operand at all?
        """
        cs = colors()
        rng = np.random.default_rng(seed)
        changed = 0
        sample = lines()[::5]
        for a, b in sample:
            other = cs[rng.integers(len(cs))]
            swapped = (other, b) if role == 0 else (a, other)
            changed += self.table.get(swapped, self.op(*swapped)) != self.table[(a, b)]
        return changed / len(sample)


@app.cell(hide_code=True)
def _():
    scouts: dict[str, Scout] = {op.name: Scout(op) for op in ALL_OPS}
    red_lines: list[tuple[Rgb, Rgb]] = [(a, b) for a, b in lines() if max(redness(a), redness(b)) >= RED_DOSE - 1e-9]
    """Every red line of the grammar, in both operand orders."""
    _partners = probe_partners(N_PROBE, PROBE_SEED)[1]
    probe_red: list[tuple[Rgb, Rgb]] = [
        (c, b)
        for c, ps in zip(colors(), _partners, strict=True)
        for b in ps
        if max(redness(c), redness(b)) >= RED_DOSE - 1e-9
    ]
    """The red lines of ex-2.2.3's shared probe draw, which reads every non-`mix` op."""
    mix_red: list[tuple[Rgb, Rgb]] = [
        (a, b) for a, b in mix_probe_lines() if max(redness(a), redness(b)) >= RED_DOSE - 1e-9
    ]
    # `mix`'s own probe set: D2.1's closed pairs.
    return mix_red, probe_red, red_lines, scouts


@app.cell(hide_code=True)
def _(scouts: dict[str, Scout]):
    _rows = []
    for _op in ALL_OPS:
        _s = scouts[_op.name]
        _rows.append(
            [
                f"<code>{_op.name}</code>",
                _op.rule,
                "yes" if _s.commutative > 0.999 else f"{_s.commutative:.0%}",
                f"{on_grid(_op):.0%}",
                f"{_s.redder:,} ({_s.redder / len(lines()):.0%})",
            ]
        )
    _caption = """
    <b>The fifteen ops.</b> The six ops of ex-2.2.3, then the nine candidates. <em>Commutative</em> is the share of unordered pairs whose answer is the same in both operand orders. <em>On grid</em> is the share of pairs the rule answers without rounding. <em>Redder than both</em> counts the lines whose answer is redder than either operand, the case the labeller never sees.
    """
    mo.Html(table_html(["op", "rule", "commutative", "on grid", "redder than both"], _rows, _caption))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Where the answers land

    Below is the answer-cloud figure from ex-2.2.3, extended to the candidates. Each panel shows the color cube seen from the gray diagonal, with one mark per grid color, sized by how many pairs answer there. An op that spreads its answers puts small marks everywhere. A saturating op piles them onto a face, an edge, or a corner.
    """)
    return


@app.cell(hide_code=True)
def _(scouts: dict[str, Scout]):
    _full_cell = 600
    _order = np.argsort(GRID_RGB @ CUBE_VIEWS["solid"].toward, kind="stable")
    _xy = project_cube(GRID_RGB[_order])

    @themed(
        name="answer-clouds",
        alt_text="""
            Fifteen color-cube panels, one per op, with a mark on each grid color sized by how many pairs answer there. The six current ops crowd a face or a corner, add and multiply most of all; the nine candidates spread small marks through the cube, with difference and the hue ops leaning dark and luminosity leaning light.
        """,
        caption=f"""
            **Where the answers of each op land.** One mark per grid color, with area proportional to the number of unordered pairs whose answer is that color. All panels share one scale: a mark that fills its grid cell stands for {_full_cell} pairs, out of {len(unordered_pairs()):,}. The top row and the first panel of the second row are the ops of ex-2.2.3; the rest are the candidates.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(3, 5, figsize=(11.5, 8.4))
        for ax, op in zip(axes.flat, ALL_OPS, strict=True):
            draw_cube_bound(ax)
            dia = grid_diameter(len(LEVELS)) * np.sqrt(scouts[op.name].counts[_order] / _full_cell)
            ax.add_collection(
                EllipseCollection(
                    widths=dia,
                    heights=dia,
                    angles=0,
                    units="xy",
                    offsets=_xy,
                    offset_transform=ax.transData,
                    facecolors=GRID_RGB[_order],
                    edgecolors=light_dark("#00000033", "#ffffff55"),
                    linewidths=0.5,
                    zorder=3,
                    clip_on=False,
                )
            )
            ax.set_title(op.name, y=1.12)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### How evenly the answers spread

    We measure evenness two ways. The first is per channel: for each op, the share of answers at each of the six levels of R, G, and B. A flat line at one sixth means the answers cover that channel evenly, and a spike at 0 or 15 means the op saturates. The second is a single number per op, the entropy of the answer distribution over the 216 colors.[^ent]

    [^ent]: Entropy in bits, so a higher number means the answers are spread over more colors.
    """)
    return


@app.cell(hide_code=True)
def _(scouts: dict[str, Scout]):
    @themed(
        name="channel-marginals",
        alt_text="""
            Fifteen small panels, one per op, each with three lines (red, green, blue) showing the share of answers at each of the six grid levels. The current ops spike at 15 (add, screen, lighten) or at 0 (multiply, darken); the candidates run close to the flat one-sixth line, with difference sloping toward 0.
        """,
        caption="""
            **Answer level shares per channel.** For each op, the share of unordered pairs whose answer has each grid level in R, G, and B, drawn in the color of that channel. The dashed line is one sixth, which is what an op with even coverage of the channel would show. For the per-channel ops and `hsvmix` the three lines coincide, because the rule treats the channels alike. Panels share the vertical scale.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(3, 5, figsize=(11.5, 6.4), sharex=True, sharey=True)
        inks = light_dark(["#c33", "#2a2", "#33c"], ["#f66", "#5d5", "#77f"])
        for ax, op in zip(axes.flat, ALL_OPS, strict=True):
            m = scouts[op.name].marginals
            ax.axhline(1 / len(LEVELS), ls="--", lw=0.8, color=light_dark("#888", "#aaa"))
            for c in range(3):
                ax.plot(LEVELS, m[c], "o-", ms=3, lw=1.4, color=inks[c])
            ax.set_title(op.name, fontsize=10)
            ax.set_xticks([0, TOP])
            ax.grid(alpha=0.25)
        for ax in axes[-1]:
            ax.set_xlabel("answer level")
        for ax in axes[:, 0]:
            ax.set_ylabel("share of pairs")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Does the answer need the red operand to be red?

    E8 of ex-2.2.3 calls a red line *dependent* when lowering the R of the red operand by one grid level changes the snapped answer. On the dependent lines of every op, removal came out complete. We apply that rule to every op here, both on every red line and on the shared probe draw. We also add a second rule beside it: drop the R of the red operand to zero instead of by one level.

    The one-level rule asks whether the op notices a small change in the red operand. The to-zero rule asks whether the answer needs *red* at all.
    """)
    return


@app.cell(hide_code=True)
def _(
    mix_red: list[tuple[Rgb, Rgb]],
    probe_red: list[tuple[Rgb, Rgb]],
    red_lines: list[tuple[Rgb, Rgb]],
    scouts: dict[str, Scout],
):
    dep: dict[str, dict[str, float]] = {
        op.name: {
            "step": scouts[op.name].dependence(red_lines, STEP),
            "step_probe": scouts[op.name].dependence(probe_red, STEP),
            "zero": scouts[op.name].dependence(red_lines, TOP),
        }
        for op in ALL_OPS
    }
    """Per op: one-level dependence on all red lines and on the shared probe draw, and to-zero dependence."""
    mix_own: dict[str, float] = {
        "step": scouts["mix"].dependence(mix_red, STEP),
        "zero": scouts["mix"].dependence(mix_red, TOP),
    }
    """`mix` on its own probe set, which is what E8 read."""
    sens: dict[str, tuple[float, float]] = {
        op.name: (scouts[op.name].sensitivity(0), scouts[op.name].sensitivity(1)) for op in ALL_OPS
    }
    # Per op: the share of lines whose answer changes when op1, or op2, is replaced at random.
    return dep, mix_own, sens


@app.cell(hide_code=True)
def _(
    dep: dict[str, dict[str, float]],
    mix_own: dict[str, float],
    mix_red: list[tuple[Rgb, Rgb]],
    probe_red: list[tuple[Rgb, Rgb]],
    red_lines: list[tuple[Rgb, Rgb]],
    scouts: dict[str, Scout],
    sens: dict[str, tuple[float, float]],
):
    _rows = []
    for _op in ALL_OPS:
        _s, _d = scouts[_op.name], dep[_op.name]
        _rows.append(
            [
                f"<code>{_op.name}</code>",
                f"{_s.spread:.2f}",
                f"{_s.light_half:.2f}",
                f"{_d['step']:.2f}",
                f"{_d['step_probe']:.2f}",
                f"{_d['zero']:.2f}",
                f"{sens[_op.name][0]:.2f} / {sens[_op.name][1]:.2f}",
            ]
        )
    _caption = f"""
    <b>Spread and dependence, per op.</b> <em>Spread</em> is the entropy of the answer distribution in bits, where 7.75 is uniform over the grid. <em>Light half</em> is the share of pairs whose answer is lighter than mid-gray. <em>One level</em> is the rule from E8: the share of red lines (dose ≥ {RED_DOSE:g}) whose answer changes when the R of the red operand drops one grid level, given over all {len(red_lines):,} red lines and over the {len(probe_red):,} red lines of the shared probe draw. <em>To zero</em> drops it to 0 instead. <em>Reads op1 / op2</em> is the share of lines whose answer changes when that operand is replaced at random. On its own probe set of {len(mix_red)} red lines, <code>mix</code> is {mix_own["step"]:.2f} by the one-level rule and {mix_own["zero"]:.2f} to zero.
    """
    mo.Html(
        table_html(
            ["op", "spread ↑", "light half", "one level ↑", "one level, probe", "to zero ↑", "reads op1 / op2"],
            _rows,
            _caption,
        )
    )
    return


@app.cell(hide_code=True)
def _(
    dep: dict[str, dict[str, float]],
    mix_own: dict[str, float],
    scouts: dict[str, Scout],
):
    @themed(
        name="spread-vs-dependence",
        alt_text="""
            Scatter of one-level dependence against answer spread, one labelled point per op. The current ops sit at the left, low spread, with dependence from 0.2 to 0.8; the candidates sit to the right at higher spread, the non-commutative HSV ops at 0.4 to 0.55 dependence and the commutative ones higher, with difference at the top right.
        """,
        caption=f"""
            **Spread against dependence.** For each op, the entropy of its answers in bits against the share of its red lines whose answer changes when the R of the red operand drops one level. Filled marks are commutative ops and open marks are not. Gray marks are the ops of ex-2.2.3, blue the candidates. The dotted line is `mix` on its own probe set ({mix_own["step"]:.2f}), the level E8 compared the other ops against.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        inks = {True: light_dark("#666", "#aaa"), False: light_dark("#2557b8", "#7fa6ff")}
        nudge = {
            "hue": (-8, -13),
            "luminosity": (5, 4),
            "value-hsv": (-52, -1),
            "sat-hsv": (5, -10),
            "saturation": (5, 7),
            "multiply": (5, 0),
            "mix": (5, -8),
        }
        ax.axhline(mix_own["step"], ls=":", lw=1, color=light_dark("#999", "#888"))
        for op in ALL_OPS:
            s = scouts[op.name]
            x, y = s.spread, dep[op.name]["step"]
            ink = inks[op in OPS]
            comm = s.commutative > 0.98
            ax.plot(x, y, "o", ms=7, mfc=ink if comm else "none", mec=ink, mew=1.4)
            ax.annotate(
                op.name, (x, y), xytext=nudge.get(op.name, (5, 4)), textcoords="offset points", fontsize=8.5, color=ink
            )
        ax.set(xlabel="spread of answers (bits)", ylabel="dependence on the red operand (one level)")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(
    dep: dict[str, dict[str, float]],
    mix_own: dict[str, float],
    scouts: dict[str, Scout],
):
    _cand = [op.name for op in CANDIDATES]
    _cur = [op.name for op in OPS]
    _hsv = ["hue", "saturation", "luminosity", "hue-hsv", "sat-hsv", "value-hsv"]
    _comm = ["difference", "exclusion", "hsvmix"]
    _hi_spread = min(scouts[n].spread for n in _cand if n != "exclusion")
    _cur_max = max(scouts[n].spread for n in _cur)
    mo.md(rf"""
    **What we saw.**

    *The candidates spread; the current ops crowd.* Every candidate except `exclusion` has an answer entropy above {_hi_spread:.1f} bits, and no current op reaches {_cur_max:.1f}. `exclusion` sits between the two groups, with a third of its answers at each middle level.

    The channel marginals say why: `add`, `screen`, and `lighten` pile up at 15, `multiply` and `darken` at 0, and the candidates run near the flat line. So on evenness the proposed hue, saturation, and brightness ops do what the item hoped, and so do the three commutative candidates.

    *The dependence rule from E8 responds to rounding as much as to the op.* By the one-level rule, `mix` itself is dependent on {mix_own["step"]:.0%} of its own red probe lines, and {dep["mix"]["step_probe"]:.0%} on the shared draw used for the other ops. That is because a one-level drop in one operand moves the mean by half a level, and the snap sends half of those cases back to the original answer.

    The hue, saturation, and brightness ops sit at {min(dep[n]["step"] for n in _hsv):.2f} to {max(dep[n]["step"] for n in _hsv):.2f}, no better than `screen` or `multiply`, for a structural reason: each takes one attribute from one operand and the rest from the other, and scaling the R of a pure red changes only its value. So whichever role reads hue or saturation from the red operand never sees the drop.

    Under the to-zero rule every op, current and candidate, is above {min(d["zero"] for d in dep.values()):.2f}. What differs between ops is sensitivity to a small change in the red operand, and that is where the commutative candidates stand out: `difference` at {dep["difference"]["step"]:.2f}, `hsvmix` at {dep["hsvmix"]["step"]:.2f}, and `exclusion` at {dep["exclusion"]["step"]:.2f}.

    *The blend modes are the first ops where operand order matters.* All six hue, saturation, and brightness variants agree with their own reverse on under {max(scouts[n].commutative for n in _hsv):.0%} of pairs. The saturation ones read op2 on only {min(scouts[n].sensitivity(1) for n in ("saturation", "sat-hsv")):.0%} of lines, since many partners share a saturation. The grammar of ex-2.2.3 is commutative throughout: the probe sets walk every color as op1, and the labeller pools both operands the same way. A non-commutative op would put role information into the grammar for the first time, a design change that would need its own experiment, and the anchored-op work does not need it.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### How often the op word is worth reading

    The D2.2 design uses a quantity called *op-relevance*: for each line of the anchored op, how many other ops in the table give the same answer. At 0 the answer identifies the op on its own. At k the op word only rules out the other k. The suppression experiments predict per-line damage from this number, so widening the table changes the prediction for every op in it.

    We compare three tables: the one from ex-2.2.3, table A (drop `add`, add the three commutative candidates), and table B (the HSV trio the item proposed, alongside the current six).
    """)
    return


@app.cell(hide_code=True)
def _():
    _by_name = {op.name: op for op in ALL_OPS}
    _blocks = []
    for _label, _names in TABLES.items():
        _ops = tuple(_by_name[n] for n in _names)
        _dists = {n: relevance(_by_name[n], _ops) for n in _names}
        _levels = sorted({k for d in _dists.values() for k, v in d.items() if v >= 0.005})
        _head = ["anchored op", *(str(k) for k in _levels)]
        _rows = [
            [f"<code>{n}</code>", *(f"{d[k]:.0%}" if d.get(k, 0) >= 0.005 else "·" for k in _levels)]
            for n, d in _dists.items()
        ]
        _blocks.append(
            table_html(
                _head,
                _rows,
                f"<b>Op-relevance under table {_label}</b> (<code>{'</code>, <code>'.join(_names)}</code>): for each anchored op, the share of its unordered pairs on which k other ops of the table give the same answer. A dot means under half a percent.",
            )
        )
    mo.vstack([mo.Html(b) for b in _blocks])
    return


@app.cell(hide_code=True)
def _(scouts: dict[str, Scout]):
    _a = TABLES["A"]
    _by_name = {op.name: op for op in ALL_OPS}
    _ops_a = tuple(_by_name[n] for n in _a)
    _mix_a = relevance(_by_name["mix"], _ops_a)
    _mix_cur = relevance(_by_name["mix"])
    mo.md(rf"""
    Widening the table makes `mix` much less distinctive. Today it is alone in its answer on {_mix_cur[0]:.0%} of pairs; under table A that falls to {_mix_a[0]:.0%}, because `hsvmix` agrees with it on a third of pairs, the ones whose operands are close in hue. We want that: the per-line relevance prediction needs more than one level to read, and `hsvmix` sharing the answers of `mix` on a known set of lines is what supplies it.

    `difference` is the most distinctive op in table A. It also has the most redder-than-both lines ({scouts["difference"].redder:,}, over a quarter of its lines), so it is where the blind-span question (E3 of ex-2.2.3) would get the most lines.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### What we make of it

    **Does it change the design?** Yes, in two places.

    The diverse table should be the commutative widening, table A: keep `mix`, `screen`, `multiply`, `lighten`, and `darken`, drop `add`, and add `difference`, `exclusion`, and `hsvmix`. We drop `add` because a fifth of its pairs go to white and it is the least dependent op by either rule. Table A gives the E8 split both kinds of op to read, as the item asks, without putting operand order into the grammar.

    The removal statistic should also be filtered by dependence when it is scored, as E8 did after the fact, but with the filter written into the prereg rather than found in the discussion. The one-level rule is the one to keep, because it matches the size of change an intervention makes. We now know it means "sensitive to a small change" rather than "needs *red*".

    **What we would do differently.** The hue, saturation, and brightness ops were the headline of the item, and they turn out to be the wrong tool for the removal test: they read one attribute of one operand, whereas *red* as the labeller defines it, r·(1 − g/2 − b/2), is a magnitude that hue and saturation do not see. They stay in the library as candidates. If the D2.2 program later wants an op that reads operand roles, these are the ones to reach for, in an experiment of its own.

    **What this does not settle.** Whether the model learns table A as well as it learned the six-op table is a training question. E4 of ex-2.2.3 found the operand cube less linearly decodable at six ops than at three, with lines per op as a confound. Eight ops at the same corpus size would sharpen that confound, so the prereg should decide whether to hold lines per op fixed instead. The scouting questions still to run may move the operating point, but they do not move the table.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    Everything in this section is computed on the grid when the notebook runs, using [`sca.data.ops`](/src/sca/data/ops.py): the six ops of ex-2.2.3 as `OPS` and the nine candidates as `CANDIDATES`, both snapped to the 216-color grid by the same `snap`.

    **Lines.** There are 46,656 ordered pairs of grid colors, each one an (op1, op2) line, and 23,436 unordered pairs, which we use where a statistic is symmetric. A *red line* has dose ≥ 0.8, where dose is the larger of the two operand rednesses, as in ex-2.2.3. The shared probe draw is the one from ex-2.2.3: every color as op1 against 27 partners drawn once from the grid, with the same partners for every op other than `mix`. The probe set for `mix` is the 5,832 closed pairs from D2.1.

    **Spread** is the Shannon entropy, in bits, of the distribution of answers over the 216 grid colors, taken over unordered pairs. **Channel marginals** are the share of unordered pairs whose answer has each grid level in each channel. **Light half** is the share of unordered pairs whose answer sums to more across its channels than mid-gray does.

    **Dependence** follows E8. For a red line, take the redder operand, lower its R by one grid level (3) or to zero, leave everything else alone, and ask whether the snapped answer changes. We report it as a share of the lines in the set named. **Reads op1 / op2** replaces that operand with a color drawn uniformly from the grid (seed 0, one draw per line, every fifth line) and asks the same question.

    **Commutative** is the share of unordered pairs on which the op gives the same answer in both orders. `hsvmix` falls short of 1 on the pairs whose hues are opposite, where the circular mean is undefined and the rule keeps the hue of op1.

    **Op-relevance** is `relevance` from the same module, now over an arbitrary table: for each unordered pair, the number of other ops in the table that give the same answer as the anchored op.
    """)
    return


if __name__ == "__main__":
    app.run()
