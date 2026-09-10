import marimo

__generated_with = "0.24.0"
app = marimo.App(
    app_title="Pilot: stochastic rounding of off-grid answers",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    # mini:manual-publish — a pilot, published to the dev pair by hand; it carries no production pin.
    import json
    import tempfile
    from dataclasses import dataclass
    from pathlib import Path
    from typing import cast

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import experiment as ex
    from mini.hf_store import HFStore
    from mini.reports import report_bundle, use_publisher
    from mini.runs import data_root
    from mini.store import LocalStore, Store, project_store
    from mini.vis import AxesRow, figure_html, light_dark, themed

    use_publisher(report_bundle(__file__))

    PROD_BUCKET = "z0u/sca2-store"
    """The production store, read for ex-2.2.3's `recipe-short` and `control-short` seeds; this pilot's own
    results live in whichever store the profile names (the dev pair, as run)."""

    SLICE_NAMES = ["emb", "1", "2", "3", "4"]
    POS_NAMES = ["op1", "op", "op2", "=", "ans", "⏎"]
    ROUNDED_OPS = ("mix", "screen", "multiply")
    """The ops whose answers round at all; `add`, `lighten` and `darken` land on the grid on every pair."""

    INK = {
        "control-stoch": ("#d0461b", "#f07a50"),
        "recipe-stoch": ("#1f6fb4", "#5fa8dd"),
        "control-near": ("#777", "#999"),
        "control-short": ("#bbb", "#666"),
        "recipe-short": ("#9ec3e6", "#3d6a8f"),
        "control": ("#bbb", "#666"),
    }
    """One ink per condition, as (light, dark) pairs; production arms draw pale."""

    None


@app.function(hide_code=True)
def prod_store() -> Store:
    """A read-only view of the production bucket, cached beside the project's own store cache."""
    return HFStore(PROD_BUCKET, cache=LocalStore(data_root() / "store-cache" / "prod"))


@app.function(hide_code=True)
def load_json(ref: str, store: Store | None = None) -> dict | None:
    store = store or project_store()
    art = store.get_refs([ref])[ref]
    if art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "data.json")])
        return json.loads(path.read_text())


@app.function(hide_code=True)
def load_npz(ref: str, store: Store | None = None) -> dict[str, np.ndarray] | None:
    store = store or project_store()
    art = store.get_refs([ref])[ref]
    if art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "arrays.npz")])
        with np.load(path) as z:
            return {k: z[k] for k in z.files}


@app.function(hide_code=True)
def span2(v: np.ndarray, fmt: str = ".3f") -> str:
    """Seed mean with half the seed range beside it, in the shared `.range` style."""
    v = np.asarray(v, float)
    if len(v) == 1:
        return f"{v[0]:{fmt}}"
    return f"{v.mean():{fmt}} <span class='range'>±{(v.max() - v.min()) / 2:{fmt}}</span>"


@app.function(hide_code=True)
def ink(cond: str):
    return light_dark(*INK[cond])


@app.function(hide_code=True)
def table_html(head: list[str], rows: list[list[str]], caption: str, *, ref_rows: frozenset[int] = frozenset()) -> str:
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        f"<tr{' class=ref' if r in ref_rows else ''}>"
        + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(row))
        + "</tr>"
        for r, row in enumerate(rows)
    )
    table = f'<div class="report-table-scroll"><table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table></div>'
    return figure_html(table, caption=caption, class_="report-figure")


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class Results:
    """The pilot's published results, and the production ex-2.2.3 arms they are read against."""

    metrics: dict
    arrays: dict[str, np.ndarray]
    geometry: dict
    prod: dict
    prod_geometry: dict

    def runs(self, cond: str) -> list[dict]:
        return sorted((r for r in self.metrics["runs"] if r["condition"] == cond), key=lambda r: r["seed"])

    def rounding(self, cond: str) -> list[dict]:
        return sorted((r for r in self.metrics["rounding"] if r["condition"] == cond), key=lambda r: r["seed"])

    def scored(self, cond: str) -> list[dict]:
        return sorted((r for r in self.metrics["scores"] if r["condition"] == cond), key=lambda r: r["seed"])

    def prod_runs(self, cond: str) -> list[dict]:
        """A production arm's eval records; `recipe-short` pools its addendum seeds (twenty in all)."""
        names = (cond, f"{cond}-more")
        return sorted((r for r in self.prod["runs"] if r["condition"] in names), key=lambda r: r["seed"])

    def prod_scored(self, cond: str) -> list[dict]:
        names = (cond, f"{cond}-more")
        return sorted((r for r in self.prod["scores"] if r["condition"] in names), key=lambda r: r["seed"])

    @staticmethod
    def stat(runs: list[dict], key: str, op: str | None = None) -> np.ndarray:
        return np.array([r[key] if op is None else r["per_op"][op][key] for r in runs], float)

    @staticmethod
    def em(runs: list[dict], op: str) -> np.ndarray:
        return np.array([r["holdout_em"][op] for r in runs], float)

    def round_stat(self, cond: str, op: str, split: str, key: str) -> np.ndarray:
        return np.array([r["per_op"][op][split][key] for r in self.rounding(cond)], float)

    def lines(self, cond: str) -> dict[str, np.ndarray]:
        """The per-line rounding table of a condition, seeds concatenated."""
        keys = [k.split("/")[-1] for k in self.arrays if k.startswith(f"{cond}-s0/lines/")]
        return {k: np.concatenate([self.arrays[f"{r['label']}/lines/{k}"] for r in self.rounding(cond)]) for k in keys}

    @staticmethod
    def score(scored: list[dict], op: str, iv: str | None, key: str, group: str) -> np.ndarray:
        out = []
        for r in scored:
            s = r["ops"][op]
            v = s["clean"][key] if iv is None else s["interventions"][iv][key]
            out.append(v[group])
        return np.array(out, float)

    def r2(self, cond: str, prod: bool = False) -> np.ndarray:
        """(seeds, slices, positions, targets, channels) strict-holdout R² of a control arm's cube probes."""
        src = self.prod_geometry if prod else self.geometry
        return np.array([r["r2_strict"] for r in src["runs"] if r["condition"] == cond], float)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Pilot: stochastic rounding of off-grid answers

    /// tip |
    <!-- tl;dr -->
    A scouting run, no gates. We retrained the six-op grammar's un-anchored control and the adopted recipe on a corpus whose off-grid answers round *stochastically* (the upper level with probability equal to the fraction of the way up), and read what that does to the exact-match ceiling, to where the model puts its answer mass, to anchoring, and to the redder-than-both counts.
    ///

    ## Observations

    /// admonition | TODO
    One line per observation, each with its deciding number and the noise it was read against.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        rf"""
    ## Why, and what we ran

    Ex-2.2.3's grammar computes each answer channel on the 0..15 scale and snaps it to the nearer grid level, so `screen`, `multiply`, and (off its on-grid pairs) `mix` are step functions of their raw value. The [stochastic-rounding item](https://github.com/z0u/sca2/blob/main/todo/science/stochastic-rounding-of-off-grid-answers.md) asks what a variant that rounds by coin flip would settle: (a) the exact-match ceiling, and whether exact match is then the wrong statistic; (b) whether the answer's representation becomes more graded; (c) whether the redder-than-both counts, and so E3, move.

    The variant is `sca.data.ops.Rounding = "stochastic"`: per channel and per line, the upper neighbour is drawn with probability equal to how far up the interval the raw value sits, so the mean target is the raw value itself and a tie is a coin flip. The (op, pair) sequence of the corpus and the eval lines are the production ones at the same seed; only the answers of rounded lines differ. Three arms, all at the adopted point's length ({ex.EPOCHS} epochs), all under ex-2.2.3's either-slot labeller:

    | condition | seeds | corpus | anchor |
    | --- | ---: | --- | --- |
    {chr(10).join(f"| `{c.name}` | {c.seeds} | {ex.ROUNDING[c.name]} | {'λ = ' + str(c.lam) if c.lam else 'none'} — {c.title} |" for c in ex.CONDITIONS)}

    `control-near` is two fresh seeds of production's `control-short`, trained here so the rounding readout and the cube probes have a same-code nearest-rounded comparison. The anchored arm reads against production's `recipe-short` at twenty seeds. Every task but the corpus build and the rounding readout is ex-2.2.3's code, loaded from its module unchanged, so the placement statistics mean what they meant there.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    _metrics = load_json(ex.METRICS_REF)
    mo.stop(_metrics is None, mo.md("_Results are not published yet; the result cells render once they are._"))
    assert _metrics is not None
    _arrays, _geometry = load_npz(ex.ARRAYS_REF), load_json(ex.GEOMETRY_REF)
    _prod = prod_store()
    _pm, _pg = load_json(ex.EX223_METRICS_REF, _prod), load_json(ex.EX223_GEOMETRY_REF, _prod)
    assert _arrays and _geometry and _pm and _pg, "a result is missing from the store"
    res: Results = Results(_metrics, _arrays, _geometry, _pm, _pg)
    return (res,)


@app.cell(hide_code=True)
def _(res: Results):
    mo.md(
        r"""
    ## The grammar under each rounding

    With no model in the loop: how much of each op rounds, the exact-match ceiling a stochastic target imposes (the mean over unordered pairs of the mode's probability, which is the most a predictor that knows the rule can match a drawn answer), and the redder-than-both count of each op's lines, where under stochastic rounding an answer counts with its probability.
    """
    )
    _g = res.metrics["grammar"]
    _rows = [
        [
            f"<code>{op}</code>",
            f"{g['rounded_frac']:.1%}",
            f"{g['ceiling']:.3f}",
            f"{g['ceiling_rounded']:.3f}",
            f"{g['redder_nearest']:,}",
            f"{g['redder_stochastic']:,.0f}",
            f"{g['redder_stochastic'] / g['redder_nearest'] - 1:+.1%}" if g["redder_nearest"] else "—",
        ]
        for op, g in _g.items()
    ]
    _head = [
        "op",
        "pairs that round",
        "EM ceiling",
        "ceiling on rounded pairs",
        "redder (nearest)",
        "redder (stochastic, expected)",
        "change",
    ]
    grammar_table = mo.Html(
        table_html(
            _head,
            _rows,
            "The six ops under the two rounding rules. The ceiling is 1 on the three ops that never round; on the other three it is the mean mode probability over all unordered pairs, and over the pairs that round. Redder-than-both counts are over each op's 46,656 lines.",
        )
    )
    grammar_table
    return


@app.cell(hide_code=True)
def _(res: Results):
    mo.md(
        r"""
    ## Exact match against a moving target

    Ex-2.2.3 reads holdout exact match against the corpus answer. On a stochastic corpus that answer is one draw, so the statistic has a ceiling below 1 on the rounded ops however well the model knows the rule. Three readings of the same eval pass, all on held-out pairs: exact match against the drawn answer (what the production statistic is), against the rule's mode (the nearest answer, which a stochastic model should still name), and the *expected* exact match, which is the chance a fresh draw of the line would match the model's guess and is what the drawn-answer statistic converges to over many draws.
    """
    )
    _ops = list(ex.ex223.OP_NAMES)
    _conds = [c.name for c in ex.CONDITIONS]
    _x = np.arange(len(_ops))
    _ceiling = np.array([res.metrics["grammar"][op]["ceiling"] for op in _ops])
    _em = {c: np.array([res.em(res.runs(c), op) for op in _ops]) for c in _conds}  # (ops, seeds)
    _mode = {c: np.array([res.round_stat(c, op, "holdout", "em_mode") for op in _ops]) for c in _conds}
    _exp = {c: np.array([res.round_stat(c, op, "holdout", "expected_em") for op in _ops]) for c in _conds}
    _prod_em = np.array([res.em(res.prod_runs("control-short"), op) for op in _ops])

    @themed(
        name="em-readings",
        alt_text="""
            Three panels side by side, each with the six ops on the horizontal axis and accuracy from 0 to 1 on the vertical. Left: exact match against the drawn answer; the two stochastic-corpus arms sit near a short black ceiling tick on mix, screen and multiply and near 1 elsewhere, while the nearest-rounded arms sit near 1 on every op. Middle: exact match against the rule's mode, where every arm sits near 1. Right: expected exact match, where the stochastic arms again meet the ceiling ticks.
        """,
        caption="""
            **Three readings of holdout accuracy, per op and condition.** Each mark is one seed. Left: exact match against the corpus answer, the production statistic. The black tick is the stochastic ceiling of the op (its mean mode probability); pale marks are production's `control-short`, on the nearest-rounded corpus. Middle: exact match against the rule's mode, the nearest answer. Right: the exact match a model would score on average over fresh draws of each line, which is what the left panel estimates with one draw.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6), sharey=True, layout="constrained")
        axes = cast(AxesRow, axes)
        off = np.linspace(-0.27, 0.27, len(_conds))
        for ax, data, title in zip(
            axes, (_em, _mode, _exp), ("vs the drawn answer", "vs the mode", "expected over draws"), strict=True
        ):
            for i, c in enumerate(_conds):
                v = data[c]
                for s in range(v.shape[1]):
                    ax.plot(_x + off[i], v[:, s], "o", ms=3.5, color=ink(c), alpha=0.85, label=c if s == 0 else None)
            if data is _em:
                for s in range(_prod_em.shape[1]):
                    ax.plot(_x - 0.4, _prod_em[:, s], "o", ms=3, color=ink("control-short"), alpha=0.7)
            for xi, cl in zip(_x, _ceiling, strict=True):
                ax.hlines(cl, xi - 0.42, xi + 0.42, color=light_dark("#000", "#fff"), lw=1.2)
            ax.set_xticks(_x, _ops, rotation=30, ha="right")
            ax.set_title(title, fontsize=9)
        axes[0].set_ylim(0.3, 1.03)
        axes[0].set_ylabel("holdout accuracy")
        axes[0].legend(fontsize=7, loc="lower left", frameon=False)
        return fig

    _plot()
    return


@app.cell(hide_code=True)
def _(res: Results):
    _conds = [_c.name for _c in ex.CONDITIONS]
    _rows = []
    for _c in _conds:
        for _op in ROUNDED_OPS:
            _rows.append(
                [
                    f"<code>{_c}</code>",
                    f"<code>{_op}</code>",
                    f"{res.metrics['grammar'][_op]['ceiling']:.3f}",
                    span2(res.em(res.runs(_c), _op)),
                    span2(res.round_stat(_c, _op, "holdout", "expected_em")),
                    span2(res.round_stat(_c, _op, "holdout", "em_mode")),
                    span2(res.round_stat(_c, _op, "holdout", "em_mode_rounded")),
                ]
            )
    _head = ["condition", "op", "ceiling", "EM vs drawn", "expected EM", "EM vs mode", "EM vs mode, rounded pairs"]
    mo.Html(
        table_html(
            _head,
            _rows,
            "The three readings on the ops that round, held-out pairs. Seed means with half the seed range. The last column restricts the mode reading to the pairs that round at all, where the two corpora differ.",
        )
    )
    return


@app.cell(hide_code=True)
def _(res: Results):
    mo.md(
        r"""
    ## Where the answer mass goes

    A model trained on a stochastic target has a reason to spread its answer probability over the two (or up to eight) candidate colors of a line in proportion to their probabilities, which is what "reads the effective geometry rather than the snap" meant in the item. The read is on the rounded eval lines only, pooled over ops and splits: the model's probability on the mode against the mode's true probability (a calibrated model sits on the diagonal; a nearest-trained one sits near 1 whatever the ceiling), and the model's total mass on the line's candidate set (near 1 for any model that has learned the rule, whichever way it rounds).
    """
    )
    _conds = [_c.name for _c in ex.CONDITIONS]
    _edges = np.linspace(0.1, 1.0, 10)
    _binned: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for _c in _conds:
        _t = res.lines(_c)
        _r = _t["rounded"]
        _b = np.clip(np.digitize(_t["ceiling"][_r], _edges) - 1, 0, len(_edges) - 2)
        xs, pm, sup, n = [], [], [], []
        for _k in range(len(_edges) - 1):
            m = _b == _k
            if m.sum() >= 20:
                xs.append(_t["ceiling"][_r][m].mean())
                pm.append(_t["p_mode"][_r][m].mean())
                sup.append(_t["support"][_r][m].mean())
                n.append(m.sum())
        _binned[_c] = (np.array(xs), np.array(pm), np.array(sup), np.array(n))

    @themed(
        name="calibration",
        alt_text="""
            Two panels. Left: the model's probability on the mode answer against the mode's true probability, from about 0.15 to 1, with a dashed diagonal. The stochastic-corpus arms run close to the diagonal; the nearest-rounded arm stays near the top whatever the true probability. Right: the model's total mass on the candidate colors against the same axis; every arm sits near 1.
        """,
        caption="""
            **Calibration of the answer distribution on rounded lines.** Lines binned by the true probability of their mode answer (the line's exact-match ceiling), pooled over ops, splits and seeds; each mark is a bin with at least twenty lines, placed at the bin's mean ceiling. Left: mean model probability on the mode; the dashed line is perfect calibration. Right: mean model mass on the line's candidate set, the colors the stochastic rule can produce.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), layout="constrained")
        axes = cast(AxesRow, axes)
        axes[0].plot([0, 1], [0, 1], "--", lw=0.8, color=light_dark("#888", "#aaa"))
        for _c in _conds:
            xs, pm, sup, _ = _binned[_c]
            axes[0].plot(xs, pm, "o-", ms=3.5, lw=1, color=ink(_c), label=_c)
            axes[1].plot(xs, sup, "o-", ms=3.5, lw=1, color=ink(_c))
        axes[0].set(xlabel="P(mode) under the rule", ylabel="model P(mode)", xlim=(0, 1.02), ylim=(0, 1.02))
        axes[1].set(
            xlabel="P(mode) under the rule", ylabel="model mass on the candidates", xlim=(0, 1.02), ylim=(0.5, 1.02)
        )
        axes[0].legend(fontsize=7, frameon=False, loc="upper left")
        return fig

    _plot()
    return


@app.cell(hide_code=True)
def _(res: Results):
    _conds = [_c.name for _c in ex.CONDITIONS]
    _rows = []
    for _c in _conds:
        for _op in ROUNDED_OPS:
            _rows.append(
                [
                    f"<code>{_c}</code>",
                    f"<code>{_op}</code>",
                    span2(res.round_stat(_c, _op, "holdout", "kl_rounded")),
                    span2(res.round_stat(_c, _op, "holdout", "p_mode_rounded")),
                    span2(res.round_stat(_c, _op, "holdout", "ceiling_rounded")),
                    span2(res.round_stat(_c, _op, "holdout", "support_rounded")),
                ]
            )
    _head = ["condition", "op", "KL(rule ‖ model) ↓", "model P(mode)", "rule P(mode)", "mass on candidates ↑"]
    mo.Html(
        table_html(
            _head,
            _rows,
            "The same readings as means over the rounded held-out lines of each _op. KL is from the rule's answer distribution to the model's, over the palette; it is zero for a model that reproduces the rule's spread and grows as the model commits to one level.",
        )
    )
    return


@app.cell(hide_code=True)
def _(res: Results):
    mo.md(
        r"""
    ## Anchoring on the stochastic corpus

    The labeller and the anchor are ex-2.2.3's; only the corpus changed. The placement statistics of the anchored arm against production's `recipe-short` at twenty seeds, on the `mix` probe lines, plus the two task-cost reads: holdout exact match on `mix` (whose held-out lines include rounded pairs) and against the mode.
    """
    )
    _p = res.prod_runs("recipe-short")
    _r = res.runs("recipe-stoch")
    _stats = [
        ("m_line", "m_line ↑"),
        ("r2_sim", "r² grading ↑"),
        ("contrast", "contrast ↑"),
        ("lead_emb", "lead ↓"),
        ("latch_pi", "latch π ↓"),
        ("alpha_op1", "ᾱ op1 ↓"),
        ("retention", "retention ↑"),
    ]
    _rows = [
        ["<code>recipe-stoch</code>", str(len(_r))]
        + [span2(res.stat(_r, k)) for k, _ in _stats]
        + [span2(res.em(_r, "mix"))],
        ["<code>recipe-short</code> (production)", str(len(_p))]
        + [span2(res.stat(_p, k)) for k, _ in _stats]
        + [span2(res.em(_p, "mix"))],
    ]
    _head = ["condition", "seeds", *[t for _, t in _stats], "holdout EM mix"]
    mo.Html(
        table_html(
            _head,
            _rows,
            "Ex-2.2.3's gated statistics on the recipe, stochastic corpus against production. Arrows are the direction the gates read as good. Seed means with half the seed range.",
            ref_rows=frozenset({1}),
        )
    )
    return


@app.cell(hide_code=True)
def _(res: Results):
    _iv = ex.ex223.PROJECTION.name
    _rows = []
    for _name, _sc in (
        ("recipe-stoch", res.scored("recipe-stoch")),
        ("recipe-short (production)", res.prod_scored("recipe-short")),
    ):
        for _op in ("mix", "multiply"):
            _rows.append(
                [
                    f"<code>{_name}</code>",
                    f"<code>{_op}</code>",
                    span2(res.score(_sc, _op, None, "acc", "red"), ".2f"),
                    span2(res.score(_sc, _op, _iv, "acc", "red"), ".2f"),
                    span2(
                        res.score(_sc, _op, None, "acc", "nonred") - res.score(_sc, _op, _iv, "acc", "nonred"), ".3f"
                    ),
                    span2(res.score(_sc, _op, None, "acc", "redder"), ".2f"),
                    span2(res.score(_sc, _op, _iv, "acc", "redder"), ".2f"),
                ]
            )
    _head = [
        "condition",
        "op",
        "red acc clean",
        "red acc projected ↓",
        "non-red deficit ↓",
        "redder acc clean",
        "redder acc projected",
    ]
    mo.Html(
        table_html(
            _head,
            _rows,
            "The eval contract's suppression reads under the full projection, on the probe lines (nearest-rounded, as production's). Red lines have dose ≥ 0.8; non-red ≤ 0.2; redder lines have an answer redder than both operands.",
            ref_rows=frozenset({2, 3}),
        )
    )
    return


@app.cell(hide_code=True)
def _(res: Results):
    mo.md(
        r"""
    ## The answer's geometry

    Question (b) of the item: does a stochastic target make the answer's representation more graded? The cube probes are ex-2.2.3's: strict-holdout ridge R² of each slot's RGB at every (slice, position) site, on `mix`'s on-grid probe lines, where no rounding ever happens. The read is the answer's own RGB at `=` (the prediction site) and at the answer position, un-anchored control on each corpus, with production's full-length `control` pale.
    """
    )
    _stoch, _near, _prod = res.r2("control-stoch"), res.r2("control-near"), res.r2("control", prod=True)
    _t = 2  # the answer target
    _x = np.arange(len(SLICE_NAMES))

    @themed(
        name="answer-r2",
        alt_text="""
            Two panels, one for the equals position and one for the answer position, each with the five residual-stream slices on the horizontal axis and strict-holdout R-squared from 0 to 1 on the vertical. Lines for the stochastic-corpus control and the nearest-rounded control track each other closely in both panels, with production's full-length control pale beside them.
        """,
        caption="""
            **Strict-holdout R² of the answer's RGB, by slice, at `=` and at the answer position.** Mean over the three channels; one line per seed. Production's `control` ran twice as long as the two pilot arms.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True, layout="constrained")
        axes = cast(AxesRow, axes)
        for ax, pos in zip(axes, (3, 4), strict=True):
            for s in _prod:
                ax.plot(_x, s[:, pos, _t].mean(-1), "-", lw=1, color=ink("control"), alpha=0.8)
            for name, arr in (("control-near", _near), ("control-stoch", _stoch)):
                for i, s in enumerate(arr):
                    ax.plot(
                        _x, s[:, pos, _t].mean(-1), "o-", ms=3, lw=1.2, color=ink(name), label=name if i == 0 else None
                    )
            ax.set_xticks(_x, SLICE_NAMES)
            ax.set_title(f"answer RGB at {POS_NAMES[pos]}", fontsize=9)
        axes[0].set(ylabel="strict R²", ylim=(0, 1.02))
        axes[0].legend(fontsize=7, frameon=False, loc="upper left")
        return fig

    _plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What we make of it

    /// admonition | TODO
    What we would do differently, and whether this changes the D2.2 design: which statistic to read under a stochastic target, and whether the variant earns a place in an anchored-op experiment.
    ///

    ## Method notes

    - **Corpus.** `sca.data.ops.sample_corpus(..., rounding="stochastic")` and `eval_sets(..., rounding="stochastic")`: the (op, pair) sequence and the eval lines are the production ones at seed 0; each rounded line's answer is one draw from its own stream. The probe lines keep the nearest answer: they read the prompt's geometry, and on `mix` they are on-grid lines that never round.
    - **Runs.** The d64-L4 nGPT and the ex-2.2.3 training step, fifty epochs (1,650 steps), the either-slot labeller at the per-slot rate; the recipe is λ = 0.1, τ = 0.1, anti-subspace 2.5 → 0.3 by 90%, annealed anchor. Seeds 0–1 (controls) and 0–2 (recipe).
    - **Rounding readout.** Per eval line, the rule's answer distribution is the product of its per-channel two-point distributions (`answer_dist`); the ceiling is the mode's probability (`mode_prob`). The model's distribution is its softmax at the pre-answer position restricted to the 216 palette tokens, renormalized for the KL only.
    - **Everything else** is ex-2.2.3's: the eval pass, the eval contract's scorer, and the cube probes, run from that module's functions on this pilot's checkpoints.
    """)
    return


if __name__ == "__main__":
    app.run()
