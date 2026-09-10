import marimo

__generated_with = "0.24.0"
app = marimo.App(
    app_title="Ex 2.2.3: the multi-op grammar, with red anchored again",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import itertools
    import json
    import tempfile
    from collections import Counter
    from dataclasses import dataclass
    from pathlib import Path
    from typing import cast

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.transforms import Affine2D

    # The design constants and the result refs come from `experiment.py` beside
    # this notebook (Marimo puts the notebook directory on sys.path). The prose
    # quotes the frozen gates, and that module carries the same numbers with each
    # gate's wording in its docstring.
    import experiment as ex
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import (
        AxesGrid,
        AxesRow,
        figure_html,
        light_dark,
        smooth_step,
        smooth_step_area,
        smooth_step_band,
        themed,
    )
    from sca.anchoring import softmin_weights
    from sca.vis import CUBE_VIEWS, draw_cube_bound, grid_diameter, project_cube
    from sca.vis_probes import draw_traces

    use_publisher(report_bundle(__file__))

    SURVEY_STATS = ("m_line", "r2_sim", "contrast", "alpha_op1", "holdout_em", "retention")
    """The statistics the survey scored its trials on, in the order the H3 table prints them."""

    SLICE_NAMES = ["emb", "1", "2", "3", "4"]
    POS_NAMES = ["op1", "op", "op2", "=", "ans", "⏎"]
    ROLES = POS_NAMES[: ex.SPAN]
    """The span roles the softmin profiles run over; `op` is the op word, where D2.1 had `+`."""

    EX221_METRICS_REF = "reports/m2/ex-2.2.1/metrics"
    EX2110_PRIMARY = "either-t100"
    EX2110_CONTROL = "lam0"
    EX221_PROJECTION = "primary"
    """Ex-2.2.1 named its full projection `primary`; the same row is `projection` here."""

    INK = {
        "control": ("#555", "#aaa"),
        "control-short": ("#999", "#777"),
        "recipe": ("#1f6fb4", "#5fa8dd"),
        "recipe-short": ("#7fb3d8", "#4f7ea6"),
        "t00": ("#d0461b", "#f07a50"),
        "t48": ("#8030c0", "#c48cff"),
        "t12": ("#1e8a5a", "#5ccf98"),
        "ex-2.1.10": ("#0000001a", "#ffffff22"),
    }
    """One ink per condition, as (light, dark) pairs for `light_dark`; the D2.1 reference draws as a ghost."""

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
def order_stat(a: np.ndarray, q: float, axis: int) -> np.ndarray:
    """The *q*-th percentile as an order statistic, as `experiment.top_quantile` reads it: no interpolation,
    so a monotone map of the values commutes with the quantile.
    """
    n = a.shape[axis]
    k = int(np.ceil(n * (1 - q / 100)))
    return np.take(np.sort(a, axis=axis), n - k, axis=axis)


@app.function(hide_code=True)
def load_survey() -> dict | None:
    """The ex-2.1.11 survey's stored results, or None if unpublished."""
    return load_json(ex.SURVEY_REF)


@app.function(hide_code=True)
def span2(v: np.ndarray, fmt: str = ".3f") -> str:
    """Seed mean with half the seed range beside it, in the shared `.range` style."""
    return f"{v.mean():{fmt}} <span class='range'>±{(v.max() - v.min()) / 2:{fmt}}</span>"


@app.function(hide_code=True)
def ink(cond: str):
    return light_dark(*INK[cond])


@app.function(hide_code=True)
def control_of(c: ex.Condition) -> ex.Condition:
    """The un-anchored arm of the same length, which H1 reads a condition against."""
    return ex.CONTROL if c.epochs == ex.EPOCHS else ex.CONTROL_SHORT


@app.function(hide_code=True)
def table_html(head: list[str], rows: list[list[str]], caption: str, *, ref_rows: frozenset[int] = frozenset()) -> str:
    """A result table in the shared classes: numeric cells right-aligned, and `ref_rows` styled as references."""
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        f"<tr{' class=ref' if r in ref_rows else ''}>"
        + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(row))
        + "</tr>"
        for r, row in enumerate(rows)
    )
    table = f'<div class="report-table-scroll"><table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table></div>'
    return figure_html(table, caption=caption, class_="report-figure")


@app.function(hide_code=True)
def op_table_md() -> str:
    """The op table: each rule, how often it lands on the grid unrounded, and its redder-than-both lines."""
    rows = []
    for op in ex.OPS:
        n = ex.line_counts(op)
        rows.append(
            f"| `{op.name}` | {op.rule} | {ex.on_grid(op):.1%} | {n['redder']:,} ({n['redder'] / n['lines']:.1%}) |"
        )
    head = "| op | per channel | on grid | redder than both |\n| --- | --- | ---: | ---: |\n"
    return head + "\n".join(rows)


@app.function(hide_code=True)
def agreement_md() -> str:
    """Pairwise agreement: the share of pairs on which two ops give the same answer."""
    names = [op.name for op in ex.OPS]
    cells = {}
    for p, q in itertools.combinations(ex.OPS, 2):
        cells[(p.name, q.name)] = cells[(q.name, p.name)] = f"{ex.agreement(p, q):.1%}"
    head = "| | " + " | ".join(f"`{n}`" for n in names) + " |\n|" + " ---: |" * (len(names) + 1) + "\n"
    rows = ["| `" + p + "` | " + " | ".join(cells.get((p, q), "—") for q in names) + " |" for p in names]
    return head + "\n".join(rows)


@app.function(hide_code=True)
def relevance_md() -> str:
    """Per candidate anchored op, the share of its lines on which k other ops give the same answer."""
    dists = {op.name: ex.relevance(op) for op in ex.OPS}
    levels = sorted({k for d in dists.values() for k, v in d.items() if v >= 0.0005})
    head = "| anchored op | " + " | ".join(str(k) for k in levels) + " |\n|" + " ---: |" * (len(levels) + 1) + "\n"
    rows = [
        f"| `{name}` | " + " | ".join(f"{d[k]:.1%}" if k in d else "·" for k in levels) + " |"
        for name, d in dists.items()
    ]
    return head + "\n".join(rows)


@app.function(hide_code=True)
def conditions_md() -> str:
    rows = []
    for c in ex.CONDITIONS:
        if c.lam == 0:
            anchor = "none"
        else:
            shape = "annealed" if c.anchor_anneal else "flat"
            anchor = f"λ_a = {c.lam:.3f} ({shape}), τ = {c.tau:.3f}, anti {c.anti_peak_ratio:.2f} → 0.30 by {c.anti_anneal_end_frac:.0%}"
        ops = "all six" if c.ops == ex.OP_NAMES else ", ".join(f"`{o}`" for o in c.ops)
        rows.append(
            f"| **{c.name}** | {c.title} | {anchor} | {ops} | {c.lines_per_op:,} | {c.epochs} ({c.steps:,}) | {c.seeds} |"
        )
    head = (
        "| condition | what | anchor | ops | lines per op | epochs (steps) | seeds |\n"
        "| --- | --- | --- | --- | ---: | ---: | ---: |\n"
    )
    return head + "\n".join(rows)


@app.function(hide_code=True)
def calibration_md() -> str:
    """The pre-freeze read: holdout exact match per op for one seed of each control arm."""
    cal = load_json(ex.CALIBRATION_REF)
    if cal is None:
        return "_The calibration has not been published yet._"
    head = "| run | " + " | ".join(f"`{o}`" for o in ex.OP_NAMES) + " |\n|" + " ---: |" * (len(ex.OP_NAMES) + 1) + "\n"
    rows = [
        f"| `{r['label']}` | " + " | ".join(f"{r['holdout_em'][o]:.3f}" for o in ex.OP_NAMES) + " |"
        for r in cal["runs"]
    ]
    return head + "\n".join(rows)


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class Results:
    """Every published result the report reads, with one accessor per shape the cells need.

    `metrics`, `traj`, `arrays` and `geometry` are this experiment's; `ex2110` and `ex221` are the D2.1
    references (the recipe's nine seeds, and the suppression figures on the one-op grammar); `survey` is
    ex-2.1.11's, whose proposals H3 re-measures.
    """

    metrics: dict
    traj: dict
    arrays: dict[str, np.ndarray]
    geometry: dict
    ex2110: dict
    ex221: dict
    survey: dict | None

    def runs(self, cond: str) -> list[dict]:
        """The eval records of a condition, in seed order."""
        return sorted((r for r in self.metrics["runs"] if r["condition"] == cond), key=lambda r: r["seed"])

    def scored(self, cond: str) -> list[dict]:
        return sorted((r for r in self.metrics["scores"] if r["condition"] == cond), key=lambda r: r["seed"])

    def stat(self, cond: str, key: str, op: str | None = None) -> np.ndarray:
        """One value per seed: a gated (`mix`) statistic, or one op's from `per_op`."""
        return np.array([r[key] if op is None else r["per_op"][op][key] for r in self.runs(cond)], float)

    def em(self, cond: str, op: str) -> np.ndarray:
        """Holdout exact match per seed on one op; NaN where the arm never trained on that op."""
        return np.array([r["holdout_em"].get(op, np.nan) for r in self.runs(cond)], float)

    def arr(self, cond: str, kind: str, name: str) -> np.ndarray:
        """One per-run array stacked over seeds: `kind` is `eval` or `score`, `name` is `{op}/{array}`."""
        return np.stack([self.arrays[f"{r['label']}/{kind}/{name}"] for r in self.runs(cond)])

    def trajectory(self, cond: str, key: str) -> np.ndarray:
        """(seeds, points) of one trajectory key, recorded every `TRAJ_STRIDE` steps."""
        return np.array([self.traj[r["label"]]["traj"][key] for r in self.runs(cond)], float)

    def val_loss(self, cond: str) -> np.ndarray:
        """(seeds, epochs) validation loss."""
        return np.array([self.traj[r["label"]]["val_loss"] for r in self.runs(cond)], float)

    def score(self, cond: str, op: str, iv: str | None, key: str, group: str | None = "red") -> np.ndarray:
        """One value per seed of a scored statistic on one op's probe lines; `iv=None` is the clean pass."""
        out = []
        for r in self.scored(cond):
            s = r["ops"][op]
            v = s["clean"][key] if iv is None else s["interventions"][iv][key]
            out.append(v if group is None else v[group])
        return np.array(out, float)

    def deficit(self, cond: str, op: str, iv: str, group: str = "nonred") -> np.ndarray:
        """Clean exact-match accuracy minus intervened accuracy, per seed, the statistic ex-2.2.1 gated on."""
        return self.score(cond, op, None, "acc", group) - self.score(cond, op, iv, "acc", group)

    def pooled(self, cond: str, key: str) -> np.ndarray:
        """A recipe arm's frozen seeds and its addendum seeds together: twenty values of one statistic."""
        v = (
            self.em(c, ex.PRIMARY_OP.name) if key == "holdout_em" else self.stat(c, key) for c in (cond, f"{cond}-more")
        )
        return np.concatenate(list(v))

    def pooled_score(self, cond: str, op: str, iv: str | None, key: str, group: str = "red") -> np.ndarray:
        return np.concatenate([self.score(c, op, iv, key, group) for c in (cond, f"{cond}-more")])

    def pooled_deficit(self, cond: str, op: str, iv: str, group: str = "nonred") -> np.ndarray:
        return np.concatenate([self.deficit(c, op, iv, group) for c in (cond, f"{cond}-more")])

    def ref_cells(self, cond: str = EX2110_PRIMARY) -> list[dict]:
        return [c for c in self.ex2110["cells"] if c["condition"] == cond]

    def ref_stat(self, key: str, cond: str = EX2110_PRIMARY) -> np.ndarray:
        """One value per seed of the D2.1 reference: nine on the primary, three on the control."""
        return np.array([c[key] for c in self.ref_cells(cond)], float)

    def ref_trajectory(self, key: str, cond: str = EX2110_PRIMARY) -> np.ndarray:
        return np.array([c["traj"][key] for c in self.ref_cells(cond)], float)

    def ref_retention(self, cond: str = EX2110_PRIMARY) -> np.ndarray:
        t = self.ref_trajectory("m_line", cond)
        return t[:, -1] / np.maximum.accumulate(t, axis=1).max(axis=1)

    def ex221_stat(self, iv: str | None, key: str, group: str | None = "red", cond: str = EX2110_PRIMARY) -> np.ndarray:
        """One value per seed of ex-2.2.1's one-op suppression figures; `iv=None` is its clean pass."""
        out = []
        for r in sorted((r for r in self.ex221["runs"] if r["condition"] == cond), key=lambda r: r["seed"]):
            v = r["clean"][key] if iv is None else r["interventions"][iv][key]
            out.append(v if group is None else v[group])
        return np.array(out, float)

    def ex221_deficit(self, iv: str, group: str = "nonred") -> np.ndarray:
        return self.ex221_stat(None, "acc", group) - self.ex221_stat(iv, "acc", group)

    def survey_point(self, c: ex.Condition) -> tuple[int, dict | None]:
        """A candidate's one-op numbers: the survey's own re-run of the recipe, or its five-seed trial."""
        if c.survey_trial is None:
            return ex.SURVEY_RECIPE[c.name]
        return 5, (self.survey["scored"].get(str(c.survey_trial)) if self.survey else None)

    def nonred_profile(self, cond: str, op: str = ex.PRIMARY_OP.name) -> np.ndarray:
        """(seeds, slices, roles) softmin profile of the non-red group, at each run's τ: the latch reads it."""
        out = []
        for r in self.runs(cond):
            cos = self.arrays[f"{r['label']}/eval/{op}/alpha_lines"].astype(np.float32)
            nonred = self.arrays[f"{r['label']}/eval/{op}/dose"] <= ex.NONRED_DOSE + 1e-9
            out.append(softmin_weights(1.0 - cos[:, nonred, : ex.SPAN], r["tau"]).mean(axis=1))
        return np.stack(out)

    def clean_alpha_band(self, cond: str, op: str, q: float = 99.0) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) signed tails of the clean alignment over the non-red lines, each (slices, positions)
        and seed-meaned: the *q*-th percentile and its reflection.

        The scorer stores only `alpha_q99_nonred`, the same percentile of |α|, which folds the two tails
        together; the per-line alignments the eval stage stores keep the sign, on the same probe lines.
        """
        hi, lo = [], []
        for r in self.runs(cond):
            cos = self.arrays[f"{r['label']}/eval/{op}/alpha_lines"].astype(np.float32)
            nonred = self.arrays[f"{r['label']}/eval/{op}/dose"] <= ex.NONRED_DOSE + 1e-9
            a = cos[:, nonred]
            hi.append(order_stat(a, q, axis=1))
            lo.append(-order_stat(-a, q, axis=1))
        return np.mean(lo, axis=0), np.mean(hi, axis=0)

    def latched(self, cond: str) -> np.ndarray:
        """Per run: the non-red group's op1 weight over the post-attention slices, above `LATCH_PI`."""
        return self.nonred_profile(cond)[:, 1:, 0].mean(axis=1) > ex.LATCH_PI


@app.cell(hide_code=True)
def _():
    _metrics = load_json(ex.METRICS_REF)
    mo.stop(_metrics is None, mo.md("_Results are not published yet; the result cells render once they are._"))
    assert _metrics is not None
    _traj, _arrays, _geometry = load_json(ex.TRAJ_REF), load_npz(ex.ARRAYS_REF), load_json(ex.GEOMETRY_REF)
    _ex2110, _ex221 = load_json(ex.EX2110_METRICS_REF), load_json(EX221_METRICS_REF)
    assert _traj and _arrays and _geometry and _ex2110 and _ex221, "a reference result is missing from the store"
    res: Results = Results(_metrics, _traj, _arrays, _geometry, _ex2110, _ex221, load_survey())
    # The design the DAG ran is the one this notebook quotes.
    assert res.metrics["design"]["n_runs"] == ex.N_RUNS
    assert len(res.metrics["runs"]) == ex.N_RUNS, len(res.metrics["runs"])
    assert {r["condition"] for r in res.metrics["scores"]} == {c.name for c in (*ex.CANDIDATES, *ex.ADDENDUM)}
    # The reference the margin gates quote, to three decimals.
    assert round(float(res.ref_stat("m_line").mean()), 3) == round(ex.REF_M_LINE, 3)
    return (res,)


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class Verdict:
    """One hypothesis's read: `status` is holds, partial, contrary, or unscored; `line` is the Findings entry."""

    status: str
    line: str

    @property
    def md(self) -> str:
        return f"**{self.status}.** {self.line}"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.2.3: the multi-op grammar, with *red* anchored again

    /// tip |
    <!-- tl;dr -->
    The grammar grows from one operation to six: `mix` (`+`), `add`, `screen`, `multiply`, `lighten`, `darken`. We retrain on the new grammar to check the recipes: the un-anchored control, the ex-2.1.10 recipe, and three proposals from the ex-2.1.11 survey. The recipe carried over as it was. The proposals place *red* with a wider margin, and the frozen rule adopted one of them, but they put the axis on the syntax rows, so projecting it out breaks most non-red lines too; a post hoc read with the gates the rule left out narrows the choice to the recipe, and D2.2 builds on its short arm, which grades better for half the compute. Removal reads as partial on the saturating ops because their answers mostly do not depend on how red the red operand is.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(
    adopted: str,
    adopted_post: str,
    carried: str,
    h1: Verdict,
    h2: Verdict,
    h3: Verdict,
    h4: Verdict,
):
    mo.md(rf"""
    ## Findings

    - [Task cost on the new grammar (H1)](#task-cost-on-the-new-grammar-h1) — {h1.md}
    - [Placement under the recipe reproduces (H2)](#placement-under-the-recipe-reproduces-h2) — {h2.md}
    - [The plateau from the survey transfers (H3)](#the-plateau-from-the-survey-transfers-h3) — {h3.md} The frozen rule adopts **`{adopted}`**; a [post hoc amendment](#post-hoc-the-selection-rule-with-the-gates-it-left-out) that adds H2's lead gate and H4's selectivity gate, with the tie between the recipe lengths broken at twenty seeds (E6), adopts **`{adopted_post}`**. D2.2 builds on **`{carried}`** by decision: it grades better and leads better at the embedding, with the same H4 read, for half the compute.
    - [Suppression transfers (H4)](#suppression-transfers-h4) — {h4.md}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this draft
    The hypotheses, the gates, the selection rule, and the method were frozen at commit `5a3233b`, before any anchored run, with the calibration recorded under [the method](#before-the-freeze). Each hypothesis section opens with its frozen prediction; its results follow in place. Anything we thought of after seeing the data is under [Exploratory analyses](#exploratory-analyses), marked as post hoc. The op table and every count in the method are computed from `experiment.py` at render time.
    ///

    ## Why this experiment

    The claims of D2.2 are about an operation, and the D2.1 grammar has only one: `mix`, spelled `+`. The [D2.2 plan](../d2.2/design.md) makes the operation a variable. But every number D2.1 produced was measured on the one-op grammar: the recipe in [ex-2.1.10](../ex-2.1.10/report.py), the operating-point plateau in the [ex-2.1.11](../ex-2.1.11/report.py) survey, and the suppression figures in [ex-2.2.1](../ex-2.2.1/report.py). The risk table in the plan names this ("recipe is grammar-specific") and sends it here.

    So this is a regression check with a decision attached. We measure the same statistics on the same `mix` lines, and we freeze a rule that turns the proposals from the survey into the operating point the anchored-op experiments adopt.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conditions

    Every condition is trained here at fresh seeds. All but the richer-op arms use the six-op corpus at D2.1's size and epoch count, so the recipe's step count is unchanged.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    {conditions_md()}

    The recipe is the D2.1 primary, `either-t100`. The only thing we changed is the grammar. The short arm of the recipe runs for as long as the proposals do, so H3 compares runs of equal length; the control has a short arm so that H1 can do the same.

    The three proposals are the operating point from the survey plus the two promoted trials the survey read as its twin and its knee. `t48` sits within one m_line band of `t00`, with more grading and contrast. `t12` holds the most grading of the promoted set, at a margin the survey could not tell apart from the one for `t00`.

    The richer-op arms are un-anchored models on one and three ops at the recipe's step count, in two matchings. The `corpus` arms keep D2.1's line count, and so have more lines per op. The `per-op` arms keep the six-op lines per op, and so have a smaller corpus, repeated over more epochs. Corpus size, lines per op, and step count cannot all be held while the op set changes, and the two matchings put the remaining difference on opposite sides, so the pair lets E4 read in both directions. They serve E4 only.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The interventions

    We score the checkpoints of every candidate through the eval contract in [`sca.intervention`](/src/sca/intervention.py), on the probe lines of each op.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    - **`projection`** — the primary from ex-2.2.1: the axis projected out at every slice and position, at full strength. Read by H4.
    - **Ride-along rows** — `operands` (the projection at the two operand positions only), `shaped` (the shaped suppression from M1, a = {ex.SHAPED["a"]:g}, b = {ex.SHAPED["b"]:g}, p = {ex.SHAPED["p"]:g}), and `ablate` (the axis weights zeroed). Reported without gates, so the intervention-tuning pass the plan schedules has the multi-op figures beside the one-op figures from ex-2.2.1.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Task cost on the new grammar (H1)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    **H1.** Anchoring costs nothing on the task, on any op. We score the recipe at both lengths and the three proposals against the control of the same length, op by op, so the gate reads {len(ex.CANDIDATES) * len(ex.OPS)} condition-op comparisons. In each, seed-mean holdout exact-match accuracy is within {ex.TASK_GATE:g} of its control's. Partial: every comparison within {ex.TASK_PARTIAL:g}, or all but one within {ex.TASK_GATE:g} and that one within {ex.TASK_PARTIAL:g}. Contrary: a condition more than {ex.TASK_PARTIAL:g} below its control on some op, which would say that the anchor and reading the op compete with each other. A condition that far *above* its control on some op misses the gate too, and we would report it as an anomaly, since nothing in the design predicts one.

    This gate only means something if the controls learn the grammar in the first place. We check that at both lengths in the calibration runs described in the [method](#before-the-freeze), before the freeze, rather than gating it here.

    If the recipe misses this gate at full length, H2 goes unscored and the refuted H1 is the finding, as ex-2.1.10 ruled for its primary. H3 still runs, since its selection rule already requires the task gate of every candidate, and H4 is then scored on the adopted point alone. If no candidate is feasible, H4 goes unscored too.
    """)
    return


@app.cell(hide_code=True)
def _(res: Results):
    # H1: every scored condition against the control of its own length, op by op.
    _gaps: dict[tuple[str, str], float] = {}
    for _c in ex.CANDIDATES:
        for _op in ex.OP_NAMES:
            _gaps[_c.name, _op] = float(res.em(_c.name, _op).mean() - res.em(control_of(_c).name, _op).mean())
    _abs = {k: abs(v) for k, v in _gaps.items()}
    _worst_key = max(_abs, key=_abs.__getitem__)
    _n_gate = sum(v <= ex.TASK_GATE for v in _abs.values())
    _n_partial = sum(v <= ex.TASK_PARTIAL for v in _abs.values())
    _above = [k for k, v in _gaps.items() if v > ex.TASK_GATE]
    task_ok: dict[str, bool] = {
        c.name: all(_abs[c.name, op] <= ex.TASK_GATE for op in ex.OP_NAMES) for c in ex.CANDIDATES
    }
    """Per candidate, the H1 gate as the selection rule reads it: within the gate on every op."""
    _n = len(_abs)
    if _n_gate == _n:
        _status = "holds"
    elif _n_partial == _n and _n_gate >= _n - 1:
        _status = "partial"
    elif _n_partial == _n:
        _status = "partial"
    else:
        _status = "contrary"
    _c, _op = _worst_key
    _line = (
        f"All {_n} condition-op comparisons are within {ex.TASK_GATE:g} of the control of the same length; "
        f"the largest gap is {_gaps[_worst_key]:+.4f}, `{_c}` on `{_op}`."
        if _status == "holds"
        else f"{_n_gate} of {_n} comparisons are within {ex.TASK_GATE:g} and {_n_partial} within {ex.TASK_PARTIAL:g}; "
        f"the largest gap is {_gaps[_worst_key]:+.4f}, `{_c}` on `{_op}`."
    )
    if _above:
        _n_above = f"{len(_above)} of them sit" if len(_above) > 1 else "One of them sits"
        _line += f" {_n_above} more than {ex.TASK_GATE:g} *above* the control, an anomaly the design did not predict."
    h1: Verdict = Verdict(_status, _line)
    h1_gaps: dict[tuple[str, str], float] = _gaps
    return h1, h1_gaps, task_ok


@app.cell(hide_code=True)
def _(res: Results):
    _head = ["condition"] + [f"{op} EM ↑" for op in ex.OP_NAMES]
    _rows, _refs = [], set()
    for _i, _c in enumerate(ex.CONDITIONS):
        _rows.append([_c.name] + [span2(res.em(_c.name, op)) if op in _c.ops else "·" for op in ex.OP_NAMES])
        if _c not in ex.CANDIDATES:
            _refs.add(_i)
    _caption = f"""
    Holdout exact match per op, by condition: the seed mean with half the seed range beside it. The greyed rows are the two controls ({ex.EPOCHS} and {ex.EPOCHS_SHORT} epochs) and the richer-op arms, which carry no gate; a dot means the arm never trained on that op.
    """
    mo.Html(table_html(_head, _rows, _caption, ref_rows=frozenset(_refs)))
    return


@app.cell(hide_code=True)
def _(h1_gaps: dict[tuple[str, str], float]):
    _gap_head = ["condition", "control"] + [f"{op} gap" for op in ex.OP_NAMES] + ["worst"]
    _gap_rows = []
    for _c in ex.CANDIDATES:
        _g = [h1_gaps[_c.name, op] for op in ex.OP_NAMES]
        _worst = max(_g, key=abs)
        _gap_rows.append(
            [_c.name, control_of(_c).name]
            + [f"<b>{g:+.4f}</b>" if abs(g) <= ex.TASK_GATE else f"{g:+.4f}" for g in _g]
            + [f"<b>{_worst:+.4f}</b>" if abs(_worst) <= ex.TASK_GATE else f"{_worst:+.4f}"]
        )
    _gap_caption = f"""
    The H1 read: each scored condition's seed-mean holdout exact match minus that of the control of the same length, per op. Bold marks a gap within the gate of {ex.TASK_GATE:g}; the partial band runs to {ex.TASK_PARTIAL:g}.
    """
    mo.Html(table_html(_gap_head, _gap_rows, _gap_caption))
    return


@app.cell(hide_code=True)
def _(res: Results):
    _conds = [c for c in ex.CONDITIONS if c not in ex.RICHER_OP_ARM and c not in ex.ADDENDUM]
    _em = {(c.name, op): res.em(c.name, op) for c in _conds for op in ex.OP_NAMES}
    _ctrl = {c.name: {op: float(res.em(control_of(c).name, op).mean()) for op in ex.OP_NAMES} for c in _conds}

    @themed(
        name="h1-holdout-dots",
        alt_text="""
            Six small panels, one per op, each listing the seven conditions down the side with a dot at the seed-mean holdout exact match and a bar for the seed range. A grey strip behind each row marks the gate around that row's control. Every dot sits inside its strip, near the right edge at accuracy 1.
        """,
        caption=f"""
            **Holdout exact match per op, against the gate.** One panel per op; conditions down the side, in their own ink. Each dot is the seed mean, the bar the seed range. The grey strip behind a row is ±{ex.TASK_GATE:g} around the seed-mean accuracy of the control of the same length, so a dot outside its strip misses the H1 gate; the two control rows carry their own strips for reference.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(7.6, 3.6), sharex=True, sharey=True, layout="constrained")
        axes = cast(AxesGrid, axes)
        strip = light_dark("#00000014", "#ffffff1a")
        ys = np.arange(len(_conds))[::-1]
        for ax, op in zip([a for row in axes for a in row], ex.OP_NAMES, strict=True):
            for y, c in zip(ys, _conds, strict=True):
                lo, hi = _ctrl[c.name][op] - ex.TASK_GATE, _ctrl[c.name][op] + ex.TASK_GATE
                ax.fill_betweenx([y - 0.42, y + 0.42], lo, hi, color=strip, lw=0, zorder=1)
                v = _em[c.name, op]
                ax.plot([v.min(), v.max()], [y, y], color=ink(c.name), lw=1.2, zorder=2, solid_capstyle="round")
                ax.plot(v.mean(), y, "o", ms=3.6, color=ink(c.name), zorder=3)
            ax.set_title(op, fontsize=9)
            ax.set_yticks(ys, [c.name for c in _conds], fontsize=7)
            ax.set_xlim(0.9, 1.005)
            ax.set_xticks([0.9, 0.95, 1.0])
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="x", alpha=0.2)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(res: Results):
    _spe = ex.steps_per_epoch()
    _fresh = {c.name: res.val_loss(c.name) for c in (ex.CONTROL, ex.RECIPE)}
    _ref = np.array([c["val_loss"] for c in res.ref_cells()], float)
    _ref_ctrl = np.array([c["val_loss"] for c in res.ref_cells(EX2110_CONTROL)], float)

    @themed(
        name="h1-val-loss",
        alt_text="""
            One log-scale line chart of validation loss over 3,300 training steps. Five thin blue lines for the recipe and five grey lines for the control fall together from about 3 to below 0.1 and stay interleaved throughout; a faint band of ghost lines from the D2.1 runs lies behind them on the same axis, reaching a lower floor.
        """,
        caption="""
            **Validation loss over training.** Five thin lines per condition on the step axis: the control in grey and the recipe in blue. The ghost lines behind them are the ex-2.1.10 primary (nine seeds) and its control (three), which trained on the one-op grammar at the same step count; their floor sits lower because their corpus held about seventeen copies of each distinct line, where this one holds most lines once. Log scale.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 2.8), layout="constrained")
        ghost = ink("ex-2.1.10")
        x_ref = (np.arange(_ref.shape[1]) + 1) * _spe
        for row in _ref_ctrl:
            ax.plot(x_ref, row, color=ghost, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for row in _ref:
            ax.plot(x_ref, row, color=ghost, lw=0.8, zorder=1)
        for cond, rows in _fresh.items():
            x = (np.arange(rows.shape[1]) + 1) * _spe
            for row in rows:
                ax.plot(x, row, color=ink(cond), lw=0.8, zorder=3 if cond == "recipe" else 2, alpha=0.85)
            ax.plot([], [], color=ink(cond), lw=1.4, label=cond)
        ax.plot([], [], color=ghost, lw=1.2, label="ex-2.1.10 primary")
        ax.set_yscale("log")
        ax.set_xlim(0, ex.RECIPE.steps)
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel("validation loss (nats per token)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(h1: Verdict):
    mo.md(rf"""
    **H1, {h1.status}.** {h1.line}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Placement under the recipe reproduces (H2)
    """)
    return


@app.cell(hide_code=True)
def _():
    # REVIEW: H2's contrary clause read a latch or containment miss as the op word alone,
    # while the paragraph under it names exposure as a second reading; the clause now says
    # "something in the new grammar" and the paragraph carries both. The added arithmetic
    # (6.4% of all pairs against 6.3% of the on-grid ones are red by dose) is what says the
    # anchor's own exposure is near unchanged. Verify: `line_counts` against ex-2.2.1's
    # 365 red of 5,832 mix probe lines.
    # The gate list is ordered seed-mean criteria first, then the two per-run ones
    # (retention, latch), so the "unless the criterion says otherwise" carries; ex-2.1.10
    # aggregated the same way. Don't reword to imply every criterion is scored at every run.
    mo.md(rf"""
    **H2.** The ex-2.1.10 recipe places *red* on the new grammar as it did on the old one. On the `mix` probe lines, with five fresh seeds, all of the ex-2.1.10 placement gates hold. Each statistic is a seed mean unless the criterion says otherwise: containment $\bar\alpha$ at op1 at most {ex.MEAN_ALIGN_GATE:g}; a leading softmin weight of the red group at the embedding of at least {ex.LEAD_GATE:g}; contrast at least {ex.CONTRAST_GATE:g}; grading $r^2$ no more than {ex.GRADE_R2_DROP:g} below the {ex.REF_R2_SIM:.3f} the survey measured on this recipe; m_line at least {ex.MARGIN_RATIO:g} of its ex-2.1.10 value of {ex.REF_M_LINE:.3f}; and, read per run rather than on the seed mean, retention at least {ex.RETENTION_GATE:g} of the running peak for every run whose peak reaches {ex.RETENTION_FLOOR:g}, and no latched run.

    Partial: every criterion holds except one of m_line in the {ex.MARGIN_PARTIAL:g}–{ex.MARGIN_RATIO:g} band or contrast in the {ex.CONTRAST_PARTIAL:g}–{ex.CONTRAST_GATE:g} band. Contrary: a latch or a containment miss. That would mean something in the new grammar changed what the pull finds. If instead the margin fell below the partial band while the other criteria held, the placement would be there but weaker, and the H3 selection would lean toward the proposals.

    Two things in the new grammar could do that, and we read a miss against both. The first is the op word. The labeller never sees it, and reads the same two operand roles on every op.

    The second is exposure. Only a sixth of the corpus is `mix`, and those lines are drawn from every pair rather than from the on-grid ones. So over a run the model sees a given `mix` probe pair about a fiftieth as often as in D2.1, and never sees most of them (see [the corpus](#the-corpus)). What the anchor sees is closer to unchanged. The corpus is the same size, and the share of lines the labeller fires on follows the operand redness: 6.4% of all pairs, against 6.3% of the on-grid ones in D2.1. So the exposure that fell is the exposure of the task, counted per pair, rather than the exposure of the anchor, counted per step.

    Two measurements separate the readings: the task accuracy of the control on `mix` (H1), and the per-op placement in E1.
    """)
    return


@app.cell(hide_code=True)
def _(res: Results, task_ok: dict[str, bool]):
    # H2's criteria on the recipe's mix lines. Seed-mean criteria first, then the two per-run ones.
    _c = ex.RECIPE.name
    _m_line, _a_op1, _lead = res.stat(_c, "m_line"), res.stat(_c, "alpha_op1"), res.stat(_c, "lead_emb")
    _contrast, _r2 = res.stat(_c, "contrast"), res.stat(_c, "r2_sim")
    _ret, _peak = res.stat(_c, "retention"), res.stat(_c, "m_line_peak")
    _ret_scored = _ret[_peak >= ex.RETENTION_FLOOR]
    _latched = res.latched(_c)
    h2_checks: dict[str, tuple[float, bool]] = {
        "containment": (float(_a_op1.mean()), bool(_a_op1.mean() <= ex.MEAN_ALIGN_GATE)),
        "lead": (float(_lead.mean()), bool(_lead.mean() >= ex.LEAD_GATE)),
        "contrast": (float(_contrast.mean()), bool(_contrast.mean() >= ex.CONTRAST_GATE)),
        "grading": (float(_r2.mean()), bool(_r2.mean() >= ex.REF_R2_SIM - ex.GRADE_R2_DROP)),
        "margin": (float(_m_line.mean()), bool(_m_line.mean() >= ex.MARGIN_RATIO * ex.REF_M_LINE)),
        "retention": (
            float(_ret_scored.min()) if len(_ret_scored) else float("nan"),
            bool(len(_ret_scored) == 0 or _ret_scored.min() >= ex.RETENTION_GATE),
        ),
        "latch": (float(res.nonred_profile(_c)[:, 1:, 0].mean(axis=1).max()), bool(not _latched.any())),
    }
    """Each H2 criterion on the recipe: (the statistic as the gate reads it, whether it holds)."""
    _misses = [k for k, (_, ok) in h2_checks.items() if not ok]
    _m_band = ex.MARGIN_PARTIAL * ex.REF_M_LINE <= _m_line.mean() < ex.MARGIN_RATIO * ex.REF_M_LINE
    _c_band = ex.CONTRAST_PARTIAL <= _contrast.mean() < ex.CONTRAST_GATE
    _ratio = _m_line.mean() / ex.REF_M_LINE
    if not task_ok[_c]:
        _v = Verdict(
            "unscored",
            "The recipe missed the H1 gate at full length, so H2 is not scored; the refuted H1 is the finding.",
        )
    elif not _misses:
        _v = Verdict(
            "holds",
            f"Every placement criterion holds on the `mix` lines at five fresh seeds: seed-mean m_line {_m_line.mean():.3f}, {_ratio:.2f} of its ex-2.1.10 value, grading r² {_r2.mean():.3f}, contrast {_contrast.mean():.2f}, containment {_a_op1.mean():.3f}, no latched run.",
        )
    elif _misses in (["margin"], ["contrast"]) and (_m_band or _c_band):
        _v = Verdict(
            "partial",
            f"Every criterion holds except {_misses[0]}, which sits in its partial band: m_line {_m_line.mean():.3f} ({_ratio:.2f} of the reference), contrast {_contrast.mean():.2f}.",
        )
    elif "latch" in _misses or "containment" in _misses:
        _v = Verdict(
            "contrary",
            f"The recipe misses {', '.join(_misses)}: containment {_a_op1.mean():.3f}, {int(_latched.sum())} latched run(s). Something in the new grammar changed what the pull finds.",
        )
    else:
        _v = Verdict(
            "contrary",
            f"The recipe misses {', '.join(_misses)}: m_line {_m_line.mean():.3f} ({_ratio:.2f} of the reference), grading r² {_r2.mean():.3f}, contrast {_contrast.mean():.2f}, lead {_lead.mean():.2f}, retention {h2_checks['retention'][0]:.2f}.",
        )
    h2: Verdict = _v
    return (h2,)


@app.cell(hide_code=True)
def _(res: Results):
    _c = ex.RECIPE.name
    _ref_ret = res.ref_retention()
    _sv_seeds, _sv = ex.SURVEY_RECIPE["recipe"]

    def _band(stat: str, n_b: int) -> str:
        return f"{ex.equiv_band(stat, 5, n_b):.3f}" if stat in ex.NOISE_RUN else "—"

    def _row(
        name: str,
        fresh: np.ndarray,
        ref: np.ndarray | None,
        ref_note: str,
        gate: str,
        ok: bool,
        stat: str,
        n_b: int,
        fmt: str = ".3f",
    ) -> list[str]:
        v = f"<b>{span2(fresh, fmt)}</b>" if ok else span2(fresh, fmt)
        if ref is None:
            return [name, v, "—", "—", "—", "—", gate]
        d = fresh.mean() - ref.mean()
        b = ex.equiv_band(stat, 5, n_b) if stat in ex.NOISE_RUN else float("nan")
        resolved = "" if np.isnan(b) else (" ✓" if abs(d) > b else " ·")
        return [
            name,
            v,
            span2(ref, fmt) if len(ref) > 1 else f"{ref.mean():{fmt}}",
            ref_note,
            f"{d:+{fmt}}{resolved}",
            _band(stat, n_b),
            gate,
        ]

    _m = res.stat(_c, "m_line")
    _rows = [
        _row(
            "m_line ↑",
            _m,
            res.ref_stat("m_line"),
            "9",
            f"≥ {ex.MARGIN_RATIO * ex.REF_M_LINE:.3f}",
            _m.mean() >= ex.MARGIN_RATIO * ex.REF_M_LINE,
            "m_line",
            9,
        ),
        _row("m_span", res.stat(_c, "m_span"), res.ref_stat("m_span"), "9", "—", False, "m_span", 9),
        _row(
            "containment ᾱ ↓",
            res.stat(_c, "alpha_op1"),
            res.ref_stat("alpha_mean_op1"),
            "9",
            f"≤ {ex.MEAN_ALIGN_GATE:g}",
            res.stat(_c, "alpha_op1").mean() <= ex.MEAN_ALIGN_GATE,
            "alpha_op1",
            9,
        ),
        _row(
            "lead weight at emb ↑",
            res.stat(_c, "lead_emb"),
            None,
            "",
            f"≥ {ex.LEAD_GATE:g}",
            res.stat(_c, "lead_emb").mean() >= ex.LEAD_GATE,
            "lead",
            9,
            ".2f",
        ),
        _row(
            "contrast ↑",
            res.stat(_c, "contrast"),
            np.array([_sv["contrast"]]),
            f"{_sv_seeds} (survey)",
            f"≥ {ex.CONTRAST_GATE:g}",
            res.stat(_c, "contrast").mean() >= ex.CONTRAST_GATE,
            "contrast",
            _sv_seeds,
        ),
        _row(
            "grading r² ↑",
            res.stat(_c, "r2_sim"),
            np.array([ex.REF_R2_SIM]),
            f"{_sv_seeds} (survey)",
            f"≥ {ex.REF_R2_SIM - ex.GRADE_R2_DROP:.3f}",
            res.stat(_c, "r2_sim").mean() >= ex.REF_R2_SIM - ex.GRADE_R2_DROP,
            "r2_sim",
            _sv_seeds,
        ),
        _row(
            "retention, min over runs ↑",
            res.stat(_c, "retention"),
            _ref_ret,
            "9",
            f"≥ {ex.RETENTION_GATE:g}",
            res.stat(_c, "retention").min() >= ex.RETENTION_GATE,
            "retention",
            9,
        ),
        _row(
            "non-red op1 weight, max over runs ↓",
            res.nonred_profile(_c)[:, 1:, 0].mean(axis=1),
            None,
            "",
            f"≤ {ex.LATCH_PI:g}",
            not res.latched(_c).any(),
            "latch",
            9,
            ".2f",
        ),
    ]
    _caption = """
    The placement statistics of the recipe on the <code>mix</code> probe lines, five fresh seeds, beside the D2.1 reference. The reference is the ex-2.1.10 primary's nine seeds where its stored metrics carry the statistic, and the survey's three-seed re-run of the same recipe (its <code>ref</code> arm) for contrast and grading, which ex-2.1.10 did not store per run; the lead weight and the latch have no stored reference. Δ is fresh minus reference; a ✓ beside it means the difference clears the band (2σ√(1/5 + 1/n) with the frozen per-run σ), a dot that it does not. Bold marks a value inside its gate. Retention and the latch are read per run, so their rows print the worst run.
    """
    _head = ["statistic", "fresh (5 seeds)", "D2.1 reference", "ref. seeds", "Δ (✓ clears band)", "band", "gate"]
    mo.Html(table_html(_head, _rows, _caption))
    return


@app.cell(hide_code=True)
def _(res: Results):
    _spe = ex.steps_per_epoch()
    _fresh = res.trajectory(ex.RECIPE.name, "m_line")
    _fresh_x = res.trajectory(ex.RECIPE.name, "epoch") * _spe
    _ctrl = res.trajectory(ex.CONTROL.name, "m_line")
    _ctrl_x = res.trajectory(ex.CONTROL.name, "epoch") * _spe
    _ref = res.ref_trajectory("m_line")
    _ref_x = res.ref_trajectory("step")

    @themed(
        name="h2-m-line-trajectory",
        alt_text="""
            A line chart of m_line over 3,300 training steps. Five thin blue lines for the recipe rise steeply in the first few hundred steps, level off, and dip slightly after step 3,000; nine faint ghost lines from the D2.1 runs follow the same path behind them. Five grey lines for the control lie flat near zero.
        """,
        caption=f"""
            **m_line over training.** The trajectory instrument reads one line per color on the <code>mix</code> probe set, every {ex.TRAJ_STRIDE} steps. Five thin blue lines are the recipe's fresh seeds; the ghost lines behind them are the nine seeds of the ex-2.1.10 primary on the one-op grammar, on the same step axis. The grey lines are the un-anchored control. The dip over the last tenth is the anchor anneal.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 2.8), layout="constrained")
        for x, y in zip(_ref_x, _ref, strict=True):
            ax.plot(x, y, color=ink("ex-2.1.10"), lw=1.0, zorder=1)
        for x, y in zip(_ctrl_x, _ctrl, strict=True):
            ax.plot(x, y, color=ink("control"), lw=0.8, zorder=2, alpha=0.8)
        for x, y in zip(_fresh_x, _fresh, strict=True):
            ax.plot(x, y, color=ink("recipe"), lw=0.9, zorder=3, alpha=0.9)
        for c, name in (("recipe", "recipe"), ("control", "control"), ("ex-2.1.10", "ex-2.1.10 primary")):
            ax.plot([], [], color=ink(c), lw=1.4, label=name)
        ax.set_xlim(0, ex.RECIPE.steps)
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel("m_line", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(res: Results):
    _pg = res.arr(ex.RECIPE.name, "eval", f"{ex.PRIMARY_OP.name}/w_group")  # (seeds, group, slices, roles)
    _mean = _pg.mean(axis=0)
    _con = res.stat(ex.RECIPE.name, "contrast")
    _x = np.arange(ex.SPAN)

    @themed(
        name="h2-group-profiles",
        alt_text="""
            Ten small panels in two columns, G1 (op1 drew) and G2 (op2 drew), five residual slices per column with the embedding at the bottom, each spanning the roles op1, op, op2, and equals. In the G1 column the weight sits at op1 on every slice; in the G2 column it sits at op2. Five per-seed hairlines hug the mean in every panel.
        """,
        caption=rf"""
            **Per-group softmin profiles for the recipe on the `mix` lines.** Columns are the label groups (G1 weights probe lines by P(only op1 drew), G2 by P(only op2 drew)); rows are residual slices with the embedding at the bottom; the shaded smooth-step is the seed-mean softmin weight over the span roles, with a hairline per seed. On the embedding row, the caret on each column's outer axis marks the lead gate ({ex.LEAD_GATE:g}; uniform is 0.25), and the number above each group's own operand is the seed-mean weight there. Deep-slice op2 contrast {_con.mean():+.2f} <span class='range'>±{_con.std(ddof=1):.2f}</span> (gate ≥ {ex.CONTRAST_GATE:g}). The layout is ex-2.1.10's, whose primary put 0.77 on the own operand at the embedding and 0.9 or more at depth.
        """,
    )
    def _plot() -> plt.Figure:
        from matplotlib.layout_engine import ConstrainedLayoutEngine

        color = ink("recipe")
        hair = light_dark("#00000055", "#ffffff55")
        gate = light_dark("#555", "#bbb")
        fig, axes = plt.subplots(5, 2, figsize=(3.1, 3.9), sharex=True, sharey=True)
        axes = cast(AxesGrid, axes)
        engine = fig.get_layout_engine()
        assert isinstance(engine, ConstrainedLayoutEngine)
        engine.set(hspace=0, h_pad=0.01, wspace=0.05)
        for col, gname in enumerate(("G1: op1 drew", "G2: op2 drew")):
            for row in range(5):
                sl = 4 - row
                ax = axes[row][col]
                smooth_step_area(ax, _x, _mean[col, sl], ramp=0.5, color=color, alpha=light_dark(0.22, 0.28))
                smooth_step(ax, _x, _mean[col, sl], ramp=0.5, color=color, lw=1.5)
                for sp in _pg:
                    smooth_step(ax, _x, sp[col, sl], ramp=0.5, color=hair, lw=0.5)
                ax.set(ylim=(-0.08, 1.08), xlim=(-0.4, ex.SPAN - 0.6), yticks=[0.0, 0.5, 1.0])
                ax.spines[:].set_visible(False)
                ax.grid(axis="y", which="major", c="#888", alpha=0.2)
                ax.tick_params(axis="x", length=0, labelbottom=False)
                ax.tick_params(axis="y", left=True, right=True, direction="in", labelleft=False, labelright=False)
                if col == 0:
                    ax.set_ylabel("emb" if sl == 0 else f"{sl}", fontsize=7.5)
            axes[0][col].set_title(gname, fontsize=8.5, color=color)
            own = 0 if col == 0 else 2
            emb = axes[4][col]
            emb.plot(
                col,
                ex.LEAD_GATE,
                marker=9 if col == 0 else 8,
                ms=3,
                color=gate,
                clip_on=False,
                zorder=6,
                transform=emb.get_yaxis_transform(),
            )
            val = float(_mean[col, 0, own])
            emb.annotate(
                f"{val:.2f}",
                (own + 0.35 if col == 0 else own, min(val + 0.14, 0.92)),
                ha="left" if col == 0 else "center",
                fontsize=7,
                color=light_dark("#444", "#bbb"),
            )
            axes[4][col].set_xticks(_x, ROLES)
            axes[4][col].tick_params(axis="x", labelbottom=True, labelsize=7.5)
        axes[4][1].tick_params(axis="y", labelright=True, labelsize=6.5, pad=2)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(h2: Verdict):
    mo.md(rf"""
    **H2, {h2.status}.** {h2.line}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The plateau from the survey transfers (H3)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    **H3.** The plateau found in the survey belongs to the recipe rather than to the one-op grammar. At least one proposal is feasible at fresh seeds on the new grammar, and its seed-mean m_line on the `mix` probe lines exceeds `recipe-short` (the recipe at the same step count) by more than one band. Feasibility is read as the survey read it, with the constraints listed once in the selection rule below. The grading margin is one of them, and so is a floor on the worst seed: every run's m_line at least {ex.MARGIN_PARTIAL:g} of its ex-2.1.10 value, since the bands use the frozen per-run σ and a candidate's own spread would otherwise not count against it.

    Partial: a proposal is feasible and its m_line sits within a band of `recipe-short`; or a proposal clears every survey constraint and exceeds `recipe-short` by more than a band, but misses the grading margin. Contrary: no proposal is feasible, or every feasible proposal sits more than a band below `recipe-short`. Either would say the plateau was specific to the one-op grammar, and the recipe would carry D2.2. A proposal that misses only the per-run floor is reported as unstable at fresh seeds rather than as weak, with the seed range printed.

    **The selection rule.** {ex.SELECTION_RULE} The Findings line for this section names the adopted point, and it lands in `experiment.py` for the D2.2 experiments that follow.

    **Winner's curse.** The numbers from the survey are proposals. So the table prints the one-op value for each candidate beside the fresh one, with the per-run σ behind the bands and the band a difference has to clear. The survey re-ran the recipe at both lengths, and those runs sit in the same table, since they are the one-op values the fresh recipe arms are read against.
    """)
    return


@app.cell(hide_code=True)
def _(res: Results, task_ok: dict[str, bool]):
    # H3: feasibility as the selection rule reads it, then the rule itself.
    _r2_floor = float(res.stat(ex.RECIPE.name, "r2_sim").mean()) - ex.GRADE_R2_DROP
    _r2_margin = ex.GRADE_MARGIN_SD * ex.NOISE_RUN["r2_sim"]
    _band = ex.equiv_band("m_line", 5, 5)

    feas: dict[str, dict[str, bool]] = {}
    """Per candidate, each feasibility check of the selection rule, in the order the table prints them."""
    for _c in ex.CANDIDATES:
        _n = _c.name
        _ret, _peak = res.stat(_n, "retention"), res.stat(_n, "m_line_peak")
        _scored = _ret[_peak >= ex.RETENTION_FLOOR]
        feas[_n] = {
            "task": task_ok[_n],
            "containment": bool(res.stat(_n, "alpha_op1").mean() <= ex.MEAN_ALIGN_GATE),
            "contrast ≥ partial": bool(res.stat(_n, "contrast").mean() >= ex.CONTRAST_PARTIAL),
            "grading": bool(res.stat(_n, "r2_sim").mean() >= _r2_floor + _r2_margin),
            "retention": bool(len(_scored) == 0 or _scored.min() >= ex.RETENTION_GATE),
            "no latch": bool(not res.latched(_n).any()),
            "m_line floor": bool(res.stat(_n, "m_line").min() >= ex.MARGIN_PARTIAL * ex.REF_M_LINE),
        }
    _feasible = [c for c in ex.CANDIDATES if all(feas[c.name].values())]
    _full = [c for c in _feasible if res.stat(c.name, "contrast").mean() >= ex.CONTRAST_GATE]
    _m = {c.name: float(res.stat(c.name, "m_line").mean()) for c in ex.CANDIDATES}
    _grade = {c.name: float(res.stat(c.name, "r2_sim").mean()) - _r2_floor for c in ex.CANDIDATES}
    _order = {c.name: i for i, c in enumerate((ex.RECIPE, ex.RECIPE_SHORT))}  # then the proposals, unranked

    if _full:
        _best = max(_m[c.name] for c in _full)
        _tied = [c for c in _full if _m[c.name] >= _best - _band]
        _win = max(_tied, key=lambda c: (_grade[c.name], -_order.get(c.name, 9)))
        adopted: str = _win.name
        _how = (
            f"{len(_tied)} candidate(s) within one band ({_band:.3f}) of the highest feasible m_line ({_best:.3f})"
            + ("; the tie went to the larger grading margin" if len(_tied) > 1 else "")
        )
    else:
        adopted = ex.RECIPE.name
        _missed = [k for k, ok in feas[adopted].items() if not ok]
        _how = "no feasible candidate clears the full contrast gate, so the full-length recipe is adopted" + (
            f"; it misses {', '.join(_missed)} itself" if _missed else ""
        )
    adoption_note: str = _how

    # The verdict reads the proposals against recipe-short.
    _short = _m[ex.RECIPE_SHORT.name]
    _feasible_props = [c for c in ex.PROPOSALS if c in _feasible]
    _unstable = [c.name for c in ex.PROPOSALS if [k for k, ok in feas[c.name].items() if not ok] == ["m_line floor"]]
    _grading_only = [
        c
        for c in ex.PROPOSALS
        if [k for k, ok in feas[c.name].items() if not ok] == ["grading"] and _m[c.name] > _short + _band
    ]
    _lead = {c.name: _m[c.name] - _short for c in ex.PROPOSALS}
    _summary = ", ".join(f"`{c.name}` {_lead[c.name]:+.3f}" for c in ex.PROPOSALS)
    if any(_lead[c.name] > _band for c in _feasible_props):
        _v = Verdict(
            "holds",
            f"A feasible proposal exceeds `recipe-short` ({_short:.3f}) by more than a band ({_band:.3f}): {_summary}. Adopted: `{adopted}`.",
        )
    elif any(abs(_lead[c.name]) <= _band for c in _feasible_props) or _grading_only:
        _v = Verdict(
            "partial",
            f"The feasible proposals sit within a band ({_band:.3f}) of `recipe-short` ({_short:.3f}): {_summary}. Adopted: `{adopted}`.",
        )
    elif not _feasible_props:
        _v = Verdict(
            "contrary",
            f"No proposal is feasible at fresh seeds ({_summary} against `recipe-short` at {_short:.3f}). Adopted: `{adopted}`.",
        )
    else:
        _v = Verdict(
            "contrary",
            f"Every feasible proposal sits more than a band ({_band:.3f}) below `recipe-short` ({_short:.3f}): {_summary}. Adopted: `{adopted}`.",
        )
    if _unstable:
        _v = Verdict(
            _v.status,
            _v.line
            + f" {', '.join(f'`{n}`' for n in _unstable)} missed only the per-run m_line floor, so it reads as unstable at fresh seeds rather than weak.",
        )
    h3: Verdict = _v
    return adopted, adoption_note, feas, h3


@app.cell(hide_code=True)
def _(res: Results):
    _head = ["candidate", "statistic", "survey", "seeds", "fresh (5 seeds)", "Δ (✓ clears band)", "σ per run", "band"]
    _rows = []
    for _c in ex.CANDIDATES:
        _seeds, _sv = res.survey_point(_c)
        for _i, _stat in enumerate(SURVEY_STATS):
            _fresh = res.em(_c.name, ex.PRIMARY_OP.name) if _stat == "holdout_em" else res.stat(_c.name, _stat)
            _sv_v = _sv[_stat] if _sv and _stat in _sv else None
            _b = ex.equiv_band(_stat, 5, _seeds) if _stat in ex.NOISE_RUN else None
            _d = (
                "—"
                if _sv_v is None
                else f"{_fresh.mean() - _sv_v:+.3f}"
                + ("" if _b is None else (" ✓" if abs(_fresh.mean() - _sv_v) > _b else " ·"))
            )
            _rows.append(
                [
                    f"<b><code>{_c.name}</code></b>" if _i == 0 else "",
                    _stat,
                    f"{_sv_v:.3f}" if _sv_v is not None else "—",
                    str(_seeds),
                    span2(_fresh),
                    _d,
                    f"{ex.NOISE_RUN[_stat]:.4f}" if _stat in ex.NOISE_RUN else "—",
                    f"{_b:.3f}" if _b is not None else "—",
                ]
            )
    _caption = """
    Each candidate's one-op survey numbers beside the fresh ones on the six-op grammar. The survey column is the trial's five-seed mean (for the proposals) or the survey's own three-seed re-run of the recipe at that length; fresh is the seed mean here with half the seed range. Δ is fresh minus survey, with a ✓ where the difference clears the band for those seed counts and a dot where it does not. σ is the frozen per-run spread from ex-2.1.10 behind every band. <code>holdout_em</code> is read on <code>mix</code>.
    """
    mo.Html(table_html(_head, _rows, _caption))
    return


@app.cell(hide_code=True)
def _(res: Results):
    _stats = [k for k in SURVEY_STATS if k in ex.NOISE_RUN]

    @themed(
        name="h3-survey-vs-fresh",
        alt_text="""
            Four small panels, one per statistic, listing the five candidates down the side. In each row a hollow marker is the survey value and a filled dot with a range bar is the fresh five-seed read; a grey strip around the survey value is the band. The fresh m_line dots sit left of their survey markers on every proposal.
        """,
        caption="""
            **Survey against fresh, per candidate.** The same numbers as the table above, drawn: the hollow marker is the survey value, the filled dot the fresh seed mean with a bar for the seed range, and the grey strip the band around the survey value for those seed counts. A dot outside its strip is a difference the resolution rule counts.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(_stats), figsize=(7.6, 1.8), sharey=True, layout="constrained")
        axes = cast(AxesRow, axes)
        strip = light_dark("#00000014", "#ffffff1a")
        ys = np.arange(len(ex.CANDIDATES))[::-1]
        for ax, k in zip(axes, _stats, strict=True):
            for y, c in zip(ys, ex.CANDIDATES, strict=True):
                n, sv = res.survey_point(c)
                v = res.em(c.name, ex.PRIMARY_OP.name) if k == "holdout_em" else res.stat(c.name, k)
                if sv and k in sv:
                    b = ex.equiv_band(k, 5, n)
                    ax.fill_betweenx([y - 0.42, y + 0.42], sv[k] - b, sv[k] + b, color=strip, lw=0, zorder=1)
                    ax.plot(sv[k], y, "o", ms=4.2, mfc="none", mec=ink(c.name), mew=1.0, zorder=3)
                ax.plot([v.min(), v.max()], [y, y], color=ink(c.name), lw=1.2, zorder=2, solid_capstyle="round")
                ax.plot(v.mean(), y, "o", ms=3.6, color=ink(c.name), zorder=4)
            ax.set_yticks(ys, [c.name for c in ex.CANDIDATES], fontsize=7)
            ax.set_title(k, fontsize=9)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="x", alpha=0.2)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(
    adopted: str,
    adoption_note: str,
    feas: dict[str, dict[str, bool]],
    res: Results,
):
    _checks = list(next(iter(feas.values())))
    _head = ["candidate", *_checks, "feasible", f"contrast ≥ {ex.CONTRAST_GATE:g}", "m_line ↑", "grading r² ↑"]
    _rows = []
    for _c in ex.CANDIDATES:
        _f = feas[_c.name]
        _ok = all(_f.values())
        _full = _ok and res.stat(_c.name, "contrast").mean() >= ex.CONTRAST_GATE
        _name = _c.name + (" ★" if _c.name == adopted else "")
        _rows.append(
            [
                _name,
                *["✓" if v else "✗" for v in _f.values()],
                "<b>✓</b>" if _ok else "✗",
                "✓" if _full else "✗",
                span2(res.stat(_c.name, "m_line")),
                span2(res.stat(_c.name, "r2_sim")),
            ]
        )
    _r2_floor = float(res.stat(ex.RECIPE.name, "r2_sim").mean()) - ex.GRADE_R2_DROP
    _caption = f"""
    Feasibility, as the selection rule reads it. Seed-mean checks: task within {ex.TASK_GATE:g} of the same-length control on every op; containment ᾱ ≤ {ex.MEAN_ALIGN_GATE:g}; contrast ≥ {ex.CONTRAST_PARTIAL:g}; grading r² at least {ex.GRADE_MARGIN_SD:g}σ ({ex.GRADE_MARGIN_SD * ex.NOISE_RUN["r2_sim"]:.3f}) above its floor of {_r2_floor:.3f}, the full-length recipe's fresh value less {ex.GRADE_R2_DROP:g}. Per-run checks: retention ≥ {ex.RETENTION_GATE:g} of the running peak for every run whose peak reaches {ex.RETENTION_FLOOR:g}; no run with more than {ex.LATCH_PI:g} of the non-red group's weight on op1; every run's m_line at least {ex.MARGIN_PARTIAL:g} of {ex.REF_M_LINE:.3f}. The selection then requires contrast at the full gate and takes the highest m_line, ties within one band ({ex.equiv_band("m_line"):.3f}) going to the larger grading margin. ★ marks the adopted point: {adoption_note}.
    """
    mo.Html(table_html(_head, _rows, _caption))
    return


@app.cell(hide_code=True)
def _(adopted: str, res: Results):
    _pts = {c.name: (res.stat(c.name, "r2_sim"), res.stat(c.name, "m_line")) for c in ex.CANDIDATES}
    _sv = {}
    for _c in ex.CANDIDATES:
        _, _s = res.survey_point(_c)
        if _s:
            _sv[_c.name] = (_s["r2_sim"], _s["m_line"])
    _short = float(res.stat(ex.RECIPE_SHORT.name, "m_line").mean())
    _band = ex.equiv_band("m_line", 5, 5)
    _r2_floor = float(res.stat(ex.RECIPE.name, "r2_sim").mean()) - ex.GRADE_R2_DROP

    @themed(
        name="h3-margin-vs-grading",
        alt_text="""
            A scatter of m_line against grading r squared for the five candidates, each in its own color: five small filled dots per candidate with a ring at the seed mean, and one hollow marker for the survey's value. A grey horizontal strip marks the band around recipe-short, and a dotted vertical line marks the grading floor.
        """,
        caption=f"""
            **m_line against grading r² for the five candidates.** Filled dots are single seeds, the ring is the seed mean, and the hollow diamond is the same candidate's one-op survey value. The grey strip is one band ({_band:.3f}) either side of <code>recipe-short</code>'s seed mean, which the H3 verdict reads the proposals against; the dotted line is the grading floor of the selection rule ({_r2_floor:.3f}, before its 1σ margin). The adopted point is labelled ★.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
        ax.axhspan(_short - _band, _short + _band, color=light_dark("#00000012", "#ffffff18"), lw=0, zorder=0)
        ax.axvline(_r2_floor, color=light_dark("#888", "#999"), lw=0.8, ls=(0, (1, 2)), zorder=1)
        for name, (r2, m) in _pts.items():
            color = ink(name)
            ax.scatter(r2, m, s=12, color=color, zorder=3, lw=0)
            ax.scatter(r2.mean(), m.mean(), s=90, facecolor="none", edgecolor=color, lw=1.3, zorder=4)
            if name in _sv:
                ax.scatter(*_sv[name], s=40, marker="D", facecolor="none", edgecolor=color, lw=1.0, zorder=4)
            ax.annotate(
                name + (" ★" if name == adopted else ""),
                (r2.mean(), m.mean()),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )
        ax.set_xlabel("grading r² (op1 alignment vs sim^1.5)", fontsize=8)
        ax.set_ylabel("m_line, mix probe lines", fontsize=8)
        ax.tick_params(labelsize=7)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(res: Results):
    _stride = ex.N_PROBE  # every color sits at op1 once per partner; op1 precedes the partner, so one read per color
    _x = ex.SIM_TARGET

    def _op1(cond: str) -> np.ndarray:
        """(seeds, colors) mean post-attention alignment at op1 on the `mix` lines."""
        a = res.arr(cond, "eval", f"{ex.PRIMARY_OP.name}/alpha_lines").astype(np.float32)  # (seeds, slices, N, T)
        return a[:, 1:, ::_stride, 0].mean(axis=1)

    @themed(
        name="h3-grading-clouds",
        alt_text="""
            Five small scatter panels, one per candidate. Each has 216 points, one per grid color drawn in that color, with similarity to red along the bottom and the alignment at op1 up the side. On every panel the points rise from left to right, tightest on the recipe and looser on the proposals, with the reds at the top right.
        """,
        caption="""
            **What grading looks like.** One panel per candidate: each point is one of the 216 grid colors, drawn in its own color, at its similarity-to-red target (sim^1.5) along the bottom and its seed-mean alignment at op1 up the side, averaged over the four post-attention slices. The r² in each title is the squared correlation of these two axes on the seed-mean cloud; the tables report the per-seed r² averaged, which is a little lower. A cloud that rises in a line is graded; one that steps from a floor to a ceiling is thresholded.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(
            1, len(ex.CANDIDATES), figsize=(7.6, 1.9), sharex=True, sharey=True, layout="constrained"
        )
        axes = cast(AxesRow, axes)
        for ax, c in zip(axes, ex.CANDIDATES, strict=True):
            y = _op1(c.name).mean(axis=0)
            ax.scatter(_x, y, s=9, c=ex.GRID_RGB, edgecolors=light_dark("#00000033", "#ffffff44"), linewidths=0.3)
            r2 = float(np.corrcoef(_x, y)[0, 1] ** 2)
            ax.set_title(f"{c.name}  r² {r2:.2f}", fontsize=8.5, color=ink(c.name))
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("α at op1", fontsize=7.5)
        axes[len(axes) // 2].set_xlabel("similarity to red, sim^1.5", fontsize=7.5)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(h3: Verdict):
    mo.md(rf"""
    **H3, {h3.status}.** {h3.line}
    """)
    return


@app.cell(hide_code=True)
def _(
    e6_reads: dict[str, tuple[float, float, float]],
    feas: dict[str, dict[str, bool]],
    res: Results,
):
    # Post hoc, after the results were read (2026-09-10). The frozen selection rule carries every H2
    # gate but the lead weight at the embedding, and none of H4; the point it adopts misses the first
    # and, under the plain projection, fails the second. This cell re-runs the rule with each gate
    # added, and the Findings entry names the outcome beside the frozen one.
    _mix = ex.PRIMARY_OP.name
    _m = {c.name: float(res.stat(c.name, "m_line").mean()) for c in ex.CANDIDATES}
    _lead = {c.name: float(res.stat(c.name, "lead_emb").mean()) for c in ex.CANDIDATES}
    _cost = {c.name: float(res.deficit(c.name, _mix, ex.PROJECTION.name).mean()) for c in ex.CANDIDATES}
    _band = ex.equiv_band("m_line")
    _lead_ok = [c for c in ex.CANDIDATES if all(feas[c.name].values()) and _lead[c.name] >= ex.LEAD_GATE]
    _pick_lead = max(_lead_ok, key=lambda c: _m[c.name]).name if _lead_ok else ex.RECIPE.name
    _sel_ok = [c for c in _lead_ok if _cost[c.name] <= ex.NONRED_DEFICIT_GATE]
    _top = max(_m[c.name] for c in _sel_ok) if _sel_ok else float("nan")
    _tied = [c for c in _sel_ok if _m[c.name] >= _top - _band]
    _d, _fb, _ff = e6_reads["m_line"]  # short minus full, at twenty seeds, and the two bands
    _wide = max(_fb, _ff)
    if len(_tied) <= 1:
        adopted_post: str = _tied[0].name if _tied else ex.RECIPE.name
        _tie = "no tie to break"
    elif -_d > _wide:
        adopted_post = ex.RECIPE.name
        _tie = (
            f"at twenty seeds the full recipe leads by {-_d:.3f}, more than the {_wide:.3f} band, so it takes the tie"
        )
    elif _d > _wide:
        adopted_post = ex.RECIPE_SHORT.name
        _tie = (
            f"at twenty seeds the short recipe leads by {_d:.3f}, more than the {_wide:.3f} band, so it takes the tie"
        )
    else:
        adopted_post = ex.RECIPE_SHORT.name
        _tie = f"at twenty seeds the two are within the {_wide:.3f} band ({_d:+.3f}), so the tie goes to the shorter arm, at half the compute"
    _leads = ", ".join(f"`{c.name}` {_lead[c.name]:.2f}" for c in ex.CANDIDATES)
    _costs = ", ".join(f"`{c.name}` {_cost[c.name]:.3f}" for c in ex.CANDIDATES)
    # The decision (2026-09-10, after E6 was read): D2.2 builds on the short arm. The rule above ranks on
    # margin and names the full recipe; the reasons for the short arm are in the paragraph below.
    carried: str = ex.RECIPE_SHORT.name
    _r2 = {c: float(res.pooled(c, "r2_sim").mean()) for c in (ex.RECIPE.name, ex.RECIPE_SHORT.name)}
    _ld = {c: float(res.pooled(c, "lead_emb").mean()) for c in (ex.RECIPE.name, ex.RECIPE_SHORT.name)}
    mo.md(rf"""
    ### Post hoc: the selection rule, with the gates it left out

    This is a deviation, written after the numbers were read. The selection rule above was frozen with two holes in it, and the point it adopts falls through both.

    The first hole is H2's lead gate. Feasibility carries every H2 gate except the lead weight of the red group at the embedding, which H2 gates at {ex.LEAD_GATE:g}; per candidate that weight is {_leads}. Add the gate, and the highest feasible m_line is `{_pick_lead}`.

    The second hole is H4. The rule ranks on placement and never looks at what the plain projection costs, and on every proposal that cost is most of the non-red `mix` lines: the deficit under `projection` is {_costs}, against H4's gate of {ex.NONRED_DEFICIT_GATE:g}. The cost sits on the syntax rows (E2, E7), and the operand-only edit routes around it, but an operating point that needs a routed edit before its first intervention is a poor base for the anchored-op experiments. Add H4's selectivity gate as well, and only the two recipe lengths remain; at five seeds they sit within a band of each other, and E6's twenty-seed read breaks the tie: {_tie}.

    The amended rule adopts **`{adopted_post}`**. The frozen H3 verdict stands as written, and the proposals keep their reads: they place *red* with a larger margin, and they pay for it on the syntax rows. Whether that cost can be removed rather than routed around is the untied-readout question in the Discussion.

    **D2.2 builds on `{carried}`, by decision rather than by rule.** Both rules rank on margin, and at twenty seeds the full recipe leads on margin by {-_d:.3f}, about two bands. The short arm is ahead on the rest of what the anchored-op experiments will lean on: it grades better (r² {_r2[carried]:.3f} against {_r2[adopted_post]:.3f}), puts more of the pull on the drawing operand at the embedding ({_ld[carried]:.2f} against {_ld[adopted_post]:.2f}), has a little more contrast, gives the same H4 read, and costs half as much to train. Those experiments read the grading and the operand lead more than they read the margin, so the graded point is the better base. The decision was taken on 2026-09-10, after E6 was read, and this paragraph is its record.
    """)
    return adopted_post, carried


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Suppression transfers (H4)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    **H4.** Projecting the axis out removes *red* on every op, at the selectivity ex-2.2.1 reached. On the recipe and on the adopted point, under `projection`, seed-mean accuracy on the red lines of each op falls to at most {ex.RED_ACC_GATE:g}, and the seed-mean non-red deficit on the `mix` lines is at most {ex.NONRED_DEFICIT_GATE:g}. Partial: exactly one clause holds, or removal holds on all but one op, or the deficit sits in the {ex.NONRED_DEFICIT_GATE:g}–{ex.NONRED_DEFICIT_PARTIAL:g} band.

    The deficit gate is the band ex-2.2.1 landed in (0.024, partial at its 0.02 gate), and we report the 0.02 read beside it. That cost came from the syntax rows, and E2 reads the rows of the six op words.

    Contrary on removal: red accuracy near the clean value on one of the new ops. That would say the model reads *red* off the axis on lines of that op, and the per-op alignment in E1 should show it first. Contrary on selectivity: a deficit above {ex.NONRED_DEFICIT_PARTIAL:g}. That would send the intervention-tuning pass to the `operands` and `shaped` rows before anything is anchored.
    """)
    return


@app.cell(hide_code=True)
def _(adopted: str, res: Results, task_ok: dict[str, bool]):
    h4_conds: list[str] = [ex.RECIPE.name] + ([adopted] if adopted != ex.RECIPE.name else [])
    """The conditions H4 gates: the recipe, and the adopted point when it is a different one."""
    _iv = ex.PROJECTION.name
    _mix = ex.PRIMARY_OP.name
    _reads = {}
    for _c in h4_conds:
        _red = {op: float(res.score(_c, op, _iv, "acc", "red").mean()) for op in ex.OP_NAMES}
        _def = float(res.deficit(_c, _mix, _iv).mean())
        _reads[_c] = (_red, _def)
    _removal_ok = {c: all(v <= ex.RED_ACC_GATE for v in r.values()) for c, (r, _) in _reads.items()}
    _removal_misses = {c: [op for op, v in r.items() if v > ex.RED_ACC_GATE] for c, (r, _) in _reads.items()}
    _select_ok = {c: d <= ex.NONRED_DEFICIT_GATE for c, (_, d) in _reads.items()}
    _select_band = {c: ex.NONRED_DEFICIT_GATE < d <= ex.NONRED_DEFICIT_PARTIAL for c, (_, d) in _reads.items()}
    _txt = "; ".join(
        f"`{c}`: red accuracy under projection {min(r.values()):.2f}–{max(r.values()):.2f} across the ops, `mix` non-red deficit {d:.3f}"
        for c, (r, d) in _reads.items()
    )
    if not task_ok[ex.RECIPE.name] and adopted == ex.RECIPE.name:
        _v = Verdict("unscored", "No candidate is feasible, so H4 is not scored.")
    elif all(_removal_ok.values()) and all(_select_ok.values()):
        _v = Verdict("holds", f"Projection removes *red* on every op at ex-2.2.1's selectivity. {_txt}.")
    elif all(
        (_removal_ok[c] or len(_removal_misses[c]) == 1) and (_select_ok[c] or _select_band[c]) for c in h4_conds
    ) and any(not _removal_ok[c] or not _select_ok[c] for c in h4_conds):
        _v = Verdict(
            "partial",
            f"{_txt}. "
            + " ".join(
                [f"Removal misses on `{', '.join(_removal_misses[c])}`." for c in h4_conds if _removal_misses[c]]
                + [
                    f"The deficit sits in the {ex.NONRED_DEFICIT_GATE:g}–{ex.NONRED_DEFICIT_PARTIAL:g} band on `{c}`."
                    for c in h4_conds
                    if _select_band[c]
                ]
            ),
        )
    else:
        _v = Verdict(
            "contrary",
            f"{_txt}. "
            + " ".join(
                [
                    f"Removal misses on `{', '.join(_removal_misses[c])}` for `{c}`."
                    for c in h4_conds
                    if _removal_misses[c]
                ]
                + [
                    f"The `mix` deficit on `{c}` is above {ex.NONRED_DEFICIT_PARTIAL:g}."
                    for c in h4_conds
                    if _reads[c][1] > ex.NONRED_DEFICIT_PARTIAL
                ]
            ),
        )
    h4: Verdict = _v
    return h4, h4_conds


@app.function(hide_code=True)
def removal_table(res: Results, cond: str) -> str:
    """H4's removal read on one condition: red-line accuracy per op under each operator, with ex-2.2.1's one-op column."""
    ivs = [ex.PROJECTION, *ex.RIDE_ALONG]
    ex221 = {"projection": EX221_PROJECTION, "operands": "operands", "shaped": "shaped", "ablate": "ablate"}

    def b(s: str, ok: bool) -> str:
        return f"<b>{s}</b>" if ok else s

    rows = []
    for iv in ivs:
        gated = iv.name == ex.PROJECTION.name
        row = [f"{cond}, {iv.name}"]
        for op in ex.OP_NAMES:
            v = res.score(cond, op, iv.name, "acc", "red")
            row.append(b(span2(v, ".2f"), bool(gated and v.mean() <= ex.RED_ACC_GATE)))
        row.append(span2(res.ex221_stat(ex221[iv.name], "acc", "red"), ".2f"))
        rows.append(row)
    rows.append(
        [f"{cond}, clean"]
        + [span2(res.score(cond, op, None, "acc", "red"), ".2f") for op in ex.OP_NAMES]
        + [span2(res.ex221_stat(None, "acc", "red"), ".2f")]
    )
    head = ["condition, operator"] + [f"{op} red acc ↓" for op in ex.OP_NAMES] + ["ex-2.2.1 (mix)"]
    caption = f"""
    Removal on the red lines of each op, for <code>{cond}</code>. Each value is seed-mean exact-match accuracy on the red lines (dose ≥ {ex.RED_DOSE:g}) under the operator, with half the seed range; the last column is the same operator on ex-2.2.1's one-op grammar (nine seeds of the D2.1 recipe). Bold marks a <code>projection</code> value inside the removal gate, at or below {ex.RED_ACC_GATE:g}. The ride-along rows carry no gate. The last row is the clean pass.
    """
    return table_html(head, rows, caption, ref_rows=frozenset({len(rows) - 1}))


@app.function(hide_code=True)
def selectivity_table(res: Results, cond: str) -> str:
    """H4's selectivity read on one condition: the non-red deficit per op under each operator."""
    ivs = [ex.PROJECTION, *ex.RIDE_ALONG]
    ex221 = {"projection": EX221_PROJECTION, "operands": "operands", "shaped": "shaped", "ablate": "ablate"}
    mix = ex.PRIMARY_OP.name

    def cell(v: np.ndarray, gated: bool) -> str:
        s = span2(v)
        return f"<b>{s}</b>" if gated and v.mean() <= ex.NONRED_DEFICIT_GATE else s

    rows = []
    for iv in ivs:
        row = [f"{cond}, {iv.name}"]
        for op in ex.OP_NAMES:
            row.append(cell(res.deficit(cond, op, iv.name), iv.name == ex.PROJECTION.name and op == mix))
        row.append(span2(res.ex221_deficit(ex221[iv.name])))
        rows.append(row)
    head = ["condition, operator"] + [f"{op} non-red deficit ↓" for op in ex.OP_NAMES] + ["ex-2.2.1 (mix)"]
    d = res.deficit(cond, mix, ex.PROJECTION.name).mean()
    caption = f"""
    Selectivity on the non-red lines of each op, for <code>{cond}</code>. The deficit is clean exact-match accuracy minus intervened accuracy on the non-red lines (dose ≤ {ex.NONRED_DOSE:g}), seed mean with half the seed range; the last column is ex-2.2.1's one-op value for the same operator. The gate reads <code>projection</code> on <code>mix</code> only, at or below {ex.NONRED_DEFICIT_GATE:g} (bold); the 0.02 read that ex-2.2.1 gated at is {"met" if d <= 0.02 else "not met"} there ({d:.3f}). The other ops are reported for E2.
    """
    return table_html(head, rows, caption)


@app.cell(hide_code=True)
def _(res: Results):
    mo.Html(removal_table(res, ex.RECIPE.name))
    return


@app.cell(hide_code=True)
def _(adopted: str, res: Results):
    # The adopted point's own table, when it is not the recipe.
    mo.Html(removal_table(res, adopted) if adopted != ex.RECIPE.name else "")
    return


@app.cell(hide_code=True)
def _(res: Results):
    mo.Html(selectivity_table(res, ex.RECIPE.name))
    return


@app.cell(hide_code=True)
def _(adopted: str, res: Results):
    mo.Html(selectivity_table(res, adopted) if adopted != ex.RECIPE.name else "")
    return


@app.cell(hide_code=True)
def _(h4_conds: list[str], res: Results):
    _iv = ex.PROJECTION.name
    _rows = []
    for _c in h4_conds:
        for _key, _title in (("argmax_dist", "decoded answer"), ("expected_dist", "expected")):
            for _pass, _ivn in (("projection", _iv), ("clean", None)):
                _rows.append(
                    [f"{_c}, {_title}, {_pass}"]
                    + [span2(res.score(_c, op, _ivn, _key, "red"), ".2f") for op in ex.OP_NAMES]
                )
    _ref = res.ex221_stat(EX221_PROJECTION, "guess_dist_red", None)
    _head = ["condition, distance, pass"] + [f"{op} ↓" for op in ex.OP_NAMES]
    _caption = f"""
    Answer distance on the red lines under <code>projection</code>, in the unit cube. The decoded row is the distance from the argmax answer to the true one; the expected row takes the expectation of that distance under the whole answer distribution over the color vocabulary. Each value is the seed mean with half the seed range; the clean pass sits under each projected row. On ex-2.2.1's one-op grammar the decoded distance under the projection was {_ref.mean():.2f} <span class='range'>±{(_ref.max() - _ref.min()) / 2:.2f}</span>; a one-level step on this grid is {1 / (len(ex.LEVELS) - 1):.2f}.
    """
    mo.Html(table_html(_head, _rows, _caption))
    return


@app.cell(hide_code=True)
def _(res: Results):
    _c = ex.RECIPE.name
    _iv = ex.PROJECTION.name
    _clean = {
        op: np.array([r["ops"][op]["clean"]["alpha_q99_nonred"] for r in res.scored(_c)], float).mean(0)
        for op in ex.OP_NAMES
    }
    # The two signed tails behind each |α| above, from the per-line alignments the eval stage keeps.
    _band = {op: res.clean_alpha_band(_c, op) for op in ex.OP_NAMES}
    _arr = {op: res.score(_c, op, _iv, "q99_alpha_nonred", None).mean(0) for op in ex.OP_NAMES}
    _ratio = {
        op: res.score(_c, op, _iv, "q99_write_nonred", None).mean(0) / np.arcsin(_clean[op]) for op in ex.OP_NAMES
    }
    _over = {op: int((_ratio[op][1:] > 1).sum()) for op in ex.OP_NAMES}
    _top = max(float(m.max()) for m in _clean.values())

    @themed(
        name="h4-write-bound-maps",
        alt_text="""
            A grid of thirty small panels, five rows for the residual slices with the embedding at the bottom and six columns for the ops, each over the six token positions, drawn around a zero line. The shaded band spans the clean alignment's two signed tails over the non-red lines; at the color positions it straddles zero, leaning negative at the embedding and positive on op2 in the deeper slices, while at the op word and the equals sign it sits wholly above zero and pinches to a line at the embedding, where every non-red line shares one token. The dashed pair is the arriving alignment's magnitude envelope under projection, mirrored about zero. The six columns look alike.
        """,
        caption=f"""
            **The bound and the write, per site and per op, for the recipe under <code>projection</code>, signed.** For each op's probe lines, at each (slice, position), seed mean. The shaded band runs from the 1st to the 99th percentile of the signed clean alignment over the non-red lines; the scorer's bound is the arcsine of the 99th-percentile |α|, which is at least the further edge and equal to it where one tail holds the extreme, so this is close to the map the earlier version drew folded onto one side. The dashed pair is the 99th-percentile |α| arriving at the operator, whose arcsine is the write, mirrored about zero because the scorer keeps only its magnitude. The embedding row matches the clean band by construction. One row per slice with the embedding at the bottom, one column per op, and every panel on the same scale, so a write that pokes above its bound reads as a dashed line clearing the further band edge. Sites where the seed-mean write exceeds the seed-mean bound, per op: {", ".join(f"<code>{op}</code> {n}" for op, n in _over.items())} of 24 post-embedding sites.
        """,
    )
    def _plot() -> plt.Figure:
        from matplotlib.layout_engine import ConstrainedLayoutEngine

        bound_c, write_c = ink("recipe"), light_dark("#8a0000", "#ff7070")
        fig, axes = plt.subplots(len(SLICE_NAMES), len(ex.OP_NAMES), figsize=(7.6, 4.2), sharex=True, sharey=True)
        axes = cast(AxesGrid, axes)
        engine = fig.get_layout_engine()
        assert isinstance(engine, ConstrainedLayoutEngine)
        engine.set(hspace=0, h_pad=0.01, wspace=0.04)
        x = np.arange(len(POS_NAMES))
        top = min(1.0, _top * 1.15)
        for col, op in enumerate(ex.OP_NAMES):
            for row in range(len(SLICE_NAMES)):
                sl = len(SLICE_NAMES) - 1 - row  # the embedding at the bottom, as the profile figures draw it
                ax = axes[row][col]
                lo, hi = _band[op][0][sl], _band[op][1][sl]
                ax.axhline(0, c="#888", lw=0.5, alpha=0.5)
                smooth_step_band(ax, x, lo, hi, ramp=0.3, color=bound_c, alpha=light_dark(0.2, 0.26), fillet=1.5)
                for edge in (lo, hi):
                    smooth_step(ax, x, edge, ramp=0.3, color=bound_c, lw=1.1, fillet=1.5)
                for sign in (1, -1):
                    smooth_step(
                        ax, x, sign * _arr[op][sl], ramp=0.3, color=write_c, lw=1.0, ls=(0, (2, 1.5)), fillet=1.5
                    )
                ax.set(
                    ylim=(-top, top),
                    xlim=(-0.5, len(POS_NAMES) - 0.5),
                    yticks=[-round(top / 2, 2), 0, round(top / 2, 2)],
                )
                ax.spines[:].set_visible(False)
                ax.grid(axis="y", c="#888", alpha=0.2)
                ax.tick_params(axis="x", length=0, labelsize=7)
                ax.tick_params(axis="y", left=True, direction="in", labelleft=col == 0, labelsize=6)
                if col == 0:
                    ax.set_ylabel(SLICE_NAMES[sl], fontsize=7.5)
            axes[0][col].set_title(op, fontsize=9)
            axes[-1][col].set_xticks(x, POS_NAMES, rotation=60, fontsize=6.5)
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], color=bound_c, lw=1.1, label="clean α, 1st–99th percentile"),
            Line2D([], [], color=write_c, lw=1.0, ls=(0, (2, 1.5)), label="±|α| arriving (projected)"),
        ]
        fig.legend(handles=handles, fontsize=6.5, frameon=False, ncol=2, loc="outside upper left")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(h4: Verdict):
    mo.md(rf"""
    **H4, {h4.status}.** {h4.line}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    Preregistered as exploratory, no gates.

    ### E1 — per-op statistics

    Every H2 statistic read on the probe lines of each op, rather than on the `mix` lines alone, for every candidate. The labeller never sees the op, so a placement that differs by op would mean the blocks carry *red* differently under different rules. Only m_line and contrast can differ: they involve op2 and the answer, which follow the op word, while grading, containment, and the lead weight are read at or before op1, where causal attention has not yet seen which rule the line uses.

    The five statistics, in a phrase each: **m_line** is how much more the labelled lines lean on the axis than the average line does (the margin the survey ranked on); **grading r²** is whether that lean rises smoothly with redness rather than switching on; **contrast** is how much the pull prefers the operand that drew the label over the other one, at depth; **containment ᾱ** is the mean lean of every color at op1, which should stay near zero; and the **lead weight** is the share of the pull that sits on the drawing operand at the embedding.
    """)
    return


@app.cell(hide_code=True)
def _(res: Results):
    _stats = (
        ("m_line", "m_line ↑"),
        ("r2_sim", "grading r² ↑"),
        ("contrast", "contrast ↑"),
        ("alpha_op1", "containment ᾱ ↓"),
        ("lead_emb", "lead weight ↑"),
    )
    _head = ["candidate", "statistic"] + [f"<code>{op}</code>" for op in ex.OP_NAMES] + ["spread"]
    _rows = []
    _spread = {}
    for _c in ex.CANDIDATES:
        for _i, (_k, _title) in enumerate(_stats):
            _means = np.array([res.stat(_c.name, _k, op).mean() for op in ex.OP_NAMES])
            _rows.append(
                [f"<b><code>{_c.name}</code></b>" if _i == 0 else "", _title]
                + [span2(res.stat(_c.name, _k, op)) for op in ex.OP_NAMES]
                + [f"{_means.max() - _means.min():.3f}"]
            )
            _spread[_c.name, _k] = float(_means.max() - _means.min())
    e1_spread: dict[tuple[str, str], float] = _spread
    _caption = """
    The placement statistics per op, for every candidate: seed mean with half the seed range. The <code>mix</code> column repeats the gated values. Spread is the range of the seed means across the six ops, to read against the band for that statistic in the H3 table.
    """
    mo.Html(table_html(_head, _rows, _caption))
    return (e1_spread,)


@app.cell(hide_code=True)
def _(res: Results):
    _stats = (("m_line", "m_line"), ("contrast", "contrast"))
    _x = np.arange(len(ex.OP_NAMES))
    # The five ops that share one partner draw; `mix` has its own, so it is drawn apart from them.
    _shared = [o for o in ex.OP_NAMES if o != ex.PRIMARY_OP.name]

    @themed(
        name="e1-per-op-dots",
        alt_text="""
            A two-by-two grid, m_line in the left column and contrast in the right, with the six ops along the
            bottom and a dashed divider separating mix from the other five. The tall upper row plots the
            statistic itself: the candidates sit in well-separated groups, each near-flat across the ops. The
            short lower row plots the same points as distances from the candidate's mean over the five, on a
            scale about ten times finer, over a grey stripe marking the equivalence band. There the contrast
            points sit inside the stripe with mix far below it, furthest on the two recipe arms; the m_line
            points scatter around the stripe with bars taller than it, darken high and add low on most
            candidates.
        """,
        caption="""
            **The two per-op placement statistics, whole and magnified.** m_line and contrast are the only
            statistics that can vary by op. One line per candidate in its ink, one dot per op at the seed mean
            with a bar for the seed range, offset a little so the candidates do not overlap. The upper row
            plots each statistic on its own scale, where the gaps between candidates set the axis. The lower
            row magnifies what that hides: the same points as a distance from the candidate's own mean over
            the five ops that share a partner draw, per seed, so the ops can be compared within a run. The
            grey stripe there is that statistic's equivalence band, the smallest seed-mean difference the
            resolution rule may call a difference. <code>mix</code> sits left of the divider, and out of the
            baseline, because its probe lines are D2.1's on-grid partners rather than the shared draw: its
            distance from the five carries the partner set as well as the op.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(
            2,
            len(_stats),
            figsize=(6.4, 3.9),
            height_ratios=(3, 2),
            sharex=True,
            layout="constrained",
        )
        axes = cast(AxesGrid, axes)
        off = np.linspace(-0.22, 0.22, len(ex.CANDIDATES))
        rule = light_dark("#555555", "#aaaaaa")
        for col, (k, title) in enumerate(_stats):
            hi, lo = axes[0][col], axes[1][col]
            half = ex.equiv_band(k) / 2
            lo.axhspan(-half, half, color=rule, alpha=0.13, lw=0, zorder=0)
            lo.axhline(0, color=rule, lw=0.6, alpha=0.5, zorder=1)
            for d, c in zip(off, ex.CANDIDATES, strict=True):
                v = np.stack([res.stat(c.name, k, op) for op in ex.OP_NAMES])  # (ops, seeds)
                # Per seed against its own mean over the five: the ops share seeds, so the paired
                # difference is the comparison, and `mix` stays out of the baseline it is read against.
                dev = v - np.stack([res.stat(c.name, k, op) for op in _shared]).mean(axis=0)
                for ax, y in ((hi, v), (lo, dev)):
                    ax.plot(_x[1:] + d, y[1:].mean(axis=1), "-", color=ink(c.name), lw=0.8, alpha=0.6, zorder=2)
                    ax.vlines(_x + d, y.min(axis=1), y.max(axis=1), color=ink(c.name), lw=1.0, zorder=2)
                    ax.plot(
                        _x + d,
                        y.mean(axis=1),
                        "o",
                        ms=3,
                        color=ink(c.name),
                        zorder=3,
                        label=c.name if (k == "m_line" and ax is hi) else None,
                    )
            hi.set_title(title, fontsize=9)
            lo.set_ylabel("Δ from the five", fontsize=6.5)
            lo.set_xticks(_x, ex.OP_NAMES, fontsize=6.5, rotation=30)
            for ax in (hi, lo):
                ax.axvline(0.5, color=rule, lw=0.6, ls=(0, (2, 2)), alpha=0.5, zorder=1)
                ax.tick_params(axis="y", labelsize=7)
                ax.grid(axis="y", alpha=0.2)
        fig.legend(fontsize=6, frameon=False, ncol=len(ex.CANDIDATES), loc="outside lower center")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(e1_spread: dict[tuple[str, str], float], res: Results):
    _shared = [o for o in ex.OP_NAMES if o != ex.PRIMARY_OP.name]

    def _five(c: str, k: str) -> np.ndarray:
        """The seed means of one statistic over the five ops that share a partner draw."""
        return np.array([res.stat(c, k, op).mean() for op in _shared])

    # Spread over the five, and how far `mix` sits from their mean: the first is a clean op read,
    # the second carries the partner set too.
    _sp = {(c.name, k): float(np.ptp(_five(c.name, k))) for c in ex.CANDIDATES for k in ("m_line", "contrast")}
    _k5 = max(_sp[c.name, "contrast"] for c in ex.CANDIDATES)
    _gk = [
        float(res.stat(c.name, "contrast", ex.PRIMARY_OP.name).mean() - _five(c.name, "contrast").mean())
        for c in ex.CANDIDATES
    ]
    _m6 = max(e1_spread[c.name, "m_line"] for c in ex.CANDIDATES)
    _k6 = max(e1_spread[c.name, "contrast"] for c in ex.CANDIDATES)
    _l6 = max(e1_spread[c.name, "lead_emb"] for c in ex.CANDIDATES)
    mo.md(rf"""
    Read across all six ops, the widest spread of seed-mean m_line on any candidate is {_m6:.3f} (band {ex.equiv_band("m_line"):.3f}) and of contrast {_k6:.3f} (band {ex.equiv_band("contrast"):.3f}), both wider than the band. But the six ops are not six comparable reads. `mix` draws its probe lines from its own on-grid partners, D2.1's set, while the other five share one draw, so only those five differ from each other in the op alone. Split that way the two statistics separate.

    **Contrast is op-blind among the five.** Its spread there is no wider than the band on any candidate ({_k5:.3f} at most, band {ex.equiv_band("contrast"):.3f}). All of the six-op spread is `mix`, which sits below the five on every candidate, by {min(-g for g in _gk):.3f} to {max(-g for g in _gk):.3f} — furthest on the two recipe arms, whose contrast is highest to begin with. Since `mix` is also the op with the different partner set, that step reads as the partner set rather than as the rule, and a `mix` probe drawn like the others would be needed to tell the two apart.

    **m_line is less settled.** Among the five it stays inside the band on the recipe arms ({_sp[ex.RECIPE.name, "m_line"]:.3f} and {_sp[ex.RECIPE_SHORT.name, "m_line"]:.3f}) and sits at it on `t00` ({_sp["t00", "m_line"]:.3f}), reaching {_sp["t48", "m_line"]:.3f} on `t48` and {_sp["t12", "m_line"]:.3f} on `t12`, the two proposals with the lowest anchor weight, against a band of {ex.equiv_band("m_line"):.3f}. The ordering repeats across candidates — `darken` high on all five, `add` low on all but `recipe-short` — but the per-seed deviations are wider than the band, so five seeds do not settle whether the ordering is real. If it is, the probe set is the first place to look: m_line is a margin over the average line of that op's set, and the five sets share their op1 colors and partners but not their answers, so an op whose answers sit differently against the red axis would move it with the placement unchanged. Either way the effect is small, and every gated read is on `mix`.

    Grading r² and containment ᾱ are identical on all six ops to every printed digit. Both are read at op1, which precedes the op word, so under causal attention they cannot see which rule the line uses; the table prints them for completeness. The lead weight is read at op1 too, but its softmin normalizer runs over the whole span, so the op word's own embedding enters it: it moves by {_l6:.3f} at most, again with `mix` apart from the rest.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### E2 — the op words

    The alignment of the six op-word embeddings and of `=` and the newline, per slice, on the recipe and the control. Ex-2.2.1 found a constant component on `+` and `=` of about 0.3–0.4 in the published map, which is where its non-red cost came from. The second question is whether the alignment of an op word predicts the non-red deficit on its lines under `projection`.
    """)
    return


@app.function(hide_code=True)
def alpha_at(res: Results, cond: str, op: str, position: int) -> np.ndarray:
    """(seeds, slices) mean alignment at one position of one op's probe lines."""
    return np.array([r["per_op"][op]["alpha_pos"] for r in res.runs(cond)], float)[:, :, position]


@app.function(hide_code=True)
def syntax_table(res: Results, cond: str) -> str:
    """E2's table for one condition: the alignment of each op word, `=`, and the newline, per slice."""
    rows = []
    for op in ex.OP_NAMES:
        a = alpha_at(res, cond, op, 1)
        rows.append([f"<code>{op}</code>"] + [span2(a[:, s], ".2f") for s in range(len(SLICE_NAMES))])
    for name, p in (("=", 3), ("⏎", 5)):
        a = np.mean([alpha_at(res, cond, op, p) for op in ex.OP_NAMES], axis=0)
        rows.append([f"<code>{name}</code>"] + [span2(a[:, s], ".2f") for s in range(len(SLICE_NAMES))])
    head = ["token"] + [f"slice {s}" for s in SLICE_NAMES]
    caption = f"""
    Alignment (cos to e₁) of the syntax tokens on <code>{cond}</code>, per slice: each op word read on its own op's probe lines, and <code>=</code> and the newline averaged over the six ops. Seed mean with half the seed range. The embedding column is the token's embedding itself; deeper columns are the residual stream at that position.
    """
    return table_html(head, rows, caption)


@app.cell(hide_code=True)
def _(res: Results):
    mo.Html(syntax_table(res, ex.RECIPE.name))
    return


@app.cell(hide_code=True)
def _(adopted: str, res: Results):
    mo.Html(syntax_table(res, adopted) if adopted != ex.RECIPE.name else "")
    return


@app.cell(hide_code=True)
def _(res: Results):
    mo.Html(syntax_table(res, ex.CONTROL.name))
    return


@app.cell(hide_code=True)
def _(adopted: str, res: Results):
    # Added after the results were read: the two proposals the adoption did not name, for the same rows.
    _others = [c.name for c in ex.PROPOSALS if c.name != adopted]
    mo.Html("".join(syntax_table(res, c) for c in _others))
    return


@app.cell(hide_code=True)
def _(res: Results):
    _c = ex.RECIPE.name
    _x = np.array([np.abs(alpha_at(res, _c, op, 1)[:, 1:]).mean() for op in ex.OP_NAMES])
    _y = np.array([res.deficit(_c, op, ex.PROJECTION.name).mean() for op in ex.OP_NAMES])
    _r = float(np.corrcoef(_x, _y)[0, 1]) if _x.std() > 0 and _y.std() > 0 else float("nan")
    _pairs = ", ".join(f"`{op}` {x:.2f} / {y:.3f}" for op, x, y in zip(ex.OP_NAMES, _x, _y, strict=True))
    mo.md(rf"""
    On the recipe, the mean |α| of each op word over the four post-attention slices, against the non-red deficit under `projection` on its lines: {_pairs}. Pearson r over the six ops is {_r:+.2f}. Six points cannot carry much, and the sign is what to read: a positive r says the ops whose word sits further along the axis lose more on their non-red lines when the axis is projected out.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### E3 — redder than both

    Every op but `lighten` has lines whose answer is redder than either operand (the [op table](#the-grammar) counts them). On those lines the strongest evidence for *red* sits at the answer, and the labeller never keys on that position. This is the blind-span case named in the scope note of ex-2.1.10. We read the alignment at `=` and at the answer position against lines of the same op and dose whose answer is not redder than both, plus the softmin profile on each group, and then restrict the H4 statistics to these lines.

    <!-- REVIEW: a labeller that reads the answer as well as the operands (so `magenta darken yellow = red` draws a *red* label from its answer) is the M3-shaped labelling, and is filed as a fast-follow under (c) of todo/science/labeling-pull-span-variants-ex-2-1.md rather than changed here. E3 reads the blind span under the operand-only labeller first, which is what makes the answer position blind. Verify: that item's 2026-09-09 note. -->
    """)
    return


@app.cell(hide_code=True)
def _(res: Results):
    _c = ex.RECIPE.name
    _bins = np.linspace(0.0, 1.0, 11)
    _ops = [op for op in ex.OP_NAMES if ex.line_counts(ex.OP_BY_NAME[op])["redder"] > 0]

    def _matched_diff(v: np.ndarray, redder: np.ndarray, dose: np.ndarray) -> float:
        """Mean over dose bins, weighted by the redder count, of (redder mean − other mean) of a per-line value."""
        b = np.clip(np.digitize(dose, _bins) - 1, 0, len(_bins) - 2)
        num = den = 0.0
        for k in np.unique(b[redder]):
            a, o = v[redder & (b == k)], v[~redder & (b == k)]
            if len(o):
                num += len(a) * (a.mean() - o.mean())
                den += len(a)
        return num / den if den else float("nan")

    _rows = []
    e3_profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _op in _ops:
        _d_eq, _d_ans, _prof_r, _prof_m = [], [], [], []
        for _r in res.runs(_c):
            cos = res.arrays[f"{_r['label']}/eval/{_op}/alpha_lines"].astype(np.float32)  # (slices, N, T)
            redder = res.arrays[f"{_r['label']}/eval/{_op}/redder"]
            dose = res.arrays[f"{_r['label']}/eval/{_op}/dose"]
            deep = cos[1:].mean(axis=0)  # (N, T)
            _d_eq.append(_matched_diff(deep[:, 3], redder, dose))
            _d_ans.append(_matched_diff(deep[:, ex.ANSWER_POS], redder, dose))
            w = softmin_weights(1.0 - cos[1:, :, : ex.SPAN], _r["tau"]).mean(axis=0)  # (N, roles)
            b = np.clip(np.digitize(dose, _bins) - 1, 0, len(_bins) - 2)
            wt = np.zeros(len(dose))
            for k in np.unique(b[redder]):
                m = ~redder & (b == k)
                if m.any():
                    wt[m] = (redder & (b == k)).sum() / m.sum()
            _prof_r.append(w[redder].mean(axis=0))
            _prof_m.append((w * wt[:, None]).sum(axis=0) / wt.sum())
        e3_profiles[_op] = (np.array(_prof_r), np.array(_prof_m))
        _n = int(res.runs(_c)[0]["per_op"][_op]["n_redder"])
        _clean = res.score(_c, _op, None, "acc", "redder")
        _proj = res.score(_c, _op, ex.PROJECTION.name, "acc", "redder")
        _comp = np.array(
            [r["ops"][_op]["interventions"][ex.PROJECTION.name]["composition_redder"] for r in res.scored(_c)], float
        )
        _miss = _comp[:, 1:]  # everything but the true answer
        _share = _miss / np.maximum(_miss.sum(axis=1, keepdims=True), 1)
        _rows.append(
            [
                f"<code>{_op}</code>",
                f"{_n:,}",
                span2(np.array(_d_eq), ".3f"),
                span2(np.array(_d_ans), ".3f"),
                span2(_clean, ".2f"),
                span2(_proj, ".2f"),
                span2(_share[:, 0], ".2f"),
                span2(_share[:, 2], ".2f"),
            ]
        )
    _head = [
        "op",
        "redder lines",
        "Δα at =",
        "Δα at the answer",
        "clean acc",
        "projection acc ↓",
        "misses → red operand",
        "misses → neighbor",
    ]
    _caption = """
    The redder-than-both lines of each op on the recipe. Δα is the alignment on those lines minus the alignment on lines of the same op in the same dose bin whose answer is not redder than both, averaged over the four post-attention slices and weighted by the redder lines' dose histogram; a positive value at the answer says the stream carries the answer's extra redness there, where the labeller never keyed. The right half restricts H4 to these lines: exact-match accuracy clean and under <code>projection</code>, and, of the projected answers that are wrong, the share that are the redder operand and the share that are a one-step neighbor of the truth. Seed means with half the seed range; <code>lighten</code> has no such lines.
    """
    mo.Html(table_html(_head, _rows, _caption))
    return (e3_profiles,)


@app.cell(hide_code=True)
def _(e3_profiles: dict[str, tuple[np.ndarray, np.ndarray]]):
    _ops = list(e3_profiles)
    _x = np.arange(ex.SPAN)

    @themed(
        name="e3-redder-profiles",
        alt_text="""
            A row of small panels, one per op with redder-than-both lines, each showing two softmin profiles over the roles op1, op, op2, and equals: a solid line for the redder lines and a dashed line for dose-matched lines whose answer is not redder than both. The two lines lie close together in every panel.
        """,
        caption="""
            **Softmin profiles on the redder-than-both lines, recipe, post-attention slices.** One panel per op that has such lines. Solid is the seed-mean profile over the span roles on the redder lines; dashed is the dose-matched comparison group of the same op. The profile is the share of the pull each role would receive at the run's τ, read from the clean alignment, so a difference between the two says the answer's redness moved where the pull lands within the span. Hairlines are seeds.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(_ops), figsize=(1.55 * len(_ops), 1.9), sharey=True, layout="constrained")
        axes = cast(AxesRow, axes)
        hair = light_dark("#00000033", "#ffffff33")
        for ax, op in zip(axes, _ops, strict=True):
            pr, pm = e3_profiles[op]
            for s in pr:
                smooth_step(ax, _x, s, ramp=0.5, color=hair, lw=0.5)
            smooth_step(ax, _x, pr.mean(0), ramp=0.5, color=ink("recipe"), lw=1.5)
            smooth_step(ax, _x, pm.mean(0), ramp=0.5, color=ink("recipe"), lw=1.1, ls=(0, (2, 1.5)))
            ax.set(ylim=(-0.05, 1.05), xlim=(-0.4, ex.SPAN - 0.6), yticks=[0, 0.5, 1])
            ax.set_xticks(_x, ROLES)
            ax.tick_params(labelsize=6.5)
            ax.set_title(op, fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### E4 — a richer op set

    The cube probes of ex-2.1.12 (ridge, ℓ₂ = 10⁻², strict per-value holdout) on the un-anchored models at one, three, and six ops: held-out $R^2$ for op1, op2, and the RGB of the answer, per slice and position. This asks whether more rules give the model a better operand geometry at the same compute. The two matchings under [the conditions](#conditions) are read as a pair, with the control as the six-op point of both. In the `corpus` arms the fewer-op models see more lines per op, so a cube that improves with the op count there is a clean positive. In the `per-op` arms the fewer-op models see the same lines per op, repeated more often, so a cube that worsens with the op count there is a clean negative. A trend that holds in both arms is read as the op count; one that holds in only one is read as its confound, lines per op or repetition.
    """)
    return


@app.function(hide_code=True)
def draw_probe_grid(ax: plt.Axes, stack: np.ndarray, *, own: int | None, row_names: bool, scale: bool) -> None:
    """Ex-2.1.12's probe panel: one row per residual slice, the six positions across, one trace per RGB channel.

    *stack* is (seed, slice, position, channel) strict R² for one target. Each row draws the seed mean over the grey area of the channel mean, with the seed envelope as hairlines; rows share one Axes, each offset by a translate, so every row spans the same 0–1 scale. Negative scores clip to the floor. *own* is the position whose color the panel decodes, drawn bold.
    """
    sw = 0.7
    _, n_slices, n_pos, _ = stack.shape
    stack = np.clip(stack, 0, 1)
    for si in range(n_slices):
        shift = Affine2D().translate(0, si) + ax.transData
        draw_traces(
            ax, stack.mean(0)[si], spread=stack[:, si], ramp=1 - sw, faint_risers=True, transform=shift, clip_on=False
        )
        ax.axhline(si, color=light_dark("#aaaa", "#333a"), lw=0.5)
    ax.set_xlim(-0.5, n_pos - 0.5)
    ax.set_ylim(0, n_slices)
    ax.set_xticks(range(n_pos), POS_NAMES)
    ax.set_yticks(np.arange(n_slices) + 0.5, SLICE_NAMES if row_names else [""] * n_slices)
    ax.tick_params(axis="x", labelsize=6, bottom=False)
    ax.tick_params(axis="y", labelsize=7, left=False)
    if own is not None:
        ax.get_xticklabels()[own].set(fontweight="bold")
    if scale:
        sec = ax.secondary_yaxis("right")
        sec.set_yticks(np.arange(n_slices + 1), ["0", "1"] + [""] * (n_slices - 1))
        sec.tick_params(labelsize=6, direction="out")
        sec.set_ylabel("R², 0–1 per row", fontsize=7)


@app.cell(hide_code=True)
def _(res: Results):
    _targets = {"op1": 0, "op2": 2, "ans": 3}
    _sites = {"op1": "own slot", "op2": "own slot", "ans": "="}
    # The arms by matching, each ordered by op count, with the control as the six-op point of both.
    _by_name = {c.name: c for c in ex.RICHER_OP_ARM}
    e4_arms: dict[str, list[ex.Condition]] = {
        "corpus": sorted([c for c in _by_name.values() if c.name.endswith("-corpus")], key=lambda c: len(c.ops))
        + [ex.CONTROL],
        "per-op": sorted([c for c in _by_name.values() if c.name.endswith("-per-op")], key=lambda c: len(c.ops))
        + [ex.CONTROL],
    }

    def _r2(cond: str) -> np.ndarray:
        """(seeds, slices, positions, targets, channels) strict R² of one arm."""
        rs = sorted((r for r in res.geometry["runs"] if r["condition"] == cond), key=lambda r: r["seed"])
        return np.array([r["r2_strict"] for r in rs], float)

    e4_r2: dict[str, np.ndarray] = {c.name: _r2(c.name) for arm in e4_arms.values() for c in arm}
    _counts = sorted({len(c.ops) for arm in e4_arms.values() for c in arm})
    _head = ["target, site", "arm"] + [f"{n} op{'s' if n > 1 else ''}" for n in _counts]
    _rows = []
    for _ti, (_t, _p) in enumerate(_targets.items()):
        for _ai, (_arm, _conds) in enumerate(e4_arms.items()):
            _row = [f"{_t} at {_sites[_t]}" if _ai == 0 else "", _arm]
            _by_count = {len(c.ops): np.clip(e4_r2[c.name][:, 1:, _p, _ti, :], 0, 1).mean(axis=(1, 2)) for c in _conds}
            _row += [span2(_by_count[n], ".2f") if n in _by_count else "·" for n in _counts]
            _rows.append(_row)
    _caption = """
    Strict held-out R² of the un-anchored models by op count, at the site that carries each target: op1 and op2 at their own slot, the answer at <code>=</code>, where it is first decodable. Each value is the mean over the RGB channels and the four post-attention slices, negative scores clipped to zero first, as a seed mean with half the seed range. The <code>corpus</code> arm keeps D2.1's line count as the op set narrows (more lines per op at fewer ops); the <code>per-op</code> arm keeps the six-op lines per op (a smaller corpus, repeated). The control is the six-op point of both.
    """
    mo.Html(table_html(_head, _rows, _caption))
    return e4_arms, e4_r2


@app.cell(hide_code=True)
def _(e4_arms: dict[str, list[ex.Condition]], e4_r2: dict[str, np.ndarray]):
    _t = 2  # the answer's RGB

    @themed(
        name="e4-answer-probes",
        alt_text="""
            Two rows of three probe grids. Each grid has one row per residual slice with the embedding at the bottom and the six token positions across, with a trace per RGB channel over a grey area. The answer's color rises at the equals slot in the upper slices in every grid; the top row is the corpus-matched arm and the bottom the per-op-matched arm, at one, three, and six ops from left to right.
        """,
        caption="""
            **Where the answer's color is decodable, by op count.** Strict held-out R² for the RGB of the answer at every site, ex-2.1.12's layout: rows are residual slices with the embedding at the bottom, columns the six positions, one trace per channel over the grey area of their mean, with the seed envelope as hairlines. The bold slot is the answer itself. Top row, the <code>corpus</code> matching; bottom row, the <code>per-op</code> matching; the six-op panel is the control in both. Negative scores clip to the floor.
        """,
    )
    def _plot() -> plt.Figure:
        n = max(len(a) for a in e4_arms.values())
        fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 5.2), squeeze=False)
        axes = cast(AxesGrid, axes)
        for row, (arm, conds) in zip(axes, e4_arms.items(), strict=True):
            for i, (ax, c) in enumerate(zip(row, conds, strict=False)):
                draw_probe_grid(
                    ax, e4_r2[c.name][:, :, :, _t, :], own=ex.ANSWER_POS, row_names=i == 0, scale=i == len(conds) - 1
                )
                ax.set_title(f"{arm}: {len(c.ops)} op{'s' if len(c.ops) > 1 else ''} ({c.name})", fontsize=7.5, pad=6)
            for ax in row[len(conds) :]:
                ax.set_axis_off()
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### E5 — the noise floor, re-measured

    The per-run σ of every gated statistic on the new grammar, from the five seeds of the recipe and of the control, beside the ex-2.1.10 values the bands used. The H3 verdict and the selection rule are scored with the frozen ex-2.1.10 bands either way. Where the fresh σ is larger, we report which H3 comparisons the wider band would leave unresolved, as a robustness read.
    """)
    return


@app.cell(hide_code=True)
def _(res: Results):
    _mix = ex.PRIMARY_OP.name

    def _sd(cond: str, stat: str) -> float:
        v = res.em(cond, _mix) if stat == "holdout_em" else res.stat(cond, stat)
        return float(v.std(ddof=1))

    _head = [
        "statistic",
        "σ, ex-2.1.10 (frozen)",
        "σ, recipe (fresh)",
        "σ, control (fresh)",
        "band, frozen",
        "band, fresh recipe",
    ]
    _rows = []
    _fresh_band = {}
    for _stat, _sigma in ex.NOISE_RUN.items():
        _r, _c = _sd(ex.RECIPE.name, _stat), _sd(ex.CONTROL.name, _stat)
        _fresh_band[_stat] = ex.equiv_band(_stat, noise={_stat: _r})
        _rows.append(
            [
                _stat,
                f"{_sigma:.4f}",
                f"<b>{_r:.4f}</b>" if _r > _sigma else f"{_r:.4f}",
                f"{_c:.4f}",
                f"{ex.equiv_band(_stat):.3f}",
                f"{_fresh_band[_stat]:.3f}",
            ]
        )
    _short = float(res.stat(ex.RECIPE_SHORT.name, "m_line").mean())
    _reads = []
    for _p in ex.PROPOSALS:
        _d = float(res.stat(_p.name, "m_line").mean()) - _short
        _frozen, _fresh = abs(_d) > ex.equiv_band("m_line"), abs(_d) > _fresh_band["m_line"]
        _reads.append(
            f"`{_p.name}` {_d:+.3f}: {'resolved' if _frozen else 'unresolved'} at the frozen band, {'resolved' if _fresh else 'unresolved'} at the fresh one"
        )
    e5_md: str = "; ".join(_reads)
    _caption = """
    The per-run spread of each gated statistic: the frozen ex-2.1.10 value behind every band in this report, and the sample standard deviation over the five fresh seeds of the recipe and of the control. Bold marks a fresh recipe σ above the frozen one. The placement statistics on the control are near zero by construction, so its σ there is the instrument's noise rather than a run-to-run spread of a placement. The last two columns are the band for a five-against-five comparison under each σ.
    """
    mo.Html(table_html(_head, _rows, _caption))
    return (e5_md,)


@app.cell(hide_code=True)
def _(e5_md: str):
    mo.md(rf"""
    The H3 comparisons of each proposal's m_line against `recipe-short`, under both bands: {e5_md}.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### E6 — the two recipe lengths at twenty seeds

    Added after the results were read. The H3 read compared `recipe` and `recipe-short` at five seeds each, and the post hoc amendment above adopts one of them; five seeds is a thin basis for a choice the next experiments build on, so fifteen more seeds of each length were run (the `-more` arms under [the conditions](#conditions)). The frozen arms keep their five seeds and their verdicts; this section reads all twenty of each length together. The question is plain: at twenty seeds, does the full-length recipe still place *red* better than the short one, by more than the band, and does either length change its H4 read?
    """)
    return


@app.cell(hide_code=True)
def _(res: Results):
    _stats = ["m_line", "contrast", "r2_sim", "alpha_op1", "lead_emb", "holdout_em"]
    _pool = {(c, k): res.pooled(c, k) for c in (ex.RECIPE.name, ex.RECIPE_SHORT.name) for k in _stats}
    _n = len(_pool[ex.RECIPE.name, "m_line"])
    _head = [
        "statistic",
        "recipe, 5 seeds",
        f"recipe, {_n} seeds",
        f"recipe-short, {_n} seeds",
        "Δ (short − full)",
        "band, frozen σ",
        "band, fresh σ",
    ]
    _rows = []
    e6_reads: dict[str, tuple[float, float, float]] = {}
    for _k in _stats:
        _a, _b = _pool[ex.RECIPE.name, _k], _pool[ex.RECIPE_SHORT.name, _k]
        _d = float(_b.mean() - _a.mean())
        _sd = float(np.sqrt((_a.var(ddof=1) + _b.var(ddof=1)) / 2))
        _frozen = ex.equiv_band(_k, _n, _n) if _k in ex.NOISE_RUN else float("nan")
        _fresh = ex.equiv_band(_k, _n, _n, noise={_k: _sd})
        e6_reads[_k] = (_d, _frozen, _fresh)
        _five = res.em(ex.RECIPE.name, ex.PRIMARY_OP.name) if _k == "holdout_em" else res.stat(ex.RECIPE.name, _k)
        _rows.append(
            [
                _k,
                f"{_five.mean():.3f}",
                f"{_a.mean():.3f} <span class='range'>&nbsp;σ {_a.std(ddof=1):.3f}</span>",
                f"{_b.mean():.3f} <span class='range'>&nbsp;σ {_b.std(ddof=1):.3f}</span>",
                f"<b>{_d:+.3f}</b>" if abs(_d) > _fresh else f"{_d:+.3f}",
                "·" if np.isnan(_frozen) else f"{_frozen:.3f}",
                f"{_fresh:.3f}",
            ]
        )
    _caption = f"""
    The two recipe lengths at {_n} seeds each: the five frozen seeds and the fifteen addendum seeds pooled. Values are seed means; σ is the sample standard deviation over the {_n} seeds. Δ is short minus full. The frozen band is 2σ√(2/{_n}) with the ex-2.1.10 per-run σ; the fresh band uses the pooled σ of the two arms in this table. Bold marks a difference outside the fresh band. The lead weight has no frozen σ.
    """
    mo.Html(table_html(_head, _rows, _caption))
    return (e6_reads,)


@app.cell(hide_code=True)
def _(res: Results):
    _stats = ["m_line", "contrast", "r2_sim", "lead_emb"]
    _conds = (ex.RECIPE.name, ex.RECIPE_SHORT.name)

    @themed(
        name="e6-recipe-lengths",
        alt_text="""
            Four small panels, one per placement statistic, each with two rows, recipe and recipe-short. Every seed is a small dot along the row; the larger marker is the twenty-seed mean, and the five frozen seeds are drawn hollow.
        """,
        caption="""
            **The two recipe lengths, seed by seed.** One panel per placement statistic on the `mix` lines; the top row of each is `recipe`, the bottom `recipe-short`. Each small dot is one seed (hollow for the five frozen seeds, filled for the fifteen addendum seeds), jittered a little so they do not stack; the large marker is the twenty-seed mean, with a bar of one σ either side.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(_stats), figsize=(7.6, 1.9), layout="constrained")
        axes = cast(AxesRow, axes)
        rng = np.random.default_rng(0)
        for ax, k in zip(axes, _stats, strict=True):
            for y, c in zip((1.0, 0.0), _conds, strict=True):
                five, more = res.stat(c, k), res.stat(f"{c}-more", k)
                allv = np.concatenate([five, more])
                jit = rng.uniform(-0.18, 0.18, len(allv))
                ax.plot(five, y + jit[: len(five)], "o", ms=3.2, mfc="none", mec=ink(c), mew=0.9, zorder=3)
                ax.plot(more, y + jit[len(five) :], "o", ms=3.2, color=ink(c), alpha=0.75, zorder=3)
                m, sd = allv.mean(), allv.std(ddof=1)
                ax.plot([m - sd, m + sd], [y - 0.42, y - 0.42], color=ink(c), lw=1.4, solid_capstyle="round", zorder=2)
                ax.plot(m, y - 0.42, "D", ms=4, color=ink(c), zorder=4)
            ax.set_yticks([1.0, 0.0], _conds, fontsize=7)
            ax.set_ylim(-0.8, 1.5)
            ax.set_title(k, fontsize=9)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="x", alpha=0.2)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(res: Results):
    _mix = ex.PRIMARY_OP.name
    _head = ["condition"] + [f"{op} red acc ↓" for op in ex.OP_NAMES] + ["mix non-red deficit ↓"]
    _rows = []
    e6_h4: dict[str, tuple[float, float]] = {}
    for _c in (ex.RECIPE.name, ex.RECIPE_SHORT.name):
        _red = [res.pooled_score(_c, op, ex.PROJECTION.name, "acc") for op in ex.OP_NAMES]
        _def = res.pooled_deficit(_c, _mix, ex.PROJECTION.name)
        e6_h4[_c] = (float(max(v.mean() for v in _red)), float(_def.mean()))
        _rows.append(
            [f"{_c}, projection, {len(_def)} seeds"]
            + [f"<b>{span2(v, '.2f')}</b>" if v.mean() <= ex.RED_ACC_GATE else span2(v, ".2f") for v in _red]
            + [f"<b>{span2(_def)}</b>" if _def.mean() <= ex.NONRED_DEFICIT_GATE else span2(_def)]
        )
    _caption = f"""
    H4's two statistics under <code>projection</code> at twenty seeds per length: seed-mean exact-match accuracy on the red lines of each op (gate ≤ {ex.RED_ACC_GATE:g}, bold) and the non-red deficit on <code>mix</code> (gate ≤ {ex.NONRED_DEFICIT_GATE:g}, bold), each with half the seed range.
    """
    mo.Html(table_html(_head, _rows, _caption))
    return (e6_h4,)


@app.cell(hide_code=True)
def _(
    e6_h4: dict[str, tuple[float, float]],
    e6_reads: dict[str, tuple[float, float, float]],
):
    _d, _frozen, _fresh = e6_reads["m_line"]
    _lead = e6_reads["lead_emb"][0]
    _full, _short = e6_h4[ex.RECIPE.name], e6_h4[ex.RECIPE_SHORT.name]
    _verdict = (
        "the full-length recipe keeps its lead"
        if -_d > max(_frozen, _fresh)
        else "the two lengths are within a band of each other"
        if abs(_d) <= max(_frozen, _fresh)
        else "the short recipe leads"
    )
    mo.md(rf"""
    At twenty seeds, `recipe-short` sits {_d:+.3f} of m_line from `recipe`, against a band of {_frozen:.3f} (frozen σ) or {_fresh:.3f} (fresh σ): {_verdict}. The lead weight at the embedding moves {_lead:+.3f} between the lengths. Under `projection`, the worst-op red accuracy is {_full[0]:.2f} on the full recipe and {_short[0]:.2f} on the short one, and the `mix` non-red deficit is {_full[1]:.3f} and {_short[1]:.3f}. So the choice between the two lengths does not turn on H4. It turns on placement, and the two lengths split it: the full recipe has the larger margin, and the short one grades better and puts more of the pull on the drawing operand at the embedding. The post hoc rule ranks on the margin, as the frozen one did, and names the full recipe; the decision recorded beside it takes the short arm, on grading, the operand lead, and cost.
    """)
    return


@app.cell(hide_code=True)
def _(adopted: str):
    mo.md(rf"""
    ### E7 — the other arms under projection

    Added after the results were read. The H4 tables above read the recipe and `{adopted}`; this reads the same statistics on the remaining scored arms, `recipe-short` and the two proposals the adoption did not name, so the removal picture (complete on `mix`, partial on the saturating ops) can be checked across every operating point rather than two. The E2 syntax rows for those proposals sit at the end of E2.
    """)
    return


@app.cell(hide_code=True)
def _(adopted: str, res: Results):
    _mix = ex.PRIMARY_OP.name
    _conds = [ex.RECIPE_SHORT.name] + [c.name for c in ex.PROPOSALS if c.name != adopted]
    _head = ["condition"] + [f"{op} red acc ↓" for op in ex.OP_NAMES] + ["mix non-red deficit ↓", "|α| of = (deep)"]
    _rows = []
    for _c in _conds:
        _red = [res.score(_c, op, ex.PROJECTION.name, "acc", "red") for op in ex.OP_NAMES]
        _def = res.deficit(_c, _mix, ex.PROJECTION.name)
        _eq = np.abs(np.mean([alpha_at(res, _c, op, 3) for op in ex.OP_NAMES], axis=0)[:, 1:]).mean(axis=1)
        _rows.append(
            [f"{_c}, projection"]
            + [f"<b>{span2(v, '.2f')}</b>" if v.mean() <= ex.RED_ACC_GATE else span2(v, ".2f") for v in _red]
            + [f"<b>{span2(_def)}</b>" if _def.mean() <= ex.NONRED_DEFICIT_GATE else span2(_def), span2(_eq, ".2f")]
        )
    _caption = f"""
    H4's statistics under <code>projection</code> on the arms the H4 section does not table, five seeds each: red-line accuracy per op (gate ≤ {ex.RED_ACC_GATE:g}, bold), the <code>mix</code> non-red deficit (gate ≤ {ex.NONRED_DEFICIT_GATE:g}, bold), and, as the E2 correlate, the mean |α| of <code>=</code> over the four post-attention slices. Seed mean with half the seed range.
    """
    mo.Html(table_html(_head, _rows, _caption))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### E8 — which red lines survive projection

    Added after the results were read. H4's removal statistic asks whether the answer is still right once the axis is projected out; on ops that saturate, many red lines have an answer that does not depend on how red the red operand is, so a model that has lost *red* can still answer them. This sorts each op's red lines by whether the answer *depends* on the red operand's redness (lowering that operand's R by one grid step changes the snapped answer) and reads accuracy on the two groups separately, clean and under `projection`. Removal that reads as partial in H4 should read as complete on the dependent lines if the axis carries the redness the answer needs.
    """)
    return


@app.cell(hide_code=True)
def _(adopted: str, res: Results):
    from sca.config import TokenizerConfig
    from sca.data.colors import redness
    from sca.data.named_colors import WordTokenizer
    from sca.data.ops import NAMES, probe_lines, vocabulary

    _stoi = WordTokenizer(TokenizerConfig(vocabulary=vocabulary())).stoi  # the corpus tokenizer, rebuilt

    _step = ex.LEVELS[1] - ex.LEVELS[0]

    def _dependent(op: str) -> tuple[np.ndarray, np.ndarray]:
        """(red, dependent) masks over the op's probe lines, in the stored order."""
        red, dep = [], []
        for ln in probe_lines(ex.OP_BY_NAME[op], ex.N_PROBE, ex.PROBE_SEED):
            a, b = (ln.lhs, ln.rhs) if redness(ln.lhs) >= redness(ln.rhs) else (ln.rhs, ln.lhs)
            red.append(max(redness(ln.lhs), redness(ln.rhs)) >= ex.RED_DOSE - 1e-9)
            a2 = (max(a[0] - _step, 0), a[1], a[2])
            dep.append(ex.OP_BY_NAME[op](a2, b) != ln.result)
        return np.array(red), np.array(dep)

    def _answers(op: str) -> np.ndarray:
        """The answer token id of each probe line, in the stored order."""
        return np.array([_stoi[NAMES[ln.result]] for ln in probe_lines(ex.OP_BY_NAME[op], ex.N_PROBE, ex.PROBE_SEED)])

    def acc(g: np.ndarray, m: np.ndarray, ans: np.ndarray) -> np.ndarray:
        return (g[:, m] == ans[m]).mean(axis=1) if m.any() else np.full(len(g), np.nan)

    _head = [
        "op",
        "condition",
        "dependent red lines",
        "other red lines",
        "clean, dependent",
        "clean, other",
        "projection, dependent ↓",
        "projection, other",
    ]
    _rows = []
    for _op in ex.OP_NAMES:
        _red, _dep = _dependent(_op)
        for _c in (ex.RECIPE.name, adopted):
            _dose = res.arr(_c, "score", f"{_op}/dose")[0]
            assert np.array_equal(_dose >= ex.RED_DOSE - 1e-9, _red), (
                "the stored probe order and the rebuilt one disagree"
            )
            _clean = res.arr(_c, "score", f"{_op}/clean/guess")
            _proj = res.arr(_c, "score", f"{_op}/{ex.PROJECTION.name}/guess")
            _ans = _answers(_op)
            _all = np.ones(len(_ans), bool)
            _stored = res.score(_c, _op, None, "acc", "all")
            assert np.allclose(acc(_clean, _all, _ans), _stored, atol=1e-6), (
                "the rebuilt answers disagree with the stored accuracy"
            )
            _d, _o = _red & _dep, _red & ~_dep
            _rows.append(
                [
                    f"<code>{_op}</code>",
                    _c,
                    f"{int(_d.sum()):,}",
                    f"{int(_o.sum()):,}",
                    span2(acc(_clean, _d, _ans), ".2f"),
                    span2(acc(_clean, _o, _ans), ".2f") if _o.any() else "·",
                    span2(acc(_proj, _d, _ans), ".2f"),
                    span2(acc(_proj, _o, _ans), ".2f") if _o.any() else "·",
                ]
            )
    _caption = f"""
    The red lines (dose ≥ {ex.RED_DOSE:g}) of each op split by whether the answer depends on the red operand's redness: a line is <em>dependent</em> if lowering that operand's R by one grid level changes the snapped answer. Exact-match accuracy on each group, clean and under <code>projection</code>, seed mean with half the seed range, on the recipe and on the point the frozen rule adopted. On <code>mix</code> the other lines are those where a one-level drop rounds back to the same answer.
    """
    mo.Html(table_html(_head, _rows, _caption))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion

    Here is what we think happened, in plain terms.

    **The anchoring recipe does not care about the grammar.** We trained on six rules instead of one, with the labeller still only ever looking at the two operands, and *red* landed on the axis the same way it did in D2.1: no task cost on any op (H1), every placement gate met (H2), and the same numbers on the lines of every op (E1). The recipe needs its own stream of labelled operands each step, the corpus kept that, and the rest of the line can do whatever it likes. So the "recipe is grammar-specific" row of the [D2.2 risk table](../d2.2/design.md) closes.

    **The survey's proposals are real, and smaller than advertised.** All three came in about 0.05 of m_line below their survey values, the recipe about 0.02 below its own. That is the winner's curse we priced in: when you pick the best of many noisy trials you tend to pick one that got lucky, so a fresh run scores lower.[^curse] About two thirds of the advantage over `recipe-short` survived, the proposals grade better than the survey said (the other side of the same luck), and they spread more across seeds than the recipe does. E5 says the H3 differences hold up under the wider band too.

    [^curse]: The winner's curse: when you pick the top result out of many noisy trials, you tend to pick one whose noise ran in its favor, so a fresh run of it usually scores lower.

    **The selection rule picked `t00`, and we are not carrying it.** The rule ranked on margin and checked most of the H2 gates, and `t00` won. But it misses the one gate the rule left out, the lead weight at the embedding (0.33 against 0.4), and its contrast is 0.36 where the recipe's is 0.86. More to the point, it costs something the rule never looked at. At five times the recipe's anchor weight, the axis ends up on the `=` and op-word rows of the shared embedding table (E2: `=` at 0.93 against 0.40), and projecting the axis out of every position then breaks 0.6 of the non-red `mix` lines, with some seeds losing nearly all of them. The same is true of `t48` and `t12` (E7). Editing only the operand positions gets around it (0.017 on `mix`), and the prereg's contrary clause already sends the intervention-tuning pass to those rows. But an operating point that needs a routed edit before its first intervention is a weak base for the anchored-op experiments, so the [post hoc read](#post-hoc-the-selection-rule-with-the-gates-it-left-out) adds the two missing gates and adopts the recipe, and D2.2 builds on its short arm. Whether the syntax rows can be made clean rather than routed around is the [tied-readout question](/todo/science/syntax-rows-carry-the-axis-via-tied-readout.md), and it wants settling before the D2.3 swap, which rotates every position.

    **Full length or half?** We ran fifteen more seeds of each recipe length to answer this properly (E6). At twenty seeds the full recipe keeps a 0.012 margin lead, about twice the band, and the short one grades better (r² 0.89 against 0.82) and leads better at the embedding (0.83 against 0.79). Neither is wrong. The post hoc rule ranks on margin, so it names the full recipe; we chose the short arm anyway, and the choice is recorded in the post hoc section. The anchored-op experiments lean on grading and on the operand lead more than on margin, the H4 read is the same, and the short arm is half the compute.

    **Removal looked partial on four ops, and mostly is not.** H4's removal statistic is whether the model still gets the answer right once the axis is projected out. On the recipe that held on `mix` (0.06 red-line accuracy) and missed on four of the new ops (0.17 to 0.32). The reason is the ops, not the anchor. On `add`, `screen`, `multiply`, `lighten`, and `darken`, most red lines have an answer that would be the same if the red operand were a little less red, because the op saturates or picks the other operand's channel; a model that has lost *red* can still answer those. E8 splits each op's red lines by whether the answer depends on the red operand's redness at all, and on the dependent lines the projection removes *red* on every op (accuracy 0.00 to 0.18 on the recipe, 0.02 or less on `t00`). The misses in the H4 table are the lines where the correct answer never needed *red*. For the next grammar we want both kinds: some ops whose answers depend on both operands and spread through the cube, and some anisotropic ones like these, whose answers sit on a face or an edge, so the E8 split has both to read; that is the [diverse-op item](/todo/science/diverse-operator-set-hue-saturation-brightness.md), and it should be checked with the answer clouds and the E8 split before anything is anchored on it.

    **Selectivity on the recipe transfers.** The `mix` non-red deficit under projection is the 0.024 from ex-2.2.1 at five seeds and 0.026 at twenty, and the other ops lose nothing to speak of. The write stays inside the bound at most sites (the stacked figure under H4), and where it pokes above, it is at the op word, which is the one new token.

    **The blind span behaves.** On the lines whose answer is redder than either operand, the answer's own redness shows on the axis at `=` and at the answer position, by 0.15 at most, and the projection leaves most of those lines intact (E3). The anchor never sees the answer, so that redness is computed off the axis, as the ex-2.1.10 scope note said it would be. `multiply` has the most such lines and the largest shift, so it is where a labeller that also reads the answer would show up first; that labeller is filed under [the span variants](/todo/science/labeling-pull-span-variants-ex-2-1.md).

    **More rules did not make a straighter cube, on this probe.** At six ops the operand's RGB is less linearly decodable at its own slot than at three (0.52 against 0.85 to 0.87), in both matchings, though the six-op seeds spread widely (±0.14) and the one-op arm of the `per-op` matching is in a memorization regime, so the read is a trend over three points rather than a result. The answer at `=` gives no clean read either way. Whether the model builds op-specific readings of its operands is a probe question for the anchored-op experiments, which read those sites anyway (E4).

    **Which op to anchor next.** The [relevance table](#the-grammar) favors `mix` (94% of its lines have an answer no other op gives) and, among the new ops, `multiply`. E3 adds that `multiply` is where the answer's redness shows on the axis most, and E8 that its red lines are about half dependent. The suppression prereg can choose on what it wants to show: `mix` for the sharpest test, `multiply` for a rule the anchor has never seen.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    Two things change for the model against D2.1: the corpus carries six rules at the same total size, and a token between the operands decides the answer.[^const] The [dependency named in the design](../d2.2/design.md#deps) lands before the DAG, in `sca.data`, and the table below is its specification.

    [^const]: Whereas previously, the operation was constant (`+`), so the model had no reason to use that position for anything other than spare compute.
    """)
    return


@app.cell(hide_code=True)
def _():
    # REVIEW: review asked whether the snap is biased. `screen` and `multiply` never land halfway
    # between two levels, and their mean signed rounding error over the pairs is zero. `mix` did
    # round up on every off-grid channel sum (D2.1's round-half-up rule, then the snap), a
    # lightening bias on 87% of its pairs; it is now the plain mean, with ties sent to the even
    # level index, per the REVIEW note on `snap` in experiment.py. Stochastic rounding was
    # considered and not taken: it would put the exact-match ceiling below 1 on the rounded ops,
    # which the calibration and H1 read, and is filed as a grammar variant in
    # todo/science/stochastic-rounding-of-off-grid-answers.md. Verify: `snap`'s docstring, and the
    # on-grid column of the op table, unchanged for `mix` at 12.9%.
    mo.md(r"""
    ### The grammar

    Six ops on the `v216` grid (six levels per channel: 0, 3, 6, 9, 12, 15), with every color one token and every op one token. Each op is computed per channel on the 0..15 scale and snapped to the nearest grid level. So every op answers every pair with a color in the vocabulary, and a line can be written for any pair. Where the rule lands on the grid by itself, no rounding happens, and `mix` on those pairs is the D2.1 op.

    Across the pairs, the snap rounds up about as often as down. `screen` and `multiply` never land halfway between two levels, so their mean signed rounding error is zero. `mix` lands halfway on half of its channel sums, and those ties go to the even level (0, 6, or 12), which sends ten of the eighteen cases down and eight up.[^bias]

    [^bias]: If the snap always broke ties the same way, every rounded `mix` answer would shift in that direction, and the model would learn the shift as part of the rule. The `mix` of D2.1 rounded half up, which on this grid meant up by a whole level on every off-grid pair. D2.1 never trained on those pairs, so the change costs nothing there, and the probe set is the on-grid pairs, which are untouched.
    """)
    return


@app.cell(hide_code=True)
def _():
    # REVIEW: the op table was drafted under D2.1's closed-pair rule, which admits only
    # pairs whose answer is already a grid color; `screen` and `multiply` are degenerate
    # under it, and were dropped. The design's deps section asks for a "grid function with
    # defined rounding" instead, so every op is now computed on the 0..15 scale and snapped
    # to the nearest level, which makes all six total and restores the two. Verify: the
    # on-grid column below shows how much each op rounds, and the agreement table that
    # the six are distinct rules.
    # REVIEW: the sentence reading the agreement table named the wrong pairs — `add`
    # agrees with `screen` (37.6%) but with `lighten` on only 17.1%; the third pair at
    # about a third is `screen`-`lighten` (37.5%), as the design's deps section has it.
    # Verify: the rendered agreement table above the sentence.
    mo.md(rf"""
    Each op has {ex.line_counts(ex.MIX)["lines"]:,} lines, of which {ex.line_counts(ex.MIX)["red"]:,} are red and {ex.line_counts(ex.MIX)["nonred"]:,} non-red by dose, the larger of the two operand rednesses. Those counts are the same for every op, since dose reads the operands.

    {op_table_md()}

    The five named rules are the blend modes of the same names in Photoshop and Krita, on the 0..15 scale; `add` is what Photoshop calls *linear dodge*. `mix` is a normal blend at half opacity, which is the per-channel mean.

    All six ops are commutative, as `mix` was in D2.1, so operand order carries no information and the two operand roles stay interchangeable.

    The set leans light. `add`, `screen`, and `lighten` can only raise a channel; `multiply` and `darken` can only lower one; `mix` sits between its operands. A `subtract` op would balance `add`, but it would also be the first op whose operand order mattered, so the set stays as it is.

    **Agreement.** How often each pair of ops gives the same answer. On a line where two ops agree, reading the op word gains nothing:

    {agreement_md()}

    `screen` agrees with `add` on about a third of pairs and with `lighten` on about a third as well, at the light end of the range where all three saturate; `multiply` agrees with `darken` at the dark end. `add` and `lighten` agree on a sixth of pairs. Every other pair agrees on a few percent at most, at the ends of the range.

    **Op-relevance.** Take the lines of one op and ask of each line: how many of the other five ops would have given the same answer for this pair? Call that count k. Where k = 0, no other op matches, so the model has to read the op word to get the line right. Where k is larger, the op word narrows the six ops to k + 1, all of which give the same answer, so the model could answer the line without reading it.

    The table gives, for each op we might anchor, the share of its lines at each k. The suppression experiments rest on this: when the op concept is removed, only the lines at k = 0 can show it, so an op with most of its lines there makes a sharper test. Which op we anchor stays open until the suppression prereg:

    {relevance_md()}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The corpus
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    {ex.N_LINES:,} lines, the same count as D2.1. Ops are drawn uniformly, and within an op the training pairs are drawn uniformly with random operand order, the way ex-2.1.10 drew the `mix` pairs. For each op, a fifth of its distinct pairs is held out, keyed on (op, pair), so a pair held out under `add` may still be trained under `mix`. Corpus seed {ex.CORPUS_SEED}.

    D2.1 drew its {ex.N_LINES:,} lines from 5,832 distinct ones, so the corpus held about seventeen copies of each. Here the {ex.N_LINES:,} lines come from six times {ex.line_counts(ex.MIX)["lines"]:,} distinct ones, so most distinct lines appear at most once, and most pairs of an op never appear under that op. In E4, the `corpus` arms keep the {ex.N_LINES:,} lines and narrow the op set, while the `per-op` arms narrow the op set and keep {ex.CONTROL.lines_per_op:,} lines per op.

    The operands are the whole grid for every op, as they were in D2.1; what differs by op is where the answers land. `add` sends a fifth of its pairs to white, `screen` crowds the light half of the cube and `multiply` the dark half, and `mix`, `lighten`, and `darken` spread their answers through it.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Where each op's answers land: the number of unordered pairs whose answer is each grid color.
    # Computed outside the plot function so `themed` renders both themes from one pass.
    _grid = np.array(ex.colors(), dtype=float) / ex.TOP
    _index = {c: i for i, c in enumerate(ex.colors())}
    _pairs = ex.unordered_pairs()
    _counts = {}
    for _op in ex.OPS:
        _n = np.zeros(len(_grid))
        for _c, _k in Counter(_op(_a, _b) for _a, _b in _pairs).items():
            _n[_index[_c]] = _k
        _counts[_op.name] = _n
    _full_cell = 600
    """Pairs at which a mark fills its grid cell. Area is proportional to the count, so `add`'s white
    corner (a fifth of all pairs) overflows its cell, which is the point."""

    @themed(
        name="answer-clouds",
        alt_text="""
            Six hexagonal color-cube panels, one per op, with a mark on each grid color sized by how many pairs answer there. mix, lighten, and darken spread their answers through the whole cube; add sends a fifth of its pairs to one large white mark at the top, screen crowds the light half, and multiply the dark half.
        """,
        caption=f"""
            **Where each op's answers land.** One mark per grid color, with area proportional to the number of unordered pairs whose answer is that color, on one scale for all six panels: a mark that fills its grid cell stands for {_full_cell} pairs, out of {len(_pairs):,}.
        """,
    )
    def _plot() -> plt.Figure:
        from matplotlib.collections import EllipseCollection

        fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.8))
        # Drawn by hand rather than through `plot_rgb_cube`, whose sized marks carry no edge: the
        # white corner of `add` and the black corner of `multiply` are the largest marks on the
        # page, and each vanishes against the cube's fill in one theme without a faint outline.
        order = np.argsort(_grid @ CUBE_VIEWS["solid"].toward, kind="stable")  # nearer draws last
        xy = project_cube(_grid[order])
        for ax, op in zip(axes.flat, ex.OPS, strict=True):
            draw_cube_bound(ax)
            dia = grid_diameter(len(ex.LEVELS)) * np.sqrt(_counts[op.name][order] / _full_cell)
            ax.add_collection(
                EllipseCollection(
                    widths=dia,
                    heights=dia,
                    angles=0,
                    units="xy",
                    offsets=xy,
                    offset_transform=ax.transData,
                    facecolors=_grid[order],
                    edgecolors=light_dark("#00000033", "#ffffff55"),
                    linewidths=0.5,
                    zorder=3,
                    clip_on=False,
                )
            )
            # Lifted clear of the overflowing corner marks, which are unclipped.
            ax.set_title(op.name, y=1.12)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Training
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    D64-L4 nGPT with the ex-2.1.3 data config. A step is a batch of {ex.BATCH} random crops of {ex.BLOCK} tokens (about ten lines each) from the tokenized corpus. An epoch is enough batches to sample a quarter of the training tokens, which the loader sets with an `oversample` of {ex.OVERSAMPLE} over the batch size. At the corpus size of D2.1 that comes to {ex.steps_per_epoch()} steps per epoch, as it did there.

    Peak LR is {ex.PEAK_LR:g}. Anchoring uses the D2.1 code path: the pooled either-operand labeller (each operand draws at redness⁸ × {ex.PER_SLOT_RATE:g}), the anchor on e₁ at every slice, and the anti-subspace term.

    Conditions are matched on compute, at the step count of D2.1, rather than on samples per op. So the per-step regime of the anchor matches D2.1, and how often the task sees each pair becomes a covariate that H1 reads.

    The recipe runs for the {ex.EPOCHS} epochs of ex-2.1.10; the proposals and its short arm run for the {ex.EPOCHS_SHORT} of the survey. The `per-op` arms of E4 keep the step count and shrink the corpus, so they get more and shorter epochs; the conditions table prints them.

    Every schedule keyframe is a fraction of training, as ex-2.1.11 restated them: LR and anchor warm-up over the first {ex.WARMUP_FRAC:.0%}, the anchor anneal for the recipe over the last {1 - ex.ANNEAL_START_FRAC:.0%} down to a {ex.ANNEAL_FLOOR:g} floor, and the anti-subspace weight annealing from its peak ratio to {ex.ANTI_HOLD_RATIO:g} of the anchor weight by its own end fraction. The parameters for the proposals are the sampled values from the survey, unrounded, in `experiment.py`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The probe set
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    The probe set is a fixed list of equations that every run is scored on, separately from training: {ex.N_PROBE} lines per color as op1, per op. For `mix` these are its on-grid partners, so the `mix` probe set is the D2.1 one with the op word changed. For the other ops we draw {ex.N_PROBE} partners per color once with seed {ex.PROBE_SEED}, and share them across every run.

    Each op then has {len(ex.mix_probe_lines()):,} probe lines, red and non-red by dose. We keep the infix frame,[^infix] so the probe positions in `sca.compute.evaluation` still read correctly.

    [^infix]: An infix operator sits between its operands, as in `c1 mix c2`, rather than before or after them. The frame is the fixed six-token layout of a line: op1, the op word, op2, `=`, the answer, and the newline sit at the same positions in every line, so the probe code can read a role by its position.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Measurements
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    The placement statistics are the ones from ex-2.1.10, computed by its code on the probe lines of each op from one clean pass per run: m_line and the softmin profiles, containment, the lead weight, contrast, grading against the sim^1.5 target, and the latch. The trajectory records m_line every {ex.TRAJ_STRIDE} steps. Holdout exact match is read on a held-out set per op, of the size that ex-2.1.10 used.

    The interventions run one teacher-forced pass per run per operator over the six probe sets, through the eval contract, giving the log-softmax at `=` and the write per (slice, line, position) as in ex-2.2.1. Red-line accuracy and the non-red deficit are read per op.

    **Answer distance.** Exact match scores a one-step miss and a far miss the same way. Under suppression, ex-2.2.1 found that most misses on red lines were one-step neighbors of the true answer. So beside each accuracy we read the distance in the unit cube from the decoded answer to the true one, as ex-2.2.1 read it on the red lines.

    A second version takes the expected value of that distance under the answer distribution, using the whole log-softmax rather than only its argmax. Both are computed per line and averaged per group, for every operator row, and neither is gated.

    **Noise floor.** The bands use the per-run σ of each statistic at the reference recipe, from the nine seeds of ex-2.1.10, printed in the H3 table; E5 re-measures them here. Gates score seed means against fixed thresholds and do not use the floor. Every difference between two conditions is quoted with its band, and a difference inside the band is reported as not resolved.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Budget
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    {ex.N_RUNS} training runs: {ex.CONTROL.seeds + ex.RECIPE.seeds} at the full length of the recipe, {sum(c.seeds for c in ex.PROPOSALS) + ex.RECIPE_SHORT.seeds + ex.CONTROL_SHORT.seeds} at half of it, and {sum(c.seeds for c in ex.RICHER_OP_ARM)} for the richer-op arms at the full length as well, each at a plain D2.1 step count on an L4. Scoring is one clean pass plus four operator passes over six probe sets per run, taking CPU seconds each, and the cube probes of E4 are ridge fits on 216 rows. That is well under the cost of ex-2.2.2, which trained 24 runs at twice the step cost. The run took 41 minutes of wall-clock on Modal at twelve containers, 15 of them training, and cost $2.33, with no task failed or retried.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ### Before the freeze

    Before the freeze we run a calibration: one seed of each control arm on the six-op corpus. It checks that the grammar is learned at both lengths, which means holdout exact match per op near 1, the level the D2.1 control reached. The rounded ops are the ones to watch. If the grammar is not learned at one length or the other, we change the corpus size, the step count, or the set of arms before the freeze, and record the change with a `REVIEW` note. Nothing anchored runs before the freeze.

    **The calibration ran.** One seed of each control arm, published under `{ex.CALIBRATION_REF}`; the table below reads it. Both lengths learn all six ops: holdout exact match is 1.0 on every op but one at each length, and that one misses a single line of its {ex.N_EVAL}. The rounded ops are as clean as the others, and held-out surprisal is below 0.02 nats everywhere. So the corpus, the step counts, and the set of arms stay as designed, and `control-short` stays too, since its numbers say it will read the proposals' task cost at their own length without a training deficit of its own. Nothing about the arms or the gates changed after this read.

    {calibration_md()}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    /// details | Glossary
    - **line** — one equation, `c1 ‹op› c2 = answer`, six word-level tokens. The op word is one token, like each color. Bare *op* is the operation; *op1* and *op2* are the operand roles.
    - **on-grid pair** — a pair whose answer under an op lands on the grid without rounding. The on-grid pairs of `mix` are the closed pairs of D2.1, and its probe lines.
    - **residual stream** — the running vector each token carries through the network, which every block reads from and writes to.
    - **slice** ($\ell$) — a depth at which the residual stream is read: the embedding (0), plus the stream after each of the four blocks.
    - **alignment** ($\alpha$) — $\cos(h, e_1)$, the cosine between a state and the anchor axis. States are unit-norm, so this is the e₁ component.
    - **dose** — how *red* a line is, the larger of the two operand rednesses ($r(1 - g/2 - b/2)$ on the unit cube). **Red lines** have dose ≥ {ex.RED_DOSE:g}; **non-red lines** dose ≤ {ex.NONRED_DOSE:g}. The labeller draws at dose⁸, so a dark red such as (12,0,0) draws at about a sixth of pure red's rate and (9,0,0) at under 2%: dark reds are barely labelled, and on the grid the red lines are the seven operands with R = 15 or (12,0,0).
    - **softmin weight** ($\pi$) — the share of the pooled pull a span position receives; sums to 1 over the span. A profile is these shares over roles (op1, the op word, op2, …) at one slice.
    - **m_line** — the margin from ex-2.1.10: at the best span role, the label-weighted mean alignment minus the unweighted mean, averaged over slices. The one statistic tight enough to rank operating points on.
    - **containment** ($\bar\alpha$) — the mean alignment over all 216 colors at op1. A pull that latches onto op1 as a position rather than a concept drives it up.
    - **retention** — the final m_line as a fraction of its running peak over training.
    - **grading** ($r^2$) — the squared Pearson correlation between op1 alignment and the sim^1.5 target across the 216 colors. It says whether alignment rises in proportion to redness rather than switching at a threshold.
    - **contrast** — the difference in op2 softmin weight between the op2-triggered and op1-triggered line groups, averaged over the four post-attention slices. It says whether the pull found the operand that carried the label.
    - **latch** — a run whose non-red group puts more than {ex.LATCH_PI:g} of its softmin weight on op1; vetoed per run.
    - **deficit** — clean exact-match accuracy minus intervened accuracy over a group of lines. This is the statistic ex-2.2.1 gated selectivity on.
    - **band** — the smallest seed-mean difference the resolution rule may call a difference: $2\sigma\sqrt{{1/n_a + 1/n_b}}$, with σ the per-run spread.
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
