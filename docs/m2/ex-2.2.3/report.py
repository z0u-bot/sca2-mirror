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
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # The design constants and the result refs come from `experiment.py` beside
    # this notebook (Marimo puts the notebook directory on sys.path). The prose
    # quotes the frozen gates, and that module carries the same numbers with each
    # gate's wording in its docstring.
    import experiment as ex
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import figure_html, light_dark, themed
    from sca.vis import CUBE_VIEWS, draw_cube_bound, grid_diameter, project_cube

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
        "ex-2.1.10": ("#00000033", "#ffffff3a"),
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
def candidates_md(survey: dict | None) -> str:
    """Each candidate's one-op survey numbers beside blank columns for the fresh ones, with the noise floor."""
    head = (
        "| candidate | statistic | survey | seeds | fresh (5 seeds) | σ per run | band |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
    )
    rows = []
    for c in ex.CANDIDATES:
        if c.survey_trial is None:
            seeds, scored = ex.SURVEY_RECIPE[c.name]
        else:
            seeds, scored = 5, (survey["scored"].get(str(c.survey_trial)) if survey else None)
        for i, stat in enumerate(SURVEY_STATS):
            label = f"**{c.name}**" if i == 0 else ""
            value = f"{scored[stat]:.3f}" if scored and stat in scored else "—"
            sigma = f"{ex.NOISE_RUN[stat]:.4f}" if stat in ex.NOISE_RUN else "—"
            band = f"{ex.equiv_band(stat, 5, seeds):.3f}" if stat in ex.NOISE_RUN else "—"
            rows.append(f"| {label} | {stat} | {value} | {seeds} | — | {sigma} | {band} |")
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.2.3: the multi-op grammar, with *red* anchored again

    /// tip |
    <!-- tl;dr -->
    The grammar grows from one operation to six: `mix` (`+`), `add`, `screen`, `multiply`, `lighten`, `darken`. We retrain on the new grammar to check the recipes: the un-anchored control, the ex-2.1.10 recipe, and three proposals from the ex-2.1.11 survey. From that, we pick the operating point for D2.2.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Findings

    - [Task cost on the new grammar (H1)](#task-cost-on-the-new-grammar-h1) —
    - [Placement under the recipe reproduces (H2)](#placement-under-the-recipe-reproduces-h2) —
    - [The plateau from the survey transfers (H3)](#the-plateau-from-the-survey-transfers-h3) —
    - [Suppression transfers (H4)](#suppression-transfers-h4) —
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this draft
    This is a preregistration: no run of this experiment has happened. Each hypothesis section opens with its frozen prediction, and the `TODO` under it says what its figure or table will show. Once this skeleton is agreed in review, the predictions freeze and we quote the commit here. Results then replace the placeholders in place. Anything we think of after seeing the data goes under [Exploratory analyses](#exploratory-analyses), marked as post hoc. The op table and every count in the method are computed from `experiment.py` at render time.
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

    /// admonition | TODO
    A table of holdout exact-match accuracy per op for every condition, giving the seed mean and the seed range. The two control rows come first; every other row also gives the gap from the control of the same length.

    Under it, the same numbers as a figure: one panel per op, conditions down the side, each a dot at the seed mean with a bar for the seed range. A shaded strip marks the gate around the control of the same length, so a dot outside the strip is a miss.

    Then the validation loss over training for the control and the recipe, five thin lines each, with the ex-2.1.10 primary drawn behind them on the step axis.
    ///
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

    /// admonition | TODO
    A table of the seven statistics for the recipe on the `mix` lines, seed mean with the seed range, beside the nine-seed values of the ex-2.1.10 primary and the band between them. Then the m_line trajectory over training, five thin lines drawn over the nine from ex-2.1.10 on the step axis, and the softmin profile over roles per slice for the red and non-red groups, in the ex-2.1.10 layout.
    ///
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

    {candidates_md(load_survey())}

    /// admonition | TODO
    The table above filled in, with the feasibility checks for each candidate as a row of ticks and the adopted point marked. Then m_line against grading $r^2$ for the five candidates, one point per seed with the seed mean ringed, the survey point for each drawn hollow beside it, and the band for `recipe-short` drawn as a horizontal strip.
    ///
    """)
    return


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

    /// admonition | TODO
    Red-line accuracy and non-red deficit per op under each operator (`projection` gated, the ride-along rows beside it), for the recipe and the adopted point, seed mean with the seed range, with the one-op figures from ex-2.2.1 in a reference column, and the answer distance beside each accuracy. Then the write-bound map from ex-2.2.1 for the recipe under `projection`, on the shared 0–1 scale, one panel per op.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    /// admonition | TODO
    Preregistered as exploratory, no gates.

    **E1 — per-op statistics.** Every H2 statistic read on the probe lines of each op, rather than on the `mix` lines alone, for every candidate. The labeller never sees the op, so a placement that differs by op would mean the blocks carry *red* differently under different rules. The per-op grading and contrast say whether the anchor is op-blind.

    **E2 — the op words.** The alignment of the six op-word embeddings and of `=` and the newline, per slice, on the recipe and the adopted point, beside the constant component ex-2.2.1 found on `+` and `=`. We also ask whether the alignment of an op word predicts the non-red deficit on its lines under `projection`.

    **E3 — redder than both.** Every op but `lighten` has lines whose answer is redder than either operand (the [op table](#the-grammar) counts them). On those lines the strongest evidence for *red* sits at the answer, and the labeller never keys on that position. This is the blind-span case named in the scope note of ex-2.1.10. We read the alignment at `=` and at the answer position, against lines of the same op and dose whose answer is not redder than both, plus the softmin profile on each group. Then we restrict the H4 statistics to these lines, asking whether removing *red* from the operands also removes it from an answer that was redder than they were.

    <!-- REVIEW: a labeller that reads the answer as well as the operands (so `magenta darken yellow = red` draws a *red* label from its answer) is the M3-shaped labelling, and is filed as a fast-follow under (c) of todo/science/labeling-pull-span-variants-ex-2-1.md rather than changed here. E3 reads the blind span under the operand-only labeller first, which is what makes the answer position blind. Verify: that item's 2026-09-09 note. -->

    **E4 — a richer op set.** The cube probes of ex-2.1.12 (ridge, ℓ₂ = 10⁻², strict per-value holdout) on the un-anchored models at one, three, and six ops: held-out $R^2$ for op1, op2, and the RGB of the answer, per slice and position. This asks whether more rules give the model a better operand geometry at the same compute. The two matchings under [the conditions](#conditions) are read as a pair, with the control as the six-op point of both. In the `corpus` arms the fewer-op models see more lines per op, so a cube that improves with the op count there is a clean positive. In the `per-op` arms the fewer-op models see the same lines per op, repeated more often, so a cube that worsens with the op count there is a clean negative. A trend that holds in both arms is read as the op count; one that holds in only one is read as its confound, lines per op or repetition.

    **E5 — the noise floor, re-measured.** The per-run σ of every gated statistic on the new grammar, from the five seeds of the recipe and of the control, beside the ex-2.1.10 values the bands used. The H3 verdict and the selection rule are scored with the frozen ex-2.1.10 bands either way. Where the fresh σ is larger, we report which H3 comparisons the wider band would leave unresolved, as a robustness read.
    ///

    ## Discussion

    /// admonition | TODO
    Interpretation only, after the results. Whether the D2.1 recipe belongs to the anchor or to the grammar it was tuned on, and what the gaps between the survey and the fresh runs say about how much of the plateau was luck. What the adopted point is, and what it costs against the recipe. Whether the six op words repeat the syntax-row cost of ex-2.2.1, and what that means for the intervention the anchored-op experiments should use. What the redder-than-both lines say about the blind span. Which op the relevance distributions favor anchoring. No re-derivation of the findings.
    ///
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
            Where each op's answers land. One mark per grid color, with area proportional to the number of unordered pairs whose answer is that color, on one scale for all six panels: a mark that fills its grid cell stands for {_full_cell} pairs, out of {len(_pairs):,}.
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
    {ex.N_RUNS} training runs: {ex.CONTROL.seeds + ex.RECIPE.seeds} at the full length of the recipe, {sum(c.seeds for c in ex.PROPOSALS) + ex.RECIPE_SHORT.seeds + ex.CONTROL_SHORT.seeds} at half of it, and {sum(c.seeds for c in ex.RICHER_OP_ARM)} for the richer-op arms at the full length as well, each at a plain D2.1 step count on an L4. Scoring is one clean pass plus four operator passes over six probe sets per run, taking CPU seconds each, and the cube probes of E4 are ridge fits on 216 rows. That is well under the cost of ex-2.2.2, which trained 24 runs at twice the step cost.
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
    - **dose** — how *red* a line is, the larger of the two operand rednesses ($r(1 - g/2 - b/2)$ on the unit cube). **Red lines** have dose ≥ {ex.RED_DOSE:g}; **non-red lines** dose ≤ {ex.NONRED_DOSE:g}.
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
