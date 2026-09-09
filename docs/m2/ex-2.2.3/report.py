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
    from pathlib import Path

    import marimo as mo

    # The design constants and the result refs come from `experiment.py` beside
    # this notebook (Marimo puts the notebook directory on sys.path). The prose
    # quotes the frozen gates, and that module carries the same numbers with each
    # gate's wording in its docstring.
    import experiment as ex
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store

    use_publisher(report_bundle(__file__))

    SURVEY_STATS = ("m_line", "r2_sim", "contrast", "alpha_op1", "holdout_em", "retention")
    """The statistics the survey scored its trials on, in the order the H3 table prints them."""

    # A cell renders its last expression, and a trailing docstring is one.
    None


@app.function(hide_code=True)
def load_survey() -> dict | None:
    """The ex-2.1.11 survey's stored results, or None if unpublished."""
    store = project_store()
    art = store.get_refs([ex.SURVEY_REF])[ex.SURVEY_REF]
    if art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "survey.json")])
        return json.loads(path.read_text())


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
        rows.append(f"| **{c.name}** | {c.title} | {anchor} | {ops} | {c.epochs} ({c.steps:,}) | {c.seeds} |")
    head = "| condition | what | anchor | ops | epochs (steps) | seeds |\n| --- | --- | --- | --- | ---: | ---: |\n"
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
    mo.md(rf"""
    ## Conditions

    Every condition is trained here, on the six-op corpus, at fresh seeds. The corpus size and the epoch count are the ones from D2.1, so the recipe's step count is unchanged.

    {conditions_md()}

    The recipe is the D2.1 primary, `either-t100`. The only thing we changed is the grammar. The short arm of the recipe runs for as long as the proposals do, so H3 compares runs of equal length; the control has a short arm so that H1 can do the same.

    The three proposals are the operating point from the survey plus the two promoted trials the survey read as its twin and its knee. `t48` sits within one m_line band of `t00`, with more grading and contrast. `t12` holds the most grading of the promoted set, at a margin the survey could not tell apart from the one for `t00`.

    The two richer-op arms are un-anchored models on one and three ops, at the same line count and step count, and so with more lines per op. They serve E4 only.

    ### The interventions

    We score the checkpoints of every candidate through the eval contract in [`sca.intervention`](/src/sca/intervention.py), on the probe lines of each op.

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

    /// admonition | TODO
    A table of holdout exact-match accuracy per op for every condition, seed mean with the seed range, with the two control rows first and, on each other row, the gap from the control of the same length. Beside it, the validation loss over training for the control and the recipe, five thin lines each, with the ex-2.1.10 primary drawn behind them on the step axis.
    <!-- Would be nice to have a figure here too (easier to read) -->
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
    **H3.** The plateau found in the survey belongs to the recipe rather than to the one-op grammar. At least one proposal is feasible at fresh seeds on the new grammar, and its seed-mean m_line on the `mix` probe lines exceeds `recipe-short` (the recipe at the same step count) by more than one band. Feasibility is read as the survey read it, with the constraints listed once in the selection rule below. The grading margin is one of them.

    Partial: a proposal is feasible and its m_line sits within a band of `recipe-short`; or a proposal clears every survey constraint and exceeds `recipe-short` by more than a band, but misses the grading margin. Contrary: no proposal is feasible, or every feasible proposal sits more than a band below `recipe-short`. Either would say the plateau was specific to the one-op grammar, and the recipe would carry D2.2.

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
    Red-line accuracy and non-red deficit per op under each operator (`projection` gated, the ride-along rows beside it), for the recipe and the adopted point, seed mean with the seed range, with the one-op figures from ex-2.2.1 in a reference column. Then the write-bound map from ex-2.2.1 for the recipe under `projection`, on the shared 0–1 scale, one panel per op.
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

    **E4 — a richer op set.** The cube probes of ex-2.1.12 (ridge, ℓ₂ = 10⁻², strict per-value holdout) on the un-anchored models at one, three, and six ops: held-out $R^2$ for op1, op2, and the RGB of the answer, per slice and position. This asks whether more rules give the model a better operand geometry at the same data and compute. Read it together with the lines-per-op confound noted under [the conditions](#conditions).

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
    ## Method

    Two things change for the model against D2.1: the corpus carries six rules at the same total size, and a token between the operands decides the answer. The [dependency named in the design](../d2.2/design.md#deps) lands before the DAG, in `sca.data`, and the table below is its specification.

    ### The grammar

    Six ops on the `v216` grid (six levels per channel: 0, 3, 6, 9, 12, 15), with every color one token and every op one token. Each op is computed per channel on the 0..15 scale and snapped to the nearest grid level. So every op answers every pair with a color in the vocabulary, and a line can be written for any pair. Where the rule lands on the grid by itself, no rounding happens, and `mix` on those pairs is the D2.1 op.

    Each op has {ex.line_counts(ex.MIX)["lines"]:,} lines, of which {ex.line_counts(ex.MIX)["red"]:,} are red and {ex.line_counts(ex.MIX)["nonred"]:,} non-red by dose. Those counts are the same for every op, since dose reads the operands.

    {op_table_md()}

    **Agreement.** For each pair of ops, how often they give the same answer. Where two ops agree, reading the op is worth nothing on that line:

    {agreement_md()}

    `screen` agrees with `add` on about a third of pairs, and with `lighten` on about a third as well, at the light end of the range where all three saturate. `multiply` agrees with `darken` at the dark end. `add` and `lighten` agree on a sixth of pairs, and `mix` and `lighten` on a tenth. Every other pair of ops agrees on only a handful of pairs, at the ends of the range.

    **Op-relevance** per candidate anchored op: the share of its lines on which 0, 1, 2, or 3 other ops give the same answer. At 0 the answer names the op. At k the op word only rules out 5 − k of the six. The choice of anchored op stays open until the suppression prereg, and these distributions are the input to it:

    {relevance_md()}

    ### The corpus

    {ex.N_LINES:,} lines, the same count as D2.1. Ops are drawn uniformly, and within an op the training pairs are drawn uniformly with random operand order, the way ex-2.1.10 drew the `mix` pairs. For each op, a fifth of its distinct pairs is held out, keyed on (op, pair), so a pair held out under `add` may still be trained under `mix`. Corpus seed {ex.CORPUS_SEED}.

    D2.1 drew its lines from 5,832 distinct ones, so each was seen about seventeen times per epoch. Here they are drawn from six times {ex.line_counts(ex.MIX)["lines"]:,}, so most lines are seen once per epoch and most pairs of an op are never seen under it. The richer-op arms keep the {ex.N_LINES:,} lines and narrow the op set.

    ### Training

    D64-L4 nGPT, the ex-2.1.3 data config (64 × 64 batches, oversample 16), peak LR {ex.PEAK_LR:g}, and the D2.1 anchoring code path: the pooled either-operand labeller (each operand draws at redness⁸ × {ex.PER_SLOT_RATE:g}), the anchor on e₁ at every slice, and the anti-subspace term. The corpus is the same size as in D2.1, so an epoch is {ex.steps_per_epoch()} steps as it was there. The recipe runs for the {ex.EPOCHS} epochs of ex-2.1.10; the proposals and the short arm of the recipe run for the {ex.EPOCHS_SHORT} of the survey.

    Every schedule keyframe is a fraction of training, as ex-2.1.11 restated them: LR and anchor warm-up over the first {ex.WARMUP_FRAC:.0%}, the anchor anneal for the recipe over the last {1 - ex.ANNEAL_START_FRAC:.0%} down to a {ex.ANNEAL_FLOOR:g} floor, and the anti-subspace weight annealing from its peak ratio to {ex.ANTI_HOLD_RATIO:g} of the anchor weight by its own end fraction. The parameters for the proposals are the sampled values from the survey, unrounded, in `experiment.py`.

    ### The probe set

    {ex.N_PROBE} lines per color as op1, per op. For `mix` these are its on-grid partners, so its probe set is the D2.1 one with the op word changed. For the other ops we draw {ex.N_PROBE} partners per color once with seed {ex.PROBE_SEED} and share them across every run. Each op then has {len(ex.mix_probe_lines()):,} probe lines, red and non-red by dose. We keep the infix frame so the probe positions in `sca.compute.evaluation` still read correctly.

    ### Measurements

    The placement statistics are the ones from ex-2.1.10, computed by its code on the probe lines of each op from one clean pass per run: m_line and the softmin profiles, containment, the lead weight, contrast, grading against the sim^1.5 target, and the latch. The trajectory records m_line every {ex.TRAJ_STRIDE} steps. Holdout exact match is read on a held-out set per op, of the size that ex-2.1.10 used.

    The interventions run one teacher-forced pass per run per operator over the six probe sets, through the eval contract, giving the log-softmax at `=` and the write per (slice, line, position) as in ex-2.2.1. Red-line accuracy and the non-red deficit are read per op.

    **Noise floor.** The bands use the per-run σ of each statistic at the reference recipe, from the nine seeds of ex-2.1.10, printed in the H3 table; E5 re-measures them here. Gates score seed means against fixed thresholds and do not use the floor. Every difference between two conditions is quoted with its band, and a difference inside the band is reported as not resolved.

    ### Budget

    {ex.N_RUNS} training runs: {ex.CONTROL.seeds + ex.RECIPE.seeds} at the full length of the recipe, {sum(c.seeds for c in ex.PROPOSALS) + ex.RECIPE_SHORT.seeds + ex.CONTROL_SHORT.seeds} at half of it, and {sum(c.seeds for c in ex.RICHER_OP_ARM)} for the richer-op arm at the full length as well, each at a plain D2.1 step count on an L4. Scoring is one clean pass plus four operator passes over six probe sets per run, taking CPU seconds each, and the cube probes of E4 are ridge fits on 216 rows. That is well under the cost of ex-2.2.2, which trained 24 runs at twice the step cost.

    ### Before the freeze

    Before the freeze we run a calibration: one seed of each control arm on the six-op corpus. It checks that the grammar is learned at both lengths, which means holdout exact match per op near 1, the level the D2.1 control reached. The rounded ops are the ones to watch. If the grammar is not learned at one length or the other, we change the corpus size, the step count, or the set of arms before the freeze, and record the change with a `REVIEW` note. Nothing anchored runs before the freeze.
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
