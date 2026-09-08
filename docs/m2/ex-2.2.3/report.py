import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="medium",
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
    """The op table with its closure and line counts, computed from `experiment.py`."""
    rows = []
    for op in ex.OPS:
        n = ex.line_counts(op)
        rows.append(
            f"| `{op.name}` | {op.rule} | {ex.closure(op):.1%} | {n['lines']:,} | {n['red']:,} | {n['nonred']:,} "
            f"| {n['redder']:,} ({n['redder'] / n['lines']:.1%}) |"
        )
    head = (
        "| op | per channel | closed pairs | lines | red | non-red | redder than both |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
    )
    return head + "\n".join(rows)


@app.function(hide_code=True)
def agreement_md() -> str:
    """Pairwise agreement: the share of pairs closed under both ops on which they give the same answer."""
    names = [op.name for op in ex.OPS]
    cells = {}
    for p, q in itertools.combinations(ex.OPS, 2):
        same, both = ex.agreement(p, q)
        cells[(p.name, q.name)] = cells[(q.name, p.name)] = f"{same:,} / {both:,} ({same / both:.1%})"
    head = "| | " + " | ".join(f"`{n}`" for n in names) + " |\n|" + " --- |" * (len(names) + 1) + "\n"
    rows = ["| `" + p + "` | " + " | ".join(cells.get((p, q), "—") for q in names) + " |" for p in names]
    return head + "\n".join(rows)


@app.function(hide_code=True)
def relevance_md() -> str:
    """The op-relevance distribution per candidate anchored op."""
    levels = sorted({v for op in ex.OPS for v in ex.relevance(op)})
    head = "| anchored op | " + " | ".join(f"{v:.2f}" for v in levels) + " |\n|" + " ---: |" * (len(levels) + 1) + "\n"
    rows = []
    for op in ex.OPS:
        r = ex.relevance(op)
        rows.append(f"| `{op.name}` | " + " | ".join(f"{r[v]:.1%}" if v in r else "·" for v in levels) + " |")
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
        ops = "all four" if c.ops == ex.OP_NAMES else ", ".join(f"`{o}`" for o in c.ops)
        rows.append(f"| **{c.name}** | {c.title} | {anchor} | {ops} | {c.epochs} ({c.steps:,}) | {c.seeds} |")
    head = "| condition | what | anchor | ops | epochs (steps) | seeds |\n| --- | --- | --- | --- | ---: | ---: |\n"
    return head + "\n".join(rows)


@app.function(hide_code=True)
def proposals_md(survey: dict | None) -> str:
    """Each proposal's survey five-seed numbers beside blank columns for the fresh ones."""
    head = "| condition | statistic | survey (5 seeds) | fresh (5 seeds) | band |\n| --- | --- | ---: | ---: | ---: |\n"
    rows = []
    for c in ex.PROPOSALS:
        scored = survey["scored"].get(str(c.survey_trial)) if survey else None
        for i, stat in enumerate(SURVEY_STATS):
            label = f"**{c.name}**" if i == 0 else ""
            value = f"{scored[stat]:.3f}" if scored and stat in scored else "—"
            band = f"{ex.equiv_band(stat):.3f}" if stat in ex.NOISE_RUN else "—"
            rows.append(f"| {label} | {stat} | {value} | — | {band} |")
    return head + "\n".join(rows)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.2.3: the multi-op grammar, with *red* anchored again

    /// tip |
    <!-- tl;dr -->
    The grammar grows from one operation to four, each spelled as a word. So every D2.1 recipe has to be checked again before we anchor an operation. We retrain the un-anchored control, the ex-2.1.10 recipe, and three proposals from the ex-2.1.11 survey on the new grammar at fresh seeds. We check whether the placement and the suppression carry over, and freeze the rule that picks the operating point for the rest of D2.2.
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
    This is a preregistration. Each hypothesis section opens with its frozen prediction, and the `TODO` under it says what its figure or table will show. The predictions freeze once this skeleton is agreed in review, and we will quote the commit that freezes them here. Results replace the placeholders in place. Anything we think of after seeing the data goes under [Exploratory analyses](#exploratory-analyses), marked as post hoc.

    No run of this experiment has happened. The gates are the ones from D2.1, quoted against the ex-2.1.10 primary and the five-seed proposals from the survey, plus the suppression figures from ex-2.2.1. The op table and every count in the method are computed from `experiment.py` at render time.
    ///

    ## Why this experiment

    The claims of D2.2 are about an operation, and the D2.1 grammar has only one: `mix`, spelled `+`. The [D2.2 plan](../d2.2/design.md) makes the operation a variable, with four ops on the same color grid, each spelled as a word between the operands. Two things change for the model: the corpus carries four rules at the same total size, and a token between the operands decides the answer.

    Every number D2.1 produced was measured on the one-op grammar: the recipe in [ex-2.1.10](../ex-2.1.10/report.py), the operating-point plateau in the [ex-2.1.11](../ex-2.1.11/report.py) survey, and the suppression figures in [ex-2.2.1](../ex-2.2.1/report.py). The risk table in the plan names this ("recipe is grammar-specific") and sends it here.

    So this is a regression check with a decision attached. We retrain the control, the recipe, and three survey proposals on the new grammar, then read the same statistics on the same `mix` lines D2.1 read them on.

    The survey set its own rule: D2.2 has to confirm its numbers at fresh seeds before any of them is quoted. The selection rule frozen under H3 then names the operating point the anchored-op experiments adopt.

    Two items ride along. First, the op table: on this grid, `screen` and `multiply` do not survive the closed-pair rule, so the op table is `mix`, `add`, `lighten`, and `darken`; the method shows the closure and agreement counts behind that.

    Second, two of the new ops can make the answer redder than both operands, which gives the blind-span question from ex-2.1.10 its first lines (E3). Which op to anchor stays open, and the relevance distributions in the method are the input to that choice.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    /// details | Glossary
    - **line** — one equation, `c1 ‹op› c2 = answer`, six word-level tokens. The op word is one token, like each color.
    - **op word** — the token between the operands: one of `mix`, `add`, `lighten`, `darken`. Bare *op* is the operation; *op1* and *op2* are the operand roles.
    - **closed pair** — a pair of colors whose answer under an op is itself a grid color, so the line can be written. `mix` is closed on {ex.closure(ex.MIX):.0%} of pairs; the other three ops on all of them.
    - **op-relevance** — how much of the answer reading the op is worth, for one line. Take the mixture over the ops closed on that pair, and measure the weight it withholds from the answer given by the anchored op. Defined in `experiment.py`.
    - **residual stream** — the running vector each token carries through the network, which every block reads from and writes to.
    - **slice** ($\ell$) — a depth at which the residual stream is read: the embedding (0), plus the stream after each of the four blocks.
    - **alignment** ($\alpha$) — $\cos(h, e_1)$, the cosine between a state and the anchor axis. States are unit-norm, so this is the e₁ component.
    - **dose** — how *red* a line is, the larger of the two operand rednesses ($r(1 - g/2 - b/2)$ on the unit cube). **Red lines** have dose ≥ {ex.RED_DOSE:g}; **non-red lines** dose ≤ {ex.NONRED_DOSE:g}.
    - **m_line** — the margin from ex-2.1.10: at the best span role, the label-weighted mean alignment minus the unweighted mean, averaged over slices. The one statistic tight enough to rank operating points on.
    - **containment** ($\bar\alpha$) — the mean alignment over all 216 colors at op1. A pull that latches onto op1 as a position rather than a concept drives it up.
    - **retention** — the final m_line as a fraction of its running peak over training.
    - **grading** ($r^2$) — the squared Pearson correlation between op1 alignment and the sim^1.5 target across the 216 colors. It says whether alignment rises in proportion to redness rather than switching at a threshold.
    - **contrast** — the difference in op2 softmin weight between the op2-triggered and op1-triggered line groups, averaged over the four post-attention slices. It says whether the pull found the operand that carried the label.
    - **latch** — a run whose non-red group puts more than {ex.LATCH_PI:g} of its softmin weight on op1; vetoed per run.
    - **holdout EM** — exact-match accuracy on held-out (op, pair) keys, read per op.
    - **deficit** — clean exact-match accuracy minus intervened accuracy over a group of lines. This is the statistic ex-2.2.1 gated selectivity on.
    - **band** — the smallest seed-mean difference the resolution rule may call a difference: $2\sigma\sqrt{{1/n_a + 1/n_b}}$, with σ the per-run spread.
    - **feasible** — a condition that passes every constraint the survey scored on, re-read on fresh seed means (H3).
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Conditions

    Every condition is trained here, on the four-op corpus, at fresh seeds. The corpus is twice the size, so we match D2.1 by steps rather than epochs (see the [method](#training)). The recipe runs the {ex.D21_STEPS:,} steps of the ex-2.1.10 primary, and each proposal runs the {ex.SURVEY_STEPS:,} steps of its survey trial.

    {conditions_md()}

    The recipe is the D2.1 primary, `either-t100`, with nothing changed but the grammar and the epoch count. The three proposals are the operating point from the survey plus the two promoted trials the survey read as its twin and its knee. `t48` sits within one m_line band of `t00`, with more grading and contrast. `t12` holds the most grading of the promoted set, at a margin the survey still could not tell apart from the one at `t00`. All three ran the anchor flat, as the survey did.

    The two richer-op arms are un-anchored models on one and two ops, at the same line count and step count, and they serve E4 only.

    ### The interventions

    We score the checkpoints of every candidate through the eval contract in [`sca.intervention`](/src/sca/intervention.py), on the probe lines of each op.

    - **`projection`** — the primary from ex-2.2.1: the axis projected out at every slice and position, at full strength. Read by H4.
    - **Ride-along rows** — `operands` (the projection at the two operand positions only), `shaped` (the shaped suppression from M1, a = {ex.SHAPED["a"]:g}, b = {ex.SHAPED["b"]:g}, p = {ex.SHAPED["p"]:g}), and `ablate` (the axis weights zeroed). We report these without gates, so that the intervention-tuning pass the plan schedules has the multi-op figures beside the one-op figures from ex-2.2.1.
    """)
    return


@app.cell(hide_code=True)
def _():
    # REVIEW: H3 had two gaps as drafted. (1) The grading margin was written as extra
    # to feasibility, but the selection rule counts it inside feasibility, which made the
    # old partial clause ("feasible, but exceeds without the grading margin") unreachable;
    # feasibility now includes it in both places. (2) A feasible proposal sitting more
    # than a band *below* the recipe named no outcome; it is now contrary, on the same
    # reading as "no proposal is feasible" — the survey's point did not transfer. Also
    # named the training-length difference between the proposals and the recipe, which
    # rides along with every H3 margin comparison. Verify: if the conditions gain a
    # recipe arm at the proposals' step count, the length caveat can go.
    mo.md(rf"""
    ## Task cost on the new grammar (H1)

    **H1.** Anchoring costs nothing on the task, on any op. For every scored condition (the recipe and the three proposals), seed-mean holdout exact-match accuracy on each op is within {ex.TASK_GATE:g} of the control on the same op. Partial: within {ex.TASK_PARTIAL:g} on every op, or within {ex.TASK_GATE:g} on all but one. Contrary: a condition more than {ex.TASK_PARTIAL:g} below the control on some op. That would say the anchor and the op reading compete.

    The control has to learn the grammar for the gate to mean anything. The calibration in the [method](#before-the-freeze) checks that before the freeze, rather than gating it here.

    /// admonition | TODO
    A table of holdout exact-match accuracy per op for every condition, seed mean with the seed range, with the control row first and the gap from it on each other row. Beside it, the validation loss over training for the control and the recipe, five thin lines each, with the ex-2.1.10 primary drawn behind them on the step axis.
    ///

    ## Placement under the recipe reproduces (H2)

    **H2.** The ex-2.1.10 recipe places *red* on the new grammar as it did on the old one. On the `mix` lines, at five fresh seeds, every one of the ex-2.1.10 placement gates holds: containment $\bar\alpha$ at op1 at most {ex.MEAN_ALIGN_GATE:g}; retention at least {ex.RETENTION_GATE:g} of the running peak for every seed whose peak reaches {ex.RETENTION_FLOOR:g}; a leading softmin weight of the red group at the embedding of at least {ex.LEAD_GATE:g}; contrast at least {ex.CONTRAST_GATE:g}; grading $r^2$ no more than {ex.GRADE_R2_DROP:g} below the {ex.REF_R2_SIM:.3f} the survey measured on this recipe; seed-mean m_line at least {ex.MARGIN_RATIO:g} of its D2.1 value of {ex.REF_M_LINE:.3f}; and no latched run.

    Partial: every criterion holds except one of m_line in the {ex.MARGIN_PARTIAL:g}–{ex.MARGIN_RATIO:g} band or contrast in the {ex.CONTRAST_PARTIAL:g}–{ex.CONTRAST_GATE:g} band. Contrary: a latch or a containment miss. That would mean the op word changed what the pull finds, since the labeller reads the same two operand roles on every op. A margin below the partial band with the other criteria intact would mean the placement is there but weaker, and would send the H3 selection toward the proposals.

    /// admonition | TODO
    A table of the seven statistics for the recipe on the `mix` lines, seed mean with the seed range, beside the nine-seed values of the ex-2.1.10 primary and the band between them. Then the m_line trajectory over training, five thin lines drawn over the nine from ex-2.1.10 on the step axis, and the softmin profile over roles per slice for the red and non-red groups, in the ex-2.1.10 layout.
    ///

    ## The plateau from the survey transfers (H3)

    **H3.** The plateau found in the survey belongs to the recipe rather than to the one-op grammar. At least one proposal is feasible at fresh seeds on the new grammar, and its seed-mean m_line on the `mix` lines exceeds the recipe by more than one band.

    Feasibility is read as the survey read it: task on every op, containment, retention, no latch, contrast at least {ex.CONTRAST_PARTIAL:g}, and grading $r^2$ no more than {ex.GRADE_R2_DROP:g} below the fresh value for the recipe, clearing that floor by at least one per-run σ. The last clause is the grading margin, and it is part of feasibility here as it is in the selection rule below.

    Partial: a proposal is feasible and its m_line sits within a band of the recipe, or a proposal clears every survey constraint and exceeds the recipe by more than a band but misses the grading margin. Contrary: no proposal is feasible, or every feasible proposal sits more than a band below the recipe. Either would say the plateau was specific to the one-op grammar, and the recipe carries D2.2.

    The proposals train for half the recipe's steps, since each runs at the length of its survey trial (see the [method](#training)). So a margin difference between a proposal and the recipe carries a training-length difference with it, and H3 asks only whether the survey's point survives transfer on its own terms. Which schedule a matched-length comparison would favor is a separate question.

    **The selection rule.** {ex.SELECTION_RULE} The Findings line for this section names the adopted point, and it lands in `experiment.py` for the D2.2 experiments that follow.

    **Winner's curse.** The five-seed numbers from the survey are proposals. The table below prints them beside the fresh five-seed values, per proposal and statistic, with the band a difference has to clear to count as a change. That gap is the correction the survey asked for when it handed these over.

    {proposals_md(load_survey())}

    /// admonition | TODO
    The table above filled in, with the feasibility checks for each proposal as a row of ticks and the adopted point marked. Then m_line against grading $r^2$ for the four candidates, one point per seed with the seed mean ringed, the five-seed survey point for each proposal drawn hollow beside it, and the band for the recipe drawn as a horizontal strip.
    ///

    ## Suppression transfers (H4)

    **H4.** Projecting the axis out removes *red* on every op, at the selectivity ex-2.2.1 reached. On the recipe and on the adopted point, under `projection`, seed-mean accuracy on the red lines of each op falls to at most {ex.RED_ACC_GATE:g}, and the seed-mean non-red deficit on the `mix` lines is at most {ex.NONRED_DEFICIT_GATE:g}. Partial: exactly one clause holds, or removal holds on all but one op, or the deficit sits in the {ex.NONRED_DEFICIT_GATE:g}–{ex.NONRED_DEFICIT_PARTIAL:g} band.

    The deficit gate is the band ex-2.2.1 landed in (0.024, partial at its 0.02 gate), and we report the 0.02 read beside it. That cost came from the syntax embeddings carrying a constant component on the axis. This grammar has four op words where D2.1 had one, and E2 reads their rows.

    Contrary on removal: red accuracy near clean on one of the new ops. That would say the model reads *red* off the axis on lines of that op, and the per-op alignment in E1 would show it first. Contrary on selectivity: a deficit above {ex.NONRED_DEFICIT_PARTIAL:g}, which would send the intervention-tuning pass to the `operands` and `shaped` rows before anything is anchored.

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

    **E2 — the op words.** The alignment of the four op-word embeddings and of `=` and the newline, per slice, on the recipe and the adopted point, beside the constant component ex-2.2.1 found on `+` and `=`. We also ask whether the alignment of an op word predicts the non-red deficit on its lines under `projection`.

    **E3 — redder than both.** `add` and `darken` have lines whose answer is redder than either operand; `mix` and `lighten` have none. On those lines the strongest evidence for *red* sits at the answer, a position the labeller never keys on. That is the blind-span case named in the scope note of ex-2.1.10. We read the alignment at `=` and at the answer position, against lines of the same op and dose whose answer is not redder than both, and the softmin profile on each group. Then we restrict the H4 statistics to these lines, which asks whether removing *red* from the operands also removes it from an answer that was redder than they were.

    **E4 — a richer op set.** The cube probes of ex-2.1.12 (ridge, ℓ₂ = 10⁻², strict per-value holdout) on the un-anchored models at one, two, and four ops: held-out $R^2$ for op1, op2, and the RGB of the answer, per slice and position. This asks whether more rules give the model a better operand geometry at the same data and compute, read together with the lines-per-op confound the conditions table notes.

    **E5 — the noise floor, re-measured.** The per-run σ of every gated statistic on the new grammar, from the five seeds of the recipe and of the control, beside the ex-2.1.10 values the bands used. The H3 verdict and the selection rule are scored with the frozen ex-2.1.10 bands either way; where the fresh σ is larger, we report which H3 comparisons the wider band would leave unresolved, as a robustness read.
    ///

    ## Discussion

    /// admonition | TODO
    Interpretation only, after the results. Whether the D2.1 recipe belongs to the anchor or to the grammar it was tuned on, and what the survey-to-fresh gaps say about how much of the plateau was luck. What the adopted point is and what it costs against the recipe. Whether the four op words repeat the syntax-row cost of ex-2.2.1, and what that means for the intervention the anchored-op experiments should use. What the redder-than-both lines say about the blind span. Which op the relevance distributions favor anchoring. No re-derivation of the findings.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Method

    ### Prerequisite: the operation as a variable

    The [dependency named in the design](../d2.2/design.md#deps) lands before the DAG, in `sca.data`: an op table (name, surface form, per-channel rule), an `op` field on `Example`, seen-pair bookkeeping keyed on (op, pair), ops spelled as words, and the infix frame kept so the probe positions in `sca.compute.evaluation` still read. The table below is the specification for that op table, and the closure and agreement counts are what chose it.

    ### The grammar

    Four ops on the `v216` grid (six levels per channel: 0, 3, 6, 9, 12, 15), every color one token, every op one token. A line can be written only for a pair closed under its op. Counts over each op's writable lines (every closed pair in both orders):

    {op_table_md()}

    `mix` is the D2.1 op with its word changed, and its {ex.line_counts(ex.MIX)["lines"]:,} lines are the D2.1 probe set. The other three are closed on every pair, so each has 46,656 lines.

    The plan named `screen` and `multiply` for two of these slots. On this grid, `screen` lands on the grid only where a channel is 0 or 15, and there it equals `add`; `multiply` lands on the grid only where a channel is 0 or where it acts as the identity. So neither contributes a line another op does not already teach. `add` and `darken` are the two ops that can make the answer redder than both operands.

    **Agreement.** For each pair of ops, how often they give the same answer on the pairs closed under both. Where two ops agree, reading the op is worth nothing on that line, and that is what makes op-relevance graded:

    {agreement_md()}

    `add` and `lighten` agree on a sixth of pairs (per channel, where one operand is 0 or the other is 15). The other agreements are the 216 self-pairs, where every op returns the color, plus a handful of pairs at the ends of the range.

    **Op-relevance** per candidate anchored op, as the fraction of the lines of that op at each level. The plan asked for these under the rounding used by the table itself, since the per-line predictions of the suppression experiments rest on them:

    {relevance_md()}

    For `add` and `lighten`, about a sixth of lines have another op giving the same answer, so their relevance has weight on a lower level. For `mix` and `darken`, nearly every line has an answer that only the one op gives. The two upper levels differ only in whether the pair is closed under `mix`, which sets how many ops the mixture ranges over. The choice of anchored op stays open until the suppression prereg.

    ### The corpus

    {ex.N_LINES:,} lines, {ex.N_LINES_PER_OP:,} per op. Ops are drawn uniformly, and within an op the closed training pairs are drawn uniformly with random operand order, the way ex-2.1.10 drew the `mix` pairs. For each op, a fifth of its distinct closed pairs is held out, keyed on (op, pair), so a pair held out under `add` may still be trained under `mix`. Corpus seed {ex.CORPUS_SEED}. The richer-op arms keep the {ex.N_LINES:,} lines and narrow the op set.

    ### Training

    D64-L4 nGPT, the ex-2.1.3 data config (64 × 64 batches, oversample 16), peak LR {ex.PEAK_LR:g}, and the D2.1 anchoring code path: the pooled either-operand labeller (each operand draws at redness⁸ × {ex.PER_SLOT_RATE:g}), the anchor on e₁ at every slice, and the anti-subspace term.

    **Matched by steps.** The loader sizes an epoch as a fixed fraction of the corpus, so doubling the lines doubles the steps per epoch ({ex.steps_per_epoch(ex.D21_LINES)} on the D2.1 corpus, {ex.steps_per_epoch(ex.N_LINES)} here). The recipe runs the {ex.D21_STEPS:,} steps of ex-2.1.10 as {ex.EPOCHS_RECIPE} epochs; each proposal runs the {ex.SURVEY_STEPS:,} steps of its survey trial as {ex.EPOCHS_PROPOSAL}.

    Every schedule keyframe is a fraction of training, as ex-2.1.11 restated them: LR and anchor warm-up over the first {ex.WARMUP_FRAC:.0%}, the anchor anneal for the recipe over the last {1 - ex.ANNEAL_START_FRAC:.0%} down to a {ex.ANNEAL_FLOOR:g} floor, and the anti-subspace weight annealing from its peak ratio to {ex.ANTI_HOLD_RATIO:g} of the anchor weight by its own end fraction.

    At matched steps the corpus is a quarter `mix`, so a `mix` pair is seen a quarter as often over the run as in D2.1: half as often per epoch, over half as many epochs. That is part of what the check measures, and it is the alternative reading of an H2 miss on the `mix` lines — less exposure to the op the statistics are read on, rather than the op word changing the placement.

    <!-- REVIEW: was "half as often per step", which is neither the per-epoch nor the run figure. Per epoch it is half (double the steps, a quarter of each batch); over the run it is a quarter, since the epoch count halves too. Verify against `steps_per_epoch` and `N_LINES_PER_OP` in experiment.py. -->

    The parameters for the proposals are the sampled values from the survey, unrounded, in `experiment.py`. Each proposal runs here as the survey ran the trial: the anchor flat, the anti-subspace schedule at the peak ratio and end fraction of that trial, and at the length of that trial.

    ### The probe set

    {ex.N_PROBE} lines per color as op1, per op. For `mix` that is every closed partner, so its probe set is the D2.1 one with the op word changed. The other ops are closed on every pair, so we draw {ex.N_PROBE} partners per color once with seed {ex.PROBE_SEED} and share them across every run. Each op then has {ex.line_counts(ex.MIX)["lines"]:,} probe lines. Red and non-red lines are defined per op by dose.

    ### Measurements

    The placement statistics are the ones from ex-2.1.10, computed by its code on the probe lines of each op from one clean pass per run: m_line and the softmin profiles, containment, the lead weight, contrast, grading against the sim^1.5 target, and the latch. The trajectory records m_line every {ex.TRAJ_STRIDE} steps. Holdout exact match is read on a held-out set per op, of the size ex-2.1.10 used.

    The interventions run one teacher-forced pass per run per operator over the four probe sets, through the eval contract, giving the log-softmax at `=` and the write per (slice, line, position) as in ex-2.2.1. Red-line accuracy and the non-red deficit are read per op.

    **Noise floor.** The bands use the per-run σ of each statistic at the reference recipe, from the nine seeds of ex-2.1.10 ({", ".join(f"{k} {v:g}" for k, v in ex.NOISE_RUN.items())}), and E5 re-measures them here. Gates score seed means against fixed thresholds and do not use the floor. Every difference between two conditions is quoted with its band, and a difference inside the band is reported as not resolved.

    ### Budget

    {ex.N_RUNS} training runs: {ex.CONTROL.seeds + ex.RECIPE.seeds} at the recipe's length, {sum(c.seeds for c in ex.PROPOSALS)} at half of it, and {sum(c.seeds for c in ex.RICHER_OP_ARM)} for the richer-op arm, each at a plain D2.1 step count on an L4. Scoring is one clean pass plus four operator passes over four probe sets per run, CPU seconds each, and the cube probes of E4 are ridge fits on 216 rows. That is well under the cost of ex-2.2.2, which trained 24 runs at twice the step cost.

    ### Before the freeze

    One calibration run precedes the freeze: the control at one seed on the four-op corpus. It checks that the grammar is learned at the step count of the recipe (holdout exact match per op near 1, as the D2.1 control reached) and that `add` does not need more. If it does, we change the corpus rule or the step count before the freeze and record the change with a `REVIEW` note. Nothing anchored runs before the freeze.
    """)
    return


if __name__ == "__main__":
    app.run()
