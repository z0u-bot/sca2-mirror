import marimo

__generated_with = "0.24.0"
app = marimo.App(
    app_title="Ex 2.2.9: the grammar handover",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np

    # The design constants come from `experiment.py` beside this notebook (Marimo puts the
    # notebook directory on sys.path). The prose quotes the frozen gates, and that module
    # carries the same numbers with each gate's wording in its docstring.
    import experiment as ex
    from mini.reports import report_bundle, use_publisher
    from sca.data.colors import redness
    from sca.data.ops import TOP, commutativity, dose, lines, probe_lines, relevance

    use_publisher(report_bundle(__file__))

    # A cell renders its last expression, and a trailing docstring is one.
    None


@app.function(hide_code=True)
def zeroed(a, b):
    """The line with the redder operand's R set to zero: ex-2.2.4's *to-zero* rule."""
    if redness(a) >= redness(b):
        return (0, a[1], a[2]), b
    return a, (0, b[1], b[2])


@app.function(hide_code=True)
def to_zero_move(op, a, b) -> float:
    """How far the rule's answer moves in the unit cube when the red operand loses its R."""
    return float(np.linalg.norm(np.subtract(op(*zeroed(a, b)), op(a, b))) / TOP)


@app.function(hide_code=True)
def removal_counts() -> dict[str, tuple[int, int]]:
    """Per op: (red probe lines, removal lines), on the op's probe set as ex-2.2.3 draws it."""
    out = {}
    for op in ex.TABLE:
        red = [
            (ln.lhs, ln.rhs) for ln in probe_lines(op, ex.N_PROBE, ex.PROBE_SEED) if dose(ln.lhs, ln.rhs) >= ex.RED_DOSE
        ]
        far = sum(to_zero_move(op, a, b) >= ex.FAR_MOVE for a, b in red)
        out[op.name] = (len(red), far)
    return out


@app.function(hide_code=True)
def table_md() -> str:
    rows = []
    for op in ex.TABLE:
        group = "kept" if op.name in ex.KEPT else ("added" if op.name in ex.ADDED else "ordered subset")
        comm = "yes" if commutativity(op) > 0.99 else f"{commutativity(op):.0%}"
        rows.append(f"| `{op.name}` | {op.rule.replace('|', '\\|')} | {comm} | {group} |")
    head = "| op | rule (0..15 scale, snapped to the grid) | commutative | in A+ as |\n| --- | --- | --- | --- |\n"
    return head + "\n".join(rows)


@app.function(hide_code=True)
def arms_md() -> str:
    rows = []
    for a in ex.ARMS:
        anchor = "none" if a.lam == 0 else f"λ_a = {a.lam:g}, τ = {ex.TAU:g}"
        labeller = "—" if a.lam == 0 else ("whole line" if a.keying == "line" else "either slot, prompt span")
        readout = "tied" if a.tie else "untied"
        role = "gated" if a.scored else "reference / exploratory"
        rows.append(
            f"| **{a.name}** | {a.title} | {anchor} | {labeller} | {readout} | {a.lines_per_op:,} | {a.epochs} ({a.steps:,}) | {a.seeds} | {role} |"
        )
    head = (
        "| arm | what | anchor | labeller | readout | lines per op | epochs (steps) | seeds | role |\n"
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |\n"
    )
    return head + "\n".join(rows)


@app.function(hide_code=True)
def relevance_md() -> str:
    """Op-relevance for the two reference ops under table A+, counted over ordered pairs."""
    ordered = lines()
    rows = []
    for name in (ex.PRIMARY_OP, ex.SECONDARY_OP):
        op = next(o for o in ex.TABLE if o.name == name)
        rel = relevance(op, ex.TABLE, ordered)
        shares = [rel.get(k, 0) for k in range(3)] + [sum(v for k, v in rel.items() if k >= 3)]
        cells = " | ".join(f"{v:.0%}" for v in shares)
        rows.append(f"| `{name}` | {cells} |")
    head = "| anchored op | alone | 1 other agrees | 2 | 3+ |\n| --- | ---: | ---: | ---: | ---: |\n"
    return head + "\n".join(rows)


@app.function(hide_code=True)
def removal_md() -> str:
    counts = removal_counts()
    rows = [f"| `{name}` | {red:,} | {far:,} ({far / red:.0%}) |" for name, (red, far) in counts.items()]
    head = "| op | red probe lines | removal lines |\n| --- | ---: | ---: |\n"
    return head + "\n".join(rows)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.2.9: the grammar handover

    /// tip |
    <!-- tl;dr -->
    **Draft, before any run.** The last four experiments each turned up one thing we would change about the setup we anchor *red* in: a wider set of operations, answers that are drawn rather than rounded, a label that covers the whole line, and a readout table kept separate from the embedding table. Each was tried on its own and looked fine.

    This experiment turns all four on at once, at twenty seeds. Does *red* still land where we put it? Can it still be removed cleanly? And do the two changes that came with a caveat, the label and the readout, earn their place? Two extra arms switch one of those back each, so we can attribute any change to the right one. At the end, one arm becomes the setup every later D2.2 experiment builds on.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Findings

    - [Does the model still learn the task? (H1)](#does-the-model-still-learn-the-task-h1) — _to come_
    - [Does *red* still land where we put it? (H2)](#does-red-still-land-where-we-put-it-h2) — _to come_
    - [Can we still take *red* out cleanly? (H3)](#can-we-still-take-red-out-cleanly-h3) — _to come_
    - [Does the separate readout keep the axis off the syntax words? (H4)](#does-the-separate-readout-keep-the-axis-off-the-syntax-words-h4) — _to come_
    - [Does the whole-line label cost selectivity? (H5)](#does-the-whole-line-label-cost-selectivity-h5) — _to come_
    - **Decision:** _to come._ {ex.DECISION}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this draft
    This is a preregistration. The arms, the gates, and the decision rule below are written before any run, and will be frozen at a named commit. Each hypothesis section opens in plain words, then gives the prediction we will be scored on. Results go into each section in place once they exist. Anything we think of after seeing the data goes under [Exploratory](#exploratory), marked as post hoc. Every count in the method is computed from `experiment.py` at render time.
    ///

    ## Why this experiment

    Ex-2.2.3 anchored *red* on a grammar of six operations and then tried to remove it. Removal came out partial on four ops, and the ops themselves were the reason. On `lighten`, say, most lines with a red operand have an answer that stays the same when the operand is a little less red, so a model that has lost *red* can still answer them. That sent us scouting.

    Four things came back, each from one experiment:

    1. **A wider table of operations** ([ex-2.2.4](../ex-2.2.4/report.py)). Drop `add`, add three ops whose answers spread through the color cube, and add three more that take one attribute from one operand and the rest from the other. Those last three are the first ops where the order of the operands matters.
    2. **Drawn answers instead of rounded ones** ([ex-2.2.5](../ex-2.2.5/report.py)). When a rule lands between two grid colors, the corpus picks one of them at random, with the odds set by where it landed. The model then learns a spread of possible answers rather than one right token, which is what a language model learns.
    3. **A label that covers the whole line** ([ex-2.2.6](../ex-2.2.6/report.py)). The anchor is told "this line is about red", and it pulls wherever in the line it finds the most red, answer included. That is the shape a document-level label will have in M3. The first pilot found it costs nothing, but a later one saw a few seeds lose selectivity under it, so it goes in with a check.
    4. **A separate readout table** ([ex-2.2.7](../ex-2.2.7/report.py)). The output layer of the model was sharing a table with its input embeddings, and that sharing put part of the *red* axis onto the words `=` and `mix`. Giving the output its own table keeps the axis off those words, which is what a clean full-line removal needs.

    [ex-2.2.8](../ex-2.2.8/report.py) added one more: the removal operator is the plain projection, which takes the axis out everywhere, with the operand-only edit beside it as the selective reference.

    Each change was tried alone. Here they are tried together, at the same number of seeds as the reference, so that whatever comes out is the grammar and recipe of record for the anchored-op experiments. We call it a handover because the grammar of record changes hands here. Until this runs it is the one from ex-2.2.3; after it, if the gates hold, it is this one.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Conditions

    Every arm trains fresh models on the new grammar: table A+, {ex.N_LINES:,} lines drawn uniformly over the {ex.N_OPS} ops, answers drawn stochastically, one fifth of the pairs of each op held out. The recipe is the point adopted in ex-2.2.3 (`{ex.EX223_REFERENCE}`), unchanged: λ_a = {ex.LAM:g} annealed over the last tenth, τ = {ex.TAU:g}, the anti-subspace weight from {ex.ANTI_PEAK_RATIO:g}× the anchor weight down to 0.3× by {ex.ANTI_ANNEAL_END_FRAC:.0%} of training, {ex.EPOCHS} epochs ({ex.HANDOVER.steps:,} steps). *Red* sits on e₁ at every slice, the embedding included.

    {arms_md()}

    **`handover`** has everything switched on. It is the candidate grammar of record, at the same twenty seeds as the reference.

    **`handover-slot`** switches the label back to the one from ex-2.2.3: only the two operands can draw a label, and the pull covers the prompt. Beside `handover` it is the selectivity check ex-2.2.7 asked for, and it is the fallback if the whole-line label does not clear its gates. It also has twenty seeds, so whichever of the two is adopted has the full count.

    **`handover-tied`** switches the readout back to the shared table. Beside `handover` it shows what untying does on this grammar. Nine seeds, as the pilot had.

    **`control`** has no anchor at all. It sets the task bar for H1 and the calibration bar for the drawn answers.

    **`handover-wide`** is exploratory. Eleven ops share the same {ex.N_LINES:,} lines, so each op gets about {ex.HANDOVER.lines_per_op:,} lines, about half of what it had at six ops. E4 in ex-2.2.3 found the operand cube less decodable at six ops than at three, and could not tell whether the op count or the lines per op was responsible. This arm holds lines per op at the six-op count ({ex.WIDE.lines_per_op:,}), using a corpus eleven sixths the size for fewer epochs so that the step count matches. It is read only in the exploratory section.

    The reference arm is not retrained. It is `{ex.EX223_REFERENCE}` from ex-2.2.3, at twenty seeds on the six-op grammar, and its stored statistics are printed beside every placement read.

    ### The removal operators

    We score every anchored checkpoint through the eval contract in [`sca.intervention`](/src/sca/intervention.py), on each op's probe lines.

    - **`projection`**: the axis projected out at every slice and every position, at full strength. This is the gated row, as ex-2.2.8 proposed.
    - **`operands`**: the same edit at the two operand positions only. It is the selective reference, since it never touches the syntax words and so cannot cost anything there.
    - **`shaped-a0.4-p0`**: a thresholded projection that leaves any state below alignment {ex.SHAPED["a"]:g} alone. This is the syntax-free candidate from ex-2.2.8, reported without a gate.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What the words mean

    A short glossary, since the reads below lean on it.

    - **Red line, non-red line.** A line is *red* when its redder operand has redness at least 0.8, and *non-red* when neither operand is above 0.2. Redness is r·(1 − g/2 − b/2) on the unit scale, so pure red is 1 and pink is lower.
    - **Removal lines.** The red lines on which the answer given by the rule would move a long way if the red operand had no red in it (its R channel set to zero). These are the lines where losing *red* has to show; on the others the op does not need it. The count per op is in the method.
    - **Expected exact match.** With drawn answers, the correct answer to a line is spread over two or more colors. Expected exact match is the chance that an answer drawn from the model agrees with one drawn from the rule. On the ops that round it cannot reach 1.
    - **m_line, ᾱ, lead, contrast, grading r², retention, latch.** The placement statistics of ex-2.2.3, unchanged: how far *red* is pushed onto the axis, whether other colors drift with it, whether the pull lands on the red operand rather than on a fixed position, and whether the placement holds to the end of training.
    - **Deficit.** How much expected exact match a model loses on the non-red lines when the axis is projected out. This is the selectivity read, and a clean removal costs nothing here.
    - **Band.** Two seed means are told apart only when they differ by more than 2σ√(1/n_a + 1/n_b), with σ the per-run spread. Smaller differences are reported as unresolved.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Does the model still learn the task? (H1)

    **In plain words.** Eleven ops in the same number of lines, with drawn answers, is a harder corpus than six ops with rounded ones. Before we read anything about the anchor we need to know that the anchored models learn the task as well as an un-anchored model does on this corpus.

    **Prediction.** For each of the three gated arms and each of the {ex.N_OPS} ops, the seed-mean expected exact match on held-out lines is within {ex.TASK_GATE:g} of the control, which gives the gate {len(ex.SCORED) * ex.N_OPS} comparisons. Partial: every comparison within {ex.TASK_PARTIAL:g}, or all but one within {ex.TASK_GATE:g} and that one within {ex.TASK_PARTIAL:g}. Contrary: an arm more than {ex.TASK_PARTIAL:g} below the control on some op. Ex-2.2.5 saw no task cost from the drawn answers and ex-2.2.7 none from the untied readout, so we expect this to hold.

    Whether the control itself learns the grammar is checked before the freeze, on one seed, rather than gated here. The number to watch is the ordered subset. An op that reads operand order asks the model for something the six-op grammar never did.

    /// admonition | TODO
        type: warning
    Results to come. The section will show expected exact match per arm and op against the control, and the calibration of the answer mass against the rule (P(mode) and KL, as ex-2.2.5 read them) beside it.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Does *red* still land where we put it? (H2)

    **In plain words.** The recipe was tuned on the six-op grammar, and here the corpus, the labeller, and the readout all change. We read the same placement statistics on the same `mix` lines, so the numbers mean what they meant in ex-2.2.3. The question is whether they are still inside the bars set there.

    **Prediction.** On the `mix` lines, for `handover` and `handover-slot` (each under its own labeller, as ex-2.2.6 read it):

    - *Margin:* seed-mean m_line at least {ex.MARGIN_RATIO:g} × {ex.REF_M_LINE:.4f} = {ex.MARGIN_RATIO * ex.REF_M_LINE:.3f}; partial from {ex.MARGIN_PARTIAL:g} × that.
    - *Grading:* grading r² no more than {ex.GRADE_R2_DROP:g} below {ex.REF_R2_SIM:.3f}.
    - *Containment:* ᾱ at op1 at most {ex.MEAN_ALIGN_GATE:g}; partial to {ex.MEAN_ALIGN_PARTIAL:g}.
    - *Concentration, attribution, retention, latch:* lead at the embedding at least {ex.LEAD_GATE:g}; contrast at least {ex.CONTRAST_GATE:g} (partial from {ex.CONTRAST_PARTIAL:g}); every run that reaches m_line {ex.RETENTION_FLOOR:g} ends at {ex.RETENTION_GATE:g} of its peak; no run latched.

    Containment is the one we expect to be close. At nine seeds, ex-2.2.7 read ᾱ at 0.13 on its untied arm and 0.23 on its untied whole-line arm, against 0.08 on the reference. The partial band is new, and it is there so that a near miss has a stated meaning. An arm inside the band can still be adopted, with the number flagged for the anchor-op prereg to watch; an arm above it cannot. We expect `handover-slot` to sit inside the gate or the partial band, and `handover` to sit above the gate and perhaps above the band.

    `handover-tied` is read on the same statistics, without a gate, so that we can attribute the containment read. If the tied arm sits with the reference and both untied arms sit higher, the untied readout is what moved it.

    /// admonition | TODO
        type: warning
    Results to come. The section will show the placement table (arm × statistic, with the reference beside) and the softmin profile over roles under each labeller.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Can we still take *red* out cleanly? (H3)

    **In plain words.** Take the axis out of every state and ask two things. On the lines that need *red*, does the model fail? On the lines that never had any, does it still answer? The first is removal, the second selectivity. Ex-2.2.3 could only show removal on two of six ops, because the other four mostly did not need *red*. Table A+ was chosen so that every op has lines that do, and the removal read is scored on those lines only.

    **Prediction.** Under `projection`, for each candidate arm:

    - *Removal:* on the removal lines of every op, the model keeps at most {ex.RED_KEPT_GATE:g} of its clean expected exact match, seed mean. This is the red-accuracy gate of ex-2.2.3, read as a ratio, because with drawn answers the clean value sits below 1 on the ops that round.
    - *Selectivity:* the seed-mean deficit on the non-red `mix` lines is at most {ex.NONRED_DEFICIT_GATE:g}; partial to {ex.NONRED_DEFICIT_PARTIAL:g}. Reported on every op beside it. On the reference, ex-2.2.8 read 0.040 on `mix`, which is inside the gate by less than a band. With the readout untied we expect the deficit to fall toward the `operands` row, which read 0.012.

    Two more reads come with a direction we expect but no gate.

    The first is how far the answer moves. On the removal lines we measure the distance in the unit cube from the answer the model decodes to the answer the rule gives, under `projection`. Beside it we put the distance the rule itself moves when the red operand loses its red. We expect the two rankings of ops to agree, so that the ops where the answer should move furthest (`value-hsv`, `hue-hsv`, `sat-hsv`) are where it does.

    The second is the operand-only row. It should remove less on the ops where the answer is read at `=` from both operands, and the same amount elsewhere.

    /// admonition | TODO
        type: warning
    Results to come. The section will show removal and selectivity per op and arm under the three operators, and the answer-distance read against the to-zero distance.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Does the separate readout keep the axis off the syntax words? (H4)

    **In plain words.** In ex-2.2.7 the model used its shared table to put some of the *red* axis on the embeddings of `=` and the op words, because that is a cheap way to predict `=` after a red operand. Giving the output its own table moved that component onto the output table and left the input embeddings mostly clean. That was on the six-op grammar at nine seeds. Does it carry to eleven ops, and does it buy the cleaner full-line removal it was adopted for?

    **Prediction.** On `handover` against `handover-tied`, same labeller, twenty seeds against nine:

    - The axis component on the syntax embeddings (`=`, the op words, and `⏎`), read from the embedding-component table of ex-2.2.7, is lower on `handover` than on `handover-tied` by more than a band. On `=`, `handover` sits within a band of the hard-zeroed ceiling from ex-2.2.7 (`{ex.EX227_CEILING}`, where the component is zero by construction). The component appears on the readout table of `handover` instead.
    - The non-red `mix` deficit under `projection` is lower on `handover` than on `handover-tied`, by more than a band.

    Neither line is a gate. Ex-2.2.7 already took the readout decision, and this section either confirms it or reports that it did not carry. If the second line fails while the first holds, then on this grammar the syntax embeddings were not where the cost came from.

    /// admonition | TODO
        type: warning
    Results to come. The section will show the embedding-component table per arm and the deficit under `projection` for the two arms.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Does the whole-line label cost selectivity? (H5)

    **In plain words.** A label that covers the whole line is the shape M3 will have, so we want it. Ex-2.2.6 found it costs nothing at three seeds. Ex-2.2.7 then ran nine seeds of its untied whole-line arm and saw a few of them lose a lot of non-red lines under the projection, where the operand-only labeller loses almost none. This section reads that at twenty seeds against twenty, and decides.

    **Prediction.** On `handover` against `handover-slot`, under `projection`:

    - The seed-mean non-red `mix` deficit differs by less than a band, and
    - the count of seeds whose deficit is above {ex.TAIL:g} (the level at which ex-2.2.7 read its tail) is no more than two higher on `handover` than on `handover-slot`.

    If both hold, the whole-line label carries and the decision rule prefers `handover`. If either fails, the tail is real, `handover-slot` becomes the grammar of record, and the whole-line label goes back to the backlog with the seed count it needs. We are not sure which way this goes. The tail in ex-2.2.7 was three seeds out of nine, which is enough to expect it and too few to be sure.

    /// admonition | TODO
        type: warning
    Results to come. The section will show the per-seed deficit under `projection` for the two arms, side by side, and the `operands` row beside it.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Decision

    {ex.DECISION}

    /// admonition | TODO
        type: warning
    To come, once H1 to H5 are read.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory

    Read without gates, and not part of the decision.

    - **Lines per op (`handover-wide`).** The operand-cube probe scan from E4 of ex-2.2.3, and expected exact match per op, on the arm that holds lines per op at the six-op count. If the cube is as decodable there as at six ops and less so on `handover`, then lines per op was the confound. If both sit together, it was the op count.
    - **The ordered subset, by slot.** For `hue-hsv`, `sat-hsv`, and `value-hsv` the probe set walks every color in both slots, so every read in H2 and H3 can be split by whether the red operand is op1 or op2. The labeller pools both operands the same way, so we expect the same placement in both slots. Removal need not match: `value-hsv` reads its value from op2, so a red operand at op1 contributes hue and saturation only.
    - **Calibration under the anchor.** P(mode) and KL from the rule on the rounded held-out lines, for every anchored arm against the control, as ex-2.2.5 read them. The anchor should not move them.
    - **The `shaped-a0.4-p0` row.** Reported on every op beside the two gated operators. Ex-2.2.8 found that it removes less than the projection does, at no cost. Here we ask whether that smaller amount still clears the removal gate on the new table.

    ## Discussion

    /// admonition | TODO
        type: warning
    About 200 words, after the results: what the handover settled, what it moved, and what the anchored-op prereg inherits.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Method

    ### The table

    {table_md()}

    Every rule is computed on the 0..15 scale and snapped to the grid. Where it lands between levels, the corpus draws the answer (*stochastic rounding*, ex-2.2.5). The three ordered ops take one HSV attribute from op2 and the other two from op1, so each agrees with its own reverse on under 2% of pairs. Their reads are reported as a subset, and their probe set walks every color in both slots.

    ### Op-relevance under A+

    For a line of the anchored op, how many other ops in the table give the same answer. This is the stimulus side of the per-line predictions in the anchored-op experiments, counted over ordered pairs. Widening the table makes `mix` less distinctive (ex-2.2.4 read 95% alone at six ops), which is what we want, since the later predictions need more than one level to read.

    {relevance_md()}

    ### The removal lines

    Per op, on its probe set: the red lines (dose ≥ {ex.RED_DOSE:g}) and, of those, the lines on which setting the R channel of the red operand to zero moves the answer of the rule by at least {ex.FAR_MOVE:g} in the unit cube. The removal read in H3 is scored on the second column.

    {removal_md()}

    ### The corpus, the probes, the labellers

    {ex.N_LINES:,} lines at seed {ex.CORPUS_SEED}, ops drawn uniformly, with {ex.HOLDOUT_FRAC:.0%} of the distinct unordered pairs of each op held out (for an ordered op, in both orders). Probe sets follow ex-2.2.3: `mix` on its 27 on-grid partners per color, and every other op on 27 partners per color drawn once at seed {ex.PROBE_SEED} and shared across ops. The ordered subset also walks every color as op2.

    The two labellers are *either slot, prompt span* (each operand draws at redness⁸ × {ex.PER_SLOT_RATE:g}, and the pull covers op1, op, op2, `=`) and *whole line* (the answer draws at its redness rate too, and the pull covers all six positions). The anchor term is the per-line mellowmax over the pulled span with a conserved per-line budget, so a wider span changes where the pull can land but not how strong it is.

    ### Before the freeze

    One seed of `control` trains first, and its expected exact match per op is recorded here, so that the gate in H1 is read against a control that learned the grammar. Two pieces of code land with the DAG: the holdout draw, now keyed on the position of the op in this table rather than in the table of ex-2.2.3 (`sca.data.ops.holdout`), and a probe draw that walks both slots for the ordered subset (`probe_partners`). Neither changes a number in the design.

    /// admonition | TODO
        type: warning
    The calibration read, and the freeze commit, to be filled in.
    ///

    ### Budget

    {ex.N_RUNS} runs of {ex.HANDOVER.steps:,} steps at d64-L4, plus scoring under three operators on {ex.N_OPS} probe sets. That is about the size of the run in ex-2.2.3.
    """)
    return


if __name__ == "__main__":
    app.run()
