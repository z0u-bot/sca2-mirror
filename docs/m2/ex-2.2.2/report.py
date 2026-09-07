import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="medium",
    app_title="Ex 2.2.2: fallback control in the anchored transformer",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import marimo as mo

    # The design constants come from `experiment.py` beside this notebook (Marimo
    # puts the notebook directory on sys.path). The skeleton quotes the frozen
    # gates in prose, and that module carries the same numbers with each gate's
    # wording in its docstring.
    from mini.reports import report_bundle, use_publisher

    use_publisher(report_bundle(__file__))

    # A cell renders its last expression, and a trailing docstring is one.
    None


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.2.2: a designed response to suppressing *red*

    /// tip |
    <!-- tl;dr -->
    Ex-2.2.1 removed *red* from the anchored transformer. What a red line decoded to afterward was left to the model: a near miss of the true answer, and the miss varied by seed.

    Here we add M1's fallback control to the D2.1 recipe. It is a training term that teaches the blocks what to answer once the concept is gone, aiming at a designed target, and it leaves the placement of the concept alone. Does the designed answer appear, is it the same across seeds, does the term cost anything, is the removal still selective, and how far does a response trained at the antipode, the state with its axis component flipped in sign, carry to a state that was only projected to zero?
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Findings

    - [The designed response (H1)](#the-designed-response-h1) —
    - [Reproducible across seeds (H2)](#reproducible-across-seeds-h2) —
    - [Task and placement intact (H3)](#task-and-placement-intact-h3) —
    - [Selectivity kept (H4)](#selectivity-kept-h4) —
    - [Carry from the antipode to zero (H5)](#carry-from-the-antipode-to-zero-h5) —
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this draft
    This is a preregistration. Each hypothesis section opens with its frozen prediction, and the `TODO` under it says what its figure or table will show. The predictions are frozen as of the commit that adds this file; results replace the placeholders in place, and anything conceived after seeing the data goes under [Exploratory analyses](#exploratory-analyses), marked as post hoc.

    No run of this experiment has happened. The gates come from the published reference numbers in ex-2.2.1, the task-gate width the D2.1 experiments used, and the fallback result from M1.
    ///

    ## Why this experiment

    [Ex-2.2.1](../ex-2.2.1/report.py) showed that projecting the anchor axis out of the D2.1 checkpoints removes *red*. Red-line accuracy falls from 1.0 to 0.09, in proportion to how red the line is, and stays inside the bound the placed geometry set.

    It also measured what the model does instead. A red line still decodes to a color, usually a one-step neighbor of the true mix, and which neighbor it is varies by seed: at least five of the nine seeds agree on 13% of red lines. Nothing in training said what a removed concept should decode to, so the answer is whatever the untrained region of the stream happens to produce. M1 called that spoofing, and it is where the seed spread in [ex-2.9.1](/docs/m1/ex-2.9.1/report.py) came from.

    The remedy in M1 was fallback control ([ex-2.9.2](/docs/m1/ex-2.9.2/report.py)): one loss term, applied to the decoder only, that pins the antipode of the anchor axis to a designed null answer, mid-gray. Reflecting a state through the axis then collapsed the response to a tight cluster on the bound the null predicted, across 32 seeds. Under plain zeroing the term added little, and that report said a trained fallback should be paired with the redirect it was trained at.

    Here we carry the term into the transformer, keeping the D2.1 grammar and recipe, before the operation work in the [D2.2 plan](../d2.2/design.md) changes the grammar. Two things are new. The readout is no longer a single linear decoder; it is every block after the redirect, so the term trains the blocks, and a stop-gradient keeps it from touching the placement. And the concept is continuous, so the designed target is a choice: the operand-averaged null for *red* is uniform over 27 colors and has no mode, so we take its center, the visible operand mixed with mid-gray.

    This addresses the third risk in the plan, that the response to suppression is undesigned. One limitation carries over from M1: the response is trained at the antipode, while the removal we deploy projects to zero. H5 reads how far the designed response carries between the two, so a limitation of this size shows up as a curve rather than a flat miss.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// details | Glossary
    - **line** — one equation, `c1 + c2 = answer`, six word-level tokens. The probe set enumerates all 5,832 closed lines of the 216-color grid: every color as op1 against each of its 27 closed partners.
    - **residual stream** — the running vector each token carries through the network, which every block reads from and writes to.
    - **slice** ($\ell$) — a depth at which the residual stream is read: the embedding (0), plus the stream after each of the four blocks.
    - **alignment** ($\alpha$) — $\cos(h, e_1)$, the cosine between a state and the anchor axis. States are unit-norm, so this is just the e₁ component.
    - **dose** — how red a line is, taken as the larger of the two operand rednesses ($r(1 - g/2 - b/2)$ on the unit cube).
    - **red lines** — dose ≥ 0.8. 365 probe lines. **Non-red lines** — dose ≤ 0.2. 1,689 lines.
    - **concept operand** — the redder of the two operands in a line (ties go to op1); the **visible operand** is the other one.
    - **target** — the designed answer for a red line once the concept is removed: the visible operand mixed with mid-gray, rounded to the grid. This is the center of the operand-averaged null.
    - **target accuracy** — the fraction of red lines whose decoded answer is the target.
    - **response** — the probability the model puts on the correct answer, read from the log-softmax at the `=` position. **Damage** is the clean response minus the intervened response, per line. **Deficit** is the clean exact-match accuracy minus the intervened accuracy over a group of lines; it is the statistic ex-2.2.1 gated selectivity on.
    - **seed agreement** — the fraction of red lines on which at least five of the nine seeds decode the same answer, under a given intervention. For the three-seed arms, it is the mean over seed pairs of the fraction of red lines the pair decodes alike.
    - **redirect** — reflecting the operand states through the axis at the embedding, α → −α. This is the state the fallback was trained at.
    - **clean** — the un-intervened forward pass of the same checkpoint.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conditions

    Two of the three conditions are ex-2.1.10 checkpoints already in the store, re-scored here through the same code path as the new runs. The third is trained here.

    | condition | training | seeds | source |
    | --- | --- | --- | --- |
    | **un-anchored** | the recipe with the anchor weight at zero | 3 | ex-2.1.10 `lam0` |
    | **no-fallback** | the D2.1 recipe: anchor + anti-subspace term | 9 | ex-2.1.10 `either-t100` |
    | **fallback** | the recipe + anti-anchor term (0.1) + fallback term (w_fb = 0.05) | 9 | new |

    The fallback condition is the primary one, and every hypothesis scores it. The no-fallback condition is the reference for it under every intervention.

    The arms below have three seeds each. We report the same statistics for them, without gates.

    | arm | what changes | question |
    | --- | --- | --- |
    | `fb-only` | no anti-anchor term | does the fallback need the antipode hemisphere cleared for it? |
    | `anti-only` | no fallback term | does the anti-anchor term alone move the response? |
    | `fb-w0.01`, `fb-w0.25` | the fallback weight, a factor of five either side | is 0.05 on a plateau? |
    | `recipe` | both new weights at zero | does the new code path reproduce the ex-2.1.10 checkpoints? |

    ### The interventions

    All of these run through the projection operator of the eval contract, which removes a fraction γ of the e₁ component and puts the state back on the sphere. At γ = 2 the operator is a reflection.

    - **`redirect`** — γ = 2 at the embedding, at both operand positions. This is close to the state the fallback was trained at, now applied to every line. The operator edits both operands because a deployed operator is not told which one carries the concept; training reflected the concept operand alone. A non-red operand has nothing on the axis, so the reflection leaves it alone, and on most red lines the two coincide. They part on the 67 of 365 red lines whose visible operand is itself red (redness ≥ 0.5), where the eval operator reflects a state training left clean. We report target accuracy split on that subset as well as pooled, so the pooled figure can be read against the 0.82 ceiling it would have if every one of those lines missed. H1, H2, and the first clause of H4 read here.
    - **`primary`** — the ex-2.2.1 intervention: γ = 1 at every slice and every position. This is the removal we deploy, and the fallback was never trained under it. Read by the second clause of H4 and the deployment row of H5.
    - **The carry sweep** — γ ∈ {0.5, 1, 1.5, 2} at the training site (embedding, operand positions). Read by H5.
    - **Ride-along rows** — the `operands`, `embedding`, `shaped`, and `ablate` arms from ex-2.2.1, run on the fallback condition without gates. This gives the operator-tuning pass in the design the fallback rows to hand (E5).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The designed response (H1)

    **H1.** Under `redirect`, the fallback condition decodes red lines to the target. Seed-mean target accuracy is at least 0.8; partial: between 0.5 and 0.8. The reference row is the no-fallback condition under the same intervention, and the fallback figure has to sit above it by a resolved margin (two pooled between-seed standard deviations). Contrary: target accuracy at the no-fallback level, which would mean the term did not train the readout. The fallback loss over training (see the method) says whether the term was ever active.

    /// admonition | TODO
    A table of target accuracy and true-answer accuracy on red lines under `redirect`, seed mean with the seed range, for the fallback and no-fallback conditions and the un-anchored calibration row. Beside it, per seed, the probability the model puts on the target on red lines, drawn as a strip. That way a split response, with some lines at the target and some elsewhere, shows up as such instead of averaging into the gate.
    ///

    ## Reproducible across seeds (H2)

    **H2.** The response is reproducible. Under `redirect`, the seed-agreement fraction on red lines for the fallback condition is at least 0.8; partial: at least 0.5. The reference is the no-fallback condition under the same intervention, measured here, together with the 13% it reached under the projection in ex-2.2.1.

    H2 can hold while H1 fails. If the seeds agree on an answer other than the target, the term pinned a response that the target did not specify, and the composition in E1 says which one.

    /// admonition | TODO
    The seed-agreement fraction under `redirect` and under `primary`, for the fallback and no-fallback conditions, with the pairwise figure beside it for the arms. Also a per-line view: for each red line, how many of the nine seeds decode its plurality answer, as a histogram per condition. A designed response piles the mass at nine, and an undesigned one spreads it out.
    ///

    ## Task and placement intact (H3)

    **H3.** The term costs nothing on the task or the placement. Clean exact-match accuracy on all probe lines is within 0.02 of the no-fallback condition (partial: within 0.05), and the seed-mean alignment margin at the end of training, the `m_span` of ex-2.1.10, is at least 0.8 of the no-fallback value. Partial: one of the two holds.

    Contrary: the fallback term competing with the anchor for the operand states, which is what the stop-gradient is there to prevent. The `recipe` arm says whether the code path on its own moved anything.

    /// admonition | TODO
    Clean accuracy on all, red, and non-red lines for every condition and arm. Then the alignment trajectory (margin over training) for the fallback condition drawn over the no-fallback condition, nine thin lines each. Beside it, the fallback loss and the anti-anchor loss over training, which check that the term switched on when the anchor placed the concept.
    ///

    ## Selectivity kept (H4)

    **H4.** Two clauses. Under `redirect`, the seed-mean non-red deficit stays at or below 0.02, since the operator touches only operand states and a non-red operand has nothing on the axis to reflect. Under `primary`, the non-red deficit in the fallback condition is not resolved above the no-fallback figure (0.024 in ex-2.2.1, re-measured here).

    Partial: the first clause holds only at the 0.05 band, or exactly one clause holds. Contrary on the second clause: the term widens the edit, which would mean the blocks now read the axis at positions or slices where they did not before.

    <!-- REVIEW: H4 was written on non-red *damage*, the probability statistic. The 0.02 width and the 0.024 reference both come from ex-2.2.1's H2, which gated the accuracy *deficit* (its H2 line: "outside the 0.02 gate but inside the 0.05 partial gate"), so the gate and its reference were being read on a statistic they were not set for. Now stated on the deficit, with the same widths, and the partial band written out. Verify: ex-2.2.1's H2 section, and the "non-red deficit" column of its arms table. Damage is still reported beside it in the table below. -->

    /// admonition | TODO
    The non-red deficit and non-red damage under `redirect` and under `primary`, seed mean and range, for the fallback and no-fallback conditions and for every arm. Also the ex-2.2.1 write-bound map for the fallback condition under `primary`, on the shared 0–1 scale, so a wider edit shows where it acts.
    ///

    ## Carry from the antipode to zero (H5)

    **H5.** The designed response carries part of the way to zero. Along the carry sweep, γ ∈ {0.5, 1, 1.5, 2} at the training site, seed-mean target accuracy on red lines is non-decreasing in γ, allowing a dip of at most 0.02 between adjacent strengths, and at γ = 1 it is at least half its value at γ = 2. Partial: monotone, but γ = 1 below half.

    Contrary: target accuracy at γ = 1 sits at the no-fallback level and the rise is confined to γ > 1. That would be the known limitation showing in full, with the response living at the antipode and not reaching the projected state. The remedy for that is the D2.3 [concept swap](/todo/science/redirect-between-two-anchored-ops.md), which targets a state training actually visits, rather than a wider bracket here.

    We report the `primary` row beside the sweep, without a gate: the deployment read, γ = 1 at every slice rather than at the training site alone. The difference between the two says how much the projection at later slices costs the designed response.

    /// admonition | TODO
    Target accuracy and true-answer accuracy against γ, seed mean with the seed range, fallback drawn over no-fallback, with the `primary` row as a separate mark at γ = 1. The x axis can also be drawn as the landing alignment of a pure-red operand, which has a closed form in γ.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    /// admonition | TODO
    Preregistered as exploratory, no gates.

    **E1 — composition.** What red lines decode to under `redirect` and under `primary`: the target, the true answer, a one-step neighbor of it, the visible operand, the red operand, or something else. Drawn as a stacked bar per seed, fallback beside no-fallback. These are the six categories from ex-2.2.1 with the target added. Also the mass outside the color vocabulary, and how far the decoded answer sits from the target and from the true mix.

    **E2 — the antipode.** The fraction of clean states with negative alignment, per slice and position, on the fallback, `fb-only`, and `anti-only` conditions. This is what the anti-anchor term is for, and it also says whether the anti-subspace term had already done the job.

    **E3 — off-axis recoverability.** This is the auditing row that the [D2.2 design](../d2.2/design.md) assigns here. We fit a ridge probe for the redness of the concept operand on the intervened operand states, per slice, under `primary` and under `redirect`, five-fold over lines. We report held-out R² for the fallback and no-fallback conditions beside the clean states.

    If the fallback raised the off-axis R², it would be keeping red readable off the axis so it knows when to emit the target; that is the masked case rather than the removed one. The target depends only on the visible operand, so there is no need for it to.

    **E4 — arms.** The H1–H4 statistics for `fb-only`, `anti-only`, the weight bracket, and `recipe`. The `recipe` row is a regression check against the stored no-fallback checkpoints, at three seeds, on clean accuracy, margin, red accuracy under `primary`, and non-red damage.

    **E5 — ride-along operators.** The H1, H2, and H4 statistics for the `operands`, `embedding`, `shaped`, and `ablate` rows on the fallback condition, beside the ex-2.2.1 figures for the same rows on the no-fallback condition.
    ///

    ## Discussion

    /// admonition | TODO
    Interpretation only, after the results. Whether the fallback in the transformer gives the intervention a designed outcome the way it did in M1. How far it carries from the trained state to the projected one, and what that says the anchored-op experiments should train toward. Whether the anti-anchor term earns its place in the recipe. And what the off-axis row says about masking. No re-derivation of the findings.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    ### Training

    We use the primary recipe from [ex-2.1.10](../ex-2.1.10/report.py) unchanged: the `v216` corpus, d64-L4, 100 epochs with a 10-epoch warm-up, the pooled either-operand labeller at τ = 0.1, the anchor at λ = 0.1 with its anneal, and the anti-subspace term at the ex-2.1.8 operating point.

    We also use the same seeds as the primary of that experiment, so the fallback and no-fallback conditions differ only in the two new terms. Both are weights on the existing anchored train step.

    **The fallback term.** On each training crop, take the lines with dose ≥ 0.8 whose concept operand has a clean embedding alignment of at least 0.5. For each one, reflect the embedding state of that operand through the axis, $h \mapsto \mathrm{normalize}(h - 2\alpha\, e_1)$, and hold the reflected state fixed with a stop-gradient. Run the blocks forward from there, with every other position clean, and take the cross-entropy at the `=` position against the target token of the line.

    The term is the mean over qualifying lines, at a constant weight of 0.05 from step 0. The alignment threshold keeps it inert until the anchor has placed the concept, which the ex-2.1.10 trajectories put inside the warm-up.

    The stop-gradient is what makes this the decoder-only term from M1. The embedding table and the placement of *red* get no gradient from it. The blocks and the unembedding do, and what they learn is a map from a state at the antipode to the target. Ex-2.2.1 read the concept from the operand states in the first two blocks, so those are the blocks with something to learn.

    **The anti-anchor term.** This is $\mathrm{mean}(\max(-\alpha, 0))$ over the same live positions and slices the anti-subspace term reads, at a constant weight of 0.1. It is a hinge on negative alignment, so it keeps clean states out of the hemisphere where the fallback lives, and being one-sided it does not oppose the anchor. The anti-subspace term already penalizes $\alpha^2$ there, so the arms that drop one term or the other say whether the hinge adds anything.

    ### The designed target

    Removing the concept operand leaves the visible operand and an unknown partner. The least committal answer is the distribution of the mix over every partner the corpus allows, which we call the *operand-averaged null*. On this grid, each channel of the visible operand has three closed partners, and their three mixes are distinct, so the null is uniform over 27 colors and has no mode.

    Its per-channel median is the mix with the middle partner. That coincides, at every level, with the visible operand mixed with mid-gray (7.5 on the 16-level channel) and rounded to the nearest grid level; `experiment.py` asserts the coincidence. That color is the target, and it is the gray fallback from M1 carried over.

    The target depends on the visible operand alone. If both operands of a red line are red, the redder one is the concept operand and the other is the visible one.

    ### Measurements

    We run one teacher-forced pass per run per intervention, over all 5,832 lines, through the eval contract in [`sca.intervention`](/src/sca/intervention.py). Each pass gives the log-softmax at the `=` position, from which we read the probability on the correct answer and on the target, the argmax, and the mass outside the color vocabulary.

    Each pass also gives the write per (slice, line, position), as in ex-2.2.1, and the clean alignment map. Target accuracy, seed agreement, and the composition all come from the argmax on red lines. The training trajectory records the fallback and anti-anchor losses beside the anchor loss and the margin, every 50 steps.

    **Noise floor.** As in ex-2.2.1: for each group statistic we take the pooled between-seed standard deviation per condition, computed before any comparison is read. A difference between conditions or arms smaller than twice that value is reported as not resolved. Gates score seed means against fixed thresholds and do not use the floor.

    ### Budget

    Twenty-four training runs of the ex-2.1.10 recipe, each with a second forward pass per step for the fallback term, plus the scoring of twelve stored checkpoints. The scoring is CPU work, at about a second per run per intervention.
    """)
    return


if __name__ == "__main__":
    app.run()
