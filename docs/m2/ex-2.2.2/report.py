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
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    # The design constants come from `experiment.py` beside this notebook (Marimo
    # puts the notebook directory on sys.path). The skeleton quotes the frozen
    # gates in prose, and that module carries the same numbers with each gate's
    # wording in its docstring.
    from mini.reports import report_bundle, use_publisher
    from mini.vis import light_dark, themed

    use_publisher(report_bundle(__file__))

    # A cell renders its last expression, and a trailing docstring is one.
    None


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.2.2: a designed response to suppressing *red*

    /// tip |
    <!-- tl;dr -->
    We add fallback control: a training term that teaches the blocks what to answer once the concept is gone, aiming at a designed fallback answer, with the concept's placement kept out of its gradient. Does the fallback appear, does the intervention stay effective and selective, and does seed variability fall?
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Findings

    - [The designed response (H1)](#the-designed-response-h1) —
    - [Task and placement intact (H2)](#task-and-placement-intact-h2) —
    - [Selectivity kept (H3)](#selectivity-kept-h3) —
    - [Transfer from the antipode to zero (H4)](#transfer-from-the-antipode-to-zero-h4) —

    <!-- REVIEW: a seed-agreement hypothesis (then H2) was cut: at the H1 gate it is close to implied by H1, and the case that separates them (seeds agreeing on an answer other than the fallback answer) is what E1's composition shows. Seed agreement is reported under E1, ungated. The later hypotheses moved up one number. -->
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this draft
    This is a preregistration. Each hypothesis section opens with its frozen prediction, and the `TODO` under it says what its figure or table will show. The hypotheses, their gates, and the method were frozen at commit `d72340e7`, before any run. Results replace the placeholders in place, and anything conceived after seeing the data goes under [Exploratory analyses](#exploratory-analyses), marked as post hoc.

    No run of this experiment has happened. The gates come from the published reference numbers in ex-2.2.1, the task-gate width the D2.1 experiments used, and the fallback result from M1.
    ///

    ## Why this experiment

    [Ex-2.2.1](../ex-2.2.1/report.py) showed that projecting the anchor axis out of the D2.1 checkpoints suppresses *red*. Red-line accuracy falls from 1.0 to 0.09, in proportion to how red the line is, and stays inside the bound the placed geometry set.

    It also measured what the model does instead. A red line still decodes to a color, about half the time a one-step neighbor of the true mix, and which neighbor it is varies by seed: at least five of the nine seeds agree on 13% of red lines. Nothing in training said what a removed concept should decode to, so the answer is whatever the untrained region of the stream happens to produce. M1 called that spoofing, and it is where the seed spread in [ex-2.9.1](/docs/m1/ex-2.9.1/report.py) came from.

    The remedy in M1 was fallback control ([ex-2.9.2](/docs/m1/ex-2.9.2/report.py)): one loss term, applied to the decoder only, that pins the antipode of the anchor axis to a designed null answer, mid-gray. Reflecting a state through the axis then collapsed the response to a tight cluster on the bound the null predicted, across 32 seeds. Under plain zeroing the term added little, and that report said a trained fallback should be paired with the redirect it was trained at.

    Here we bring the term into the transformer, keeping the D2.1 grammar and recipe. This comes before the operation work in the [D2.2 plan](../d2.2/design.md) changes the grammar.

    The term is one loss, but it touches the model in three places:
    the edit at the embedding, where every position is reflected through the axis and then held fixed;
    the loss, read at the `=` position of each red line;
    the gradient, which reaches the four blocks and the unembedding but not the embedding.
    """)
    return


@app.cell(hide_code=True)
def _():
    @themed(
        name="fallback-term",
        alt_text="""
            A schematic of the residual stream as a grid: four columns for the positions of a red line (red, +, blue, =), and rows from the embedding at the bottom through blocks 1 to 4 to the unembedding, which has a logits box at the = position only. The embedding row is outlined in orange and labelled as the edit, reflecting every position; an orange dashed line above it is labelled stop-gradient. The block and unembedding rows are shaded blue. A blue arrow points down into the logits box from a label reading loss, cross-entropy at = against the fallback answer, and a second blue arrow runs down the right margin from the unembedding to the dashed line, labelled gradient reaches the blocks and unembedding.
        """,
        caption="""
            **Where the fallback term acts.** One red line, `red + blue =`, as positions (columns) against slices (rows). The numbered annotations are the three places above; the blocks run forward from the edited embedding as usual, mixing positions through attention, and the dashed line is where the gradient stops.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
        ink = light_dark("#333", "#ccc")
        faint = light_dark("#0002", "#fff2")
        edit = light_dark("#b5551d", "#ffab6e")
        grad = light_dark("#2a6fdb", "#8fbaff")
        cols = ["red", "+", "blue", "="]
        rows = ["embedding\n(slice 0)", "block 1", "block 2", "block 3", "block 4", "unembedding"]
        top = len(rows) - 1
        w, h = 0.72, 0.5
        right = len(cols) - 0.5
        # The rows the gradient reaches: the blocks and the unembedding.
        ax.add_patch(Rectangle((-0.5, 0.5), len(cols), top + 0.5, color=grad, alpha=0.08, lw=0, zorder=0))
        for j, name in enumerate(rows):
            for i, tok in enumerate(cols):
                if j == top and i != 3:
                    continue  # only the `=` position is decoded
                ax.add_patch(
                    FancyBboxPatch(
                        (i - w / 2, j - h / 2),
                        w,
                        h,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        fc=light_dark("#fff", "#1c1f1e"),
                        ec=edit if j == 0 else ink,
                        lw=1.2 if j == 0 else 0.8,
                        zorder=2,
                    )
                )
                label = tok if j == 0 else ("logits" if j == top else "")
                ax.text(i, j, label, ha="center", va="center", fontsize=8, color=ink, zorder=3)
            ax.text(-0.65, j, name, ha="right", va="center", fontsize=8, color=ink)
        # The stream between rows, per position.
        for i in range(len(cols)):
            end = top if i == 3 else top - 1
            ax.plot([i, i], [h / 2, end - h / 2], color=faint, lw=3, zorder=1, solid_capstyle="butt")
        # The stop-gradient, between the embedding row and block 1.
        ax.plot([-0.5, right], [0.5, 0.5], color=edit, lw=1.4, ls=(0, (4, 2)), zorder=4)
        ax.text(right + 0.15, 0.5, "stop-gradient", ha="left", va="center", fontsize=8, color=edit)
        ax.text(
            right + 0.15,
            0,
            "1. edit: reflect every position\n(α ↦ −α), then detach",
            ha="left",
            va="center",
            fontsize=8,
            color=edit,
        )
        ax.annotate(
            "2. loss: cross-entropy at =\nagainst the fallback answer",
            xy=(3, top + h / 2),
            xytext=(3, top + 1.05),
            ha="center",
            va="center",
            fontsize=8,
            color=grad,
            arrowprops=dict(arrowstyle="-|>", color=grad, lw=1),
        )
        # The gradient runs down the right margin and stops at the bar.
        ax.annotate(
            "",
            xy=(right + 0.08, 0.62),
            xytext=(right + 0.08, top),
            arrowprops=dict(arrowstyle="-|>", color=grad, lw=1.4),
        )
        ax.text(
            right + 0.15,
            (top + 0.6) / 2,
            "3. gradient reaches the\nblocks and unembedding",
            ha="left",
            va="center",
            fontsize=8,
            color=grad,
        )
        ax.set_xlim(-2.2, len(cols) + 1.9)
        ax.set_ylim(-0.6, top + 1.7)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig

    mo.md(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In M1 the decoder took a constant input, the antipode itself, so the term was decoder-only by construction. Here a stop-gradient[^sg] gives the reflected state that same role.

    [^sg]: An identity in the forward pass with a gradient of zero (`.detach()` in PyTorch), so a loss downstream of it cannot move anything upstream of it. The other losses see the clean pass and are unaffected.

    The fallback answer for a continuous concept also has to be chosen. We use the center of the operand-averaged null, which works out to be the visible operand mixed with *mid-gray*; the [method](#the-fallback-answer) derives it.

    This addresses the third risk in the plan, that the response to suppression is undesigned. One mismatch carries over from M1: the response is trained at the antipode, while the ex-2.2.1 projection lands the state at zero. H4 measures how far the response transfers between the two.

    The nearest published analogue is LUNAR (arXiv:2502.07218): one matrix edit after training, redirecting the activations of the data to forget into the model's own refusal region. LUNAR designs the response by choosing a region the model already produces; ours is trained at a state the model never otherwise visits, which is where the mismatch above comes from. E6 fits a LUNAR-style edit to the no-fallback checkpoints, so the two designs can be compared.

    **Natural language.** The three parts all transfer to the natural language domain:
    the edit would be the reflection at the first anchored slice, applied at every position;
    the loss would be computed on the continuation (the tokens of the response), rather than at one `=` position;
    the gradient stops before the edit.
    The edit happens based on alignment of the states to the anchor, so it's grammar-agnostic and token labels are not needed.

    The fallback answer still has to be chosen; some candidates:
    **(a)** a template continuation, so the training data is pairs of context and designed response;
    **(b)** the model's own refusal region, which is the LUNAR choice (and which could be anchored separately);
    **(c)** a true null, the continuation a reference model gives when it has never seen the concept, matched with a KL term.[^kl]

    Spoofing in that setting is confabulation: a plausible answer drawn from the neighbors of the concept. The *mid-gray* answer here is a trained abstention that keeps the model on task (rather than, say, emitting a syntax token like `+` or `=` where the answer should be). A natural language version would want that same property.

    [^kl]: A KL term penalizes the distance between the next-token distribution of the model and that of the reference.
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
    - **dose** — how *red* a line is, taken as the larger of the two operand rednesses ($r(1 - g/2 - b/2)$ on the unit cube).
    - **red lines** — dose ≥ 0.8. 365 probe lines. **Non-red lines** — dose ≤ 0.2. 1,689 lines.
    - **concept operand** — the redder of the two operands in a line (ties go to op1); the **visible operand** is the other one, and is *clean* when its redness is below 0.5.
    - **margin** (`m_span`) — the pooled alignment margin of ex-2.1.10: at each slice, how much more aligned the red states are than the rest, taken at the best span role, then averaged over slices.
    - **fallback answer** — the designed answer for a *red* line under intervention: the visible operand mixed with *mid-gray*, rounded to the grid. This is the center of the operand-averaged null.
    - **fallback accuracy** — the fraction of *red* lines whose argmax at `=` is the fallback answer.
    - **response** — the probability the model puts on the correct answer, read from the log-softmax at the `=` position. **Damage** is the clean response minus the intervened response, per line. **Deficit** is the clean exact-match accuracy minus the intervened accuracy over a group of lines; it is the statistic ex-2.2.1 gated selectivity on.
    - **seed agreement** — the fraction of *red* lines on which at least five of the nine seeds decode the same answer, under a given intervention.
    - **redirect** — reflecting the embedding state through the axis, α → −α, at every position. The edit at a position is twice its alignment, so a state with nothing on the axis does not move. This is the state the fallback was trained at.
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

    Every hypothesis scores against the fallback condition. The no-fallback condition is the reference for it under every intervention.

    The arms below have three seeds each. We report the same statistics for them, without gates.

    | arm | what changes | question |
    | --- | --- | --- |
    | `fb-only` | no anti-anchor term | does the fallback need the antipode hemisphere cleared for it? |
    | `anti-only` | no fallback term | does the anti-anchor term alone move the response? |
    | `fb-w0.01`, `fb-w0.25` | the fallback weight, a factor of five either side | is 0.05 on a plateau? |
    | `recipe` | both new weights at zero | does the new code path reproduce the ex-2.1.10 checkpoints? |

    ### The interventions

    All of these run through the projection operator of the eval contract: it removes a fraction γ of the e₁ component, then puts the state back on the sphere. At γ = 2 the operator is a reflection, flipping the component instead of shrinking it.

    - **`redirect`** — γ = 2 at the embedding, at every position, with no labels. This is the same edit training applied, so H1 reads the readout the term trained. Read by H1 and the first clause of H3.
    - **`projection`** — the ex-2.2.1 intervention: γ = 1 at every slice and every position. The fallback was never trained under it. Read by the second clause of H3 and the projection row of H4.
    - **The transfer sweep** — γ ∈ {0.5, 1, 1.5, 2} at the embedding, every position. Its γ = 2 point is `redirect`, and its γ = 1 point is the `embedding` arm of ex-2.2.1, which differs from `projection` in leaving the later slices alone. Read by H4.
    - **Ride-along rows** — the `operands`, `shaped`, and `ablate` arms from ex-2.2.1, run on the fallback condition without gates. That way the intervention-tuning pass in the design has the fallback rows to hand (E5). `operands` is the one row here that needs position labels; it stays so we can compare with ex-2.2.1.

    <!-- REVIEW: earlier drafts restricted `redirect` to operand positions, then to the concept operand of each line, to match a training term that reflected one state per line. Neither is an edit a model without labelled positions can run. Now `redirect` reflects every position at the embedding, training reflects the same, and the red lines with no defined fallback answer (a visible operand that is itself red) leave the term and the gate of H1 rather than the edit. Verify: the training paragraph in the method, and VISIBLE_RED_DOSE in experiment.py. -->
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The designed response (H1)

    <!-- REVIEW: 298 clean / 67 both → 296 / 69. The frozen count read `redness < 0.5` without roundoff slack, so one color on the 0.5 contour (cf96, whose redness evaluates to 0.49999999999999994) landed on the clean side while its five siblings at the same redness landed on the red side. The rule is unchanged ("≥ 0.5 is red"); applying it to all six moves two lines. Verify: `read_lines` in experiment.py asserts the counts. -->
    **H1.** Under `redirect`, the model's answer on red lines is the fallback answer. Seed-mean fallback accuracy over the red lines with a clean visible operand (296 of the 365) is at least 0.8; partial: between 0.5 and 0.8. The other 69 red lines have no defined fallback answer, so we report them beside the gated figure, unscored. The reference row is the no-fallback condition under the same intervention. The fallback figure has to sit above that reference by a resolved margin, meaning two pooled between-seed standard deviations. A figure that clears 0.8 without a resolved margin counts as partial. Contrary: fallback accuracy at the no-fallback level, which would mean the term did not train the readout. The fallback loss over training (see the method) says whether the term was ever active.

    /// admonition | TODO
    A table of fallback accuracy and true-answer accuracy on red lines under `redirect`, seed mean with the seed range, for the fallback and no-fallback conditions and the un-anchored calibration row, with the 69 lines whose visible operand is red as their own row. Beside it, per seed, the probability the model puts on the fallback answer on red lines, drawn as a strip. That way a split response, with some lines at the fallback answer and some elsewhere, shows up as such instead of averaging into the gate.
    ///

    ## Task and placement intact (H2)

    **H2.** The term costs nothing on the task or the placement. Clean exact-match accuracy on all probe lines is within 0.02 of the no-fallback condition, and the seed-mean alignment margin at the end of training, the `m_span` of ex-2.1.10, is at least 0.8 of the no-fallback value. Partial: exactly one of the two holds, or both hold with accuracy read at the wider 0.05 band.

    Contrary: the fallback term competing with the anchor for the operand states, which is what the stop-gradient is supposed to prevent.

    <!-- REVIEW: H2 (then H3) gave two partial rules for the same clause ("partial: within 0.05" inline, and "one of the two holds" after), which do not decide a run whose accuracy sits between 0.02 and 0.05 with the margin intact. Merged into one rule; both bands are unchanged, and `TASK_PARTIAL` keeps its documented role. -->

    <!-- REVIEW: the same merge applies to H3's first clause, which quoted only the 0.02 gate although `TASK_PARTIAL`'s docstring assigns it a 0.05 partial band there too. -->


    /// admonition | TODO
    Clean accuracy on all, red, and non-red lines for every condition and arm. Then the alignment trajectory (margin over training) for the fallback condition drawn over the no-fallback condition, nine thin lines each. Beside it, the fallback loss and the anti-anchor loss over training, which check that the term switched on when the anchor placed the concept.
    ///

    ## Selectivity kept (H3)

    **H3.** Under `redirect`, the seed-mean non-red deficit stays at or below 0.02. Under `projection`, the non-red deficit in the fallback condition is not resolved above the no-fallback figure (0.024 in ex-2.2.1, re-measured here).

    On a non-red line the operands have nearly nothing on the axis, so the first clause rests on the syntax embeddings. Zeroing their constant component is what cost the ex-2.2.1 projection its 0.024, and the reflection flips it instead. Training runs the reflected pass on every crop that carries a qualifying red line, so the blocks see flipped syntax states, and read them as syntax, throughout training. We predict that this transfers to the non-red lines the term never scored.

    Partial: the first clause holds only at the 0.05 band, or exactly one clause holds.

    Contrary on the first clause: a deficit at or above the ex-2.2.1 figure, which would say the flipped syntax reads as a different syntax, and would send the intervention-tuning pass toward a thresholded reflection, the `shaped` falloff at strength 2. Contrary on the second clause: the term widens the edit, which would mean the blocks now read the axis at positions or slices where they did not before.

    <!-- REVIEW: H3 (then H4) was written on non-red *damage*, the probability statistic. The 0.02 width and the 0.024 reference both come from ex-2.2.1's H2, which gated the accuracy *deficit* (its H2 line: "outside the 0.02 gate but inside the 0.05 partial gate"), so the gate and its reference were being read on a statistic they were not set for. Now stated on the deficit, with the same widths, and the partial band written out. Verify: ex-2.2.1's H2 section, and the "non-red deficit" column of its arms table. Damage is still reported beside it in the table below. -->

    /// admonition | TODO
    The non-red deficit and non-red damage under `redirect` and under `projection`, seed mean and range, for the fallback and no-fallback conditions and for every arm. Also the ex-2.2.1 write-bound map for the fallback condition under `projection`, on the shared 0–1 scale, so a wider edit shows where it acts.
    ///

    ## Transfer from the antipode to zero (H4)

    **H4.** The designed response transfers (generalizes) part of the way to zero. Along the transfer sweep, γ ∈ {0.5, 1, 1.5, 2} at the embedding, seed-mean fallback accuracy on red lines is non-decreasing in γ, allowing a dip of at most 0.02 between adjacent strengths, and at γ = 1 it is at least half its value at γ = 2. Partial: either clause holds on its own — monotone with γ = 1 below half, or γ = 1 at half or more with a dip larger than the allowance.

    The γ = 1 point is a real removal, not an inert midpoint. It is the `embedding` arm of ex-2.2.1, where zeroing the axis at the embedding alone took red accuracy down to 0.16. The sweep adds how the fallback responds along the way from the trained state to that one. At γ = 1 the operator keeps whatever the state carries off the axis and rescales it by $1/\sqrt{1-\alpha^2}$, so the landing direction is defined as long as the state is not perfectly aligned, and no state is: pure red arrives at the embedding at about α = 0.9, a gain of about 2.3.

    Contrary: fallback accuracy at γ = 1 sits at the no-fallback level and the rise is confined to γ > 1. That would be the mismatch showing in full, with the response living at the antipode and not reaching the projected state. The [concept swap](/todo/science/redirect-between-two-anchored-ops.md) filed for D2.3 would address it by targeting a state training already visits; widening the bracket here would not.

    <!-- REVIEW: H4's (then H5's) partial band covered only the monotone-but-short case, leaving no verdict for a run that transfers to γ = 1 through a dip; both single-clause cases are now partial. The D2.3 sentence was in the present indicative ("the remedy is"), which reads as a scheduled follow-up; softened to the conditional, since that item is a backlog entry. -->

    We report the `projection` row beside the sweep, without a gate: γ = 1 at every slice rather than at the embedding alone. The difference between the two says how much the projection at later slices costs the designed response.

    /// admonition | TODO
    Fallback accuracy and true-answer accuracy against γ, seed mean with the seed range, fallback drawn over no-fallback, with the `projection` row as a separate mark at γ = 1. The x axis can also be drawn as the landing alignment of a pure-red operand, which has a closed form in γ.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    /// admonition | TODO
    Preregistered as exploratory, no gates.

    **E1 — composition.** What red lines decode to under `redirect` and under `projection`: the fallback answer, the true answer, a one-step neighbor of it, the visible operand, the red operand, or something else. Drawn as a stacked bar per seed, fallback beside no-fallback. These are the six categories from ex-2.2.1 with the fallback answer added. Also the mass outside the color vocabulary, and how far the decoded answer sits from the fallback answer and from the true mix. Seed agreement goes here too, the statistic ex-2.2.1 read at 13% under the projection, per condition and intervention, and per red line how many seeds decode its plurality answer, whose mean over lines is the threshold-free companion to the five-of-nine count.

    **E2 — the antipode.** The fraction of clean states with negative alignment, per slice and position, on the fallback, `fb-only`, and `anti-only` conditions. This is what the anti-anchor term is for, and it also says whether the anti-subspace term had already done the job.

    **E3 — off-axis recoverability.** This is the auditing row that the [D2.2 design](../d2.2/design.md) assigns here. We fit a ridge probe for the redness of the concept operand on the intervened operand states, per slice, under `projection` and under `redirect`, five-fold over lines. We report held-out R² for the fallback and no-fallback conditions beside the clean states, and beside the floor the task itself sets: the same ridge fit to the raw RGB values, which ex-2.1.7 put at 0.863. The number sizes what a linear readout can still find, and [it does not bound the intervention](/todo/science/off-axis-probe-r2-does-bound-intervention.md); the read here is the fallback figure against the no-fallback one, resolved or not.

    If the fallback raised the off-axis R², that would mean it keeps red readable off the axis so that it knows when to emit the fallback answer. Red would then be masked rather than removed. It should not need to do this, since the fallback answer depends only on the visible operand. Nine seeds resolve only a shift larger than the noise floor, so a small shift may read as not resolved. If that happens, we file a follow-up at more seeds rather than reading the row either way.

    **E4 — arms.** The H1–H3 statistics for `fb-only`, `anti-only`, the weight bracket, and `recipe`. The `recipe` row is a regression check against the stored no-fallback checkpoints, at three seeds, on clean accuracy, margin, red accuracy under `projection`, and non-red damage.

    **E5 — ride-along interventions.** The H1 and H3 statistics for the `operands`, `shaped`, and `ablate` rows on the fallback condition, beside the ex-2.2.1 figures for the same rows on the no-fallback condition.

    **E6 — a LUNAR-style redirect, fitted after training.** We freeze each no-fallback checkpoint and fit one 64×64 matrix at the embedding, applied at every position, on the same qualifying red lines the fallback term uses. Two loss parts: the fallback cross-entropy at `=`, plus an identity term on every other state (the mean squared distance from the unedited state). The fitted matrix runs through the eval contract like any other operator, and we report the H1 and H3 statistics for it beside the fallback condition under `redirect`. This model has no refusal region of its own, so the fallback answer stands in for it.
    ///

    ## Discussion

    /// admonition | TODO
    Interpretation only, after the results. Whether the fallback in the transformer gives the intervention a designed outcome the way it did in M1. How far it transfers from the trained state to the projected one, and what that says the anchored-op experiments should train toward. Whether the anti-anchor term earns its place in the recipe. What the off-axis row says about masking, read against the [Most Forbidden Technique paragraph](../d2.2/design.md#fallback-control) in the design, and what the fitted edit of E6 says about designing at a region the model already has. No re-derivation of the findings.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    ### Training

    We use the primary recipe from [ex-2.1.10](../ex-2.1.10/report.py) unchanged: the `v216` corpus, d64-L4, 100 epochs with a 10-epoch warm-up, the pooled either-operand labeller at τ = 0.1, the anchor at λ = 0.1 with its anneal, and the anti-subspace term at the ex-2.1.8 operating point. The [ex-2.1.11](../ex-2.1.11/report.py) survey proposed a different point. But survey numbers are proposals until they are re-measured at fresh seeds, and the design schedules that confirmation on the multi-op grammar. Keeping the ex-2.1.10 recipe also keeps its stored checkpoints as the no-fallback condition, so nothing has to be retrained for the comparison.

    We also use the same seeds as the primary of that experiment, so the fallback and no-fallback conditions differ only in the two new terms. Both are weights on the existing anchored train step.

    **The fallback term.** On each training crop, we reflect the embedding state of every position through the axis, $h \mapsto \mathrm{normalize}(h - 2\alpha\, e_1)$, and hold the reflected states fixed with a stop-gradient. We then run the blocks forward from there and take the cross-entropy at the `=` position of each qualifying line against its fallback token. A line qualifies on three counts: its dose is at least 0.8, the redness of its visible operand is below 0.5, and the clean embedding alignment of its concept operand is at least 0.5. The reflection singles out no positions, so it is the same edit `redirect` applies at eval, and it applies unchanged to a model whose positions are not labelled.

    **Compute.** The reflected pass is a second forward and backward through the blocks, so a step that carries a qualifying line costs about twice a plain step. A crop holds about ten lines, and about one line in twenty qualifies, so roughly 40% of crops carry one. Run per batch, that is nearly every step, and the budget below counts the term as a doubling. Three things should make it cheap at scale. The pass can be skipped when no position clears the alignment threshold, which the clean pass at the edit slice already reports, so it costs nothing before the anchor has placed the concept. It can run on a subsample of the qualifying sequences, or on every k-th step, since the term is a small weight on a slowly moving target. And it reruns only the part of the model after the edit, so an edit deeper than the embedding shares the earlier layers with the clean pass.

    The term is the mean over qualifying lines, at a constant weight of 0.05 from step 0. We take that weight from M1, as a starting point rather than a derived value: there it scaled a mean squared error on one decoder, and here it scales a cross-entropy through four blocks. Checking it is what the bracket arms are for.

    The alignment threshold keeps the term inert until the anchor has placed the concept, which the ex-2.1.10 trajectories put inside the warm-up. If the concept never arrived at the edit slice, the term would stay inert rather than train toward the wrong state, and the margin gate of H2 would report the anchoring failure.

    <!-- REVIEW: the frozen text said the embedding table gets no gradient from the term. nGPT ties the unembedding to the embedding table, so the table does see the term through the readout. Corrected to say what the detach does and does not cut; the design (reflect once, detach, train what follows) is unchanged. Verify: `reflected_logits` in src/sca/fallback.py multiplies by `wte.T`. -->
    **Where the term acts.** The stop-gradient makes this similar to the decoder-only term from M1: the reflected embedding states are detached, so nothing flows back through them to the placement of *red* at the edit, while the four blocks and the readout do learn. The readout is the embedding table itself, since nGPT ties the two, so the table sees the term through the readout only: the answer rows move toward the reflected pass's final state, and every other row, *red*'s included, moves a little away from it. That is a decoder-side nudge on the rows rather than a pull on where the concept sits, and the margin gate of H2 is where it would show. Together the blocks and the readout learn a map from a state at the antipode to the fallback answer. In ex-2.2.1 the concept was read from the operand states in the first two blocks, so those are the blocks with something to learn.

    We reflect once, at the first anchored slice, which in this recipe is the embedding. Reflecting at every slice, as the ex-2.2.1 projection does, would need a detach at every slice, and then only the layers after the last detach would train; reflecting everywhere and detaching only the first edit is the rehearsal the design rejected.

    The stop-gradient does leave one thing open. The blocks after the edit are shared with the clean pass, so the term can reshape how the concept is carried downstream of the edit, including keeping it readable off the axis. That is what the off-axis row (E3) and the margin gate watch.

    **The anti-anchor term.** This is $\mathrm{mean}(\max(-\alpha, 0))$ over the same live positions and slices the anti-subspace term reads, on the clean forward pass only, at a constant weight of 0.1. The reflected states that the fallback term produces are not live positions of that pass, so the two terms do not pull against each other. The term is a hinge on negative alignment: zero when the alignment is positive, and growing linearly as it goes negative. So it keeps clean states out of the half of the space where the fallback lives, and because it acts on one side only, it does not oppose the anchor. The anti-subspace term already penalizes $\alpha^2$ there, so the arms that drop one term or the other say whether the hinge adds anything.

    ### The fallback answer

    Removing the concept operand leaves the visible operand and an unknown partner. The least committal answer is the distribution of the mix over every partner the corpus allows, which we call the *operand-averaged null*. On this grid, each channel of the visible operand has three closed partners, and their three mixes are distinct, so the null is uniform over 27 colors and has no mode.

    Its per-channel median is the mix with the middle partner. That coincides, at every level, with the visible operand mixed with mid-gray (7.5 on the 16-level channel) and rounded to the nearest grid level; `experiment.py` asserts the coincidence. That color is the fallback answer, and it is the gray fallback from M1 carried over.

    The fallback answer depends on the visible operand alone, so it is defined only when that operand is clean. On a red line whose visible operand is itself red (redness ≥ 0.5), both operand states are reflected. The answer would then be the mix with both operands replaced by mid-gray, which is 7.5 on every channel, halfway between grid levels 6 and 9. With nothing visible, the null has no center on the grid; when one operand is visible, its level always breaks the tie. Those 69 lines take no fallback loss and sit outside the gate of H1.

    ### Measurements

    We run one teacher-forced pass per run per intervention, over all 5,832 lines, through the eval contract in [`sca.intervention`](/src/sca/intervention.py). Each pass gives the log-softmax at the `=` position, from which we read the probability on the correct answer and on the fallback answer, the argmax, and the mass outside the color vocabulary.

    Each pass also gives the write per (slice, line, position), as in ex-2.2.1, and the clean alignment map. Fallback accuracy, seed agreement, and the composition all come from the argmax on red lines. The training trajectory records the fallback and anti-anchor losses beside the anchor loss and the margin, every 50 steps.

    **Noise floor.** As in ex-2.2.1: for each group statistic we take the pooled between-seed standard deviation per condition, computed before any comparison is read. A difference between conditions or arms smaller than twice that value is reported as not resolved. Gates score seed means against fixed thresholds and do not use the floor.

    ### Budget

    Twenty-four training runs of the ex-2.1.10 recipe, each at about twice the cost of a plain run (see [Compute](#training)), plus the scoring of twelve stored checkpoints. The scoring is CPU work, at about a second per run per intervention. E6 adds nine fits of a 64×64 matrix, one per no-fallback seed, at a few CPU minutes each.
    """)
    return


if __name__ == "__main__":
    app.run()
