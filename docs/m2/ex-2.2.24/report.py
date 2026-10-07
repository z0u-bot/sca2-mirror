# ruff: noqa: B018
# title: Ex 2.2.24: Where the edit spills from

# The design constants come from `experiment.py` beside this script (the directory of the script is on sys.path
# while it runs). This is the preregistration draft: the only computed figure is the lightness of the answers of
# each op, from the op rules alone.
import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.vis import figure_html, light_dark, themed

# --- Helpers -------------------------------------------------------------------------------------------------


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str, *, text_cols: int = 1) -> str:
    """An authored table in the shared report style; the first *text_cols* columns are text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>"
        + "".join(f"<td{' class=num' if i >= text_cols else ''}>{cell_html(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def num_word(n: int) -> str:
    return ("none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")[n]


ROLE_NAMES = ("op1", "`?`", "op2", "`=`", "answer", "`,`")


def role_name(r: int) -> str:
    i, j = divmod(r, ex.UNIT)
    unit = "query" if i == ex.K else f"example {i + 1}"
    return f"{unit} {ROLE_NAMES[j]}"


def sites_table() -> str:
    rows = [[s.name, ", ".join(role_name(r) for r in s.roles), s.why] for s in ex.SITES]
    rows = [[r[0], r[1] if len(r[1]) < 60 else "every other position before the query `=`", r[2]] for r in rows]
    return table_html(
        ["site", "positions", "why"],
        rows,
        caption="**The sites of H2.** Together they cover every position up to and including the query `=`, where the answer "
        "is read. The edit at every position, as ex-2.2.23 scored it, is repeated beside them.",
        text_cols=3,
    )


# --- The answers of each op, from the rules alone ------------------------------------------------------------


@memo
def answer_lightness() -> dict[str, np.ndarray]:
    """The lightness of the unrounded answer of each op, over every ordered pair of grid colors, and of the operands."""
    from sca.data.ops import CANDIDATE_BY_NAME, OP_BY_NAME, TOP, colors

    ops = OP_BY_NAME | CANDIDATE_BY_NAME
    cs = colors()
    out = {name: np.array([np.mean(ops[name].raw(a, b)) / TOP for a in cs for b in cs]) for name in ex.OP_NAMES}
    out["operands"] = np.array([ex.lightness(c) for c in cs])
    return out


def lightness_figure() -> str:
    data = answer_lightness()
    caption = f"""
        **How light the answers of each op are**, from the op rules over every pair of grid colors, before rounding.
        Dots are means; bars span the middle half. The top row is the operands, for reference. `{ex.ANCHORED_OP}`
        is highlighted.
    """
    alt = f"""
        A horizontal chart with one row per op and one for the operands, showing the mean lightness of the answers and
        the middle half of their spread. The operands sit at one half. `{ex.ANCHORED_OP}` answers are darker on average,
        between `mix` and `darken`; `lighten` is the lightest and `darken` the darkest; the HSV ops sit with the operands.
    """
    return lightness_draw({k: v.tolist() for k, v in data.items()}, caption, alt)


@memo
def lightness_draw(data: dict[str, list[float]], caption: str, alt_text: str) -> str:
    @themed(name="ex-2.2.24-answer-lightness", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        rows = ["operands", *ex.OP_NAMES]
        fig, ax = plt.subplots(figsize=(4.6, 2.6), layout="constrained")
        ink = light_dark("#333", "#ccc")
        accent = light_dark("#b4531f", "#e39a6a")
        for i, name in enumerate(rows):
            x = np.asarray(data[name])
            c = accent if name == ex.ANCHORED_OP else ink
            lo, hi = np.percentile(x, [25, 75])
            ax.plot([lo, hi], [i, i], color=c, lw=2.2, alpha=0.5, solid_capstyle="butt")
            ax.plot([x.mean()], [i], "o", color=c, ms=5, mew=0)
        ax.axvline(float(np.mean(data["operands"])), color=ink, lw=0.6, alpha=0.4, ls=":")
        ax.set_yticks(range(len(rows)), rows, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("lightness (mean of the channels)", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        return fig

    return _plot()


# %%

rf"""
# Ex 2.2.24: Where the edit spills from

/// tip |
<!-- lede -->
A look inside the runs of ex-2.2.23, to find out why the edit spills onto other ops once the model has learned the HSV ops. We ask whether the example answers hold both the op and the color, which positions the spill comes from, and whether e₁ follows how light a color is. No new training is needed for these.
///

Ex-2.2.23 found that removing e₁ lowers `darken`, `lighten`, and `value-hsv` along with `{ex.ANCHORED_OP}` on nearly every run that has learned the HSV ops. This experiment scores the {len(ex.SEEDS) * len(ex.LENGTHS) * len(ex.CONDITIONS)} checkpoints that ex-2.2.23 trained, at 200 and 400 epochs, with three new measurements. An optional arm adds {num_word(ex.N_ARM_RUNS)} runs at {ex.ARM_EPOCHS} epochs.
"""

# %%

r"""
## Findings

- [A map of every position (E1)](#a-map-of-every-position-e1) — what we saw.
- [The op and the color share the example answers (H1)](#the-op-and-the-color-share-the-example-answers-h1) — **verdict**. One sentence.
- [The spill comes from the example answers (H2)](#the-spill-comes-from-the-example-answers-h2) — **verdict**. One sentence.
- [A position that removes without spilling (E2)](#a-position-that-removes-without-spilling-e2) — what we saw.
- [The anchored direction follows lightness (H3)](#the-anchored-direction-follows-lightness-h3) — **verdict**. One sentence.
- [The anchor at examples of other ops (E3)](#the-anchor-at-examples-of-other-ops-e3) — what we saw.
- [The anchor at the query `=` (E4)](#the-anchor-at-the-query-e4) — what we saw.
- [The schedules at the rise (E5)](#the-schedules-at-the-rise-e5) — what we saw.
- [Trained for 600 epochs (H4)](#trained-for-600-epochs-h4) — **verdict**. One sentence.

/// admonition | How to read this report
This report is a preregistration draft. Once the design is agreed, the predictions will be frozen at a commit quoted here, before any measurement of this experiment. Each section opens with what we expect, and the results will replace the placeholders in place.
///
"""

# %%

rf"""
## Why this experiment

The anchor pulls the states of `{ex.ANCHORED_OP}` contexts toward e₁, and the edit removes e₁ to take `{ex.ANCHORED_OP}` away. That works cleanly on a model that has learned only the first four ops. Once the model learns the HSV ops as well, the edit also lowers other ops, most of all `darken`, `lighten`, and `value-hsv`, and it does so more after 400 epochs than after 200 (ex-2.2.23). Those three ops all depend on how light their operands are. So it looks as though e₁ comes to hold some lightness along with the op, and the edit removes both.

There are two ways that could happen, and they may be two sides of one story.

The first is about where the anchor lands. The label marks a whole context, but most of the alignment ends up on the example answers (ex-2.2.16, ex-2.2.21). An example answer is a color, and it is also the evidence for the op. And the answers of `{ex.ANCHORED_OP}` are darker than most:

{lightness_figure()}

So a model that learns to put "this context uses `{ex.ANCHORED_OP}`" on e₁ at the answers might also learn to put "this answer is dark" there, since in `{ex.ANCHORED_OP}` contexts the two go together. Removing e₁ would then shift the lightness the model sees in the examples, or in the query.

A re-analysis of ex-2.2.22 (the example-evidence page) has since made the picture at the answers sharper. There, the alignment at an example answer follows how well that one example fits `{ex.ANCHORED_OP}`, given its operands, and hardly depends on the examples before it; the answer color alone explains little of it. So if lightness reaches e₁ at the answers, it would come through that judgement of fit rather than through the color. And a judgement of fit could also fire on examples of other ops whose answers happen to look like `{ex.ANCHORED_OP}`, so that removing e₁ reaches those ops by another route.

The second is about time. The anti-subspace term keeps everything except `{ex.ANCHORED_OP}` off e₁, and its weight eases off over training. The model learns the HSV ops late, while the term is easing off, and a 400-epoch run spends more steps learning while it is weak. The schedules are those of ex-2.1.10, chosen when *red* was anchored in a grammar of one equation per line, and have not been revisited since the in-context grammar.

This experiment tests the first story directly, and looks for signs of the second. If the spill comes from one kind of position, an edit that leaves those positions alone may be selective. If e₁ follows lightness even where it carries no evidence about the op, the anchor term alone is not keeping it clean, and the schedules are the next thing to look at.
"""

# %%

rf"""
## Runs

Every run of ex-2.2.23: the recipe of record (`anchor`) and the `control`, at {len(ex.SEEDS)} seeds, trained for 200 and for 400 epochs. Ex-2.2.23 found that every run made the second rise at 400 epochs, and that a few runs at 200 epochs had not; we call those *half-trained*, by ex-2.2.23's rule (the worst HSV op below {ex.RULE[1]:g}). The hypotheses are scored on the 400-epoch runs, and the 200-epoch runs are shown beside them, since the half-trained runs are the ones whose edit stayed clean.

The optional arm trains both conditions for {ex.ARM_EPOCHS} epochs at model seeds {" and ".join(map(str, ex.ARM_SEEDS))}, with the recipe otherwise unchanged; H4 says what it is for.

/// admonition | Open decision
Whether to run the {ex.ARM_EPOCHS}-epoch arm in this experiment. It costs about \$2 for {num_word(ex.N_ARM_RUNS)} runs (the controls are needed so the spill can be netted at the same seed and length). The alternative is to leave it for a later experiment, once H2 and H3 say what is worth measuring at that length. To check: whether the trend from 200 to 400 epochs is enough to act on, or whether a third length would change the next design.
///
"""

# %%

rf"""
## Glossary

The measurements this report uses. Each definition also shows in the margin beside the first use of its term in a section.

Spill
:   How much the edit lowers the other ops. For a run and an edit, it is the largest drop in expected exact match on any op other than `{ex.ANCHORED_OP}`, at any dose, net of what the same edit does to the control at the same seed and length. Ex-2.2.21 set a criterion of {ex.SELECTIVITY_GATE:g} on it; a run *spills* when it goes past that.

Removal
:   How far the edit at full dose moves the answers on `{ex.ANCHORED_OP}` contexts toward the target null, as in ex-2.2.22: the share of the distance from the clean model to the null that it closes. Ex-2.2.21 asked for at least {ex.GRADING_MIN_DAMAGE:.0%}.

Site
:   The positions an edit acts on. The edit always acts at every slice. H2 compares four sites that between them cover every position up to and including the query `=`.

Lightness
:   For a grid color, the mean of its three channels, from 0 (black) to 1 (white).

Op recovery
:   How well a linear probe reads the op of a context from the state at one position, on held-out contexts. A probe is a small classifier fitted on the states: if it can read the op, the op is there in a form the next layers could use. We give its accuracy as a share of what is possible, since an example answer only says so much about the op: (probe accuracy − 1/7) / (Bayes accuracy − 1/7), where the Bayes accuracy is that of the best guess given the examples up to that answer, and 1/7 is chance. 1 means the probe reads all the evidence there is, and 0 means none.

Lightness tracking
:   How closely the component of a state along e₁ follows the lightness of the color at that position. It is the squared correlation ($r^2$) between the two, over the operand positions of held-out contexts of the six ops other than `{ex.ANCHORED_OP}`, at each slice. We leave out `{ex.ANCHORED_OP}` contexts because the anchor pulls them, and their answers are dark, so on those contexts the two would go together by design.
"""

# %%

r"""
## A map of every position (E1)

Before the scored sections, a map of what each position holds, at every slice. Taking a measurement everywhere costs no more than taking it at one position, so we take three: the alignment with e₁, the op probe, and a probe for the color at that position (its three channels; at a syntax token, the color just before it, and at the query `=`, the answer to come). Each hypothesis below still scores one position, named in advance, so that the map cannot choose where a gate is read. The map is there to show whether the named positions are typical of their neighbors.

/// admonition | TODO
Three heatmaps per condition and length, position along the context against slice: the alignment with e₁ (seed mean, the control subtracted for the anchored runs), op recovery, and color $R^2$. A row of tokens under each heatmap marks the roles.
///
"""

# %%

# REVIEW: the lightness probe is a check, with no gate (review round 0). The state at an answer holds the embedding
# of its own color token, so a lightness R² above 0.5 could be predicted from the method alone. The op probe is the
# informative half and carries the verdict.
rf"""
## The op and the color share the example answers (H1)

This tests the premise of the first story on the control, which has never been pulled toward e₁: is the evidence an example gives about the op already present at its answer, along with the color? If it is, the anchor there takes up a place the model uses for both. This is evidence the line itself shows, so a pass would not mean the model has inferred the op there; E4 asks that at the query `=`. We score the first example answer, since there the evidence of that one example and the evidence of the examples so far are the same thing; the example-evidence page found the anchored runs following the first and hardly the second.

**What we expect.** At the first example answer of the 400-epoch controls, we expect a linear probe to read the op with a recovery of at least {ex.OP_RECOVERY_GATE:g}, at the best slice after the embedding. That would be a pass, and less a miss. If the seed mean clears the gate while fewer than three-quarters of the seeds do, the result would be outside the plan, and the verdict would be Unresolved.

Beside it, a second probe reads the lightness of the answer color. That the color is there is close to certain, since the state at an answer holds the embedding of its own token, so this is a check on the probes rather than a test.

/// admonition | TODO
Op recovery and lightness $R^2$ against slice, one panel each, at the three example answers and the query `=` (the readout, for reference), with the recovery at the later answers given against both the one-example posterior and the posterior so far, seed means with the seed range, for the control and the anchored runs at 400 epochs. E1 has the same probes at every position.
///
"""

# %%

# REVIEW: the query `=` is a site of its own (review round 0). It is the readout, downstream of every other site, so
# in "the rest" it would hide a route through the examples behind an edit at the answer itself. The sites differ in
# size, and an edit at one changes what later positions read from it, so the single-site spills need not add up.
rf"""
## The spill comes from the example answers (H2)

The edit at every position spills; here it acts at one site at a time. If the spill follows the anchor onto the example answers, editing them alone should spill about as much. If it comes through the query operands instead, a shift in the lightness of the colors being combined is the likelier route. The query `=` is where the answer is read, so a spill there alone would say the edit acts on the answer itself.

{sites_table()}

**What we expect.** We expect the example answers to account for the spill: on at least {ex.SITE_RUN_SHARE:.0%} of the 400-epoch anchored runs that spill, the edit at the example answers alone spills at least {ex.SITE_SHARE:.0%} as much as the edit at every position, and more than any other site. That would be a pass. If another site accounts for it instead, that would be a miss, and which one did is the finding. If no site alone reaches {ex.SITE_SHARE:.0%} on most runs, the spill needs the edit at several sites together; that would also be a miss. If the runs split between sites, with none accounting for the spill on {ex.SITE_RUN_SHARE:.0%} of them, the result would be outside the plan, and the verdict would be Unresolved.

/// admonition | TODO
Spill against dose for each site and for every position, one line per seed, anchored runs at 400 epochs, with the spill gate marked. Beside it, which op takes the largest drop at each site.
///

The sites are of different sizes, from one position to many, so a small site that spills as much as a large one says more. And an edit at one site also changes what later positions read from it through attention, so the spills of the sites need not add up to that of every position.
"""

# %%

rf"""
## A position that removes without spilling (E2)

Whether any one site, or any one position, gets most of the removal with little of the spill. Beside the four sites of H2, the edit acts on each position up to the query `=` alone, at full dose. The removal and the spill of each, per run, at both lengths. This is a measurement with no gate: H2 says where the spill comes from, and this says whether a narrower edit would be worth scoring as a recipe.

/// admonition | TODO
Removal and spill at full dose for the edit at each single position, as two strips along the context, seed means for the anchored runs at each length. Below it, removal against spill, one dot per run, a panel per site, with the removal criterion ({ex.GRADING_MIN_DAMAGE:.0%}) and the spill gate ({ex.SELECTIVITY_GATE:g}) drawn as lines, so the selective corner is visible. The half-trained runs are marked.
///
"""

# %%

# REVIEW: (b) is scored within the 400-epoch runs (review round 0). Pooled over lengths, a correlation could come
# from length alone, since the 400-epoch runs spill more. The operand state can hold the op posterior through
# attention; the claim is that the operand token carries no evidence, and its lightness is independent of the op.
rf"""
## The anchored direction follows lightness (H3)

This looks at e₁ at positions whose token says nothing about the op: the operands of contexts of the other six ops. The operands are drawn at random whatever the op is, so the lightness of an operand is unrelated to the op, even though the state there may hold what the model has inferred about the op from earlier examples. So if e₁ follows how light an operand is, the model has put lightness on e₁ for reasons other than the op, and removing e₁ would change the lightness of the colors the answer is computed from.

**What we expect.** We expect (a) the lightness tracking at the operands, averaged over slices, to be higher on the 400-epoch anchored runs than on the controls, by more than [the band](term:band); and (b) across the 400-epoch anchored runs, a rank correlation of at least {ex.TRACKING_RANK_GATE:g} between lightness tracking and spill. Both would be a pass, (a) alone a partial, and neither a miss. If (b) holds and (a) does not, e₁ would follow lightness about as much in the control, and the anchor would only decide how much that matters to the edit; that would be outside the plan, and the verdict would be Unresolved.

/// admonition | TODO
Lightness tracking against slice, for the control and the anchored runs at each length, seed means with the seed range, with the half-trained runs drawn on their own. Beside it, spill against lightness tracking, one dot per anchored run, at each length (the 200-epoch runs for comparison). A second panel shows the same at the example answers, with the one-example posterior on `difference` partialled out, since there the color is also evidence.
///
"""

# %%

rf"""
## The anchor at examples of other ops (E3)

The example-evidence page looked only at `{ex.ANCHORED_OP}` contexts, where the alignment at an example answer follows how well that example fits `{ex.ANCHORED_OP}`. Here we look at the example answers of the other six ops. If the model judges each example on its own, an example of `darken` or `lighten` whose answer happens to fit `{ex.ANCHORED_OP}` should sit on e₁ too, and the edit would change what the model infers from it. We compare the alignment at those answers with the one-example posterior on `{ex.ANCHORED_OP}`, per op, and ask whether the ops with the most alignment of this kind are the ones that spill.

/// admonition | TODO
Alignment at the example answers of each of the six other ops against the one-example posterior on `{ex.ANCHORED_OP}`, in bins, seed means for the anchored runs and the control at 400 epochs. Beside it, per op, the mean alignment at those answers against the spill on that op, one dot per op and run.
///
"""

# %%

rf"""
## The anchor at the query `=` (E4)

At an example answer, the line itself shows how well that example fits `{ex.ANCHORED_OP}`, so an alignment there need not mean the model has inferred the op. The query `=` is the one position where the line says nothing new about the op: its only evidence is the examples before it. So if the alignment there rises with the posterior on `{ex.ANCHORED_OP}` given the examples, the anchor holds the op the model inferred. This is the test the example-evidence page proposed next, and the one the backlog item on grading with the posterior waits on. Ex-2.2.23 already stores the alignment at the query `=` and the posterior for every held-out context, so it needs no new compute. Earlier experiments found the anchor weak at the query `=`, so a small rise is the likely baseline. The posterior takes only a few distinct values at three examples, so it is binned coarsely.

/// admonition | TODO
Alignment at the query `=` against the posterior on `{ex.ANCHORED_OP}` given the examples, over the held-out contexts of every op, in a few bins, per slice, seed means for the anchored runs and the control at each length.
///
"""

# %%

r"""
## The schedules at the rise (E5)

Where the second rise falls on the anchor and anti-subspace schedules, from the trajectories ex-2.2.23 recorded, with no new compute. The time story says the model learns the HSV ops after the anti-subspace term has eased. This shows whether it does, at each length, and how the lean (the mean alignment of the states of every context) moves through training.

/// admonition | TODO
Anchor and anti-subspace weights against epoch, as a share of training, with each run's rise epoch marked; below it, the lean through training, one line per anchored run, at both lengths.
///
"""

# %%

rf"""
## Trained for {ex.ARM_EPOCHS} epochs (H4)

Ex-2.2.23 found more spill at 400 epochs than at 200. A longer run might give the model time to separate the op from lightness again, or let it mix them further. This arm runs only if the open decision under Runs says so.

**What we expect.** We expect the spill to grow again: at both seeds, the {ex.ARM_EPOCHS}-epoch run spills more than its 400-epoch twin. That would be a pass, and less spill at both a miss. If the two seeds disagree, the result would be outside the plan, and the verdict would be Unresolved. The measurements of H2 and H3 are reported for these runs too.

/// admonition | TODO
Spill against dose at 200, 400, and {ex.ARM_EPOCHS} epochs for seeds {" and ".join(map(str, ex.ARM_SEEDS))}, with the lightness tracking of each run beside it.
///
"""

# %%

r"""
## Discussion

/// admonition | TODO
Written after the results, with a discussion round first.
///
"""

# %%

rf"""
## Method

**Checkpoints.** Ex-2.2.23's checkpoints, resolved by ref: its new runs, and the 200-epoch runs at seeds 700 to 704 that it reused from ex-2.2.21. Every pass runs on ex-2.2.21's held-out set, the one ex-2.2.23 scored on.

**The site edits.** Ex-2.2.22's suppression pass, with the edit restricted to the positions of a site by a mask, at every slice and each dose. The target null, the doses, and the scoring are unchanged, so the edit at every position reproduces ex-2.2.23's numbers, which is a check on the pass.

**Probes.** For E1 and H1, a multinomial logistic regression (op) and a ridge regression (color, or lightness) on the 64-dimensional state at each position and slice, fitted per run on {ex.PROBE_SPLIT:.0%} of the held-out contexts, split by context and stratified by op, and scored on the rest. The regularization is chosen by cross-validation inside the fitting half. The Bayes accuracy at an answer uses the posterior over ops given the examples up to and including that answer, as ex-2.2.22 computed it. The one-example posterior is the posterior given that example alone, from a uniform prior over the seven ops, as on the example-evidence page; E3 uses it too.

**The query `=`.** E4 uses ex-2.2.23's stored eval arrays (the alignment at the query `?` and `=`, and the posterior over ops, per held-out context). The query operands are drawn whatever the op is, which the method checks on the held-out set, so they add no evidence about the op.

**Lightness tracking.** For H3, the component along e₁ of the state at every operand position of the held-out contexts of the six other ops, and the lightness of the operand color, per slice. The band is the shared one (see the glossary), from the seed spread of each condition at 12 seeds. The rank correlation in (b) is over the 400-epoch anchored runs, using each run's spill from H2's every-position edit.

**Cost.** One pass over the 48 checkpoints of ex-2.2.23, on an L4, plus {ex.N_ARM_RUNS} training runs at {ex.ARM_EPOCHS} epochs if the arm runs: under the planned budget of \${ex.BUDGET_USD}.
"""
