# title: Ex 2.2.19: training length and seeds for the seven-op set

# The design constants come from `experiment.py` beside this script. The runs this draft builds on are published
# already: the 400-epoch `no-four` and `full` runs of ex-2.2.18, and the three ex-2.2.17 seeds of the same recipe on
# the full op set, which are the yardstick for a difference between runs.
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import experiment as ex
from mini.store import project_store
from mini.vis import figure_html

FOUR = ex.OP_SET.name
YARDSTICK = tuple(f"sweep-{ex.PEAK_LR:g}-s{s}" for s in range(3))


# --- Helpers -------------------------------------------------------------------------------------------------


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str, *, text_cols: int = 1) -> str:
    """An authored result table in the shared report style; the first *text_cols* columns are text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>"
        + "".join(f"<td{' class=num' if i >= text_cols else ''}>{cell_html(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


# --- Loading ------------------------------------------------------------------------------------------------


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists: one `get_refs` and one `get_many`."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict:
    assert path is not None, "not published yet"
    return json.loads(path.read_text())


with tempfile.TemporaryDirectory() as _tmp:
    _files = fetch([ex.ex2218.EVAL_REF, ex.ex2217.EVAL_REF], Path(_tmp))
    SCOUT_18 = {r["label"]: r for r in read_json(_files[ex.ex2218.EVAL_REF])["runs"]}
    PRIOR = {r["label"]: r for r in read_json(_files[ex.ex2217.EVAL_REF])["runs"] if r["label"] in YARDSTICK}


def score(run: dict, key: str = "eem", op: str | None = None) -> float:
    r = run["task"][key]
    return r["all"] if op is None else r["per_op"][list(run.get("ops", ex.ex2216.OP_NAMES)).index(op)]


def gap(run: dict, op: str | None = None) -> float:
    return score(run, "ceiling", op) - score(run, "eem", op)


YARD_GAP = [gap(PRIOR[s]) for s in YARDSTICK]
YARD_RANGE = max(YARD_GAP) - min(YARD_GAP)
# The seed range of each op's gap in ex-2.2.17's three runs, floored at the partial band: the per-op tolerance.
OP_TOL = {
    op: max(max(gap(PRIOR[s], op) for s in YARDSTICK) - min(gap(PRIOR[s], op) for s in YARDSTICK), ex.PARTIAL_TOL)
    for op in ex.OP_SET.ops
}
COST_1 = len(ex.SCOUT_LRS) * sum(ex.cost_per_run(e) for e in ex.SCOUT_EPOCHS)
COST_2_MAX = len(ex.CONFIRM_SEEDS) * sum(
    ex.cost_per_run(e) for e in (ex.REFERENCE_EPOCHS, *ex.chosen_lengths(min(ex.SCOUT_EPOCHS)))
)

rf"""

# Ex 2.2.19: training length and seeds for the seven-op set

/// tip |
<!-- tl;dr -->
We look for the shortest training run on the seven-op set (`no-four`) that keeps most of the skill of ex-2.2.17's 400-epoch recipe: a scout at one seed picks a length, and three fresh seeds check it.
///

## Findings

- [The scout (S1)](#the-scout-s1) —
- [A shorter run keeps most of the skill (H1)](#a-shorter-run-keeps-most-of-the-skill-h1) —
- [Skill curves (E1)](#skill-curves-e1) —
- [Calibration (E2)](#calibration-e2) —
- [Op confusion (E3)](#op-confusion-e3) —
- [The seven-op set stays closer to its ceiling (H2)](#the-seven-op-set-stays-closer-to-its-ceiling-h2) —
- [Spread over seeds (E4)](#spread-over-seeds-e4) —

/// admonition | How to read this draft
This is a preregistration: the selection rule, the hypotheses, and their gates were frozen at commit `0e0a17c0`, before any run of this experiment. Each section opens with what we expect, and a `TODO` marks where its evidence will go. Results will replace the placeholders in place, and analyses conceived after seeing the data will be marked post hoc.
///

## Why

Ex-2.2.18 trained the center control on several smaller op sets, one run each. Dropping `screen`, `multiply`, `hsvmix`, and `exclusion` gave a seven-op set, `no-four`, with a higher Bayes ceiling (defined [below](#the-runs)). The model came within {gap(SCOUT_18[FOUR]):.3f} of that ceiling, against {gap(SCOUT_18["full"]):.3f} for the full set. But that was one seed. The three ex-2.2.17 seeds of the same recipe on the full set had gaps spread over {YARD_RANGE:.3f}, so the difference could be seed variation.

We will probably adopt `no-four` for the anchoring experiments that follow, and each of those costs more the longer its runs: about \${ex.cost_per_run(ex.REFERENCE_EPOCHS):.2f} for 400 epochs.

The skill curves of ex-2.2.18 leveled off well before the end: the last fifth of training added at most about {ex.SHORTFALL_TOL} of held-out expected exact match, which suggests a shorter run could do nearly as well. But the learning-rate schedule is a cosine, so a shorter run anneals sooner,[^anneal] and its curve is not just a truncated longer one.

[^anneal]: The learning rate decays along a cosine curve that reaches its low point at the end of the run, so a shorter run lowers its rate earlier, rather than stopping partway down the curve of a longer run.

A scout trains `no-four` at shorter lengths from the initialization of the ex-2.2.18 run, and a rule fixed in advance picks a length. Three fresh seeds then train at that length and at 400 epochs, and the preregistered comparison uses those seeds alone: the scout seed helped pick the length, so it would flatter the pick.

The fresh seeds also give the first replicates of `no-four` at 400 epochs, so we can check the narrower gap of ex-2.2.18 as well.

## The runs

Every run uses the recipe of ex-2.2.17 and ex-2.2.18, on the `no-four` corpus that ex-2.2.18 built, with a different number of epochs. At each shorter length the scout also tries a higher peak learning rate, since a shorter run may want one. The warmup (the opening stretch in which the learning rate ramps up to its peak) stays at {ex.WARMUP_EPOCHS:g} epochs at every length.

"""

table_html(
    ["stage", "epochs", "peak learning rate", "model seeds", "runs"],
    [
        ["ex-2.2.18 (reused)", f"{ex.REFERENCE_EPOCHS}", f"{ex.PEAK_LR:g}", f"{ex.SEED_OFFSET + ex.SCOUT_SEED}", "1"],
        [
            "1: scout",
            ", ".join(str(e) for e in ex.SCOUT_EPOCHS),
            ", ".join(f"{lr:g}" for lr in ex.SCOUT_LRS),
            f"{ex.SEED_OFFSET + ex.SCOUT_SEED}",
            str(len(ex.SCOUT_EPOCHS) * len(ex.SCOUT_LRS)),
        ],
        [
            "2: confirmation",
            f"the pick T, 2T if shorter than {ex.REFERENCE_EPOCHS}, and {ex.REFERENCE_EPOCHS}",
            f"the better scout rate at T and 2T; {ex.PEAK_LR:g} at {ex.REFERENCE_EPOCHS}",
            f"{ex.SEED_OFFSET + min(ex.CONFIRM_SEEDS)}–{ex.SEED_OFFSET + max(ex.CONFIRM_SEEDS)}",
            f"up to {len(ex.CONFIRM_SEEDS) * (1 + ex.MAX_CONFIRM_LENGTHS)}",
        ],
    ],
    f"""
        **The runs.** All train `no-four`. The scout costs about \\${COST_1:.2f} and the confirmation at most about
        \\${COST_2_MAX:.2f}, scaled from the cost of an ex-2.2.18 run.
    """,
    text_cols=4,
)

rf"""

Model seeds {ex.SEED_OFFSET + 1} and {ex.SEED_OFFSET + 2} are the initializations of ex-2.2.17's `{YARDSTICK[1]}` and `{YARDSTICK[2]}`, and seed {ex.SEED_OFFSET} of `{YARDSTICK[0]}` and of the ex-2.2.18 `full` run. So each 400-epoch `no-four` run at those seeds has a `full` run that started from the same weights. Seed {ex.SEED_OFFSET + 3} has no such partner.

Held-out *expected exact match* (EEM) is the probability that an answer drawn from the model at the query `=` is a correct answer of the true op. The *Bayes ceiling* is the same score for an ideal predictor, one that weighs each op by how well it explains the examples. The *gap* is the ceiling minus EEM. All three are computed on the op set of the run, as in ex-2.2.18.

Given a fixed corpus, the ceiling is the same at every length, so a difference in gap between two lengths is a difference in EEM. A run *falls short* of another by how much lower its EEM is. We compare runs that share a model seed, and average those paired differences over seeds.

## The scout (S1)

The scout is a procedure, with no hypothesis; the rule below picks the length that stage 2 confirms.

**The rule.** At each scout length, take the run at the peak rate with the higher EEM. Of those runs, pick the shortest, T, that falls short of the ex-2.2.18 run by at most {ex.SHORTFALL_TOL}, with no op falling short by more than its own tolerance (below). If no scout length passes, stage 2 trains only the 400-epoch runs, and H1 is unresolved.

Stage 2 confirms T, and also 2T when 2T is shorter than {ex.REFERENCE_EPOCHS} epochs, so that a lucky pass at T still leaves a length to adopt. Each trains at the rate taken at its length.

The overall tolerance is the largest last-fifth gain of any ex-2.2.18 run. The tolerance for each op is its seed range in the three ex-2.2.17 runs, or {ex.PARTIAL_TOL}, whichever is larger, since some ops varied little across three seeds and a single scout run is noisier than that.

**What we expect.** At 50 epochs the run falls short by more than the tolerance: ex-2.2.17 needed several times that length for the HSV-channel ops. We don't have a strong expectation between 100 and 200 epochs. We expect the higher rate to help more the shorter the run, and at 200 epochs to make little difference. The HSV-channel ops may be the exception: in ex-2.2.17 they were learned while the rate passed down through about {ex.ex2217.HOLD_LR:g}, during the cosine, so a higher peak only delays that stretch, and a short run has the least time to spare.

/// admonition | TODO
A line chart of held-out EEM against epochs (log scale), with skill on a second axis, for the seed-{ex.SEED_OFFSET} runs, one line per peak rate meeting at the 400-epoch run, with the ceiling as a dashed line and the tolerance as a band under the 400-epoch point; beside it, a table of the shortfall overall and per op at each length, with the pick marked.
///

## A shorter run keeps most of the skill (H1)

**What we expect.** On the fresh seeds, the run at T falls short of the run at {ex.REFERENCE_EPOCHS} epochs by at most {ex.SHORTFALL_TOL} on average, and no op falls short by more than its tolerance on average. That is a pass.

A shortfall between {ex.SHORTFALL_TOL} and {ex.PARTIAL_TOL} (the seed range of ex-2.2.17) is a partial pass. A shortfall beyond {ex.PARTIAL_TOL}, or an op beyond its tolerance, is a miss for H1, and would mean the scout result was flattered by its seed. 2T, when trained, gets a verdict by the same rule.

**Which length we adopt.** The verdicts inform this choice without settling it, since a rule written now can't anticipate everything that might matter: a short run that matches on EEM could be poorly calibrated ([E2](#calibration-e2)), or spread more widely over seeds ([E4](#spread-over-seeds-e4)). We expect to adopt the shortest confirmed length that passes, and will say why if we choose otherwise.

/// admonition | TODO
The shortfall at each confirmed length, paired by model seed, one dot per seed and a bar for the mean, with the tolerance and the partial band shaded; and a table of the mean shortfall per op against each tolerance.
///

## Skill curves (E1)

Exploratory, with no prediction. The skill curves of every run, to see whether the HSV-channel ops are learned later, or not at all, in the shorter runs.

/// admonition | TODO
Skill against the fraction of training completed, one line per length, split by op.
///

## Calibration (E2)

Exploratory, with no prediction. The calibration KL of each run (the KL divergence from the Bayes answer distribution to that of the model) beside its EEM, since ex-2.2.17 found a model can score well and be poorly calibrated.

/// admonition | TODO
Calibration KL against EEM, one dot per run, colored by length.
///

## Op confusion (E3)

Exploratory, with no prediction. The op confusion matrix of each run at the chosen length and at {ex.REFERENCE_EPOCHS} epochs: the mass the model puts on the answers of each op, on contexts of each true op, as in ex-2.2.17. A shorter run might keep more of its mass on a similar op.

/// admonition | TODO
The confusion matrices, at the chosen length and at {ex.REFERENCE_EPOCHS} epochs, averaged over the fresh seeds.
///

## The seven-op set stays closer to its ceiling (H2)

**What we expect.** At {ex.REFERENCE_EPOCHS} epochs, `no-four` has a smaller gap than `full` at each of the three model seeds where both exist ({ex.SEED_OFFSET}, {ex.SEED_OFFSET + 1}, and {ex.SEED_OFFSET + 2}). This is a check on ex-2.2.18, with no gate, since we are likely to adopt `no-four` for its higher ceiling either way. If the gaps overlap, the narrower gap of ex-2.2.18 was seed variation.

The `full` runs are reused: those at seeds {ex.SEED_OFFSET + 1} and {ex.SEED_OFFSET + 2} are from ex-2.2.17 and trained on the ex-2.2.16 corpus, and the one at seed {ex.SEED_OFFSET} is from ex-2.2.18, on a corpus rebuilt the same way, so the pairs share a starting point and differ in corpus draw as well as op set.

/// admonition | TODO
The gap of `no-four` and of `full` at {ex.REFERENCE_EPOCHS} epochs, one pair of dots per model seed joined by a line, with the `no-four` run at seed {ex.SEED_OFFSET + 3} unpaired.
///

## Spread over seeds (E4)

Exploratory, with no prediction. The spread of EEM over the four `no-four` seeds at {ex.REFERENCE_EPOCHS} epochs, against the spread of the three `full` seeds, and the spread at each confirmed length.

/// admonition | TODO
EEM per seed at each length and op set, with the range marked.
///

## What it means for what follows

TODO after the results.

## Method

**Recipe.** The unanchored d64-L4 model with an untied readout and the newline mask, trained with a cosine schedule after a linear warmup, as in ex-2.2.18. The corpus condition is `k3-r0.3` (three examples per context, with a replacement rate of 0.3) on the `no-four` op set (the eleven ops of ex-2.2.16 less `screen`, `multiply`, `hsvmix`, and `exclusion`), with the corpus, held-out set, and probe set that ex-2.2.18 built, so that the runs differ from the ex-2.2.18 run in length and model seed alone.

**Warmup and learning rate.** The warmup is {ex.WARMUP_EPOCHS:g} epochs rather than a fixed share of the run, as in ex-2.2.17 from its second round, and the peak learning rate is {ex.PEAK_LR:g}. A shorter run spends less of its schedule near the peak, so it may do better at a higher peak rate. The scout tries {ex.SCOUT_LRS[1]:g} beside it at each shorter length: the next rate up on the ex-2.2.17 grid, which at 400 epochs scored level with {ex.PEAK_LR:g} at one seed, where 0.01 scored lower. The 400-epoch runs stay at {ex.PEAK_LR:g}, so H1 compares a shorter recipe, rate included, with the recipe we have.

**Measurements.** EEM, ceiling, floor, and calibration KL are measured on the {ex.ex2216.HOLDOUT_CONTEXTS:,} held-out contexts per op, with the evaluation of ex-2.2.18. Skill curves are logged at about {ex.ex2217.N_TRAJ_POINTS} points per run on {ex.ex2217.N_TRAJ_EEM_PER_OP} held-out contexts per op.

**Per-op tolerances.**

"""

table_html(
    ["op", "tolerance"],
    [[f"`{op}`", f"{OP_TOL[op]:.3f}"] for op in ex.OP_SET.ops],
    f"""
        **Per-op tolerances.** The seed range of the gap of each op over ex-2.2.17's three runs, or {ex.PARTIAL_TOL},
        whichever is larger.
    """,
)
