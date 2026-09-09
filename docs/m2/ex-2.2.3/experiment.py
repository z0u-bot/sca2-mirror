"""
Experiment 2.2.3: the multi-op grammar, with *red* anchored again.

The D2.2 plan makes the operation a variable: six operations on the same color
grid, each spelled as a word, so a line reads `c1 <op> c2 = answer`. Every D2.1
recipe was tuned on the one-op grammar, so this experiment is the regression
check the plan schedules before any operation is anchored: an un-anchored
control, the ex-2.1.10 reference recipe, and the ex-2.1.11 survey's proposals,
all retrained on the new grammar at fresh seeds and scored through the eval
contract (`sca.intervention`) beside the D2.1 numbers. A selection rule frozen
here names the operating point the anchored-op experiments adopt.

The grammar itself lives in `sca.data.ops`; this module re-exports the parts the
report renders, so its counts and the corpus code cannot disagree. The design
constants come first, then the DAG: one corpus per (op set, size), a train and
an eval step per run, the eval contract on the candidates, cube probes on the
un-anchored arms, and one publish step.

    EX223_STAGES=calibration bin/mini run docs/m2/ex-2.2.3/experiment.py --app modal --max-containers 2
    bin/mini run docs/m2/ex-2.2.3/experiment.py --app modal --max-containers 8
    bin/mini status ex-2.2.3
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sca.anchoring import PROMPT_SPAN, softmin_weights
from sca.colorcube import redness as cube_redness, sim_to_red
from sca.config import DataConfig, ModelConfig
from sca.data.batches import batches_per_epoch
from sca.data.colors import mix, redness
from sca.data.ops import (
    ADD,
    DARKEN,
    GRID,
    LEVELS,
    LIGHTEN,
    MIX,
    MULTIPLY,
    OP_BY_NAME,
    OP_NAMES,
    OPS,
    PALETTE,
    SCREEN,
    TOKENS_PER_LINE,
    TOP,
    Op,
    agreement,
    colors,
    dose,
    is_on_grid,
    lines,
    mix_probe_lines,
    on_grid,
    relevance,
    snap,
    unordered_pairs,
)
from mini import Ctx, Experiment, get_data_dir

__all__ = [
    "ADD",
    "DARKEN",
    "GRID",
    "LEVELS",
    "LIGHTEN",
    "MIX",
    "MULTIPLY",
    "OPS",
    "OP_BY_NAME",
    "OP_NAMES",
    "SCREEN",
    "TOKENS_PER_LINE",
    "TOP",
    "Op",
    "agreement",
    "colors",
    "dose",
    "experiment",
    "is_on_grid",
    "lines",
    "mix_probe_lines",
    "on_grid",
    "relevance",
    "snap",
    "unordered_pairs",
]

#: Result refs the report reads.
METRICS_REF = "reports/m2/ex-2.2.3/metrics"
ARRAYS_REF = "reports/m2/ex-2.2.3/arrays"
TRAJ_REF = "reports/m2/ex-2.2.3/trajectories"
PROBE_REF = "reports/m2/ex-2.2.3/probes"
GEOMETRY_REF = "reports/m2/ex-2.2.3/geometry"
CALIBRATION_REF = "reports/m2/ex-2.2.3/calibration"
"""The pre-freeze stage: one seed of each control arm, published apart from the frozen results."""

CHECKPOINT_REF = "reports/m2/ex-2.2.3/checkpoints/{label}"
"""Every run's checkpoint, for the intervention-tuning pass and the anchored-op experiments."""

#: Stored results the report compares against: the D2.1 primary (nine seeds) and the survey's proposals.
EX2110_METRICS_REF = "reports/m2/ex-2.1.10/metrics"
SURVEY_REF = "reports/m2/ex-2.1.11/survey"

# --- The operation table -----------------------------------------------------------------
# The table, the snap, and the line sets are `sca.data.ops`; see its docstrings for the rounding
# decision. What stays here is what only this experiment reads: the dose groups and the counts
# the method section renders.

GRID_RGB = np.asarray(colors(), dtype=float) / TOP
"""The 216 colors as unit-cube RGB, in palette order: the row order of every per-color array."""

REDNESS = cube_redness(GRID_RGB)
SIM_TARGET = sim_to_red(GRID_RGB, power=1.5)
"""The grading target: similarity to red, as ex-2.1.9 and ex-2.1.10 read it."""

SPAN = PROMPT_SPAN
"""The four prompt roles the mellowmax pools over: op1, the op word, op2, `=`."""


def line_counts(op: Op) -> dict[str, int]:
    """Over the op's lines: the total, the red and non-red lines (the same for every op, since dose reads the
    operands), and the redder-than-both lines.
    """
    ls = lines()
    return {
        "lines": len(ls),
        "red": sum(dose(a, b) >= RED_DOSE for a, b in ls),
        "nonred": sum(dose(a, b) <= NONRED_DOSE for a, b in ls),
        "redder": sum(redness(op(a, b)) > dose(a, b) + 1e-9 for a, b in ls),
    }


def check_table() -> None:
    """The table's invariants: every answer is a grid color, every op is commutative, `mix` is D2.1's op where
    D2.1 defined it, only `mix` ties, and the three exact ops never round.
    """
    cs = set(colors())
    for a, b in unordered_pairs()[::97]:
        for op in OPS:
            assert op(a, b) in cs, op.name
            assert op(a, b) == op(b, a), op.name
            if op is not MIX:
                assert not any(abs(v - snap(v)) == 1.5 for v in op.raw(a, b)), op.name
    for a, b in mix_probe_lines()[::37]:
        assert MIX(a, b) == mix(a, b)
    for op in (ADD, LIGHTEN, DARKEN):
        assert on_grid(op) == 1.0, op.name
    assert len(mix_probe_lines()) == 5832, "mix's on-grid lines are D2.1's probe set"


# --- The lines -----------------------------------------------------------------------------

RED_DOSE = 0.8
"""*Red lines* have dose ≥ 0.8, the ex-2.2.1 definition: 365 of `mix`'s 5,832 probe lines."""

NONRED_DOSE = 0.2
"""*Non-red lines* have dose ≤ 0.2: 1,689 of `mix`'s probe lines."""

check_table()

# --- The corpus ------------------------------------------------------------------------------

N_LINES = 100_000
"""D2.1's corpus size, unchanged. Ops are drawn uniformly, so each op has about a sixth of the lines, and
within an op the training pairs are drawn uniformly with random operand order, as ex-2.1.10 drew `mix`'s.
D2.1 drew its 100k lines from 5,832 distinct `mix` lines; here they are drawn from six times 46,656, so most
lines are seen once per epoch and most pairs of an op are never seen under it."""

CORPUS_SEED = 0
HOLDOUT_FRAC = 0.2
"""Of the distinct pairs, held out per op: bookkeeping is keyed on (op, pair), so a pair held out under `add`
may be trained under `mix`."""

BLOCK, BATCH = 64, 64
OVERSAMPLE, TRAIN_SPLIT = 16, 0.9
"""Ex-2.1.3's DataConfig, unchanged through D2.1."""


def steps_per_epoch(n_lines: int = N_LINES) -> int:
    """Optimizer steps per epoch, from the loader's own sizing rule: an epoch covers `oversample / batch_size`
    of the training split, so the count is close to 0.9 · 6 · n_lines / 16,384, which is 33 for 100k lines.
    """
    n_tokens = int(TRAIN_SPLIT * n_lines * TOKENS_PER_LINE)
    model = ModelConfig(vocab_size=256, block_size=BLOCK, n_embd=64, n_head=8, n_head_dim=8, n_ff=256, n_layer=4)
    data = DataConfig(batch_size=BATCH, oversample=OVERSAMPLE, train_split=TRAIN_SPLIT, padding_chance=0.1)
    return batches_per_epoch(n_tokens, data, model)


EPOCHS = 100
"""Ex-2.1.10's length: 100 epochs of 100k lines, 3,300 steps. The corpus size is D2.1's, so the epoch is too,
and the recipe runs here with nothing changed but the grammar."""

EPOCHS_SHORT = 50
"""The survey's trial length, 1,650 steps. Its proposals run at it, and so does the recipe's short arm."""

assert EPOCHS * steps_per_epoch() == 3300

# --- Training, shared with D2.1 --------------------------------------------------------------

PEAK_LR = 1e-2
WARMUP_FRAC = 0.10
ANNEAL_START_FRAC = 0.90
ANNEAL_FLOOR = 0.1
ANTI_HOLD_RATIO = 0.30
"""Ex-2.1.10's schedule keyframes as fractions of training: LR and anchor warm-up over the first tenth, the
anchor anneal over the last tenth to a 0.1 floor, and the anti-subspace weight annealing from its peak ratio
to 0.30 of the anchor weight over its own end fraction."""

SCORING_LAMBDA = 0.1
TAU_REF = 0.1
ANTI_PEAK_RATIO_REF = 2.5
ANTI_ANNEAL_END_FRAC_REF = 0.90
"""The reference recipe's anchor weight, mellowmax temperature, and anti-subspace schedule: ex-2.1.10's
`either-t100`."""

RED_RATE = 0.08
PER_SLOT_RATE = RED_RATE / 2
"""The pooled either-operand labeller, unchanged: each operand draws at redness⁸ × 0.04, and a line is
labelled when either draws. The op word is never an operand, so the labeller reads the same roles on every op."""

ANCHOR_AXIS = 0
"""*Red* sits on e₁ at every slice, as in D2.1."""

TRAJ_STRIDE = 50

# --- Conditions ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """A set of runs, all trained here on the new grammar."""

    name: str
    seeds: int
    title: str
    lam: float
    tau: float = TAU_REF
    epochs: int = EPOCHS
    anti_peak_ratio: float = ANTI_PEAK_RATIO_REF
    anti_anneal_end_frac: float = ANTI_ANNEAL_END_FRAC_REF
    anchor_anneal: bool = True
    """Ex-2.1.10 anneals the anchor weight over the last tenth of training; the survey's trials ran it flat
    (its ablation found the two within a band), so its proposals are flat here too."""
    ops: tuple[str, ...] = OP_NAMES
    """Which ops the corpus carries. Only the richer-op arms narrow this."""
    n_lines: int = N_LINES
    """Corpus size. D2.1's for every condition but the per-op-matched arms, whose corpus shrinks with the op
    set so that lines per op stay at the six-op value."""
    survey_trial: int | None = None
    """For a proposal, the ex-2.1.11 trial it re-measures; the report reads that trial's five-seed numbers
    from `SURVEY_REF` and prints them beside the fresh ones."""

    @property
    def steps(self) -> int:
        return self.epochs * steps_per_epoch(self.n_lines)

    @property
    def lines_per_op(self) -> int:
        return self.n_lines // len(self.ops)


CONTROL = Condition("control", 5, "un-anchored", lam=0.0)
"""Nothing placed on the axis: the anchor and anti-subspace weights at zero. The task reference for H1 at the
full length, and the six-op point of both richer-op arms."""

# REVIEW: added `control-short`, so H1 reads every 50-epoch condition against an un-anchored model of the
# same length. Per-pair exposure fell about fiftyfold from D2.1, so a 50-epoch arm could miss the task gate
# for want of training rather than because of the anchor, and the full-length control cannot tell those
# apart. Verify: the calibration read at 50 epochs; if it reaches holdout EM near 1 on every op, the arm
# changes nothing and could be dropped.
CONTROL_SHORT = Condition("control-short", 5, "un-anchored, at the proposals' length", lam=0.0, epochs=EPOCHS_SHORT)
"""The task reference for H1 at the proposals' length."""

RECIPE = Condition("recipe", 5, "the ex-2.1.10 recipe", lam=SCORING_LAMBDA)
"""The D2.1 primary retrained on the new grammar, unchanged: the placement reference for H2, and the incumbent
in H3's selection."""

RECIPE_SHORT = Condition(
    "recipe-short", 5, "the recipe at the proposals' length", lam=SCORING_LAMBDA, epochs=EPOCHS_SHORT
)
"""The recipe at 50 epochs, so H3 compares each proposal with the recipe at the same step count. The survey
ran this arm too (`short`), and found its m_line within a band of the full-length recipe's."""

T00 = Condition(
    "t00",
    5,
    "survey proposal t00",
    lam=0.5573771884288644,
    tau=0.2652855688754236,
    epochs=EPOCHS_SHORT,
    anti_peak_ratio=1.3764374809867324,
    anti_anneal_end_frac=0.5369576041586697,
    anchor_anneal=False,
    survey_trial=0,
)
"""The survey's proposed operating point: the feasible trial with the highest five-seed m_line."""

T48 = Condition(
    "t48",
    5,
    "survey proposal t48",
    lam=0.4313002226158571,
    tau=0.22225105244233356,
    epochs=EPOCHS_SHORT,
    anti_peak_ratio=2.948510815850172,
    anti_anneal_end_frac=0.5462400943506509,
    anchor_anneal=False,
    survey_trial=48,
)
"""The twin: within one m_line band of t00 at the survey, with more grading and contrast, at a lower anchor weight."""

T12 = Condition(
    "t12",
    5,
    "survey proposal t12",
    lam=0.36133018466356065,
    tau=0.23333255019137145,
    epochs=EPOCHS_SHORT,
    anti_peak_ratio=1.1284021176786219,
    anti_anneal_end_frac=0.9193784880917519,
    anchor_anneal=False,
    survey_trial=12,
)
"""The knee: the promoted trial with the most grading, at a margin the survey could still not tell from t00's."""

PROPOSALS = (T00, T48, T12)
CANDIDATES = (RECIPE, RECIPE_SHORT, *PROPOSALS)
"""The set H3's selection rule chooses from."""

SURVEY_RECIPE = {
    "recipe": (
        3,
        {
            "m_line": 0.4204,
            "r2_sim": 0.7823,
            "contrast": 0.8562,
            "alpha_op1": 0.039,
            "holdout_em": 0.9961,
            "retention": 0.9897,
        },
    ),
    "recipe-short": (
        3,
        {
            "m_line": 0.4218,
            "r2_sim": 0.8065,
            "contrast": 0.8491,
            "alpha_op1": 0.0262,
            "holdout_em": 1.0,
            "retention": 0.9968,
        },
    ),
}
"""The survey's own re-runs of the recipe at both lengths (its `ref` and `short` conditions, three seeds each),
as (seeds, seed means), so the H3 table can print every candidate's one-op value beside its fresh one. The
proposals' values are read from `SURVEY_REF` at render time."""

ONE_OP = ("mix",)
THREE_OPS = ("mix", "add", "multiply")


def per_op_matched(name: str, seeds: int, title: str, ops: tuple[str, ...]) -> Condition:
    """An un-anchored arm at the six-op lines per op and the recipe's step count. The corpus shrinks with the
    op set, and the epoch count grows to fill the same steps, so what differs from the control is the op set
    and how often each line repeats. The loader rounds steps per epoch, so the three-op arm lands two steps
    short of 3,300; the conditions table prints the actual count.
    """
    n_lines = N_LINES * len(ops) // len(OPS)
    epochs = round(EPOCHS * steps_per_epoch() / steps_per_epoch(n_lines))
    return Condition(name, seeds, title, lam=0.0, ops=ops, n_lines=n_lines, epochs=epochs)


RICHER_OP_ARM = (
    Condition("ops-1-corpus", 3, "un-anchored, mix only, D2.1's corpus size", lam=0.0, ops=ONE_OP),
    Condition("ops-3-corpus", 3, "un-anchored, mix, add, multiply, D2.1's corpus size", lam=0.0, ops=THREE_OPS),
    per_op_matched("ops-1-per-op", 3, "un-anchored, mix only, six-op lines per op", ONE_OP),
    per_op_matched("ops-3-per-op", 3, "un-anchored, mix, add, multiply, six-op lines per op", THREE_OPS),
)
"""Un-anchored models at one and three ops, probed for the cube as ex-2.1.12 probed D2.1's; the control is
the six-op point of both arms. Unscored (E4). Every arm runs at the recipe's step count. The `corpus` arms
keep D2.1's line count, so fewer ops means more lines per op; the `per-op` arms keep the six-op lines per op,
so fewer ops means a smaller corpus repeated over more epochs. Corpus size, lines per op, and step count
cannot all be held while the op set changes, and the two matchings put the remaining difference on opposite
sides: a cube that is better at six ops in the corpus arm, or worse at six ops in the per-op arm, is read
against the confound, and the pair together reads in both directions."""

CONDITIONS = (CONTROL, CONTROL_SHORT, *CANDIDATES, *RICHER_OP_ARM)
N_RUNS = sum(c.seeds for c in CONDITIONS)
assert N_RUNS == 47

# --- The probe set ---------------------------------------------------------------------------

N_PROBE = 27
"""Probe lines per color as op1, per op. For `mix` these are its on-grid partners, so its probe set is D2.1's
5,832 lines with the op word changed, and every gated statistic is read on the lines D2.1 read it on. For the
other ops, 27 partners per color are drawn once with a fixed seed; each op then has 5,832 probe lines."""

PROBE_SEED = 0
PRIMARY_OP = MIX
"""Every gated statistic is read on `mix`'s probe lines; the other ops' values are reported beside them (E1)."""

# --- Gates -----------------------------------------------------------------------------------

TASK_GATE = 0.02
"""H1: "seed-mean holdout exact-match accuracy within 0.02 of the control's, on every op", for every scored
condition. The width every D2.1 task gate used."""

TASK_PARTIAL = 0.05
"""H1 partial: "every condition-op comparison within 0.05, or all but one within TASK_GATE and that one within
0.05". The unit is one condition read against its control on one op, as it is for the gate and the contrary
clause, rather than a whole op or a whole condition."""

REF_M_LINE = 0.4202
"""Ex-2.1.10's primary, nine seeds: the seed-mean m_line every margin gate here is quoted against. The report
reads the stored metrics and asserts they agree to three decimals."""

REF_R2_SIM = 0.782
"""The same recipe's grading r² (squared Pearson between op1 alignment and the sim^1.5 target), as the
survey's three-seed re-run of it measured it: ex-2.1.11's `ref` condition, which set the survey's grading
floor at this less 0.10. It happens to round to the same three decimals as trial t12's own five-seed
r2_sim (0.7819); the two are independent measurements (ref's seeds: 0.791, 0.775, 0.781)."""

MARGIN_RATIO = 0.8
"""H2: "the recipe's seed-mean m_line on the mix lines is at least 0.8 of its ex-2.1.10 value"."""

MARGIN_PARTIAL = 0.6
"""H2 partial: "between 0.6 and 0.8 of it". H3's feasibility also reads it per run: a candidate with any seed
below 0.6 of the reference m_line is not adopted, whatever its seed mean."""

MEAN_ALIGN_GATE = 0.1
"""H2 and H3 (containment): "ᾱ, the mean alignment over all 216 colors at op1 on the mix lines, at most 0.1".
Carried from ex-2.1.8 on; the contamination detector."""

RETENTION_FLOOR = 0.2
RETENTION_GATE = 0.8
"""H2 and H3 (retention): "for every run whose running peak m_line reaches 0.2, the final value is at least 0.8
of that peak", quoted as the minimum over seeds."""

LEAD_GATE = 0.4
"""H2 (concentration): "the leading softmin weight of the red group at the embedding is at least 0.4"."""

CONTRAST_GATE = 0.2
CONTRAST_PARTIAL = 0.1
"""H2 (attribution): "the between-group contrast in op2 weight, mean over the four post-attention slices, is at
least 0.2", partial from 0.1. H3's feasibility reads the 0.1 floor, as the survey did; its selection asks
for 0.2."""

GRADE_R2_DROP = 0.10
"""H2 (grading): "the recipe's grading r² is no more than 0.10 below its D2.1 value". A difference in r²
against a reference arm rather than a fixed threshold, as ex-2.1.9 and ex-2.1.10 read it. H3's feasibility
reads each candidate's r² against the full-length recipe's fresh value at the same width."""

LATCH_PI = 0.5
"""The latch veto, per run: a softmin weight of the non-red group above 0.5 at op1 means the pull latched a
position rather than a concept. Any latched run fails the condition, on H2 and on H3."""

RED_ACC_GATE = 0.2
"""H4 (removal): "under the plain projection, seed-mean accuracy on the red lines of every op is at most 0.2".
Ex-2.2.1 measured 0.09 on the one-op grammar."""

NONRED_DEFICIT_GATE = 0.05
"""H4 (selectivity): "the seed-mean non-red deficit under the projection, on the mix lines, is at most 0.05",
the band ex-2.2.1 landed in (0.024) and called partial at its 0.02 gate. The 0.02 read is reported beside it."""

NONRED_DEFICIT_PARTIAL = 0.10
"""H4 partial: "between 0.05 and 0.10"."""

RESOLUTION_SD = 2.0
"""A difference between two seed means smaller than 2σ√(1/n_a + 1/n_b), with σ the per-run spread, is reported as
not resolved."""

GRADE_MARGIN_SD = 1.0
"""H3's selection: a candidate counts as feasible only with its grading r² at least one per-run σ above its
floor, so a point that passes its floor by less than the noise is not adopted on that pass."""

NOISE_RUN = {
    "m_line": 0.0088,
    "alpha_op1": 0.0212,
    "holdout_em": 0.0087,
    "retention": 0.0061,
    "r2_sim": 0.024,
    "contrast": 0.0057,
}
"""Per-run σ of each statistic at the reference recipe, from ex-2.1.10's nine seeds as ex-2.1.11 tabulated
them. The bands use these; E5 re-measures them on the new grammar, and the report prints both."""


def equiv_band(stat: str, n_a: int = 5, n_b: int = 5, noise: dict[str, float] = NOISE_RUN) -> float:
    """The smallest seed-mean difference the resolution rule may call a difference."""
    return RESOLUTION_SD * noise[stat] * np.sqrt(1 / n_a + 1 / n_b)


# REVIEW: the selection rule had three ways to leave the choice to the person running it.
# (1) "two candidates within one m_line band of each other tie" is not transitive over three
# candidates, so ties are now read against the highest value rather than pairwise. (2) "then to
# the recipe" did not say which of the two recipe arms. (3) The full contrast gate could empty
# the feasible set with no fallback named; it now falls to the full-length recipe, as an empty
# feasible set already did. Verify: none of these changes which candidate wins in the case where
# one feasible candidate leads by more than a band.
# REVIEW: added a per-run m_line floor at the partial margin. The band is built from the frozen
# per-run σ, so a candidate's own spread could not change its verdict, and a seed mean can win
# with one seed well below the margin. A floor on the worst run is the same shape as the
# retention and latch checks and reads cleanly at five seeds, where a fresh-σ gate would not.
# The feasibility list is now split by what it is read on. Verify: the floor is read against
# REF_M_LINE, the same reference as the seed-mean margin, and applies to the recipe arms too.
SELECTION_RULE = """\
Among the candidates (the recipe at both lengths and the three proposals), keep those feasible at fresh seeds. \
On the seed mean: task within the gate on every op against the control of the same length, containment, \
contrast at least the partial floor, and grading r² no more than 0.10 below the full-length recipe's fresh \
value, clearing that floor by at least one per-run σ. On every run: retention, no latch, and m_line at least \
the partial margin of its ex-2.1.10 value. Among those, require contrast at the full gate, and take the highest \
seed-mean m_line on the mix probe lines. Every candidate within one m_line band of that highest value ties \
with it, and the tie goes to the larger grading margin, then to the full-length recipe, then to \
`recipe-short`. If no other candidate is feasible, or none of the feasible candidates clears the full contrast \
gate, the full-length recipe is adopted, and if it is not itself feasible the report says which check it \
missed beside the adoption. Whichever point is adopted, the D2.2 experiments carry it."""

# --- Interventions ---------------------------------------------------------------------------

SLICES = (0, 1, 2, 3, 4)
OPERAND_POSITIONS = (0, 2)
DECODE_POS = 3


@dataclass(frozen=True)
class Intervention:
    """One row of the intervention results: an operator and where it acts."""

    name: str
    kind: str
    """`projection`, `shaped`, or `ablate`, from the eval contract."""
    slices: tuple[int, ...] = SLICES
    positions: tuple[int, ...] | None = None
    gamma: float = 1.0


PROJECTION = Intervention("projection", "projection")
"""Ex-2.2.1's primary: full projection of the axis at every slice and position. H4 reads it."""

RIDE_ALONG = (
    Intervention("operands", "projection", positions=OPERAND_POSITIONS),
    Intervention("shaped", "shaped"),
    Intervention("ablate", "ablate", slices=()),
)
"""Ex-2.2.1's other rows, scored without gates on every candidate, so the intervention-tuning pass has the
multi-op figures. `shaped` at a = 0.5, b = 1, p = 1, as there."""

SHAPED = dict(a=0.5, b=1.0, p=1.0)

# --- The DAG ---------------------------------------------------------------------------------

N_EVAL = 256
"""Eval lines per op and split (held-out pairs, trained pairs), as ex-2.1.10's holdout set."""

ANSWER_POS = 4
"""The answer token's position; every decode reads the logits one position before it."""

COMPOSITION = ("true", "red_operand", "visible_operand", "neighbor", "other")
"""What a decoded answer is, as ex-2.2.1 classified it: the true answer, the operand carrying the dose, the
other operand, a one-step neighbor of the true answer in the cube, or anything else."""


def corpus_key(ops: tuple[str, ...], n_lines: int) -> str:
    """One corpus per (op set, size): `ops6-100k` for the main arms."""
    return f"ops{len(ops)}-{n_lines // 1000}k"


def prepare_corpus(
    ops: tuple[str, ...],
    n_lines: int,
    seed: int,
    holdout_frac: float,
    n_probe: int,
    probe_seed: int,
    per_slot_rate: float,
    red_rate: float,
) -> dict:
    """Build one corpus, its eval sets, and the alignment probe set of every op it carries.

    The design constants arrive as arguments rather than being read off the module, so changing one re-runs
    this step instead of silently reusing a corpus built under the old value. The vocabulary always holds all
    six op words, so every arm's model has the same width whatever its op set.
    """
    from collections import Counter

    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data import ops as grammar
    from sca.data.named_colors import WordTokenizer
    from mini.store import put

    key = corpus_key(ops, n_lines)
    table = tuple(grammar.OP_BY_NAME[o] for o in ops)
    corpus = grammar.sample_corpus(n_lines, seed, table, holdout_frac)

    tokenizer_config = TokenizerConfig(vocabulary=grammar.vocabulary())
    tokenizer = WordTokenizer(tokenizer_config)
    tokens = grammar.encode_corpus(corpus, tokenizer.stoi)
    n_chars = sum(len(w) for line in corpus for w in line.words)
    meta = CorpusMetadata(
        tokenizer_config=tokenizer_config,
        total_tokens=len(tokens),
        total_chars=n_chars,
        sources=[DatasetMetadata(title=f"multi-op color corpus ({key})", fixes=[], total_chars=n_chars)],
    )
    corpus_dir = get_data_dir() / "corpora" / key
    save_data(tokens, meta, corpus_dir)

    evals = grammar.eval_sets(N_EVAL, seed, table, holdout_frac)

    # The labeller's table by token id, and the per-color affinity the per-color statistics weight with,
    # both recomputed from the palette and checked against the module's constants.
    rgb = np.asarray(grammar.colors(), dtype=float) / TOP
    np.testing.assert_allclose(rgb, GRID_RGB, rtol=0, atol=1e-12)
    color_redness = cube_redness(rgb)
    slot_p = np.zeros(tokenizer.vocab_size)  # the tokenizer's width, which counts the pad token
    for name, sp in zip(grammar.PALETTE, color_redness**8 * per_slot_rate, strict=True):
        slot_p[tokenizer.stoi[name]] = sp
    affinity = color_redness**8 * red_rate
    arrays: dict[str, np.ndarray] = {"slot_p": slot_p, "weights": affinity / affinity.sum(), "redness": color_redness}

    # Per op: the probe lines, each operand's redness, and P(labelled) under the either-slot labeller.
    for op in table:
        probe = grammar.probe_lines(op, n_probe, probe_seed)
        r1 = cube_redness(np.asarray([ln.lhs for ln in probe], dtype=float) / TOP)
        r2 = cube_redness(np.asarray([ln.rhs for ln in probe], dtype=float) / TOP)
        p1, p2 = r1**8 * per_slot_rate, r2**8 * per_slot_rate
        arrays |= {
            f"{op.name}/tokens": np.array([tokenizer.encode_words(ln.words) for ln in probe], dtype=np.int32),
            f"{op.name}/r1": r1,
            f"{op.name}/r2": r2,
            f"{op.name}/line_p": p1 + p2 - p1 * p2,
        }

    held = grammar.holdout(seed, holdout_frac, table)
    stats = {
        "key": key,
        "ops": list(ops),
        "n_lines": n_lines,
        "total_tokens": int(len(tokens)),
        "vocab_size": tokenizer.vocab_size,
        "lines_per_op": dict(Counter(ln.op for ln in corpus)),
        "distinct_per_op": {o: len({ln.pair for ln in corpus if ln.op == o}) for o in ops},
        "n_pairs": len(grammar.unordered_pairs()),
        "n_holdout_per_op": len(held) // len(ops),
        "eval_n": N_EVAL,
        "n_probe_lines": int(len(arrays[f"{ops[0]}/tokens"])),
        "line_label_rate": float(arrays["mix/line_p"].mean()) if "mix" in ops else None,
    }
    return {
        "key": key,
        "meta": meta,
        "stats": stats,
        "corpus": put(corpus_dir, name=f"ex-2.2.3-{key}-corpus"),
        "evals": put(grammar.dump_lines(evals), name=f"ex-2.2.3-{key}-evals.json"),
        "probes": put(_npz(**arrays), name=f"ex-2.2.3-{key}-probes.npz"),
    }


def _npz(**arrays) -> bytes:
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _make_config(vocab_size: int, seed: int, epochs: int):
    """The d64-L4 config from ex-2.1.3, unchanged; only the schedule's length varies."""
    from sca.config import OptimizerConfig, TokenizerConfig, TrainingConfig

    return TrainingConfig(
        model=ModelConfig(
            vocab_size=vocab_size,
            block_size=BLOCK,
            n_embd=64,
            n_head=8,
            n_head_dim=8,
            n_ff=256,
            n_layer=4,
        ),
        tokenizer=TokenizerConfig(vocabulary=[]),
        data=DataConfig(batch_size=BATCH, oversample=OVERSAMPLE, train_split=TRAIN_SPLIT, padding_chance=0.1),
        optimizer=OptimizerConfig(weight_decay=0, learning_rate=PEAK_LR, betas=(0.9, 0.95)),
        scheduler=scheduler_config(epochs),
        seed=seed,
    )


def scheduler_config(epochs: int):
    """The LR schedule for a condition of the given length, as ex-2.1.11 set it."""
    from sca.config import SchedulerConfig

    return SchedulerConfig(epochs=epochs, warmup_epochs=round(epochs * WARMUP_FRAC), min_lr_factor=0.01)


def schedules(c: Condition) -> tuple[dict, dict | None]:
    """The `AnchorSpec` and `AntiSpec` keyword dicts for one condition, as ex-2.1.11 spelled them.

    Both travel as plain dicts of every keyframe, so a run's memo key carries the whole shape of its pull and
    its repulsion. A proposal's anchor is flat, as the survey ran it; its anti-subspace schedule anneals from
    the trial's peak ratio to the hold ratio by the trial's end fraction. An un-anchored arm gets a zero
    anchor and no anti term.
    """
    anchor = {
        "peak": c.lam,
        "warmup_epochs": c.epochs * WARMUP_FRAC,
        "anneal_start": c.epochs * ANNEAL_START_FRAC,
        "anneal_end": float(c.epochs),
        "floor": ANNEAL_FLOOR,
        "span": SPAN,
        "tau": c.tau,
        "shape": "min-jerk" if c.anchor_anneal else "flat",
    }
    anti = (
        {
            "lam": c.lam,
            "peak_ratio": c.anti_peak_ratio,
            "hold_ratio": ANTI_HOLD_RATIO,
            "anneal_end": c.epochs * c.anti_anneal_end_frac,
            "anchor_anneal_start": c.epochs * ANNEAL_START_FRAC,
            "anchor_anneal_end": float(c.epochs),
            "floor": ANNEAL_FLOOR,
            "shape": "min-jerk",
        }
        if c.lam > 0
        else None
    )
    return anchor, anti


def cells(
    conditions: tuple[Condition, ...], preps: dict[str, dict], seeds: dict[str, list[int]] | None = None
) -> list[dict]:
    """One row per run: the config, both schedules, the corpus it trains on, and its labels."""
    from sca.utils import align

    rows = []
    for c in conditions:
        prep = preps[corpus_key(c.ops, c.n_lines)]
        tc = prep["meta"].tokenizer_config
        anchor, anti = schedules(c)
        for seed in range(c.seeds) if seeds is None else seeds.get(c.name, []):
            config = _make_config(align(tc.vocab_size, 64), seed, c.epochs)
            config.tokenizer = tc.model_copy()
            rows.append(
                {
                    "config": config,
                    "anchor": anchor,
                    "anti": anti,
                    "condition": c.name,
                    "seed": seed,
                    "label": f"{c.name}-s{seed}",
                    "prep": prep,
                }
            )
    return rows


def train_one(config, anchor: dict, anti: dict | None, corpus, traj_stride: int, probes, label: str) -> dict:
    """Train one run under the either-slot labeller, recording the m_line trajectory on the mix probe lines.

    Ex-2.1.11's step with the corpus arriving as a tree artifact rather than a volume path.
    """
    from sca.anchoring import AnchorSpec, AntiSpec, LabelSpec
    from sca.compute.training import train_anchored
    from mini.store import get, put

    workdir = get_data_dir() / "cells" / label
    corpus_dir = get(corpus, workdir / "corpus")
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens, slot_p, weights, line_p = z["mix/tokens"], z["slot_p"], z["weights"], z["mix/line_p"]

    # One line per color for the trajectory: at op1 the partner cannot matter; at the other span roles it
    # can, so the trajectory reads one fixed partner per color where the endpoint eval averages all 27.
    stride = len(probe_tokens) // len(weights)
    first_of_color = probe_tokens[::stride]
    line_w = line_p[::stride] / line_p[::stride].sum()

    _, metrics, traj = train_anchored(
        config,
        corpus_dir,
        anchor=AnchorSpec(**anchor),
        anti=AntiSpec(**anti) if anti is not None else None,
        label_p=LabelSpec(p=slot_p, keying="either", pull="span"),
        probe_tokens=first_of_color,
        probe_weights=weights,
        probe_line_w=line_w,
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
    )
    keep = ("epoch", "m_line", "m_op1", "m_span", "alpha_op1", "val_loss", "weight", "anti_weight")
    return {
        "label": label,
        "val_loss": [m.val_loss for m in metrics],
        "train_loss": [m.train_loss for m in metrics],
        "traj": {k: traj[k].tolist() for k in keep if k in traj},
        "checkpoint": put(workdir / "model", name=f"ex-2.2.3-{label}-ckpt"),
    }


def _load(trained: dict, workdir):
    """The run's model and tokenizer, plus the palette's token ids and the token → palette-index map."""
    from sca.compute.model import load_checkpoint
    from sca.data.named_colors import WordTokenizer
    from sca.model import NGPT
    from mini.store import get

    get(trained["checkpoint"], workdir / "model")
    model, config, _ = load_checkpoint(workdir)
    assert isinstance(model, NGPT), "the contract's operators act on nGPT's between-block stream"
    tokenizer = WordTokenizer(config.tokenizer)
    color_ids = np.array([tokenizer.stoi[n] for n in PALETTE])
    tok2color = np.full(model.transformer.wte.shape[0], -1)
    tok2color[color_ids] = np.arange(len(PALETTE))
    return model, tokenizer, color_ids, tok2color


def _probe_ops(z) -> list[str]:
    """The ops a probe file carries, in table order."""
    return [o for o in OP_NAMES if f"{o}/tokens" in z.files]


def eval_one(trained: dict, evals, probes, tau: float, condition: str, seed: int, label: str) -> dict:
    """The eval: behavior per op, and the alignment map, weight profiles and retention per op's probe lines.

    The gated statistics are the mix lines' (`PRIMARY_OP`), lifted to the top level; every op's are under
    `per_op` for E1 and E2. Per-line alignment goes to the store in half precision for E3.
    """
    from sca.anchoring import alignment
    from sca.data.ops import load_lines
    from mini.store import get, put

    workdir = get_data_dir() / "eval" / label
    model, tokenizer, color_ids, tok2color = _load(trained, workdir)
    vocab_rgb = GRID_RGB.astype(np.float32)
    palette_index = {rgb: i for i, rgb in enumerate(PALETTE.values())}

    # --- Behavior: one teacher-forced pass per (op, split), read at the pre-answer position.
    sets: dict[str, dict[str, dict]] = {}
    for op, splits in load_lines(get(evals, workdir / "evals.json").read_bytes()).items():
        sets[op] = {}
        for split, lns in splits.items():
            logp = _answer_logprobs(model, tokenizer, [ln.prompt for ln in lns])[:, color_ids]
            true_idx = np.array([palette_index[ln.result] for ln in lns])
            rows = np.arange(len(lns))
            guess = logp.argmax(axis=1)
            dists = np.linalg.norm(vocab_rgb[None] - vocab_rgb[true_idx][:, None], axis=2)
            sets[op][split] = {
                "n": len(lns),
                "accuracy": float((guess == true_idx).mean()),
                "nll": float(-logp[rows, true_idx].mean()),
                "guess_dist": float(dists[rows, guess].mean()),
            }

    # --- Alignment per op: cos(h, e₁) per slice × line × position, contracted per color for the grading
    #     and containment statistics and per line for m_line; then the softmin profiles at this run's τ.
    per_op: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    with np.load(get(probes, workdir / "probes.npz")) as z:
        weights = z["weights"]
        probe = {o: (z[f"{o}/tokens"], z[f"{o}/r1"], z[f"{o}/r2"], z[f"{o}/line_p"]) for o in _probe_ops(z)}
    for op, (tokens, r1, r2, line_p) in probe.items():
        n_partners = len(tokens) // len(weights)
        line_w = line_p / line_p.sum()
        g1, g2 = group_weights(r1, r2)
        cos = alignment(model, tokens)  # (L1, C*P, T)
        alpha = cos.reshape(cos.shape[0], len(weights), n_partners, cos.shape[2]).mean(axis=2)  # (L1, C, T)
        w_line = softmin_weights(1.0 - cos[:, :, :SPAN], tau)  # (L1, C*P, SPAN)
        w_color = w_line.reshape(cos.shape[0], len(weights), n_partners, SPAN).mean(axis=2)
        pi = np.einsum("lct,c->lt", w_color, weights)
        w_group = np.stack([np.einsum("lnt,n->lt", w_line, g) for g in (g1, g2)])  # (2, L1, SPAN)
        dose_ = np.maximum(r1, r2)
        redder = REDNESS[tok2color[tokens[:, ANSWER_POS]]] > dose_ + 1e-9
        per_op[op] = {
            "m_line": line_margin(cos, line_w),
            "m_span": pooled_margin(alpha, weights),
            "alpha_op1": float(alpha[:, :, 0].mean()),
            "r2_sim": r2_sim(alpha[:, :, 0].mean(axis=0)),
            "contrast": float(w_group[1, 1:, 2].mean() - w_group[0, 1:, 2].mean()),
            "latch_pi": float(max(pi[1:, 1].mean(), pi[1:, 3].mean())),
            "lead_emb": float(w_group[0, 0, 0]),
            "pi": pi.tolist(),
            "alpha_pos": cos.mean(axis=1).tolist(),  # (L1, T): the mean alignment at each position, E2
            "alpha_abs_pos": np.abs(cos).mean(axis=1).tolist(),
            "n_redder": int(redder.sum()),
        }
        arrays |= {
            f"{op}/alpha": alpha.astype(np.float32),
            f"{op}/alpha_lines": cos.astype(np.float16),
            f"{op}/pi": pi.astype(np.float32),
            f"{op}/w_group": w_group.astype(np.float32),
            f"{op}/dose": dose_.astype(np.float32),
            f"{op}/redder": redder,
        }

    traj = np.asarray(trained["traj"]["m_line"], dtype=float)
    peak = float(np.maximum.accumulate(traj).max())
    primary = per_op[PRIMARY_OP.name]
    return {
        "label": label,
        "condition": condition,
        "seed": seed,
        "tau": tau,
        "sets": sets,
        "holdout_em": {op: s["holdout"]["accuracy"] for op, s in sets.items()},
        **{k: primary[k] for k in ("m_line", "m_span", "alpha_op1", "r2_sim", "contrast", "latch_pi", "lead_emb")},
        "m_line_peak": peak,
        "retention": float(traj[-1] / peak) if peak > 0 else float("nan"),
        "per_op": per_op,
        "arrays": put(_npz(**arrays), name=f"ex-2.2.3-{label}-arrays.npz"),
    }


def _answer_logprobs(model, tokenizer, prompts: list[list[str]]) -> np.ndarray:
    """Full-vocabulary log-probabilities at the last prompt position, per prompt."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    forward = eqx.filter_jit(model.__call__)
    seq = np.array([tokenizer.encode_words(p) for p in prompts])
    out = [
        np.asarray(jax.nn.log_softmax(forward(jnp.asarray(seq[i : i + 256]))[:, -1], axis=-1))
        for i in range(0, len(seq), 256)
    ]
    return np.concatenate(out)


# --- Statistics shared with the report ---------------------------------------------------------


def p_slot(r: np.ndarray | float, rate: float = PER_SLOT_RATE) -> np.ndarray | float:
    """The labeller's per-operand rate at redness *r*."""
    return r**8 * rate


def line_margin(alpha_lines: np.ndarray, line_w: np.ndarray) -> float:
    """m_line: per-line selectivity, ex-2.1.10's scored statistic, unchanged.

    Per slice: the line-weighted mean alignment at each span role minus the unweighted mean, keeping the
    largest role; then the mean over slices.
    """
    m = np.einsum("n,lnt->lt", line_w, alpha_lines) - alpha_lines.mean(axis=1)  # (L1, T)
    return float(m[:, :SPAN].max(axis=1).mean())


def pooled_margin(alpha: np.ndarray, weights: np.ndarray) -> float:
    """m_span: the per-color margin maxed over the span, kept for continuity with ex-2.1.9."""
    m = np.einsum("c,lct->lt", weights, alpha) - alpha.mean(axis=1)  # (L1, T)
    return float(m[:, :SPAN].max(axis=1).mean())


def r2_sim(alpha_op1: np.ndarray) -> float:
    """The grading statistic: r² between the per-color op1 response and SIM_TARGET."""
    return float(np.corrcoef(alpha_op1, SIM_TARGET)[0, 1] ** 2)


def group_weights(r1: np.ndarray, r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-line weights for the two localization groups the contrast compares.

    G1 weights each probe line by P(op1 drew and op2 didn't); G2 by the reverse. Ex-2.1.10's construction.
    """
    p1, p2 = np.asarray(p_slot(r1)), np.asarray(p_slot(r2))
    g1 = p1 * (1.0 - p2)
    g2 = (1.0 - p1) * p2
    return g1 / g1.sum(), g2 / g2.sum()


def top_quantile(a, q: float = 99, axis: int = -1):
    """The *q*-th percentile as an order statistic: the ⌈n(1 − q/100)⌉-th largest value along *axis*, as
    ex-2.2.1 read it, so a monotone map of the values commutes with the quantile.
    """
    a = np.asarray(a)
    n = a.shape[axis]
    k = int(np.ceil(n * (1 - q / 100)))
    return np.take(np.sort(a, axis=axis), n - k, axis=axis)


# --- The eval contract on the candidates -------------------------------------------------------


def score_one(trained: dict, probes, condition: str, seed: int, label: str) -> dict:
    """Score one checkpoint on every op's probe lines: the clean pass, then each intervention.

    Ex-2.2.1's scorer with the op as an outer loop and two additions: the expected answer distance under the
    whole answer distribution, and a `redder` group (lines whose answer is redder than both operands) beside
    `all`, `red` and `nonred`. Per-run statistics return as the result; per-line arrays go to the store.
    """
    from sca.intervention import Subspace
    from mini.store import get, put

    workdir = get_data_dir() / "score" / label
    model, _, color_ids, tok2color = _load(trained, workdir)
    sub = Subspace.axis(model.transformer.wte.shape[1], ANCHOR_AXIS)
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe = {o: (z[f"{o}/tokens"], z[f"{o}/r1"], z[f"{o}/r2"]) for o in _probe_ops(z)}

    ops: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    for op, (tokens, r1, r2) in probe.items():
        stats, per_line = _score_op(model, sub, tokens, r1, r2, color_ids, tok2color)
        ops[op] = stats
        arrays |= {f"{op}/{k}": v for k, v in per_line.items()}
    return {
        "label": label,
        "condition": condition,
        "seed": seed,
        "ops": ops,
        "arrays": put(_npz(**arrays), name=f"ex-2.2.3-{label}-score.npz"),
    }


class _Readout:
    """Per-line readings of one answer distribution, on one op's probe lines.

    P(answer), the argmax and its distance, the expected distance under the whole answer distribution (the
    color mass renormalized), the off-vocabulary mass, and what the argmax is (`COMPOSITION`).
    """

    def __init__(self, tokens: np.ndarray, r1: np.ndarray, r2: np.ndarray, color_ids: np.ndarray, tok2color):
        self.tokens, self.color_ids, self.tok2color = tokens, color_ids, tok2color
        self.rows = np.arange(len(tokens))
        lc = tok2color[tokens]  # (N, T) palette index, −1 at the syntax positions
        assert (lc[:, [0, 2, ANSWER_POS]] >= 0).all() and (lc[:, [1, 3, 5]] < 0).all()
        self.answer = tokens[:, ANSWER_POS]
        self.ans_idx = lc[:, ANSWER_POS]
        self.dose = np.maximum(r1, r2)
        eps = 1e-9
        self.groups = {
            "all": np.ones(len(tokens), bool),
            "red": self.dose >= RED_DOSE - eps,
            "nonred": self.dose <= NONRED_DOSE + eps,
            "redder": REDNESS[self.ans_idx] > self.dose + eps,
        }
        self.dists = np.linalg.norm(GRID_RGB[None] - GRID_RGB[self.ans_idx][:, None], axis=2)  # (N, 216)
        self.red_operand = np.where(r1 >= r2, 0, 2)
        self.cube = np.stack(np.unravel_index(np.arange(len(PALETTE)), (6, 6, 6)), axis=1)

    def by_group(self, v: np.ndarray) -> dict[str, float]:
        return {g: float(np.nanmean(v[m])) if m.any() else float("nan") for g, m in self.groups.items()}

    def __call__(self, logits) -> dict[str, np.ndarray]:
        from sca.intervention import answer_logprobs

        lp = answer_logprobs(logits, ANSWER_POS)
        p_color = np.exp(lp[:, self.color_ids])
        guess = lp.argmax(1)
        g = self.tok2color[guess]
        step = np.abs(self.cube[np.maximum(g, 0)] - self.cube[self.ans_idx]).max(1)
        tests = [
            guess == self.answer,
            guess == self.tokens[self.rows, self.red_operand],
            guess == self.tokens[self.rows, 2 - self.red_operand],
            (g >= 0) & (step == 1),
        ]
        return {
            "p_ans": np.exp(lp[self.rows, self.answer]),
            "guess": guess,
            "argmax_dist": np.where(g >= 0, self.dists[self.rows, np.maximum(g, 0)], np.nan),
            "expected_dist": (p_color / p_color.sum(1, keepdims=True) * self.dists).sum(1),
            "offvocab": 1.0 - p_color.sum(1),
            "composition": np.select(tests, [0, 1, 2, 3], default=4),
        }


def _score_op(model, sub, tokens, r1, r2, color_ids, tok2color) -> tuple[dict, dict[str, np.ndarray]]:
    """One op's probe lines through the clean pass and every intervention: (statistics, per-line arrays)."""
    from sca.intervention import ablate_weights, angle_between, apply, projection, shaped_suppression, write_angle

    read = _Readout(tokens, r1, r2, color_ids, tok2color)
    n_pos = tokens.shape[1]
    nonred = read.groups["nonred"]

    # --- The clean pass: the scorer's own control, and the reference every intervention is read against.
    clean = apply(model, tokens, projection(sub), slices=())
    alpha_clean = clean.pre[..., ANCHOR_AXIS]  # (L1, N, T)
    alpha_q99 = top_quantile(np.abs(alpha_clean[:, nonred]), axis=1)  # (L1, T)
    bound = np.arcsin(alpha_q99)
    c = read(clean.logits)
    stats: dict[str, Any] = {
        "n": {g: int(m.sum()) for g, m in read.groups.items()},
        "clean": {
            "acc": read.by_group(c["guess"] == read.answer),
            "p_ans": read.by_group(c["p_ans"]),
            "argmax_dist": read.by_group(c["argmax_dist"]),
            "expected_dist": read.by_group(c["expected_dist"]),
            "bound": bound.tolist(),
            "alpha_q99_nonred": alpha_q99.tolist(),
        },
        "interventions": {},
    }
    arrays: dict[str, np.ndarray] = {
        "clean/p_ans": c["p_ans"].astype(np.float32),
        "clean/guess": c["guess"].astype(np.int16),
        "clean/expected_dist": c["expected_dist"].astype(np.float32),
        "dose": read.dose.astype(np.float32),
        "redder": read.groups["redder"],
    }

    # --- Each intervention: build the triple, score it, check the contract, contract per-line quantities.
    for iv in (PROJECTION, *RIDE_ALONG):
        positions = None if iv.positions is None else np.isin(np.arange(n_pos), iv.positions).astype(np.float32)
        match iv.kind:
            case "projection":
                target, operator = model, projection(sub, gamma=iv.gamma)
            case "shaped":
                target, operator = model, shaped_suppression(sub, a=SHAPED["a"], b=SHAPED["b"], p=SHAPED["p"])
            case "ablate":
                target, operator = ablate_weights(model, sub), projection(sub)
            case _:
                raise ValueError(iv.kind)
        out = apply(target, tokens, operator, slices=iv.slices, positions=positions)
        r = read(out.logits)
        theta = angle_between(out.pre, out.post)  # (L1, N, T): the write at every site
        alpha_pre = out.pre[..., ANCHOR_AXIS]
        disp = angle_between(out.post[-1, :, DECODE_POS], clean.post[-1, :, DECODE_POS])

        # The contract's promises: the embedding sees the clean stream unless the weights were edited,
        # nothing moves where the operator was skipped, a projection's write is the closed form of the
        # alignment it saw, and an ablated model never carries the axis.
        if iv.kind != "ablate":
            np.testing.assert_allclose(alpha_pre[0], alpha_clean[0], rtol=0, atol=0)
        skipped = np.ones((len(SLICES), n_pos), bool)
        for s in iv.slices:
            skipped[s] = False if positions is None else positions == 0
        off = skipped.nonzero()
        np.testing.assert_allclose(theta[off[0], :, off[1]], 0.0, rtol=0, atol=1e-6)
        if iv.kind == "projection":
            on = (~skipped).nonzero()
            expected = write_angle(alpha_pre[on[0], :, on[1]], iv.gamma)
            np.testing.assert_allclose(theta[on[0], :, on[1]], expected, rtol=0, atol=2e-3)
        if iv.kind == "ablate":
            assert np.abs(alpha_pre).max() < 1e-6, "an ablated model never carries the axis"
        q99_write = top_quantile(theta[:, nonred], axis=1)  # (L1, T)
        if iv.name == PROJECTION.name:  # H4's embedding identity, to float32 roundoff
            np.testing.assert_allclose(q99_write[0], bound[0], rtol=0, atol=1e-3)

        stats["interventions"][iv.name] = {
            "kind": iv.kind,
            "acc": read.by_group(r["guess"] == read.answer),
            "p_ans": read.by_group(r["p_ans"]),
            "deficit": read.by_group(c["p_ans"] - r["p_ans"]),
            "argmax_dist": read.by_group(r["argmax_dist"]),
            "expected_dist": read.by_group(r["expected_dist"]),
            "offvocab": read.by_group(r["offvocab"]),
            "disp": read.by_group(disp),
            "q99_write_nonred": q99_write.tolist(),
            "q99_alpha_nonred": top_quantile(np.abs(alpha_pre[:, nonred]), axis=1).tolist(),
            "composition_red": np.bincount(r["composition"][read.groups["red"]], minlength=len(COMPOSITION)).tolist(),
            "composition_redder": np.bincount(
                r["composition"][read.groups["redder"]], minlength=len(COMPOSITION)
            ).tolist(),
        }
        arrays |= {
            f"{iv.name}/p_ans": r["p_ans"].astype(np.float32),
            f"{iv.name}/guess": r["guess"].astype(np.int16),
            f"{iv.name}/expected_dist": r["expected_dist"].astype(np.float32),
            f"{iv.name}/composition": r["composition"].astype(np.int8),
        }
    return stats, arrays


# --- Cube probes on the un-anchored arms (E4) -------------------------------------------------


def probe_one(trained: dict, probes, condition: str, seed: int, label: str) -> dict:
    """Strict-holdout ridge probes for the RGB of op1, op2 and the answer at every (slice, position) site.

    Read on the mix probe lines of every arm, since every arm carries `mix` and the lines are the ones
    ex-2.1.12 probed on D2.1's models. The per-value holdout is `sca.compute.geometry.strict_r2`'s: to score
    channel k at level v, every line holding level v in channel k of any slot leaves the fit together.
    """
    import equinox as eqx
    import jax.numpy as jnp

    from sca.compute.geometry import strict_r2
    from mini.store import get

    workdir = get_data_dir() / "probe" / label
    model, _, _, tok2color = _load(trained, workdir)
    with np.load(get(probes, workdir / "probes.npz")) as z:
        tokens = z["mix/tokens"]
    targets = {"op1": 0, "op2": 2, "ans": ANSWER_POS}
    rgb = GRID_RGB[tok2color[tokens[:, list(targets.values())]]]  # (N, 3 roles, 3 ch)
    slots = [rgb[:, i] for i in range(rgb.shape[1])]

    stream = eqx.filter_jit(model.residual_stream)
    acts = np.concatenate(
        [np.asarray(stream(jnp.asarray(tokens[i : i + 1024]))) for i in range(0, len(tokens), 1024)], axis=1
    )  # (slices, N, positions, width)
    n_slices, _, n_pos, _ = acts.shape

    r2 = np.full((n_slices, n_pos, len(targets), 3), np.nan, np.float32)
    for ti, y in enumerate(slots):
        for k in range(3):
            folds = [
                (np.any([np.abs(s[:, k] - v) < 1e-6 for s in slots], axis=0), np.abs(y[:, k] - v) < 1e-6)
                for v in np.unique(y[:, k])
            ]
            for si in range(n_slices):
                for pi in range(n_pos):
                    r2[si, pi, ti, k] = strict_r2(acts[si, :, pi], y[:, k], folds)
    return {
        "label": label,
        "condition": condition,
        "seed": seed,
        "targets": list(targets),
        "r2_strict": r2.tolist(),  # (slice, position, target, channel)
    }


# --- Publishing ------------------------------------------------------------------------------


def design() -> dict[str, Any]:
    """The design constants the report checks its rendered values against."""
    return {
        "n_runs": N_RUNS,
        "conditions": [asdict(c) | {"steps": c.steps, "lines_per_op": c.lines_per_op} for c in CONDITIONS],
        "candidates": [c.name for c in CANDIDATES],
        "richer_op_arm": [c.name for c in RICHER_OP_ARM],
        "primary_op": PRIMARY_OP.name,
        "n_probe": N_PROBE,
        "probe_seed": PROBE_SEED,
        "span": SPAN,
        "epochs": {"long": EPOCHS, "short": EPOCHS_SHORT},
        "steps_per_epoch": steps_per_epoch(),
        "noise_run": NOISE_RUN,
        "gates": {
            "task": TASK_GATE,
            "task_partial": TASK_PARTIAL,
            "ref_m_line": REF_M_LINE,
            "ref_r2_sim": REF_R2_SIM,
            "margin_ratio": MARGIN_RATIO,
            "margin_partial": MARGIN_PARTIAL,
            "mean_align": MEAN_ALIGN_GATE,
            "retention_floor": RETENTION_FLOOR,
            "retention": RETENTION_GATE,
            "lead": LEAD_GATE,
            "contrast": CONTRAST_GATE,
            "contrast_partial": CONTRAST_PARTIAL,
            "grade_r2_drop": GRADE_R2_DROP,
            "latch": LATCH_PI,
            "red_acc": RED_ACC_GATE,
            "nonred_deficit": NONRED_DEFICIT_GATE,
            "nonred_deficit_partial": NONRED_DEFICIT_PARTIAL,
            "resolution_sd": RESOLUTION_SD,
            "grade_margin_sd": GRADE_MARGIN_SD,
        },
        "interventions": [asdict(iv) for iv in (PROJECTION, *RIDE_ALONG)],
        "shaped": SHAPED,
        "dose": {"red": RED_DOSE, "nonred": NONRED_DOSE},
        "composition": list(COMPOSITION),
    }


def _slim(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("arrays", "traj", "val_loss", "train_loss")}


def publish_calibration(evaled: list[dict]) -> dict:
    """The pre-freeze calibration: one seed of each control arm, as plain metrics under `CALIBRATION_REF`."""
    import json

    from mini.store import put, set_ref

    payload = {"runs": [_slim(r) for r in evaled], "design": design()}
    set_ref(CALIBRATION_REF, put(json.dumps(payload, indent=2).encode(), name="ex-2.2.3-calibration.json"))
    return {"stage": "calibration", "holdout_em": {r["label"]: r["holdout_em"] for r in evaled}}


def publish_results(
    trained: list[dict], evaled: list[dict], scored: list[dict], probed: list[dict], corpora: list[dict], probes
) -> dict:
    """Publish the frozen stage: metrics (JSON), trajectories (JSON), stacked per-run arrays (npz), the
    six-op probe set, the cube probes (JSON), and every checkpoint under its own ref.
    """
    import json

    from mini.store import get, put, set_ref

    metrics = {
        "runs": [_slim(r) for r in evaled],
        "scores": [_slim(r) for r in scored],
        "corpora": corpora,
        "design": design(),
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.2.3-metrics.json"))
    traj = {t["label"]: {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.3-trajectories.json"))
    set_ref(GEOMETRY_REF, put(json.dumps({"runs": probed}).encode(), name="ex-2.2.3-geometry.json"))
    set_ref(PROBE_REF, probes)
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])

    arrays = {}
    for r in evaled + scored:
        kind = "eval" if "per_op" in r else "score"
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}-{kind}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{kind}/{name}": z[name] for name in z.files}
    set_ref(ARRAYS_REF, put(_npz(**arrays), name="ex-2.2.3-arrays.npz"))
    return {
        "stage": "full",
        "n_runs": len(evaled),
        "n_scored": len(scored),
        "n_probed": len(probed),
        "holdout_em": {r["label"]: r["holdout_em"] for r in evaled},
    }


# --- Orchestration ----------------------------------------------------------------------------


def _stages() -> set[str]:
    """Which stage this wake may launch — `EX223_STAGES`, `full` by default.

    `calibration` trains and evaluates one seed of each control arm and stops, so the corpus can be checked
    for learnability before the design freezes. The full stage re-uses those two runs.
    """
    import os

    return {s.strip() for s in os.environ.get("EX223_STAGES", "full").split(",")}


def _train_and_eval(ctx: Ctx, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    n = len(rows)
    trained = ctx.map(
        train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [r["prep"]["corpus"] for r in rows],
        [TRAJ_STRIDE] * n,
        [r["prep"]["probes"] for r in rows],
        [r["label"] for r in rows],
        role="train",
    )
    evaled = ctx.map(
        eval_one,
        trained,
        [r["prep"]["evals"] for r in rows],
        [r["prep"]["probes"] for r in rows],
        [r["anchor"]["tau"] for r in rows],
        [r["condition"] for r in rows],
        [r["seed"] for r in rows],
        [r["label"] for r in rows],
        role="eval",
    )
    return trained, evaled


def main(ctx: Ctx) -> dict:
    keys = list(dict.fromkeys(corpus_key(c.ops, c.n_lines) for c in CONDITIONS))
    specs = {corpus_key(c.ops, c.n_lines): (c.ops, c.n_lines) for c in CONDITIONS}
    prepped = ctx.map(
        prepare_corpus,
        [specs[k][0] for k in keys],
        [specs[k][1] for k in keys],
        [CORPUS_SEED] * len(keys),
        [HOLDOUT_FRAC] * len(keys),
        [N_PROBE] * len(keys),
        [PROBE_SEED] * len(keys),
        [PER_SLOT_RATE] * len(keys),
        [RED_RATE] * len(keys),
        role="prep",
    )
    preps = dict(zip(keys, prepped, strict=True))

    if _stages() == {"calibration"}:
        rows = cells((CONTROL, CONTROL_SHORT), preps, seeds={CONTROL.name: [0], CONTROL_SHORT.name: [0]})
        _, evaled = _train_and_eval(ctx, rows)
        return ctx.run(publish_calibration, evaled, role="prep")

    rows = cells(CONDITIONS, preps)
    trained, evaled = _train_and_eval(ctx, rows)
    by_label = dict(zip([r["label"] for r in rows], zip(rows, trained, strict=True), strict=True))

    candidates = [by_label[r["label"]] for r in rows if r["condition"] in {c.name for c in CANDIDATES}]
    scored = ctx.map(
        score_one,
        [t for _, t in candidates],
        [r["prep"]["probes"] for r, _ in candidates],
        [r["condition"] for r, _ in candidates],
        [r["seed"] for r, _ in candidates],
        [r["label"] for r, _ in candidates],
        role="score",
    )
    cube = [by_label[r["label"]] for r in rows if r["condition"] in {c.name for c in (CONTROL, *RICHER_OP_ARM)}]
    probed = ctx.map(
        probe_one,
        [t for _, t in cube],
        [r["prep"]["probes"] for r, _ in cube],
        [r["condition"] for r, _ in cube],
        [r["seed"] for r, _ in cube],
        [r["label"] for r, _ in cube],
        role="probe",
    )
    six = corpus_key(CONTROL.ops, CONTROL.n_lines)
    return ctx.run(
        publish_results,
        trained,
        evaled,
        scored,
        probed,
        [preps[k]["stats"] for k in keys],
        preps[six]["probes"],
        role="prep",
    )


experiment = Experiment(
    name="ex-2.2.3",
    main=main,
    roles={
        # Corpus sampling is a plain-numpy loop over 100k lines; six probe sets of 5,832 lines beside it.
        "prep": dict(cpu=2, timeout=900),
        # 3,300 steps of 64×64 tokens (half that for the short arms), as ex-2.1.11. The watchdog is sized
        # for the gap after the last step: the checkpoint upload emits no step progress.
        "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
        # Twelve teacher-forced eval sets and up to six probe passes over 5,832 lines each.
        "eval": dict(gpu="L4", timeout=1200),
        # Four operators × six ops, each a forward pass over 5,832 lines keeping the whole stream.
        "score": dict(gpu="L4", timeout=1800),
        # One stream capture and 270 grouped ridge fits on 5,832 × 64 activations: CPU work.
        "probe": dict(cpu=4, timeout=1800),
    },
)
