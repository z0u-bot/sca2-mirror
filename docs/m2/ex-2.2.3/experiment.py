"""
Experiment 2.2.3: the multi-op grammar, with *red* anchored again.

Design constants only, for now. The report imports these so its prose and the
gates cannot drift apart; the DAG lands once the preregistration is frozen, and
`DESIGN_ONLY` goes with it.

The D2.2 plan makes the operation a variable: four operations on the same color
grid, each spelled as a word, so a line reads `c1 <op> c2 = answer`. Every D2.1
recipe was tuned on the one-op grammar, so this experiment is the regression
check the plan schedules before any operation is anchored: an un-anchored
control, the ex-2.1.10 reference recipe, and the ex-2.1.11 survey's proposals,
all retrained on the new grammar at fresh seeds and scored through the eval
contract (`sca.intervention`) beside the D2.1 numbers. A selection rule frozen
here names the operating point the anchored-op experiments adopt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sca.config import DataConfig, ModelConfig
from sca.data.batches import batches_per_epoch
from sca.data.colors import N_LEVELS, Rgb, mix, redness
from sca.data.named_colors import GRIDS

DESIGN_ONLY = True

#: Result refs the report will read.
METRICS_REF = "reports/m2/ex-2.2.3/metrics"
ARRAYS_REF = "reports/m2/ex-2.2.3/arrays"
TRAJ_REF = "reports/m2/ex-2.2.3/trajectories"
PROBE_REF = "reports/m2/ex-2.2.3/probes"
GEOMETRY_REF = "reports/m2/ex-2.2.3/geometry"

#: Stored results the report compares against: the D2.1 primary (nine seeds) and the survey's proposals.
EX2110_METRICS_REF = "reports/m2/ex-2.1.10/metrics"
SURVEY_REF = "reports/m2/ex-2.1.11/survey"

# --- The operation table -----------------------------------------------------------------

GRID = "v216"
LEVELS = GRIDS[GRID]
"""The six channel levels of the corpus grid: 0, 3, 6, 9, 12, 15. Unchanged from D2.1."""

TOP = N_LEVELS - 1


@dataclass(frozen=True)
class Op:
    """One operation: the word the model sees, and the per-channel rule on the 0..15 scale."""

    name: str
    """The op's name in code and prose, and its surface form: the one token between the operands."""
    channel: Callable[[int, int], int]
    rule: str
    """The per-channel rule, for the method's table."""

    def __call__(self, a: Rgb, b: Rgb) -> Rgb:
        r, g, b_ = (self.channel(x, y) for x, y in zip(a, b, strict=True))
        return (r, g, b_)


OPS = (
    Op("mix", lambda x, y: (x + y + 1) // 2, "⌊(x + y + 1) / 2⌋"),
    Op("add", lambda x, y: min(x + y, TOP), "min(x + y, 15)"),
    Op("lighten", max, "max(x, y)"),
    Op("darken", min, "min(x, y)"),
)
"""The first op table. `mix` is D2.1's op (the round-half-up mean, spelled `+` there and `mix` here). The
other three are the blend modes that stay closed on the grid: saturating `add`, and the per-channel `max`
and `min` (Photoshop's *lighten* and *darken*).

The D2.2 design named `screen` and `multiply` for the second and third slots. On the six-level grid neither
survives the closed-pair rule: `screen` lands on the grid only where a channel is 0 or 15, where it equals
`add`, and `multiply` only where it is 0 or the identity, so each contributes no line that another op does
not already teach. `agreement()` and `closure()` below are what settled it, and the report renders them.

All four ops are commutative, so operand order carries no information, as in D2.1."""

MIX, ADD, LIGHTEN, DARKEN = OPS
OP_NAMES = tuple(op.name for op in OPS)


def colors() -> list[Rgb]:
    """The 216 colors of the grid, in (r, g, b) order."""
    return [(r, g, b) for r in LEVELS for g in LEVELS for b in LEVELS]


def unordered_pairs() -> list[tuple[Rgb, Rgb]]:
    """Every unordered pair of grid colors, self-pairs included (23,436 of them)."""
    cs = colors()
    return [(a, b) for i, a in enumerate(cs) for b in cs[i:]]


def is_closed(op: Op, a: Rgb, b: Rgb) -> bool:
    """A pair is closed under an op when the op's answer is itself a grid color, so the line can be written."""
    return all(v in LEVELS for v in op(a, b))


def closed_pairs(op: Op) -> list[tuple[Rgb, Rgb]]:
    return [(a, b) for a, b in unordered_pairs() if is_closed(op, a, b)]


def closure(op: Op) -> float:
    """The fraction of unordered pairs closed under the op."""
    return len(closed_pairs(op)) / len(unordered_pairs())


def agreement(p: Op, q: Op) -> tuple[int, int]:
    """(pairs where p and q give the same answer, pairs closed under both)."""
    both = [(a, b) for a, b in unordered_pairs() if is_closed(p, a, b) and is_closed(q, a, b)]
    return sum(p(a, b) == q(a, b) for a, b in both), len(both)


def relevance(anchored: Op) -> dict[float, float]:
    """The *op-relevance* distribution over the anchored op's closed pairs.

    For a line, relevance is the weight the mixture over ops closed on that pair withholds from the anchored
    op's answer: 1 − (share of those ops whose answer equals the anchored op's). It is the fraction of the
    answer that reading the op is worth on that line. Returned as {relevance: fraction of lines}; the D2.2
    design asks for this per candidate anchored op, under the table's own rounding.
    """
    values = []
    for a, b in closed_pairs(anchored):
        answers = [op(a, b) for op in OPS if is_closed(op, a, b)]
        values.append(round(1 - answers.count(anchored(a, b)) / len(answers), 4))
    vals, counts = np.unique(values, return_counts=True)
    return {float(v): float(c) / len(values) for v, c in zip(vals, counts, strict=True)}


def ordered_lines(op: Op) -> list[tuple[Rgb, Rgb]]:
    """Every writable line of the op: each closed pair in both orders (self-pairs once)."""
    return [(x, y) for a, b in closed_pairs(op) for (x, y) in ({(a, b), (b, a)})]


def dose(a: Rgb, b: Rgb) -> float:
    """How *red* a line is: the larger of its two operand rednesses."""
    return max(redness(a), redness(b))


def line_counts(op: Op) -> dict[str, int]:
    """Over the op's writable lines: the total, the red and non-red lines, and the redder-than-both lines."""
    lines = ordered_lines(op)
    return {
        "lines": len(lines),
        "red": sum(dose(a, b) >= RED_DOSE for a, b in lines),
        "nonred": sum(dose(a, b) <= NONRED_DOSE for a, b in lines),
        "redder": sum(redness(op(a, b)) > dose(a, b) + 1e-9 for a, b in lines),
    }


def check_table() -> None:
    """The table's invariants: `mix` is the D2.1 op, every op is commutative, and the three new ops are closed everywhere."""
    for a, b in unordered_pairs()[::97]:
        assert MIX(a, b) == mix(a, b)
        for op in OPS:
            assert op(a, b) == op(b, a)
    for op in (ADD, LIGHTEN, DARKEN):
        assert closure(op) == 1.0, op.name
    assert len(ordered_lines(MIX)) == 5832, "mix's writable lines are D2.1's probe set"


# --- The lines -----------------------------------------------------------------------------

RED_DOSE = 0.8
"""*Red lines* have dose ≥ 0.8: 365 of `mix`'s 5,832 lines, the ex-2.2.1 definition."""

NONRED_DOSE = 0.2
"""*Non-red lines* have dose ≤ 0.2: 1,689 of `mix`'s lines."""

check_table()

# --- The corpus ------------------------------------------------------------------------------

N_LINES_PER_OP = 50_000
"""Lines per op; the corpus samples ops uniformly, and within an op draws closed training pairs uniformly with
random operand order, as ex-2.1.10 draws `mix`'s. Half of D2.1's count per op, so the corpus is twice D2.1's
size and every `mix` pair is seen half as often per epoch, and a quarter as often over the matched-step run."""

N_LINES = N_LINES_PER_OP * len(OPS)
CORPUS_SEED = 0
HOLDOUT_FRAC = 0.2
"""Of the distinct closed pairs, held out per op: bookkeeping is keyed on (op, pair), so a pair held out under
`add` may be trained under `mix`."""

TOKENS_PER_LINE = 6
BLOCK, BATCH = 64, 64
OVERSAMPLE, TRAIN_SPLIT = 16, 0.9
"""Ex-2.1.3's DataConfig, unchanged through D2.1."""


def steps_per_epoch(n_lines: int) -> int:
    """Optimizer steps per epoch for a corpus of *n_lines*, from the loader's own sizing rule.

    An epoch covers `oversample / batch_size` of the training split, so the count is close to
    0.9 · 6 · n_lines / 16,384: 33 steps for D2.1's 100k lines, 66 for this corpus.
    """
    n_tokens = int(TRAIN_SPLIT * n_lines * TOKENS_PER_LINE)
    model = ModelConfig(vocab_size=256, block_size=BLOCK, n_embd=64, n_head=8, n_head_dim=8, n_ff=256, n_layer=4)
    data = DataConfig(batch_size=BATCH, oversample=OVERSAMPLE, train_split=TRAIN_SPLIT, padding_chance=0.1)
    return batches_per_epoch(n_tokens, data, model)


D21_LINES = 100_000
D21_EPOCHS = 100
"""Ex-2.1.10's corpus and length: the reference recipe ran 100 epochs of 100k lines."""

SURVEY_EPOCHS = 50
"""The survey's trial length, on the same 100k lines: its proposals were made at half the reference's steps."""

D21_STEPS = D21_EPOCHS * steps_per_epoch(D21_LINES)
SURVEY_STEPS = SURVEY_EPOCHS * steps_per_epoch(D21_LINES)

EPOCHS_RECIPE = round(D21_STEPS / steps_per_epoch(N_LINES))
EPOCHS_PROPOSAL = round(SURVEY_STEPS / steps_per_epoch(N_LINES))
"""Training is matched to D2.1 by *steps*, since the corpus is twice the size: the recipe runs the same number
of steps as ex-2.1.10 (50 epochs here), and each proposal the same number as its survey trial (25). Every
schedule keyframe is a fraction of training, as ex-2.1.11 restated them, so the shapes are unchanged."""

assert (EPOCHS_RECIPE, EPOCHS_PROPOSAL) == (50, 25), (EPOCHS_RECIPE, EPOCHS_PROPOSAL)

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
    epochs: int = EPOCHS_RECIPE
    anti_peak_ratio: float = ANTI_PEAK_RATIO_REF
    anti_anneal_end_frac: float = ANTI_ANNEAL_END_FRAC_REF
    anchor_anneal: bool = True
    """Ex-2.1.10 anneals the anchor weight over the last tenth of training; the survey's trials ran it flat
    (its ablation found the two within a band), so its proposals are flat here too."""
    ops: tuple[str, ...] = OP_NAMES
    """Which ops the corpus carries. Only the richer-op arm narrows this, and its corpus keeps the full line
    count, so fewer ops means more lines per op at the same step count."""
    survey_trial: int | None = None
    """For a proposal, the ex-2.1.11 trial it re-measures; the report reads that trial's five-seed numbers
    from `SURVEY_REF` and prints them beside the fresh ones."""

    @property
    def steps(self) -> int:
        return self.epochs * steps_per_epoch(N_LINES)


CONTROL = Condition("control", 5, "un-anchored", lam=0.0)
"""Nothing placed on the axis: the anchor and anti-subspace weights at zero. The task reference for H1 and the
4-op point of the richer-op arm."""

RECIPE = Condition("recipe", 5, "the ex-2.1.10 recipe", lam=SCORING_LAMBDA)
"""The D2.1 primary retrained on the new grammar at the same step count: the placement reference for H2, and
the incumbent in H3's selection."""

T00 = Condition(
    "t00",
    5,
    "survey proposal t00",
    lam=0.5573771884288644,
    tau=0.2652855688754236,
    epochs=EPOCHS_PROPOSAL,
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
    epochs=EPOCHS_PROPOSAL,
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
    epochs=EPOCHS_PROPOSAL,
    anti_peak_ratio=1.1284021176786219,
    anti_anneal_end_frac=0.9193784880917519,
    anchor_anneal=False,
    survey_trial=12,
)
"""The knee: the promoted trial with the most grading, at a margin the survey could still not tell from t00's."""

PROPOSALS = (T00, T48, T12)
CANDIDATES = (RECIPE, *PROPOSALS)
"""The set H3's selection rule chooses from."""

RICHER_OP_ARM = (
    Condition("ops-1", 3, "un-anchored, mix only", lam=0.0, ops=("mix",)),
    Condition("ops-2", 3, "un-anchored, mix and add", lam=0.0, ops=("mix", "add")),
)
"""Un-anchored models at one and two ops, on a corpus of the same size (200k lines) at the same step count,
probed for the cube as ex-2.1.12 probed D2.1's; the control is the four-op point. Unscored (E4). Holding the
line count fixed means the one-op model sees each `mix` pair four times as often, so a better cube at four ops
would be a clean positive, and a worse one is confounded with lines per op."""

CONDITIONS = (CONTROL, *CANDIDATES, *RICHER_OP_ARM)
N_RUNS = sum(c.seeds for c in CONDITIONS)
assert N_RUNS == 31

# --- The probe set ---------------------------------------------------------------------------

N_PROBE = 27
"""Probe lines per color as op1. For `mix` that is every closed partner, so its probe set is D2.1's 5,832 lines
with the op word changed. The other ops are closed on every pair, so 27 partners per color are drawn with a
fixed seed; each op then has 5,832 probe lines, and the probe set is four times D2.1's."""

PROBE_SEED = 0
PRIMARY_OP = MIX
"""Every gated statistic is read on `mix`'s lines, the lines D2.1's numbers were read on; the other ops' values
are reported beside them (E1)."""

# --- Gates -----------------------------------------------------------------------------------

TASK_GATE = 0.02
"""H1: "seed-mean holdout exact-match accuracy within 0.02 of the control's, on every op", for every scored
condition. The width every D2.1 task gate used."""

TASK_PARTIAL = 0.05
"""H1 partial: "within 0.05"."""

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
"""H2 partial: "between 0.6 and 0.8 of it"."""

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
reads each proposal's r² against the recipe's fresh value at the same width."""

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
them. The bands below use these until the recipe condition here gives fresh ones on the new grammar; the
report prints both."""


def equiv_band(stat: str, n_a: int = 5, n_b: int = 5, noise: dict[str, float] = NOISE_RUN) -> float:
    """The smallest seed-mean difference the resolution rule may call a difference."""
    return RESOLUTION_SD * noise[stat] * np.sqrt(1 / n_a + 1 / n_b)


SELECTION_RULE = """\
Among the candidates (the recipe and the three proposals), keep those feasible at fresh seeds on every seed \
mean: task within the gate on every op, containment, retention, no latched run, contrast at least the partial \
floor, and grading r² no more than 0.10 below the recipe's fresh value, clearing that floor by at least one per-run σ. Among \
those, require contrast at the full gate; adopt the highest seed-mean m_line on the mix lines. Two candidates \
within one m_line band of each other tie, and the tie goes to the larger grading margin, then to the recipe. \
If no proposal is feasible, the recipe is adopted, and the D2.2 experiments carry its point."""

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
