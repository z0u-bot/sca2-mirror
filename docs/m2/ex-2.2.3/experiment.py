"""
Experiment 2.2.3: the multi-op grammar, with *red* anchored again.

Design constants only, for now. The report imports these so its prose and the
gates cannot drift apart; the DAG lands once the preregistration is frozen, and
`DESIGN_ONLY` goes with it.

The D2.2 plan makes the operation a variable: six operations on the same color
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


# REVIEW: `mix` was D2.1's round-half-up mean, (x + y + 1) // 2 on the 0..15 scale. On this grid a
# channel sum is always a multiple of 3, so where it is odd the rule rounded up by half a unit and the snap
# by another whole one: the answer rose by 1.5 on half of every op1-op2 channel sum, a lightening bias
# on the 87% of pairs that round. `mix` is now the plain mean, and the snap sends a tie to the even level
# index (0, 6, 12), so ties round down on 10 of the 18 channel-sum cases and up on 8: the mean signed
# error per channel is −0.08 rather than +0.75. No other op ties. The on-grid pairs, and so the probe
# set, are unchanged. Verify: `check_table` still asserts `MIX` agrees with D2.1's `mix` on every probe
# line, and `on_grid(MIX)` is unchanged.
def snap(v: float) -> int:
    """The grid level nearest to *v*. A tie (only `mix` has them, on channel sums with an odd multiple of 3)
    goes to the even level index, 0, 6, or 12, so ties round down about as often as up.
    """
    lo = max((level for level in LEVELS if level <= v), default=LEVELS[0])
    hi = min((level for level in LEVELS if level >= v), default=LEVELS[-1])
    if abs(v - lo) != abs(hi - v):
        return lo if abs(v - lo) < abs(hi - v) else hi
    return lo if LEVELS.index(lo) % 2 == 0 else hi


@dataclass(frozen=True)
class Op:
    """One operation: the word the model sees, and the per-channel rule on the 0..15 scale.

    The rule is computed on the continuous scale and snapped to the nearest grid level, so every op is
    defined on every pair and each answer is a vocabulary token. Where the rule already lands on a level
    (every pair for `add`, `lighten`, and `darken`; the on-grid fraction for the rest) no rounding happens,
    and on those pairs `mix` is D2.1's op unchanged. The five rules are the blend modes of the same names
    in Photoshop and Krita (`add` is *linear dodge* in Photoshop), on the 0..15 scale; `mix` is a normal
    blend at half opacity, the per-channel mean.
    """

    name: str
    """The op's name in code and prose, and its surface form: the one token between the operands."""
    channel: Callable[[int, int], float]
    rule: str
    """The per-channel rule, for the method's table."""

    def raw(self, a: Rgb, b: Rgb) -> tuple[float, float, float]:
        r, g, b_ = (self.channel(x, y) for x, y in zip(a, b, strict=True))
        return (r, g, b_)

    def __call__(self, a: Rgb, b: Rgb) -> Rgb:
        r, g, b_ = (snap(v) for v in self.raw(a, b))
        return (r, g, b_)


OPS = (
    Op("mix", lambda x, y: (x + y) / 2, "(x + y) / 2"),
    Op("add", lambda x, y: min(x + y, TOP), "min(x + y, 15)"),
    Op("screen", lambda x, y: TOP - (TOP - x) * (TOP - y) / TOP, "15 − (15 − x)(15 − y) / 15"),
    Op("multiply", lambda x, y: x * y / TOP, "x · y / 15"),
    Op("lighten", max, "max(x, y)"),
    Op("darken", min, "min(x, y)"),
)
"""The op table: the four blend modes the D2.2 design named (`mix` is D2.1's mean, spelled `+` there and
`mix` here, and the same answer on every pair D2.1 defined it on; saturating `add`; `screen`; `multiply`),
plus the per-channel `max` and `min` (Photoshop's *lighten* and *darken*). Each is computed on the 0..15
scale and snapped to the nearest level of the grid, the "defined rounding" the design's deps section asks
for. The snap is unbiased on average: `screen` and `multiply` never tie, and their mean signed rounding
error per channel is zero over the pairs; `mix` ties on half its channel sums, and `snap` sends those to
the even level index.

All six are commutative, so operand order carries no information, as in D2.1. A `subtract` to balance
`add` would be the first op where it did, which is why the set leans light (three ops lighten, two darken,
`mix` is neutral). They are distinct rules: no two agree on more than 38% of pairs (`agreement()`), and
every op but `lighten` has lines whose answer is redder than both operands (`line_counts()`)."""

MIX, ADD, SCREEN, MULTIPLY, LIGHTEN, DARKEN = OPS
OP_NAMES = tuple(op.name for op in OPS)
OP_BY_NAME = {op.name: op for op in OPS}


def colors() -> list[Rgb]:
    """The 216 colors of the grid, in (r, g, b) order."""
    return [(r, g, b) for r in LEVELS for g in LEVELS for b in LEVELS]


def unordered_pairs() -> list[tuple[Rgb, Rgb]]:
    """Every unordered pair of grid colors, self-pairs included (23,436 of them)."""
    cs = colors()
    return [(a, b) for i, a in enumerate(cs) for b in cs[i:]]


def lines() -> list[tuple[Rgb, Rgb]]:
    """Every line of an op: each unordered pair in both orders, self-pairs once (46,656)."""
    return [(x, y) for a, b in unordered_pairs() for (x, y) in ({(a, b), (b, a)})]


def is_on_grid(op: Op, a: Rgb, b: Rgb) -> bool:
    """The op's rule lands on the grid for this pair without rounding."""
    return all(v in LEVELS for v in op.raw(a, b))


def on_grid(op: Op) -> float:
    """The fraction of unordered pairs the op answers without rounding."""
    return sum(is_on_grid(op, a, b) for a, b in unordered_pairs()) / len(unordered_pairs())


def mix_probe_lines() -> list[tuple[Rgb, Rgb]]:
    """`mix`'s on-grid lines: D2.1's closed pairs in both orders, the 5,832 lines its statistics were read on."""
    return [(a, b) for a, b in lines() if is_on_grid(MIX, a, b)]


def agreement(p: Op, q: Op) -> float:
    """The fraction of unordered pairs on which p and q give the same answer."""
    return sum(p(a, b) == q(a, b) for a, b in unordered_pairs()) / len(unordered_pairs())


def relevance(anchored: Op) -> dict[int, float]:
    """How often reading the op word is worth something on the anchored op's lines.

    For a line, count the *other* ops whose answer equals the anchored op's. At 0 the answer names the op;
    at k the op word only rules out 5 − k of the six. Returned as {k: fraction of lines}. The D2.2 design asks
    for this per candidate anchored op, under the table's own rounding.
    """
    counts = np.zeros(len(OPS), dtype=int)
    for a, b in unordered_pairs():
        answer = anchored(a, b)
        counts[sum(op(a, b) == answer for op in OPS if op is not anchored)] += 1
    return {k: float(c) / len(unordered_pairs()) for k, c in enumerate(counts) if c}


def dose(a: Rgb, b: Rgb) -> float:
    """How *red* a line is: the larger of its two operand rednesses."""
    return max(redness(a), redness(b))


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

TOKENS_PER_LINE = 6
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
