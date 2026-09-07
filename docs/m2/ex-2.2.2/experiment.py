"""
Experiment 2.2.2: fallback control for *red* in the anchored transformer.

Design constants only, for now. The report imports these so its prose and the
gates cannot drift apart; the DAG lands once the preregistration is frozen, and
`DESIGN_ONLY` goes with it.

The experiment adds M1's fallback control (ex-2.9.2) to the D2.1 recipe
(ex-2.1.10 `either-t100`): a training term that teaches the blocks what to
answer once *red* has been removed, at a designed target, without touching the
placement of the concept. Every checkpoint is scored through the eval contract
(`sca.intervention`) on the 5,832-line probe set the D2.1 experiments share,
beside the stored ex-2.1.10 checkpoints ex-2.2.1 scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sca.data.colors import N_LEVELS
from sca.data.named_colors import GRIDS

DESIGN_ONLY = True

#: Result refs the report will read.
METRICS_REF = "reports/m2/ex-2.2.2/metrics"
ARRAYS_REF = "reports/m2/ex-2.2.2/arrays"
TRAJ_REF = "reports/m2/ex-2.2.2/trajectories"

# --- The recipe ------------------------------------------------------------------

RECIPE = "ex-2.1.10/either-t100"
"""Everything about training that this experiment does not change: the `v216` corpus (216 colors, one word
each), d64-L4, 100 epochs with a 10-epoch warm-up, the pooled either-operand labeller at τ = 0.1, the anchor at
λ = 0.1 with its anneal, and the anti-subspace term at the ex-2.1.8 operating point. Imported from that module
when the DAG lands."""

ANCHOR_AXIS = 0
"""*Red* sits on e₁ at every slice, as in D2.1."""

# --- The fallback term ---------------------------------------------------------------

FALLBACK_SLICE = 0
"""Where the redirect acts during training: the embedding. Everything after it is the readout the term
trains, which is the transformer form of M1's decoder-only term."""

FALLBACK_POSITION = "concept operand"
"""The redirect reflects one state per line: the redder operand's (ties go to op1)."""

FALLBACK_THRESHOLD = 0.5
"""The term applies only to lines whose concept operand has a clean embedding alignment of at least this
much, so a reflection moves the state by at least one unit of alignment. Below it the term is inert, which
is what makes a constant weight safe from step 0: the anchor places the concept inside its warm-up, and the
term switches on as it does."""

FALLBACK_WEIGHT = 0.05
"""w_fb, constant over training. M1's value (ex-2.9.2), on a cross-entropy rather than an MSE."""

FALLBACK_BRACKET = (0.01, 0.25)
"""The weight bracket, a factor of five either side, three seeds each: whether 0.05 sits on a plateau."""

ANTI_ANCHOR_WEIGHT = 0.1
"""The anti-anchor term's constant weight, equal to the anchor's peak. The term is a hinge on negative
alignment, mean(max(−α, 0)) over the same live positions and slices the anti-subspace term reads, so it
keeps the antipode hemisphere empty of clean states without opposing the anchor."""

GRAY = (N_LEVELS - 1) / 2
"""Mid-gray on the 16-level channel scale (7.5): the know-nothing operand the designed target mixes with."""

LEVELS = np.asarray(GRIDS["v216"], dtype=float)
"""The six channel levels of the corpus grid: 0, 3, 6, 9, 12, 15."""


def fallback_target(visible: np.ndarray) -> np.ndarray:
    """The designed answer for a line whose concept operand has been removed: the *visible* operand mixed
    with mid-gray, each channel rounded to the nearest grid level.

    Also the per-channel median of the operand-averaged null (the mix with each of the visible operand's
    three closed partners per channel), so it is the center of the distribution a removed concept leaves.
    The two definitions agree on every level of the grid, and `check_target()` asserts it.
    """
    v = np.asarray(visible, dtype=float)
    mean = (v + GRAY) / 2
    return LEVELS[np.abs(mean[..., None] - LEVELS).argmin(-1)]


def closed_partners(level: float) -> np.ndarray:
    """The grid levels *c* for which the corpus mix, ⌊(v + c + 1) / 2⌋, lands back on the grid."""
    mixes = np.floor((level + LEVELS + 1) / 2)
    return LEVELS[np.isin(mixes, LEVELS)]


def check_target() -> None:
    """The gray target coincides with the median closed-partner mix at every level, and the null has no mode."""
    for v in LEVELS:
        partners = closed_partners(v)
        assert len(partners) == 3, (v, partners)
        mixes = np.sort(np.floor((v + partners + 1) / 2))
        assert len(set(mixes)) == 3, "the null per channel is uniform over three distinct answers"
        assert fallback_target(np.array([v]))[0] == mixes[1], (v, mixes)


check_target()

# --- The lines -------------------------------------------------------------------------

RED_DOSE = 0.8
"""*Red lines* have dose ≥ 0.8 (the larger of the two operands' rednesses): 365 probe lines. The fallback
term applies to the corpus lines that meet the same threshold."""

NONRED_DOSE = 0.2
"""*Non-red lines* have dose ≤ 0.2: 1,689 probe lines."""

OPERAND_POSITIONS = (0, 2)
DECODE_POS = 3
"""The `=` position: the state the answer is decoded from."""

SLICES = (0, 1, 2, 3, 4)

# --- The interventions --------------------------------------------------------------------


@dataclass(frozen=True)
class Intervention:
    """One row of the results: an operator, where it acts, and which gate or row it serves."""

    name: str
    kind: str
    """`projection`, `shaped`, or `ablate`."""
    slices: tuple[int, ...] = SLICES
    positions: tuple[int, ...] | None = None
    gamma: float = 1.0


REDIRECT = Intervention("redirect", "projection", slices=(FALLBACK_SLICE,), positions=OPERAND_POSITIONS, gamma=2.0)
"""The trained state, at eval: reflect both operand states through the axis at the embedding (γ = 2 in the
projection operator). A non-red operand has nothing on the axis, so the reflection leaves it alone; training
reflected the concept operand only, and this is the same edit on every line that qualified. H1, H2, H4."""

PRIMARY = Intervention("primary", "projection")
"""Ex-2.2.1's primary intervention: full-strength projection at every slice and position. The deployment
read, and the removal the fallback was never trained under. H4, H5."""

GAMMAS = (0.5, 1.0, 1.5, 2.0)
"""The carry sweep (H5): the projection operator at the training site, from half removal through zero to the
reflection. γ = 2 is `redirect`."""

SWEEP = tuple(
    Intervention(f"gamma-{g}", "projection", slices=(FALLBACK_SLICE,), positions=OPERAND_POSITIONS, gamma=g)
    for g in GAMMAS
)

RIDE_ALONG = (
    Intervention("operands", "projection", positions=OPERAND_POSITIONS),
    Intervention("embedding", "projection", slices=(0,)),
    Intervention("shaped", "shaped"),
    Intervention("ablate", "ablate", slices=()),
)
"""Ex-2.2.1's arms, re-run on the fallback condition without gates, so the operator-tuning pass the design
schedules has the fallback rows to hand. Same definitions as there: `shaped` at a = 0.5, b = 1, p = 1."""

SHAPED = dict(a=0.5, b=1.0, p=1.0)

# --- Conditions --------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """A set of runs: stored (ex-2.1.10) or trained here."""

    exp: str
    name: str
    seeds: int
    title: str
    w_fb: float = 0.0
    w_aa: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.exp}/{self.name}"

    @property
    def stored(self) -> bool:
        return self.exp != "ex-2.2.2"


CONTROL = Condition("ex-2.1.10", "lam0", 3, "un-anchored")
"""Nothing placed on the axis: the anchor weight at zero."""

NO_FALLBACK = Condition("ex-2.1.10", "either-t100", 9, "no-fallback")
"""The D2.1 recipe as ex-2.2.1 scored it: the reference every hypothesis compares against, re-scored here
through the same code path as the new runs."""

FALLBACK = Condition("ex-2.2.2", "fallback", 9, "fallback", w_fb=FALLBACK_WEIGHT, w_aa=ANTI_ANCHOR_WEIGHT)
"""The recipe plus both new terms. The primary condition, and the one every hypothesis scores."""

ARMS = (
    Condition("ex-2.2.2", "fb-only", 3, "fallback, no anti-anchor", w_fb=FALLBACK_WEIGHT),
    Condition("ex-2.2.2", "anti-only", 3, "anti-anchor, no fallback", w_aa=ANTI_ANCHOR_WEIGHT),
    Condition("ex-2.2.2", "fb-w0.01", 3, "fallback at w_fb = 0.01", w_fb=FALLBACK_BRACKET[0], w_aa=ANTI_ANCHOR_WEIGHT),
    Condition("ex-2.2.2", "fb-w0.25", 3, "fallback at w_fb = 0.25", w_fb=FALLBACK_BRACKET[1], w_aa=ANTI_ANCHOR_WEIGHT),
    Condition("ex-2.2.2", "recipe", 3, "the recipe through the new code path"),
)
"""Each changes one thing from the primary condition; reported at the same statistics without gates."""

N_NEW_RUNS = FALLBACK.seeds + sum(a.seeds for a in ARMS)
assert N_NEW_RUNS == 24

# --- Gates ---------------------------------------------------------------------------------

TARGET_ACC_GATE = 0.8
"""H1: "seed-mean exact-match accuracy against the designed target on red lines, under `redirect`, is at
least 0.8"."""

TARGET_ACC_PARTIAL = 0.5
"""H1 partial: "between 0.5 and 0.8"."""

AGREE_SEEDS = 5
"""H2: a red line "agrees" when at least this many of the nine seeds decode the same answer. Ex-2.2.1's
definition, whose 13% under the projection is the reference."""

AGREE_GATE = 0.8
"""H2: "the seed-agreement fraction on red lines under `redirect` is at least 0.8"."""

AGREE_PARTIAL = 0.5
"""H2 partial: "at least 0.5"."""

TASK_GATE = 0.02
"""H3 and H4: the width every D2.1 task gate used. H3: "clean exact-match accuracy on all probe lines within
0.02 of the no-fallback condition's". H4: "seed-mean non-red damage under `redirect` at most 0.02"."""

TASK_PARTIAL = 0.05
"""H3 and H4 partial: "within 0.05"."""

MARGIN_RATIO = 0.8
"""H3: "the seed-mean alignment margin at the end of training (ex-2.1.10's m_span) is at least 0.8 of the
no-fallback condition's"."""

GRADE_DIP = 0.02
"""H5: "target accuracy is non-decreasing along the sweep, allowing a dip between adjacent strengths of at
most 0.02"."""

CARRY_FRAC = 0.5
"""H5: "target accuracy at γ = 1 is at least half of its value at γ = 2"."""

RESOLUTION_SD = 2.0
"""Differences between conditions or arms smaller than this many pooled between-seed standard deviations of
the statistic are reported as not resolved. H1's and H4's comparisons against no-fallback use it."""

#: What a red line decodes to under intervention, in the order the composition is stored. Ex-2.2.1's five
#: categories plus the target; a target that is also a one-step neighbor of the true answer counts as target.
COMPOSITION = ("target", "true", "neighbor", "visible_operand", "red_operand", "other")

#: Ridge strength of the off-axis recoverability probe (E3), matching ex-2.1.12 and ex-2.2.1.
DECODE_L2 = 1e-2

#: Groups the per-line statistics are contracted over.
GROUPS = ("all", "red", "nonred")
