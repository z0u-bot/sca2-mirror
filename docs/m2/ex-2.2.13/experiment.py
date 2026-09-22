"""Ex-2.2.13: does a heavier anchor make the leftover predictable?

Design constants only, during preregistration. One ladder: the anchor weight λ_a at four levels, crossed
with the home of *red* (an axis or a plane), at twenty fresh seeds a condition on ex-2.2.11's handover setup.
The question is the spread of the leftover across seeds rather than its size. Every run is new; nothing is
served from the store, and no seed here has been trained before.
"""

from dataclasses import dataclass

DESIGN_ONLY = True  # read by tests/mini/test_experiments_e2e.py via getattr

# --- What is inherited ---------------------------------------------------------------------------------

REFERENCE_EXPERIMENT = "m2/ex-2.2.11"
"""Every gate, the grammar (table A+), the stochastic corpus, the whole-line labeller, the untied readout,
the removal lines chosen by hue, and the recipe apart from λ_a (τ 0.1, the anti-subspace weight from 2.5× the
anchor weight to 0.3× by 90% of training, 50 epochs at d64-L4): all as ex-2.2.11 froze them. This module
names what the ladder changes and nothing else; the implementation binds the rest from ex-2.2.11's module."""

PLANE_EXPERIMENT = "m2/ex-2.2.12"
"""The plane conditions are ex-2.2.12's `plane`: the same anchor, anti-subspace, and removal machinery
taking a pair of axes where they took one. Its scoring fixes and its control-on-the-plane comparison are
inherited too."""

MISSED_OP = "hue-hsv"
"""The one op that missed removal at ex-2.2.11 and at every condition of ex-2.2.12's sweep. Its removal
lines have red at op2, and the answer takes its hue from that operand."""

KEPT_GATE = 0.2
"""Ex-2.2.11's gate on the kept share, unchanged. Under it, the projection is taken to have removed *red*
from an op's red-dependent answers. The implementation binds this from ex-2.2.11 and asserts the value."""

CONTROL = "control"
"""The un-anchored control, ex-2.2.11's `control` retrained at this experiment's seeds: the task reference
(ex-2.2.11's task gate is a seed-mean within a band of the control's) and the ᾱ baseline for the axis
conditions. Retraining it keeps every comparison paired within this experiment and keeps the claim that no
checkpoint here was seen before true; it costs one rung's worth of runs."""

CONTROL_PLANE = "control-plane"
"""The same control checkpoints scored on the plane: the comparison for every plane measurement, since an
unsigned two-dimensional alignment sits higher than a signed one-dimensional one for every state, anchored
or not. A scoring pass, not a training condition."""

# --- The ladder ----------------------------------------------------------------------------------------

LADDER = (0.1, 0.14, 0.2, 0.28)
"""The anchor weight, four levels on a √2 ladder. 0.1 is ex-2.2.11's recipe and 0.2 the one step ex-2.2.12
took; the ratio is constant so the levels are evenly spaced in log, which is the scale a weight is read on.
The top stops short of where ex-2.1.11's survey first saw task failures (λ_a ≥ 0.38 on the six-op grammar):
the ladder is here to test the region between the recipe and that point, and where the useful range ends is
left open."""

SUBSPACES = ("axis", "plane")
"""The home of *red*: the first axis e₁, or the plane spanned by e₁ and e₂. Ex-2.2.12 settled against the
plane's first rationale (that a single axis cannot hold which side of red a color sits on), but its one plane
condition at a heavier weight was the tightest across seeds in the sweep, and the plane at the recipe's weight
was seen at five seeds only. The cross says whether a narrowing belongs to the weight, to the subspace, or
to the two together."""

SEED_OFFSET = 200
"""Condition seed *i* trains at model seed `SEED_OFFSET + i`. Ex-2.2.11 and ex-2.2.12 trained at offset 100,
and the observation that motivated this experiment was read off those checkpoints, so none of them can test
it. Every seed here is fresh, and the reference is retrained beside the ladder at the same seeds."""

SEEDS = 20
"""Seeds a condition. Twenty is set by the spread question rather than the mean: a one-sided F-test on the
variance ratio between the top and the bottom of the ladder has power 0.90 at twenty seeds a group for a
halved standard deviation, 0.81 at fifteen, and 0.63 at ten. A difference in means of the size we care
about would be resolved by half as many."""

CONDITIONS = tuple(f"{s}-{lam:g}" for s in SUBSPACES for lam in LADDER)
assert len(CONDITIONS) == 8

N_RUNS = (len(CONDITIONS) + 1) * SEEDS  # the eight rungs and the control
assert N_RUNS == 180


@dataclass(frozen=True)
class Condition:
    name: str
    subspace: str
    lam: float

    @property
    def is_reference(self) -> bool:
        return self.name == REFERENCE


REFERENCE = "axis-0.1"
"""Ex-2.2.11's recipe, retrained at this experiment's seeds: the lowest rung of the ladder and the comparison for
every claim about what the ladder moves. Retraining it also replicates ex-2.2.11's `handover` at fresh seeds."""

GRID = tuple(Condition(f"{s}-{lam:g}", s, lam) for s in SUBSPACES for lam in LADDER)

# --- The measurements ----------------------------------------------------------------------------------

SPREAD_STATISTIC = "kept share on the missed op's removal lines"
"""What H1 is about. The anchor term acts on the alignment of a state with the anchored subspace; the kept
share is taken after training, on held-out lines, through a projection the term never sees, so its spread
across seeds is not a quantity the treatment optimizes. The line margin is the manipulation check and is
reported beside it rather than gated."""

SD_RATIO_GATE = 0.5
"""H1 holds when the across-seed standard deviation of the kept share at the top of the ladder is at most
half of its value at the lowest rung, within a subspace. Half is the effect ex-2.2.12 saw at five seeds
(`plane-lam-0.2` at 0.036 against `plane` at 0.125, a ratio near a third) rounded back toward no effect. A
standard deviation from five draws has a 95% interval of about 0.6× to 2.9× its true value, so the gate
asks for less than the observation that motivated it."""

TREND_ALPHA = 0.05
"""The significance check on H1. Each run's absolute deviation from its own condition's median is regressed
on log λ_a within a subspace, and H1 wants a negative slope at this level. A trend contrast asks the
question H1 actually poses — does the spread shrink as the weight rises — where Levene's test would only
say the four levels differ somehow, and would count a spread that rose and fell as a pass."""

UCB_LEVEL = 0.95
"""The adoption rule's confidence level. A condition's kept share is summarized by the one-sided upper
confidence bound on its seed mean, Student's t at n − 1 degrees of freedom."""

MEAN_BAND = 0.05
"""H2's band: the seed-mean kept share on the missed op is called flat across the ladder when no level
differs from the reference by more than this. It is about the paired-difference resolution at twenty seeds
at the reference's spread, so H2 is a statement about what we can see rather than about what is there."""

WORST_OP_MARGIN = 0.03
"""H3 wants the worst of the ten other ops to fall by at least this much from its value at the reference,
measured in this experiment at twenty seeds rather than quoted from ex-2.2.12. The margin is about the
binomial noise floor on a bounded proportion at these line counts, so a drop smaller than it is not a drop
we can resolve."""

COST_STATISTICS = (
    "the task",
    "the line margin",
    "the lead at the embedding",
    "the contrast between red and non-red",
    "the grading r²",
    "the deficit on the non-red lines",
)
"""What a heavier anchor might spend, each against ex-2.2.11's gate: held-out expected exact match on the
worst op (`task`), the line margin (`m_line`), the lead at the embedding (`lead_emb`), the separation of the
red and non-red groups along the anchored subspace (`contrast`), the squared correlation of the α response
with redness (`r2_sim`), and the deficit on the non-red lines under the projection (`nonred_deficit`)."""

COST_LIST = ", ".join(COST_STATISTICS[:-1]) + f", and {COST_STATISTICS[-1]}"
"""The cost statistics as a prose list."""

ADOPTION = f"""\
A condition qualifies when the one-sided {UCB_LEVEL:.0%} upper confidence bound on its seed-mean kept share \
sits under the {KEPT_GATE:g} gate on `{MISSED_OP}` and on each of the ten other ops, {COST_LIST} are inside \
ex-2.2.11's gates on the seed mean, and its ᾱ at op1 stands in no higher ratio to the un-anchored baseline for \
its own subspace than the reference stands to the axis one. Among those that qualify, the one with the lowest \
upper bound on `{MISSED_OP}` is adopted, and the smaller λ_a breaks a tie. The rule sees the measurements it \
names; if a qualifying condition looks harmful on something it does not name, the report says what and keeps \
the reference, and that override is marked as a decision made after the data. As in ex-2.2.12, this experiment \
proposes: the adopted recipe is confirmed at fresh seeds by the anchored-op experiment that inherits it, and \
no number quoted here for it is free of the selection. If none qualifies, the recipe stays at `{REFERENCE}` \
and the report carries the ladder's best characterization of the leftover — its level, its spread, and which \
lines it is made of — for the anchored-op preregistration to treat as a bounded confound."""

OLD_BAND = 0.06
"""Ex-2.2.12's fixed band: the seed spread its reference showed at twenty seeds."""

OLD_RULE = f"""\
Ex-2.2.12 asked instead for a seed mean under the gate by at least a fixed band of {OLD_BAND:g}, the spread its \
reference showed at twenty seeds. That band is a statement about one condition's resolution, and applying it \
to a condition three times tighter charges it for a spread it does not have. The upper bound above asks the \
same question — is this condition's leftover under the gate, or does the seed spread leave that unsettled — \
of each condition at its own precision. It is the stricter rule on a wide condition and the looser one on a \
tight one, and it is fixed here before the runs exist. Both verdicts are reported."""
