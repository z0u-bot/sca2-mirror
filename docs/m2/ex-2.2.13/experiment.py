"""Ex-2.2.13: does a heavier anchor make the leftover predictable?

Design constants only, during preregistration. One ladder: the anchor weight λ_a at five levels, crossed
with the home of *red* (an axis or a plane), at twenty seeds a condition on ex-2.2.11's handover setup,
plus one arm at the top rung that holds the repulsion at its foot-of-the-ladder strength. The question is
the spread of the leftover across seeds rather than its size, and the adoption rule is decided on the
second half of the seeds and quoted from the first.
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
"""The plane arm is ex-2.2.12's `plane`: the same anchor, anti-subspace, and removal machinery taking a pair
of axes where they took one. Its scoring fixes and its control-on-the-plane comparison are inherited too."""

MISSED_OP = "hue-hsv"
"""The one op that missed removal at ex-2.2.11 and at every condition of ex-2.2.12's sweep. Its removal
lines have red at op2, and the answer takes its hue from that operand."""

KEPT_GATE = 0.2
"""Ex-2.2.11's gate on the kept share, unchanged. Under it, the projection is taken to have removed *red*
from an op's red-dependent answers. The implementation binds this from ex-2.2.11 and asserts the value."""

CONTROL_PLANE = "control-plane"
"""Ex-2.2.11's un-anchored `control` checkpoints, scored on the plane: the comparison for every plane
measurement, since an unsigned two-dimensional alignment sits higher than a signed one-dimensional one for
every state, anchored or not."""

# --- The ladder ----------------------------------------------------------------------------------------

LADDER = (0.1, 0.14, 0.2, 0.28, 0.4)
"""The anchor weight, five levels on a √2 ladder. 0.1 is ex-2.2.11's recipe and 0.2 the one step ex-2.2.12
took; the ratio is constant so the levels are evenly spaced in log, which is the scale a weight is read on.
The top level is where ex-2.1.11's survey first saw task failures (λ_a ≥ 0.38) on the six-op grammar, so the
ladder reaches the point where something is expected to give way rather than stopping short of it."""

SUBSPACES = ("axis", "plane")
"""The home of *red*: the first axis e₁, or the plane spanned by e₁ and e₂. The plane's own rationale — that
a single axis cannot hold which side of red a color sits on — was settled against by ex-2.2.12, and it is
here only to say whether a narrowing of the seed spread belongs to the weight or to the subspace."""

SEEDS = 20
"""Seeds a condition, at the twenty seeds ex-2.2.11's `handover` trained, so every condition pairs with
every other seed for seed. Twenty is set by the spread question rather than the mean: telling a halved
standard deviation from an unchanged one needs about this many runs a group, where a difference in means of
the size we care about would be resolved by half as many."""

SELECT_RANGE = (11, 20)
"""The seeds the adoption rule is decided on. Ex-2.2.12 published seeds 1–5 of three of these conditions, and
the tight spread that motivated this experiment was read off them, so those seeds sit outside the deciding
half: preregistration fixes the rule in advance, and it cannot make data we have already seen unseen."""

QUOTE_RANGE = (1, 10)
"""The seeds the adopted condition's numbers are quoted from, held out of the rule. They carry the
winner's-curse correction, so it is paid inside this experiment instead of in the next one."""

FIXED_ANTI_ARM = "axis-0.4-anti-fixed"
"""One arm off the top rung of the axis ladder, at the same twenty seeds: the anchor weight at 0.4 with the
anti-subspace term held at the absolute strength it has at λ_a = 0.1 (its ratios divided by four). Every
other rung raises the pull and the repulsion together, so the ladder on its own cannot say which of the two
any effect belongs to. This arm separates them at the one place an effect is most likely to show. It rides
along outside the ladder and outside the adoption rule."""

CONDITIONS = tuple(f"{s}-{lam:g}" for s in SUBSPACES for lam in LADDER)
assert len(CONDITIONS) == 10

MEMOIZED = {"axis-0.1": 20, "axis-0.2": 5, "plane-0.1": 5, "plane-0.2": 5}
"""Runs already in the store at these seeds: ex-2.2.11's `handover`, and ex-2.2.12's `lam-0.2`, `plane`, and
`plane-lam-0.2`. A condition that changes nothing is the same run, so memoization serves them."""

NEW_RUNS = (len(CONDITIONS) + 1) * SEEDS - sum(MEMOIZED.values())
assert NEW_RUNS == 185


@dataclass(frozen=True)
class Condition:
    name: str
    subspace: str
    lam: float

    @property
    def is_reference(self) -> bool:
        return self.name == REFERENCE


REFERENCE = "axis-0.1"
"""Ex-2.2.11's recipe, at its own twenty seeds: the foot of the ladder and the comparison for every claim
about what the ladder moves."""

GRID = tuple(Condition(f"{s}-{lam:g}", s, lam) for s in SUBSPACES for lam in LADDER)

# --- The measurements ----------------------------------------------------------------------------------

SPREAD_STATISTIC = "kept share on the missed op's removal lines"
"""What H1 is about. The anchor term acts on the alignment of a state with the anchored subspace; the kept
share is taken after training, on held-out lines, through a projection the term never sees, so its spread
across seeds is not a quantity the treatment optimizes. The line margin is the manipulation check and is
reported beside it rather than gated."""

SD_RATIO_GATE = 0.5
"""H1 holds when the across-seed standard deviation of the kept share at the top of the ladder is at most
half of its value at the foot, within a subspace. Half is the effect ex-2.2.12 saw at five seeds
(`plane-lam-0.2` at 0.036 against `plane` at 0.125, a ratio near a third) rounded back toward no effect, so
the gate asks for less than the observation that motivated it."""

TREND_ALPHA = 0.05
"""The significance check on H1. Each run's absolute deviation from its own condition's median is regressed
on log λ_a within a subspace, and H1 wants a negative slope at this level. A trend contrast asks the
question H1 actually poses — does the spread shrink as the weight rises — where Levene's test would only
say the five levels differ somehow, and would count a spread that rose and fell as a pass."""

UCB_LEVEL = 0.95
"""The adoption rule's confidence level. A condition's kept share is summarized by the one-sided upper
confidence bound on its seed mean, Student's t at n − 1 degrees of freedom on the deciding seeds."""

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
A condition qualifies when, on seeds {SELECT_RANGE[0]}–{SELECT_RANGE[1]}, the one-sided {UCB_LEVEL:.0%} upper \
confidence bound on its seed-mean kept share sits under the {KEPT_GATE:g} gate on `{MISSED_OP}` and on each of the \
ten other ops, {COST_LIST} are inside ex-2.2.11's gates on the seed mean, and its ᾱ at op1 stands in no higher \
ratio to the un-anchored baseline for its own subspace than the reference stands to the axis one. Among those \
that qualify, the one with the lowest upper bound on `{MISSED_OP}` is adopted, and the smaller λ_a breaks a tie. \
Its numbers are then quoted from seeds {QUOTE_RANGE[0]}–{QUOTE_RANGE[1]}, which the rule never saw. If none \
qualifies, the recipe stays at `{REFERENCE}` and the report carries the ladder's best characterization of the \
leftover — its level, its spread, and which lines it is made of — for the anchored-op preregistration to treat \
as a bounded confound."""

OLD_BAND = 0.06
"""Ex-2.2.12's fixed band: the seed spread its reference showed at twenty seeds."""

OLD_RULE = f"""\
Ex-2.2.12 asked instead for a seed mean under the gate by at least a fixed band of {OLD_BAND:g}, the spread its \
reference showed at twenty seeds. That band is a statement about one condition's resolution, and applying it \
to a condition three times tighter charges it for a spread it does not have. The upper bound above asks the \
same question — is this condition's leftover under the gate, or does the seed spread leave that unsettled — \
of each condition at its own precision. It is the stricter rule on a wide condition and the looser one on a \
tight one, and it is fixed here before the runs exist. Both verdicts are reported."""
