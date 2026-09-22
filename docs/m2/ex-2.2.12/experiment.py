"""Ex-2.2.12: what the stream holds on `hue-hsv`, and a small recipe sweep.

Design constants only, during preregistration. Two parts, independent of each other: a scoring-only pass
over ex-2.2.11's stored `handover` checkpoints (part 1), and a training sweep of eight conditions at five
seeds on the handover setup (part 2), compared with ex-2.2.11's twenty `handover` seeds as the reference.
Nothing here is scored; the report proposes, and the handover re-run that follows adopts at fresh seeds.
"""

from dataclasses import dataclass

DESIGN_ONLY = True  # noqa: read by tests/mini/test_experiments_e2e.py via getattr

# --- What is inherited -------------------------------------------------------------------------------

REFERENCE_EXPERIMENT = "m2/ex-2.2.11"
"""Every gate, the grammar (table A+), the stochastic corpus, the whole-line labeller, the untied readout,
the recipe (λ_a 0.1, τ 0.1, the anti-subspace weight from 2.5× the anchor weight to 0.3× by 90% of training,
50 epochs at d64-L4), and the removal lines chosen by hue: all as ex-2.2.11 froze them. This module names
what the sweep changes and nothing else; the implementation binds the rest from ex-2.2.11's module."""

REFERENCE_CONDITION = "handover"
"""The reference condition of the sweep is ex-2.2.11's `handover` at its twenty seeds. Memoization makes it
free: a condition that changes nothing is the same run."""

HANDOVER_CHECKPOINT_REF = "reports/m2/ex-2.2.11/checkpoints/{label}"
"""Part 1 scores these, one per `handover` run."""

# --- Part 1: three measurements on the stored checkpoints -------------------------------------------------

MISSED_OP = "hue-hsv"
"""The one op that missed removal at ex-2.2.11. Its removal lines have red at op2, and the answer takes its
hue from that operand."""

SIDE_GROUPS = ("G = B", "G > B", "G < B")
"""The three sides of red, by the red operand's green and blue channels. A color with G = B sits on the red
axis of the cube and has hue 0 exactly; G > B leans toward orange and G < B toward pink. The anchored axis
measures how red a color is, which is the same on both sides, so the side is the part of the hue the axis
cannot hold."""

BYPASS_EDITS = ("all", "embedding", "blocks", "op2-all", "op2-embedding")
"""Where the projection is applied on the missed op's removal lines: every slice at every position (the
operator ex-2.2.11 scored), the embedding slice only, the blocks only (slices 1 and up), and the op2
position only at every slice or at the embedding. Where removal bites says where the off-axis copy of the
hue is written."""

CONTAINMENT_SPLIT = ("labelled by operand", "labelled by answer")
"""ᾱ at op1 on the red lines, split by how the whole-line labeller earned the line's label. Under the
whole-line label a line whose answer draws is pulled at every position, op1 included, so a non-red op1 is
pulled whenever the answer draws. That is the containment item's candidate for the labeller's half of the
rise, and this measurement tests it."""

# --- Part 2: the sweep --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    name: str
    seeds: int
    title: str
    tau: float = 0.1
    lam: float = 0.1
    anti_peak: float = 2.5
    n_layer: int = 4
    subspace: str = "axis"

    @property
    def changes(self) -> int:
        """How many factors differ from the reference: the tie-break prefers the smaller change."""
        ref = REF
        return sum(getattr(self, k) != getattr(ref, k) for k in ("tau", "lam", "anti_peak", "n_layer", "subspace"))


SWEEP_SEEDS = 5
"""Five seeds a condition, at the seeds ex-2.2.11's `handover` trained (its first five), so each condition
pairs with the reference seed for seed."""

REF = Condition(REFERENCE_CONDITION, 20, "ex-2.2.11's candidate, unchanged")

TAU_CONDITIONS = (
    Condition("tau-0.03", SWEEP_SEEDS, "mellowmax τ 0.03", tau=0.03),
    Condition("tau-0.01", SWEEP_SEEDS, "mellowmax τ 0.01", tau=0.01),
)
"""The fixed-τ question: does the whole-line label want a sharper pool? The number to watch is ᾱ at op1.
These conditions are not expected to move the missed op."""

FORCE_CONDITIONS = (
    Condition("lam-0.2", SWEEP_SEEDS, "anchor weight doubled", lam=0.2),
    Condition("anti-5", SWEEP_SEEDS, "anti-subspace peak doubled", anti_peak=5.0),
)
"""The recipe's two force factors, one step up each. If the missed op is a matter of one axis holding half
of an angle, neither can help, and a null here is what says so."""

SHAPE_CONDITIONS = (
    Condition("L6", SWEEP_SEEDS, "six blocks", n_layer=6),
    Condition("plane", SWEEP_SEEDS, "red anchored to a plane", subspace="plane"),
    Condition("L6-plane", SWEEP_SEEDS, "six blocks, red anchored to a plane", n_layer=6, subspace="plane"),
    Condition(
        "plane-lam-0.2", SWEEP_SEEDS, "red anchored to a plane, anchor weight doubled", lam=0.2, subspace="plane"
    ),
)
"""Depth × subspace, with the reference as the fourth corner, plus the plane at the doubled anchor weight.
The plane is the structural candidate for the missed op; depth is the other capacity factor, and the
factorial says whether the two trade or stack. The last condition is there in case the plane needs more
pull than the axis did: without it, a null on the force conditions would only say that the axis cannot be
pushed harder, and a partial result on `plane` could not be told from a plane pulled too gently."""

CONDITIONS: tuple[Condition, ...] = (REF, *TAU_CONDITIONS, *FORCE_CONDITIONS, *SHAPE_CONDITIONS)
NEW_RUNS = sum(c.seeds for c in CONDITIONS if c is not REF)
assert NEW_RUNS == 40

# --- The plane -----------------------------------------------------------------------------------------

PLANE_AXES = (0, 1)
"""The home of *red* under the plane conditions: e₁ and e₂ together. The alignment of a state with the plane is the
length of its projection onto the pair (states are unit-norm, so that is a cosine too, and unsigned). The
anchor term pulls that length toward one on the labelled lines, the anti-subspace term is its square over
every live position, and the removal projects the whole plane out. Margin, lead, and contrast are measured on
that alignment; the isotropic value of an unsigned two-dimensional alignment is higher than a signed
one-dimensional one, so every plane measurement is compared with the control's checkpoints scored the same
way."""

# --- The promotion rule --------------------------------------------------------------------------------

KEPT_BAND = 0.06
"""The seed band ex-2.2.11 resolved on the missed op's kept share at twenty seeds. A condition that clears the
gate by less than this has not been shown to clear it."""

PROMOTION = f"""\
A condition is proposed as the fix for `{MISSED_OP}` when, at {SWEEP_SEEDS} seeds, its kept share on the \
`{MISSED_OP}` removal lines is under the gate by at least {KEPT_BAND:g}, its task, margin, lead, contrast, \
and non-red deficit are inside ex-2.2.11's gates, and its ᾱ at op1 is no higher than the reference's by \
more than the band. Among several, the one that changes fewest factors is proposed, and ᾱ at op1 breaks a \
tie. If no condition qualifies, the re-run gates removal on the ten other ops and reports `{MISSED_OP}` \
beside them as the op where a single axis has a known blind spot; the recipe is then the reference, with \
a τ change carried only if the τ conditions moved ᾱ at op1 by more than the band at no cost."""

TAU_PROPOSAL = f"""\
Separately, a τ is proposed when its condition lowers ᾱ at op1 by more than {KEPT_BAND:g} against the reference \
with task, margin, lead, and contrast inside their gates. This proposal stands on its own and can be carried \
with or without a fix for `{MISSED_OP}`."""
