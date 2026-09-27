"""Ex-2.2.16: the in-context grammar pilot — round 2 of the D2.2 quick route.

The first experiment on the grammar the pivot proposed: each line is a context of a few solved examples of one op,
written with `?` in place of the op word, then a query under the same op. The pilot trains the control at three
corpus conditions (example count and replacement rate), anchored `difference` with the whole-line label and its
variants, and the arms the design lists, at a few seeds each, and proposes by frozen rules what round 3 adopts.

This module holds the design constants only. The DAG comes with the grammar generator, which is being written
separately; until then the report imports these constants and computes its method section (the posterior over
ops and the Bayes ceiling) from the op table alone, through `posterior.py` beside it.
"""

from __future__ import annotations

from dataclasses import dataclass

from sca.data.ops import CANDIDATE_BY_NAME, OP_BY_NAME, Op
from sca.data.incontext import context_length

DESIGN_ONLY = True
# Constants without a DAG; `tests/mini/test_experiments_e2e.py` skips the load check while this line is here.
# Delete it in the change that adds `main(ctx)`.

# --- The grammar -----------------------------------------------------------------------------------------

OP_NAMES: tuple[str, ...] = (
    "mix",
    "screen",
    "multiply",
    "lighten",
    "darken",
    "difference",
    "exclusion",
    "hsvmix",
    "hue-hsv",
    "sat-hsv",
    "value-hsv",
)
TABLE: tuple[Op, ...] = tuple(OP_BY_NAME[n] if n in OP_BY_NAME else CANDIDATE_BY_NAME[n] for n in OP_NAMES)
"""Table A+ as ex-2.2.9 assembled it and every experiment since has used: five ops from ex-2.2.3, three commutative
additions, and the three HSV blend modes that read operand order. Every op is total on the grid. The order is the
one the reports print."""
N_OPS = len(TABLE)

ANCHORED_OP = "difference"
"""The op the anchored arms label, as in ex-2.2.14 and ex-2.2.15. It is total on the grid, so its ceiling is not
capped by rounding, and it is commutative."""

ROUNDING = "stochastic"
"""The corpus rounds each channel of a raw answer up or down at random, in proportion to where the raw value sits
between grid levels (`sca.data.ops.Rounding`), as every corpus since the handover has. The posterior uses the same
rounding, so the likelihood of a shown answer under an op is the probability that rounding under that op gives it."""


def context_tokens(k: int, verify: bool = False) -> int:
    """`a ? b = y ,` is six tokens, and so is the query `a ? b = y ⏎`: a context of k examples is `6k + 6` tokens,
    the layout `sca.data.incontext` generates; a verification line is two tokens longer, for the marker and the verdict.
    """
    return context_length(k, verify=verify)


# --- The posterior scan (the method section) ---------------------------------------------------------------

K_GRID: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
RHO_GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5)
"""The grid the method section scans: example counts and replacement rates. The rate ρ is per example: each
example shows, with probability ρ, a draw from the answer distribution of another op, uniform over the other ten,
in place of one from the true op. A replacement is invisible when the other op agrees with the true op on the
pair, which is what makes the posterior smooth rather than a count of fitting ops."""

CUBE_GRID: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1)
"""Cube-noise rates κ scanned at the proposed conditions: with probability κ an example shows a color drawn
uniformly from the 216-color grid, which usually no op produces. The pivot allows it at a low rate so that the model
learns to discount examples that fit nothing; the scan says what it costs the ceiling."""

N_CONTEXTS = 40_000
POSTERIOR_SEED = 2216
"""Contexts sampled per grid point by the method section, and the seed of the stream they are drawn from. At this
count the standard error of a ceiling is about 0.002, well under the differences between grid points."""

MIDDLE_BAND = (0.5, 0.95)
"""The band of the posterior on the true op that the pivot calls graded: above it the context all but names the
op, below it the evidence is split or points to another op. The share of contexts in the band is reported beside the spread."""

# --- The proposed corpus conditions ------------------------------------------------------------------------

GRAMMAR_CONDITIONS: tuple[tuple[int, float], ...] = ((3, 0.2), (3, 0.3), (4, 0.3))
CENTRE: tuple[int, float] = (3, 0.3)
"""The three (examples, ρ) conditions the control trains at, proposed by the method section from the scan. The
center is the working point of the pivot. The second differs in ρ alone, a step down in inference difficulty on the same
line length, so it is the fallback within the same block size if the control falls short of the ceiling at the
center. The third adds one example at the same ρ as the center, which raises the ceiling by about 0.06 and keeps most of the
spread; it asks whether more evidence buys a higher ceiling without flattening the stimulus."""
# REVIEW: proposed from the scan in the method section, not yet reviewed. The alternatives weighed there are
# (3, 0.4), which adds no middle-band share over the center and costs 0.09 of ceiling, and (4, 0.35), which has a
# little more spread than the third condition (0.32 against 0.29) and 0.04 less ceiling. Verify: the grid table in the method section.

CUBE_RATE: float = 0.02
"""The cube-noise rate of the corpus. The ceiling and floor of record at each condition are computed at this rate
(the cube scan in the method section); the κ = 0 scan is the design tool that picked the conditions."""
# REVIEW: proposed at the lowest scanned rate, which costs about 0.013 of ceiling at the center condition (the
# cube table in the method section), so that the model meets examples that fit no op, as the pivot asks, without
# the noise eating into the margin of rule (a). The alternative is κ = 0 for the pilot, with cube noise deferred to
# round 3, which keeps the κ = 0 scan as the ceiling of record. Verify: the cube table.
assert CUBE_RATE in CUBE_GRID, "the ceiling of record is read from the cube scan"

# --- What is inherited --------------------------------------------------------------------------------------

MODEL = "d64-L4"
LARGE_MODEL = "d128-L4"
"""The larger control is wider, not deeper: the worry in the pivot is the per-example lookup over eleven ops, which
is width-bound, and the same depth keeps the slice axis comparable with every other arm."""
# REVIEW: "wider or deeper" in the design; the alternative is d64-L6, which the layer sweep (round 4) would then
# have a first point for. Verify: the slice count is what every alignment figure shares across arms.

SEEDS = 3
"""Per arm, as the pilot section of the design sets, the larger control included."""

CROP_POLICY = "whole"
"""The anchor pulls only labelled lines wholly inside the training window, as ex-2.2.15 proposed after review:
a whole context always shows its evidence. Its rule chose `half`; the reasons for `whole` are in the section
"The rule for the pilot" of that report."""

# The `knowable` oracle of ex-2.2.15 pulled a whole labelled line only when its op was in view. Under `whole` every
# pulled context is wholly in view, so on this grammar the oracle is the same arm as the primary, and the pilot drops it.

# --- The corpus and the recipe -------------------------------------------------------------------------------

N_LINES = 300_000
"""Contexts in the corpus, one per line: the line count of ex-2.2.14, so that the anchored op has the same number of
lines (about 27k) and the labeller the same number of draws. A context is four times the tokens of an op-word
line, so an epoch is about four times the steps."""

BLOCK = 96
"""The training window. A context of four examples is 30 tokens, so a random crop of 96 holds on average about two
whole contexts of four examples and three of three, with a fragment at each end. The window of 64 in ex-2.2.15 held
about one whole context of four examples."""
# REVIEW: the pivot asks for "at least two whole contexts"; 96 meets it at both example counts. The alternative
# is 64, which keeps the window of ex-2.2.15 and would fit two whole three-example contexts only just.

VERIFY_RATE = 0.3
"""In the verification arms, the share of contexts written as verification lines (`sca.data.incontext`,
`verify_rate`): a candidate equation and a `TRUE` or `FALSE` verdict in place of the completion. The rest of the
corpus is unchanged, so completion has 70% of the lines it has in the other arms."""
# REVIEW: a proposal. Higher gives D2.3 more verification data on the same checkpoints; lower keeps completion
# closer to the other arms. Rule (d) scores the difference either way.

LABEL_RATE = 0.02
"""Per context of the anchored op, the chance it draws a label, as ex-2.2.14 and ex-2.2.15 set it: about 545
labelled contexts in the corpus. The label covers the whole context under the whole-line label; the variants
narrow it."""

EPOCHS = 50
"""The count of ex-2.2.9, unchanged since; the recipe (λ_a = 0.1 annealed to a 0.1 floor over the last tenth, τ = 0.1,
the anti-subspace weight from 2.5× to 0.3× by 90% of training, the untied readout) is that of ex-2.2.14."""

HOLDOUT_CONTEXTS = 2_000
"""Held-out contexts per op for the task and alignment measurements, drawn from the same generator at a seed
the corpus does not use. Contexts rather than pairs are held out: a context is the unit the model is scored on,
and a pair recurs across contexts anyway."""

# --- The arms ----------------------------------------------------------------------------------------------

LABEL_VARIANTS: tuple[tuple[str, str], ...] = (
    ("whole", "the whole context, every slice"),
    ("no-emb", "(a) the whole context, the embedding slice left out"),
    ("latter", "(b) the latter half of the context, where the prefix posterior is near its final value"),
    ("prefix", "(c) a position once the posterior given the tokens before it clears the threshold"),
    ("sampled", "(d) the whole context, the label drawn with probability equal to the posterior on the op"),
)
"""The whole-line label and the four variants of `/todo/science/label-variants-in-context-op.md`, by the short
name each arm carries and the positions it pulls on a labelled context."""

PREFIX_THRESHOLD = 0.5
"""Label variant (c): a position is pulled when the posterior on the anchored op, given the examples before it,
is at least this. The bottom of the middle band, so a context whose evidence never clears it is not pulled at all."""

HINGE_CAP = 0.8
"""The hinge arm: the pull on a position is max(0, cap − cos) in place of 1 − cos. It is a clip, with no remapping: a
state at or above this alignment has zero anchor gradient, so it is not pulled further, and the anti-subspace term
(which still acts on it) pushes it back down, so it settles at or below the cap. Set below the cosine near 1 the op word reached in ex-2.2.14, and above what the
whole-line pull put at the use sites there (about 0.1)."""
# REVIEW: a proposal; the alternative is 0.5, which would leave the pulled states with half their length off the
# axis. Rule (c) reads saturation on the uncapped arm, so the cap only matters if the capped arm goes forward.


@dataclass(frozen=True)
class Arm:
    """One arm of the pilot: what it trains and why it is here."""

    name: str
    group: str
    """Which question the arm serves: `control`, `label`, `hinge`, `verify`, or `mask`."""
    condition: tuple[int, float] = CENTRE
    anchored: bool = False
    label: str = "whole"
    """The label-variant short name (`LABEL_VARIANTS`); ignored on an unanchored arm."""
    policy: str = CROP_POLICY
    hinge: bool = False
    verify: bool = False
    mask: bool = False
    """The newline mask: attention does not cross a line break, so a context never reads the ones before it."""
    model: str = MODEL
    seeds: int = SEEDS
    note: str = ""


ARMS: tuple[Arm, ...] = (
    *(
        Arm(f"control-k{k}-r{rho:g}", "control", (k, rho), note="the un-anchored control at each corpus condition")
        for k, rho in GRAMMAR_CONDITIONS
    ),
    Arm(
        "control-large",
        "control",
        model=LARGE_MODEL,
        note="the larger control, trained either way and scored only if rule (a) needs it",
    ),
    *(Arm(f"anchor-{label}", "label", anchored=True, label=label, note=desc) for label, desc in LABEL_VARIANTS),
    Arm("anchor-hinge", "hinge", anchored=True, hinge=True, note="the whole-line pull capped by a hinge"),
    Arm("control-verify", "verify", verify=True, note="the control with verification lines"),
    Arm("anchor-verify", "verify", anchored=True, verify=True, note="the whole-line arm with verification lines"),
    Arm("control-mask", "mask", mask=True, note="the control with the newline mask"),
    Arm("anchor-mask", "mask", anchored=True, mask=True, note="the whole-line arm with the newline mask"),
)
"""The arms the pilot section of the design lists, in its order. Every anchored arm anchors `ANCHORED_OP` at the
center condition; the control at the center is the reference for all of them."""

PRIMARY = "anchor-whole"
CONTROL = f"control-k{CENTRE[0]}-r{CENTRE[1]:g}"
N_RUNS = sum(a.seeds for a in ARMS)
assert PRIMARY in {a.name for a in ARMS} and CONTROL in {a.name for a in ARMS}
assert N_RUNS == 42


def arm(name: str) -> Arm:
    return next(a for a in ARMS if a.name == name)


# --- The rules -----------------------------------------------------------------------------------------------

CEILING_MARGIN = 0.03
"""Rule (a): a control passes at a condition when its seed-mean held-out expected exact match is at least the
ceiling of record less this margin. One-sided: a control above the ceiling passes, and the calibration check
says whether it got there by sharpening past calibration. About a fifteenth of the room between floor and ceiling
at the center condition, and less than the ceiling gives up when one of three examples is ignored (the method
section computes that shortfall)."""
# REVIEW: a proposal. The alternatives are the task gate, 0.02, which earlier experiments used for "no
# measurable difference" between arms, or a skill-score floor (0.9 would be about 0.04 at the center). Verify:
# the room and the one-example shortfall in the conditions table and the calibration table.

SPREAD_STATISTIC = "sd"
"""Rule (a) ranks the passing conditions by the standard deviation of the posterior on the true op across
contexts; the middle-band share is reported beside it and breaks a tie."""

SEED_BAND_SD = 2.0
"""The seed band between two arms at n seeds each is `SEED_BAND_SD · σ · √(2/n)`, with σ the seed standard
deviation of the statistic pooled over the two arms: the smallest difference of seed means the comparison
resolves, as ex-2.2.9 defined it. Rules (b) and (d) use it on held-out expected exact match at the center
condition."""

MARGIN_KEEP = 0.9
"""Rules (b) and (c): an arm "holds the anchor" when its seed-mean op margin is at least this share of the
margin of the whole-line arm, the bar of ex-2.2.15."""

SATURATION_LEVEL = 0.9
"""Rule (c): the uncapped whole-line arm saturates at the query `?` when the seed-mean cosine with e₁ at that
position, on the held-out contexts of the anchored op at the last block, is at least this. The op word in
ex-2.2.14 sat near 1; *red* at its own token reached about 0.5."""
# REVIEW: a proposal, set so that only a state that is nearly all concept counts. The alternative is to read
# saturation as the dose collapse itself, on the suppression pass: a projection whose damage arrives only in
# the last few percent of γ at the query `?`. Verify: the alignment-by-role figure of rule (c).

KL_CALIBRATED = 0.05
"""The calibration check: a model is calibrated at a condition when the mean over held-out contexts of the KL
divergence from the Bayes predictive to its answer distribution is under this, in nats. The method section
gives reference values: a tenth of the floor mixed into the Bayes answer is about this large, and one example
ignored is several times it."""
# REVIEW: a proposal; the check gates nothing in the pilot, it flags a control that scores above the ceiling and
# it is the calibration measurement the design asks for. The alternative is a threshold relative to the
# irreducible loss H(q), such as a twentieth of it. Verify: the calibration table in the method section.

DOSE_GAMMAS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
"""Rule (e), the dose axis of the projection: the edit removes a fraction γ of the component on e₁ and re-normalizes.
γ = 1 is the plain projection."""

REPULSION_LANDINGS: tuple[float, ...] = (0.25, 0.0)
"""Rule (e), the dose axis of the repulsion: the alignment a state above the threshold is sent to. Both landings sit
between the threshold and zero, the range the operator was designed for."""
# REVIEW: an earlier draft carried negative landings down to the antipode (−1), which is outside the use the
# repulsion was designed for; Sandy has not yet settled how, or whether, it should act on states that lie almost on
# e₁. Treat these two landings as a placeholder until then.

REPULSION_THRESHOLD = 0.5
"""States below this alignment are left where they are by the repulsion."""

REFLECT_GAMMA = 2.0
"""The reflection, `projection` at γ = 2: one dose only, so it enters rule (e) as a reference and cannot win it."""

EDIT_SITES: tuple[str, ...] = ("query ?", "query =", "every position")
"""Where each operator acts, at every slice: the two query sites separately, and every position of the context.
Rule (e) is scored on the every-position edit; the query sites are the first bypass measurement and gate nothing."""
# REVIEW: scoring on every position is the reading that does not depend on where the op turns out to sit. The
# alternative is to score at the query site with the larger full-dose damage, which is what round 3 would
# suppress at. Verify: the per-site damage table of rule (e).

SELECTIVITY_GATE = 0.02
"""Rule (e): on each of the other ten ops, the seed-mean drop in held-out expected exact match under an edit is at
most this at every dose: the task gate ex-2.2.11 set."""

GRADING_MIN_DAMAGE = 0.5
"""Rule (e): the damage an operator does "grades with dose" when the seed-mean drop on the contexts of the anchored op is
non-decreasing along its dose axis, and the drop at full dose is at least this share of the way from the clean
score to the target null."""
