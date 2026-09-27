"""Ex-2.2.16: the in-context grammar pilot — round 2 of the D2.2 quick route.

The first experiment on the grammar the pivot proposed: each line is a context of a few solved examples of one op,
written with `?` in place of the op word, then a query under the same op. The pilot trains the control at three
grammar conditions (example count and replacement rate), anchored `difference` with the whole-line label and its
variants, and the arms the design lists, at a few seeds each, and proposes by frozen rules what round 3 adopts.

This module holds the design constants only. The DAG comes with the grammar generator, which is being written
separately; until then the report imports these constants and computes its method section (the posterior over
ops and the Bayes ceiling) from the op table alone, through `posterior.py` beside it.
"""

from __future__ import annotations

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
"""Table A+ as ex-2.2.9 assembled it and every experiment since has used: five of ex-2.2.3's ops, three commutative
additions, and the three HSV blend modes that read operand order. Every op is total on the grid. The order is the
one the reports print."""
N_OPS = len(TABLE)

ANCHORED_OP = "difference"
"""The op the anchored arms label, as in ex-2.2.14 and ex-2.2.15. It is total on the grid, so its ceiling is not
capped by rounding, and it is commutative."""

ROUNDING = "stochastic"
"""The corpus rounds each channel of a raw answer up or down at random, in proportion to where the raw value sits
between grid levels (`sca.data.ops.Rounding`), as every corpus since the handover has. The posterior uses the same
rounding, so the likelihood of a shown answer under an op is the probability that op's rounding gives it."""


def context_tokens(k: int) -> int:
    """`a ? b = y ,` is six tokens, and so is the query `a ? b = y ⏎`: a context of k examples is `6k + 6` tokens,
    the layout `sca.data.incontext` generates.
    """
    return context_length(k)


# --- The posterior scan (the method section) ---------------------------------------------------------------

K_GRID: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
RHO_GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5)
"""The grid the method section scans: example counts and replacement rates. The rate ρ is per example: each
example shows, with probability ρ, a draw from the answer distribution of another op, uniform over the other ten,
in place of a draw from the true op's. A replacement is invisible when the other op agrees with the true op on the
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
op, below it the evidence favors another op. The share of contexts in the band is reported beside the spread."""

# --- The proposed grammar conditions ------------------------------------------------------------------------

GRAMMAR_CONDITIONS: tuple[tuple[int, float], ...] = ((3, 0.2), (3, 0.3), (4, 0.3))
CENTRE: tuple[int, float] = (3, 0.3)
"""The three (examples, ρ) conditions the control trains at, proposed by the method section from the scan. The
center is the pivot's working point. The second differs in ρ alone, a step down in inference difficulty on the same
line length, so it is the fallback within the same block size if the control falls short of the ceiling at the
center. The third adds one example at the same ρ as the center, which raises the ceiling by about 0.06 and keeps most of the
spread; it asks whether more evidence buys a higher ceiling without flattening the stimulus."""
# REVIEW: proposed from the scan in the method section, not yet reviewed. The alternatives weighed there are
# (3, 0.4), which adds no middle-band share over the center and costs 0.08 of ceiling, and (4, 0.35), which the third
# condition dominates on ceiling at the same spread. Verify: the grid table in the method section.

CUBE_RATE: float | None = None
"""The cube-noise rate of the corpus. Left open until the plan is reviewed; the method section gives the cost at
each rate in `CUBE_GRID`."""

# --- What is inherited --------------------------------------------------------------------------------------

MODEL = "d64-L4"
SEEDS = 3
"""Per arm, as the design's pilot section sets. The larger control arm is the exception noted there."""

CROP_POLICY = "whole"
"""The anchor pulls only labelled lines wholly inside the training window, as ex-2.2.15 proposed after review:
a whole context always shows its evidence. Its rule chose `half`; the reasons for `whole` are in that report's
"The rule for the pilot" section."""
