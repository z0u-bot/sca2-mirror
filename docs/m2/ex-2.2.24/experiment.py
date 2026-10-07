"""Ex-2.2.24: where the edit spills from — diagnostics on ex-2.2.23's checkpoints.

Ex-2.2.23 found that the edit (projecting e₁ out of every state) spills onto other ops on nearly every run that has
learned the HSV ops, mostly onto `darken`, `lighten`, and `value-hsv`. This experiment trains nothing for its main
questions. It scores the checkpoints ex-2.2.23 already has, at 200 and 400 epochs, on three questions: whether the
example answers hold both the op and the lightness of a color in the control (H1), which positions the spill comes
from (H2), and whether e₁ tracks the lightness of a color at positions where it carries no evidence about the op (H3).

This is the preregistration draft: the constants below are the design, and the DAG lands when the design is frozen.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass

DESIGN_ONLY = True


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as in ex-2.2.16 to ex-2.2.23."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / name / "experiment.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex2223 = _load_sibling("ex-2.2.23", "ex2224_ex2223")
ex2222 = ex2223.ex2222
ex2221 = ex2223.ex2221

# --- What is inherited, unchanged --------------------------------------------------------------------------

OP_NAMES = ex2223.OP_NAMES
ANCHORED_OP = ex2223.ANCHORED_OP
HSV_OPS = ex2223.HSV_OPS
K = ex2223.K
SEEDS = ex2223.SEEDS
LENGTHS = ex2223.LENGTHS
SHORT, LONG = LENGTHS
CONDITIONS = tuple(c.name for c in ex2223.CONDITIONS)
DOSE_GAMMAS = ex2223.DOSE_GAMMAS
SELECTIVITY_GATE = ex2223.SELECTIVITY_GATE
GRADING_MIN_DAMAGE = ex2223.GRADING_MIN_DAMAGE
RULE = ex2223.RULE
# Ex-2.2.23's runs, every one of them: both conditions, twelve seeds, two lengths, 48 checkpoints. The half-trained
# runs are the ones ex-2.2.23's rule (S1) marks: the worst HSV op below 0.3 at the end of training.

SCORED_LENGTH = LONG
# The hypotheses are scored on the 400-epoch runs, where every run made the second rise (ex-2.2.23, E2) and which
# the recipe now trains at. The 200-epoch runs are shown beside them.

# --- The positions -----------------------------------------------------------------------------------------

UNIT = 6
"""Tokens per equation: op1, `?`, op2, `=`, answer, then `,` in an example or `\\n` in the query."""
QUERY = UNIT * K
"""The role of the query op1; the query `=`, where the answer is read, is at `QUERY + 3`."""
READOUT = QUERY + 3
assert READOUT == ex2221.ex2216.query_role(K, ex2221.ex2216.QUERY_EQ)

EXAMPLE_ANSWERS: tuple[int, ...] = tuple(UNIT * i + 4 for i in range(K))
EXAMPLE_OPERANDS: tuple[int, ...] = tuple(UNIT * i + j for i in range(K) for j in (0, 2))
QUERY_OPERANDS: tuple[int, ...] = (QUERY, QUERY + 2)
COLOR_ROLES: tuple[int, ...] = tuple(sorted(EXAMPLE_OPERANDS + EXAMPLE_ANSWERS + QUERY_OPERANDS))
"""Every position that holds a color, up to the readout."""


@dataclass(frozen=True)
class Site:
    name: str
    roles: tuple[int, ...]
    why: str


_UP_TO_READOUT = tuple(range(READOUT + 1))
SITES: tuple[Site, ...] = (
    Site(
        "example answers",
        EXAMPLE_ANSWERS,
        "where the whole-line label put most of the alignment, and where a color is also the evidence for the op",
    ),
    Site(
        "query operands",
        QUERY_OPERANDS,
        "the two colors the answer is computed from, so a change in their lightness reaches the answer of every op "
        "that depends on it",
    ),
    Site(
        "the rest",
        tuple(r for r in _UP_TO_READOUT if r not in EXAMPLE_ANSWERS + QUERY_OPERANDS + (READOUT,)),
        "the example operands and every syntax token before the query `=`",
    ),
    Site(
        "query `=`",
        (READOUT,),
        "where the answer is read, downstream of every other site, so a spill here acts on the answer itself",
    ),
)
"""H2: the edit at each site alone, at every slice and each dose. The four sites split every position up to the
readout; positions after it cannot reach the answer, since attention is causal. The edit at every position, as
ex-2.2.23 scored it, is repeated beside them as the reference."""
assert sorted(r for s in SITES for r in s.roles) == list(_UP_TO_READOUT)

SINGLE_POSITION_DOSE = 1.0
"""E2: the edit at each single position up to the readout alone, at full dose only, as a map beside the sites."""
SINGLE_POSITIONS: tuple[int, ...] = _UP_TO_READOUT

# --- The measurements --------------------------------------------------------------------------------------


def lightness(rgb) -> float:
    """The lightness of a grid color: the mean of its three channels on the 0..1 scale. `darken` and `lighten` work
    channel by channel, so the mean of the channels is the lightness they move; `value-hsv` takes the largest.
    """
    from sca.data.ops import TOP

    return sum(rgb) / (3 * TOP)


PROBE_SPLIT = 0.5
"""H1: each probe is fit on this share of the held-out contexts of a run (split by context, stratified by op) and
scored on the rest."""
PROBE_SLICES: tuple[int, ...] = (1, 2, 3, 4)
"""H1 scores each probe at the best of the slices after the embedding. At the embedding slice an answer state is the
embedding of its color token alone, which holds no op."""

OP_RECOVERY_GATE = 0.5
"""H1: at the first example answer, the op probe recovers at least this share of what the Bayes posterior gets
above chance: (probe accuracy − 1/7) / (Bayes accuracy − 1/7)."""

SITE_SHARE = 0.5
"""H2: a site accounts for the spill on a run when its spill (the largest drop on any other op, net of the control
under the same edit) is at least this share of the spill of the edit at every position."""
SITE_RUN_SHARE = 2 / 3
"""H2: the example answers account for the spill on at least this share of the 400-epoch anchored runs that spill."""

TRACKING_RANK_GATE = 0.4
"""H3 (b): across the 400-epoch anchored runs, the rank correlation (Spearman)
between lightness tracking and spill is at least this."""

# --- The 600-epoch arm (open decision) ---------------------------------------------------------------------

ARM_EPOCHS = 600
ARM_SEEDS: tuple[int, ...] = (700, 701)
"""The first two seeds, chosen before any result. Both conditions at each, so the spill can be netted against a
control at the same seed and length: four runs."""
N_ARM_RUNS = len(CONDITIONS) * len(ARM_SEEDS)

BUDGET_USD = 8
"""About \\$0.13 for a 200-epoch run on an L4, so about \\$0.40 at 600 epochs; four arm runs, plus the diagnostic
pass over the 48 checkpoints of ex-2.2.23 (five edits at four doses, one edit per position at full dose, and the
states at every position and slice for the probes)."""
