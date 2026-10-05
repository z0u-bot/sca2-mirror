"""Ex-2.2.22: localized by depth, various pull caps, and contexts of varying length — a scout.

Ex-2.2.21 left three questions about how to anchor the inferred op. The `no-emb` condition, which leaves the
embedding slice out of the pull, had the best task score but edited less selectively. The hinge condition, whose
pull stops at an alignment of 0.8, edited selectively, and the uncapped condition just missed the gate, so a cap
between the two may hold more of the anchor and keep the selectivity. And the posterior on the op takes about three
distinct values given a whole context of three examples, too few to test whether the anchor grades with it. The hinge
condition is the best recipe so far, so every new condition here changes one setting from it (or from a new condition
of this scout), and ex-2.2.21's runs are reused as references, paired by model seed.

Design only for now: the constants below are frozen with the report skeleton. The DAG reuses ex-2.2.21's (the
corpus prep, `cells`, the training step, the eval, and the suppression pass), with three additions: a corpus whose
contexts draw their example count per line, a held-out set with every count, and the landing measurements of H2.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass

DESIGN_ONLY = True


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as in ex-2.2.16 to ex-2.2.21."""
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


ex2221 = _load_sibling("ex-2.2.21", "ex2222_ex2221")
ex2216 = ex2221.ex2216

# --- What is inherited, unchanged --------------------------------------------------------------------------

OP_NAMES = ex2221.OP_NAMES
ANCHORED_OP = ex2221.ANCHORED_OP
MODEL = ex2221.MODEL
N_LAYER = 4
"""The depth of `MODEL` (d64-L4): five slices, the embedding and the output of each block."""
assert MODEL.endswith(f"L{N_LAYER}")
N_SLICES = N_LAYER + 1

EPOCHS = ex2221.EPOCHS
CENTRE = ex2221.CENTRE
"""Three examples at ρ = 0.3: the corpus of every condition but `k-mixed`, and of every reference."""
K, RHO = CENTRE
LABEL_RATE = ex2221.LABEL_RATE
HINGE_SOFTNESS = ex2221.HINGE_SOFTNESS
VERIFY = False
"""No verification lines, as in the hinge condition of ex-2.2.21. Ex-2.2.21 (S2) found they leave
completion unchanged."""

SEED_BAND_SD = ex2221.SEED_BAND_SD
REGRESSION_TOL = ex2221.REGRESSION_TOL
SELECTIVITY_GATE = ex2221.SELECTIVITY_GATE
GRADING_MIN_DAMAGE = ex2221.GRADING_MIN_DAMAGE
DOSE_GAMMAS = ex2221.DOSE_GAMMAS
TASK_COST_TOL = ex2221.TASK_COST_TOL
"""The task tolerance, the seed band, and E2 of ex-2.2.21 (the grading criterion, the selectivity gate, and the dose
axis of the projection), unchanged."""

LAMBDA_A = 0.1
"""The anchor weight of the recipe (ex-2.2.14), before its anneal to a floor over the last tenth of training."""

# --- The conditions ----------------------------------------------------------------------------------------

SLICE_SETS: dict[str, tuple[int, ...]] = {
    "all": tuple(range(N_SLICES)),
    "no-emb": tuple(range(1, N_SLICES)),
    "no-last": tuple(range(N_SLICES - 1)),
    "middle": tuple(range(1, N_SLICES - 1)),
}
"""Which residual-stream slices the anchor and anti-subspace terms act on. Slice 0 is the embedding and slice 4 the
input to the readout. `middle` leaves out both."""


def per_slice_weight(slices: str) -> float:
    """The anchor weight that keeps the effective pull what it is with every slice pulled. Both terms average over
    the slices they act on, so dropping slices at a fixed weight pulls each remaining slice harder, by
    `N_SLICES / len(slices)`: by a quarter for `no-emb` and `no-last`, and by two thirds for `middle`.
    """
    return LAMBDA_A * len(SLICE_SETS[slices]) / N_SLICES


HINGE_CAP = ex2221.HINGE_CAP
"""The cap of ex-2.2.21's hinge condition, the base recipe here."""


@dataclass(frozen=True)
class Condition:
    """One condition: its settings, and the condition it differs from in one of them."""

    name: str
    reference: str
    """The condition this one differs from in one setting, paired by model seed: an ex-2.2.21 condition, or a new
    condition of this scout for the `-matched` twins."""
    anchored: bool = True
    slices: str = "all"
    weight: float = LAMBDA_A
    """The anchor weight before its anneal. The anneal floor and the anti-subspace schedule scale with it."""
    cap: float | None = HINGE_CAP
    counts: tuple[int, ...] = (K,)


MIXED_COUNTS: tuple[int, ...] = (1, 2, 3, 4, 5)
"""The example counts of the `k-mixed` condition, one drawn uniformly per context. The mean is three, so the corpus
has the same number of tokens on average as the fixed-count corpus, and an epoch the same number of steps. Each count
adds levels to the posterior on `difference` given the whole context that the others lack (the report has the
figure).
One example contributes the most contexts in doubt, and pooled over the counts, the share of `difference` contexts
whose examples favour some other op is about what it is at three examples. Two whole contexts at five examples fit in
the window of 96 tokens."""

BASE = "anchor-hinge"

REFERENCES: tuple[Condition, ...] = (
    Condition("control", "", anchored=False, cap=None),
    Condition(BASE, ""),
    Condition("anchor-whole", "", cap=None),
    Condition("anchor-no-emb", "", slices="no-emb", cap=None),
)
"""The ex-2.2.21 runs this scout reuses, at the seeds below. `anchor-hinge` is the base of every new condition;
`anchor-whole` is the uncapped end of the cap ladder, and `anchor-no-emb` the uncapped twin of `no-emb`. The control
and `anchor-whole` had five seeds there; only the first three pair with this scout."""

CONDITIONS: tuple[Condition, ...] = (
    *(Condition(s, BASE, slices=s) for s in ("no-emb", "no-last", "middle")),
    *(Condition(f"{s}-matched", s, slices=s, weight=per_slice_weight(s)) for s in ("no-emb", "no-last", "middle")),
    Condition("cap-0.9", BASE, cap=0.9),
    Condition("cap-0.95", BASE, cap=0.95),
    Condition("k-mixed", BASE, counts=MIXED_COUNTS),
)
"""Every new condition is the hinge condition of ex-2.2.21 with one setting changed, except the `-matched` twins,
which change the weight from a slice condition of this scout."""

SEEDS = 3
SEED_OFFSET = ex2221.SEED_OFFSET
"""The model seeds of ex-2.2.21's three-seed conditions, 700 to 702, so each new run pairs with its reference by
seed. Seed 702 took the slow path through training on most anchored conditions there, so the pairing also shows
whether a setting changes that."""

REPLICATE_SEEDS: tuple[int, ...] = (703, 704, 705)
"""H2: three more runs of `anchor-hinge`, at seeds it was not trained at in ex-2.2.21, so H2 is scored on runs its
prediction was not drawn from. The first two pair with the five-seed control of ex-2.2.21."""
assert not set(REPLICATE_SEEDS) & set(range(SEED_OFFSET, SEED_OFFSET + SEEDS))

N_RUNS = SEEDS * len(CONDITIONS) + len(REPLICATE_SEEDS)
assert N_RUNS == 30


def settings(c: Condition) -> dict[str, object]:
    """The settings a design table compares between a condition and its reference."""
    return {"slices": c.slices, "weight": c.weight, "cap": c.cap, "counts": c.counts}


def by_name(name: str) -> Condition:
    return next(c for c in (*REFERENCES, *CONDITIONS) if c.name == name)


for _c in CONDITIONS:
    _diff = [k for k, v in settings(_c).items() if settings(by_name(_c.reference))[k] != v]
    assert len(_diff) == 1, (_c.name, _diff)

# --- The measurements --------------------------------------------------------------------------------------

LANDING_FRACTION = 0.75
"""H2: at full dose, the edit at every position closes at least this share of the clean model's distance from the
target null, in total variation, as a ratio of means over held-out `difference` contexts. Proposed; the report has the
check against ex-2.2.21."""

GRADING_SITES: tuple[str, ...] = ("example answers", "query answer")
"""E3: the roles where the alignment is compared with the posterior on `difference` given the pairs up to and
including that answer. The example answers are where ex-2.2.21 (E1) found the anchor; at the query answer the state
already holds the answer, so its posterior counts the query pair too, and there ex-2.2.21 found the alignment climbing
with depth."""
