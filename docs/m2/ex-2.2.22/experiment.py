"""Ex-2.2.22: where the pull acts, how far it goes, and contexts of varying length — a scout before round 3.

Ex-2.2.21 (round 2 of the D2.2 route) left three questions about how round 3 should anchor the inferred op. The
`no-emb` arm, which leaves the embedding slice out of the pull, had the best task score but edited less selectively.
The hinge arm, whose pull stops at an alignment of 0.8, edited selectively, and the uncapped arm just missed the
gate, so a cap between the two may hold more of the anchor and keep the selectivity. And the posterior on the op takes
about three distinct values at three examples, too few for round 3 to test whether the anchor grades with it. This
scout trains new arms for each question, scores them as ex-2.2.21 did, and reuses ex-2.2.21's runs as references,
paired by model seed.

Design only for now: the constants below are frozen with the report skeleton. The DAG reuses ex-2.2.21's (the
corpus prep, `cells`, the training step, the eval, and the suppression pass), with two additions: a corpus whose
contexts draw their example count per line, and a held-out set with every count.
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
"""Three examples at ρ = 0.3: the corpus of every arm but the two count arms, and of every reference."""
K, RHO = CENTRE
LABEL_RATE = ex2221.LABEL_RATE
HINGE_SOFTNESS = ex2221.HINGE_SOFTNESS
VERIFY = False
"""No verification lines, so every new arm pairs with its ex-2.2.21 reference in one setting alone. Ex-2.2.21 (S2)
found they leave completion unchanged, and round 3 adds them back."""

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

# --- The arms ----------------------------------------------------------------------------------------------

SLICE_SETS: dict[str, tuple[int, ...]] = {
    "all": tuple(range(N_SLICES)),
    "no-emb": tuple(range(1, N_SLICES)),
    "no-last": tuple(range(N_SLICES - 1)),
    "middle": tuple(range(1, N_SLICES - 1)),
}
"""Which residual-stream slices the anchor and anti-subspace terms act on. Slice 0 is the embedding and slice 4 the
input to the readout. `middle` leaves out both."""


def per_slice_weight(slices: str) -> float:
    """The anchor weight that keeps the pull on each slice what it is with every slice pulled. Both terms average
    over the slices they act on, so dropping slices at a fixed weight pulls each remaining slice harder, by
    `N_SLICES / len(slices)`: by a quarter for `no-emb` and `no-last`, and by two thirds for `middle`.
    """
    return LAMBDA_A * len(SLICE_SETS[slices]) / N_SLICES


@dataclass(frozen=True)
class Arm:
    """One arm: the one thing it changes from its reference, and the question it serves."""

    name: str
    group: str
    """`slices`, `cap`, or `counts`."""
    reference: str
    """The arm this one differs from in one setting, paired by model seed: an ex-2.2.21 arm, or a new arm of this
    scout for the `-matched` twins of `no-last` and `middle`."""
    anchored: bool = True
    slices: str = "all"
    weight: float = LAMBDA_A
    """The anchor weight before its anneal. The anneal floor and the anti-subspace schedule scale with it."""
    cap: float | None = None
    counts: tuple[int, ...] = (K,)
    note: str = ""


MIXED_COUNTS: tuple[int, ...] = (1, 2, 3, 4, 5)
"""The example counts of the two count arms, one drawn uniformly per context. The mean is three, so the corpus has
the same number of tokens on average as the fixed-count corpus, and an epoch the same number of steps. Each count
adds levels to the posterior on `difference` that the others lack (the report has the figure). Two whole contexts at
five examples fit in the window of 96 tokens."""

ARMS: tuple[Arm, ...] = (
    Arm("no-last", "slices", "anchor-whole", slices="no-last", note="the last slice left out of the pull"),
    Arm("middle", "slices", "anchor-whole", slices="middle", note="the embedding and the last slice left out"),
    *(
        Arm(
            f"{s}-matched",
            "slices",
            f"anchor-{s}" if s == "no-emb" else s,
            slices=s,
            weight=per_slice_weight(s),
            note=f"`{s}` at the weight that keeps the pull on each slice as it is with every slice pulled",
        )
        for s in ("no-emb", "no-last", "middle")
    ),
    Arm("cap-0.9", "cap", "anchor-hinge", cap=0.9, note="the whole-line pull, capped at 0.9"),
    Arm("cap-0.95", "cap", "anchor-hinge", cap=0.95, note="the whole-line pull, capped at 0.95"),
    Arm("control-mixed", "counts", "control", anchored=False, counts=MIXED_COUNTS, note="the control, mixed counts"),
    Arm("cap-0.8-mixed", "counts", "anchor-hinge", cap=0.8, counts=MIXED_COUNTS, note="the hinge arm, mixed counts"),
)
"""Every new arm changes one setting from an ex-2.2.21 arm, except the `-matched` arms of `no-last` and `middle`,
which change the weight from a new arm of this scout."""

REFERENCES: tuple[str, ...] = ("control", "anchor-whole", "anchor-no-emb", "anchor-hinge")
"""The ex-2.2.21 runs this scout reuses, at the seeds below. The control and the whole-line arm had five seeds there;
only the first three pair with this scout."""

SEEDS = 3
SEED_OFFSET = ex2221.SEED_OFFSET
"""The model seeds of ex-2.2.21's three-seed arms, 700 to 702, so each new run pairs with its reference by seed.
Seed 702 took the slow path through training on most anchored arms there, so the pairing also shows whether a
setting changes that."""

N_RUNS = SEEDS * len(ARMS)
assert N_RUNS == 27

# --- The measurements --------------------------------------------------------------------------------------

LANDING_FRACTION = 0.75
"""H2: at full dose, the edit at every position closes at least this share of the gap between the clean model and
the target null, in KL(target null ‖ model), the only direction that is finite, averaged over held-out `difference` contexts."""

SEED_AGREEMENT_RATIO = 1.5
"""H2: the edited runs agree with one another about their answers on `difference` contexts at least as closely as
this many times the disagreement of the clean runs, in the Jensen-Shannon divergence between the predictives of each
pair of seeds."""

GRADING_SITES: tuple[str, ...] = ("example answers",)
"""E3: the role where ex-2.2.21 (E1) found the anchor, and where the alignment is compared with the posterior on
`difference` given the examples up to and including that answer."""
