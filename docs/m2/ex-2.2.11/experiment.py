"""
The handover re-run: ex-2.2.9's conditions at fresh seeds, scored with the three reads ex-2.2.10 proposed.

Ex-2.2.9 put every proposal of the scouting round together (table A+, drawn answers, the whole-line
labeller, the untied readout) and did not adopt the result: the removal gate missed on the three ops that
take one HSV attribute from their second operand, and one seed in twenty ended under the retention gate.
Ex-2.2.10 then read the stored runs and found both misses in the reads rather than in the model. The
removal miss is the line-picking rule's: the projection acts on the red operand like a change of hue, and
the to-zero rule counts lines whose answer takes only red's saturation or value. The retention drop happens
under the constant anchor weight, before the anneal the read was meant to check. And the op1 alignment
rises under either half of the handover, so its old reference belongs to the old grammar.

Reads chosen after looking at the data cannot score the same data, so this is the same experiment again,
at seeds ex-2.2.9 never trained, with the reads fixed in advance:

1. Removal lines are chosen by hue: a red line is a removal line when some channel permutation of its red
   operand moves the true answer by at least `FAR_MOVE`. The slots the rule drops (the ones whose answer
   takes only red's saturation or value) become a second read, with no gate.
2. Retention is the final alignment over the alignment at the start of the anneal, gated at 0.8, and the
   level at the end of training is reported beside the references.
3. ᾱ at op1 is a report line with `handover-slot` and `handover-tied` as its references, and gates nothing.

Design only: constants and the removal rule, no DAG yet. The DAG is ex-2.2.9's with the seed offset,
the hue rule in the probe arrays, the reference conditions scored under the projection too, and
checkpoints kept on the trajectory stride for a few seeds. `DESIGN_ONLY` goes when it lands.

    bin/mini run docs/m2/ex-2.2.11/experiment.py --app modal --max-containers 5
    bin/mini status ex-2.2.11
"""

from __future__ import annotations

import importlib.util
import itertools
import sys

import numpy as np

DESIGN_ONLY = True


def _load_ex229():
    """Ex-2.2.9's module, loaded by path and left out of `sys.modules`, as ex-2.2.10 does, so the task bodies
    here still cloudpickle by value for a remote worker.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "ex-2.2.9" / "experiment.py"
    spec = importlib.util.spec_from_file_location("ex229", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex229 = _load_ex229()

# --- What stays as ex-2.2.9 had it --------------------------------------------------------------

Cond = ex229.Cond
OP_NAMES = ex229.OP_NAMES
TABLE = ex229.TABLE
N_OPS = ex229.N_OPS
KEPT = ex229.KEPT
ADDED = ex229.ADDED
ORDER_SENSITIVE = ex229.ORDER_SENSITIVE
PRIMARY_OP = ex229.PRIMARY_OP
N_LINES = ex229.N_LINES
EPOCHS = ex229.EPOCHS
CORPUS_SEED = ex229.CORPUS_SEED
HOLDOUT_FRAC = ex229.HOLDOUT_FRAC
N_PROBE = ex229.N_PROBE
PROBE_SEED = ex229.PROBE_SEED
PROBE_BOTH_SLOTS = ex229.PROBE_BOTH_SLOTS
RED_DOSE = ex229.RED_DOSE
NONRED_DOSE = ex229.NONRED_DOSE
LAM = ex229.LAM
TAU = ex229.TAU
TRAJ_STRIDE = ex229.TRAJ_STRIDE
OPERATORS = ex229.OPERATORS
OPERATOR_SPEC = ex229.OPERATOR_SPEC
SYNTAX_WORDS = ex229.SYNTAX_WORDS
"""The grammar, the corpus, the probe sets, the recipe, and the operators: all as ex-2.2.9 froze them. The
same corpus at the same seed, so the held-out pairs are the same and the task read is on the same lines."""

TASK_GATE = ex229.TASK_GATE
TASK_PARTIAL = ex229.TASK_PARTIAL
REF_M_LINE = ex229.REF_M_LINE
MARGIN_RATIO = ex229.MARGIN_RATIO
MARGIN_PARTIAL = ex229.MARGIN_PARTIAL
REF_R2_SIM = ex229.REF_R2_SIM
GRADE_R2_RATIO = ex229.GRADE_R2_RATIO
LEAD_GATE = ex229.LEAD_GATE
CONTRAST_GATE = ex229.CONTRAST_GATE
CONTRAST_PARTIAL = ex229.CONTRAST_PARTIAL
LATCH_PI = ex229.LATCH_PI
RED_KEPT_GATE = ex229.RED_KEPT_GATE
NONRED_DEFICIT_GATE = ex229.NONRED_DEFICIT_GATE
NONRED_DEFICIT_PARTIAL = ex229.NONRED_DEFICIT_PARTIAL
DEFICIT_NOISE = ex229.DEFICIT_NOISE
TAIL = ex229.TAIL
RESOLUTION_SD = ex229.RESOLUTION_SD
NOISE_RUN = ex229.NOISE_RUN
COMPONENT_NOISE = ex229.COMPONENT_NOISE
"""Every gate ex-2.2.9 set, at the same value. The re-run changes which lines the removal gate is read on,
what the retention ratio divides by, and whether ᾱ is gated; it changes no threshold."""

# REVIEW: bound RESOLUTION_SD, NOISE_RUN, and COMPONENT_NOISE above as well. Every "within a band" or
# "more than a band" read in H1, H2, and H4 needs a frozen σ source, and the draft named one only for
# the deficit. Verify: ex-2.2.9's module carries the same three names, with the same sources.

# --- Conditions -------------------------------------------------------------------------------------

SEED_OFFSET = 100
"""Every run trains at a seed ex-2.2.9 never used: condition seed *i* here is model seed `SEED_OFFSET + i`.
Ex-2.2.10 chose the reads on ex-2.2.9's seeds 0..19, so those checkpoints cannot score them. The corpus
seed is unchanged, so the task is the same; only the initialization and the batch order are fresh."""

CONTROL = Cond("control", 5, "un-anchored", "reference", lam=0.0)
"""As ex-2.2.9: the task reference for H1."""

HANDOVER = Cond("handover", 20, "the recipe, untied readout, whole-line labeller", "candidate")
"""The one candidate, at the reference's seed count. The gates are read on this condition alone."""

SLOT = ex229.SLOT
"""Ex-2.2.3's either-slot labeller, twenty seeds. Reference for the ᾱ line (it undid half the rise in
ex-2.2.9), for the whole-line label's selectivity cost (H4), and for the saturation-and-value read, where it
says whether the shortfall is the operator's or this checkpoint's. Not a fallback: the labeller needs the
operand positions, which M3 will not have."""

TIED = ex229.TIED
"""The tied readout, nine seeds. Reference for the ᾱ line (it undid the other half), for the syntax-token
read (H4), and for the saturation-and-value read. Nine seeds are enough for a line with no gate on it."""

CONDS: tuple[Cond, ...] = (CONTROL, HANDOVER, SLOT, TIED)
ANCHORED: tuple[Cond, ...] = tuple(c for c in CONDS if c.lam > 0)
N_RUNS = sum(c.seeds for c in CONDS)
assert N_RUNS == 54
"""`handover-narrow` does not return: its lines-per-op question was exploratory and ex-2.2.9 answered it."""

SCORED_UNDER_PROJECTION: tuple[Cond, ...] = (HANDOVER, SLOT, TIED)
"""Every anchored condition is scored under the operators, so the saturation-and-value read and the
selectivity comparisons have the references beside the candidate. Ex-2.2.9 scored them too; ex-2.2.10's
answer-mass read did not, which is the gap it asked the re-run to close."""

# --- The removal lines, by hue ----------------------------------------------------------------------

FAR_MOVE = ex229.FAR_MOVE
"""Unchanged at 0.4: a move of the true answer in the unit cube that counts as far."""

# REVIEW: ex-2.2.9 chose removal lines by zeroing the red operand's R channel. Ex-2.2.10 found that
# rule counts lines whose answer takes only red's saturation or value, which the projection leaves
# mostly alone (zeroing R on pure red gives black, which loses the saturation and value too), and
# that the projection behaves like a change of hue. The rule here asks about the hue alone. Verify:
# the counts in the report's method, per op and slot, against ex-2.2.10's counterfactual table.
HUE_RULE = "channel permutation"
"""A red line is a *removal line* when some permutation of its red operand's channels moves the true answer by
at least `FAR_MOVE`. A permutation keeps the operand's saturation and value and moves its hue, so these are
the lines whose answer needs the hue of *red*. On `mix` this is the to-zero set exactly; on `hsvmix` the two
rules differ on a few lines; on the six other channel-wise ops the hue rule is wider, since a permutation
moves two channels where zeroing R moved one. On the HSV ops it drops the slots whose answer takes only
saturation or value from red."""

HUE_ROTATION_CHECK = True
"""The permutation rule reaches six hues. The method also counts the lines a finer rule would pick, rotating
the red operand's hue in HSV in steps and snapping to the grid, and reports every line where the two rules
disagree. If they disagree anywhere, the finer rule is the one to keep, and the report says so before the
freeze. This constant is a reminder that the check is part of the method, and its result is not a gate."""

HUE_ROTATION_STEPS = 12
"""Hue steps for the finer rule: every 30 degrees, snapped to the grid, with the operand's saturation and
value held."""


def hue_move(op, a, b) -> float:
    """The furthest the true answer moves in the unit cube under any channel permutation of the redder
    operand. A line is a removal line when this is at least `FAR_MOVE`.
    """
    from sca.data.colors import redness
    from sca.data.ops import TOP

    red_first = redness(a) >= redness(b)
    red = a if red_first else b
    truth = op(a, b)
    moves = []
    for p in set(itertools.permutations(red)):
        if p == tuple(red):
            continue
        za, zb = (p, b) if red_first else (a, p)
        moves.append(float(np.linalg.norm(np.subtract(op(za, zb), truth)) / TOP))
    return max(moves) if moves else 0.0


def sv_line(op, a, b) -> bool:
    """A red line the hue rule drops: red by dose, and no permutation of the red operand moves its answer far.
    These are the saturation-and-value lines, read without a gate.
    """
    from sca.data.colors import redness

    return max(redness(a), redness(b)) >= RED_DOSE and hue_move(op, a, b) < FAR_MOVE


# --- Retention, against the start of the anneal -----------------------------------------------------

RETENTION_FLOOR = ex229.RETENTION_FLOOR
RETENTION_GATE = ex229.RETENTION_GATE
ANNEAL_WEIGHT_RATIO = 0.99
"""H2 (retention), a line and no gate: every run whose alignment at the start of the anneal reaches `RETENTION_FLOOR` ends at
`RETENTION_GATE` of that value. The anneal starts at the first trajectory point after the anchor weight's
peak where the weight is under `ANNEAL_WEIGHT_RATIO` of it, and the alignment at the start is the last point
before that, as ex-2.2.10 read it. The old ratio divided by the run's peak, which on `handover` is the high
point of a noisy plateau reached thirty epochs before the anneal."""

LEVEL_REFS: tuple[Cond, ...] = (SLOT, TIED)
"""The level line: `handover`'s final alignment, seed mean and range, beside the same on these references and
on ex-2.2.3's adopted point. No gate. Ex-2.2.10 read 0.66 against 0.72 and 0.70; the drift that produced the
gap happens under the constant anchor weight and is the [training-dynamics
item](/todo/science/training-dynamics-under-the-retention-drift.md)'s question."""

# --- Containment, as a line -----------------------------------------------------------------------------

ALPHA_REFS: tuple[Cond, ...] = (SLOT, TIED)
"""ᾱ at op1 on `handover`, beside the same on these two references and on ex-2.2.3's adopted point. No gate:
ex-2.2.9 read 0.28 against 0.18 and 0.16, each reference undoing about half the rise, and no mechanism is
named for either half. A gate returns once one is."""

EOL_ROW = "\n"
"""The `⏎` embedding row's axis component, on every anchored condition, as a line. Ex-2.2.9 read 0.17 on
`handover` and about zero on `handover-slot`, so the whole-line labeller is what puts it there."""

# --- Checkpoints on the trajectory stride --------------------------------------------------------------

TRAJ_CHECKPOINT_SEEDS = 3
"""The first three seeds of each anchored condition keep a checkpoint at every trajectory point, so a
training-dynamics read (the local learning coefficient, the whole-geometry read) can be run on the plateau
after the fact. About `TRAJ_STRIDE` checkpoints per run, a few MB each. The read itself is not part of this
experiment."""

# --- Refs -------------------------------------------------------------------------------------------

METRICS_REF = "reports/m2/ex-2.2.11/metrics"
CALIBRATION_REF = ex229.CALIBRATION_REF
"""Ex-2.2.9's calibration stands: same corpus, same point, same control. It is not repeated."""
ARRAYS_REF = "reports/m2/ex-2.2.11/arrays"
TRAJ_REF = "reports/m2/ex-2.2.11/trajectories"
PROBE_REF = "reports/m2/ex-2.2.11/probes"
CHECKPOINT_REF = "reports/m2/ex-2.2.11/checkpoints/{label}"
TRAJ_CHECKPOINT_REF = "reports/m2/ex-2.2.11/checkpoints/{label}/{point}"
RUN_ARRAYS_REF = "reports/m2/ex-2.2.11/arrays/{label}/{kind}"

EX229_METRICS_REF = ex229.METRICS_REF
EX229_TRAJ_REF = ex229.TRAJ_REF
"""Ex-2.2.9's numbers, printed beside every read as the before column."""

# --- The decision rule --------------------------------------------------------------------------------

DECISION = f"""\
The handover is adopted, and `{HANDOVER.name}` becomes the grammar and recipe of record for the anchored-op \
experiments, when it clears H1, H2 (margin, grading, and contrast, all in full), and H3 in full on the removal lines chosen by hue; every partial band is a reporting level. \
Otherwise it is not adopted, and the report says which gate was missed and what the references say about \
which change is responsible."""
