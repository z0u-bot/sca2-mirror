"""
The grammar handover: ex-2.2.3's recipe on table A+, with the stochastic corpus, the whole-line
labeller, and the untied readout.

Preregistered. The design comes first: the table, the conditions, the probe rule, and the gates, each
with its wording in a docstring, so the report quotes the same numbers it will be scored on. The DAG
follows: two corpora (the 300k one and the narrow one), a train and an eval step per run, the eval contract
under three operators on every run, cube probes on the un-anchored and the exploratory arms, and one
publish step. A `calibration` stage trains one control seed and stops.

    EX229_STAGES=calibration bin/mini run -w docs/m2/ex-2.2.9/experiment.py --app modal --max-containers 2

A miss on a commutative op sends the corpus size and epoch count back for a look: `EX229_CALIBRATION_EPOCHS=50,100,150`
trains one control seed at each count, `EX229_CALIBRATION_LINES=100000,200000` one at each corpus size, and
`EX229_CALIBRATION_MODELS=64x4,64x6` one at each width×depth (the design's own point is memoised), or
`EX229_CALIBRATION_POINTS=100000/50/64x4,200000/50/64x4` names the points one by one. Every look is published
under the calibration ref, merged with the looks before it, so the report shows them all.

The scouting round (ex-2.2.4) and the pilots (ex-2.2.5 to 2.2.8) each proposed one change to the
grammar or the recipe. This experiment adopts them together, at fresh seeds, and asks whether the
anchoring still lands, whether the removal reads are cleaner, and what the two changes whose pilots
were inconclusive (the labeller, the readout) each do. One reference condition changes each of
those back, so each read is a two-condition comparison on the new grammar.

    bin/mini run docs/m2/ex-2.2.9/experiment.py --app modal --max-containers 8 --budget 3h
    bin/mini status ex-2.2.9
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from sca.data.ops import CANDIDATE_BY_NAME, OP_BY_NAME, Op, Rounding
from mini import Ctx, Experiment, get_data_dir


def _load_ex223():
    """Ex-2.2.3's module, loaded by path and left out of `sys.modules`, as `mini.load_experiment` does,
    so its task functions still cloudpickle by value for a remote worker once the DAG lands.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "ex-2.2.3" / "experiment.py"
    spec = importlib.util.spec_from_file_location("ex223", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex223 = _load_ex223()

# Bound by name so a task body never references the module object itself.
Condition = ex223.Condition
steps_per_epoch = ex223.steps_per_epoch
REDNESS = ex223.REDNESS
GRID_RGB = ex223.GRID_RGB
PALETTE = ex223.PALETTE
ANSWER_POS = ex223.ANSWER_POS
DECODE_POS = ex223.DECODE_POS
N_EMBD, N_LAYER = 64, 4
"""The model's width and depth: the d64-L4 of every experiment since ex-2.1.3. The calibration look can try
another shape (`EX229_CALIBRATION_MODELS`); the freeze names the one adopted."""
SLICES = tuple(range(N_LAYER + 1))
"""The residual-stream slices the operators act on: the embedding and the output of every block."""
OPERAND_POSITIONS = ex223.OPERAND_POSITIONS
COMPOSITION = ex223.COMPOSITION
TRAJ_STRIDE = ex223.TRAJ_STRIDE
N_EVAL = ex223.N_EVAL
RED_RATE = ex223.RED_RATE
_npz = ex223._npz


def _make_config(vocab_size: int, seed: int, epochs: int, n_embd: int = 64, n_layer: int = 4):
    """Ex-2.2.3's config (d64-L4 from ex-2.1.3) with the model's width and depth as parameters, so the
    pre-freeze look can try another shape on the new grammar. The width is a multiple of 64 (`ModelConfig` wants
    eight heads of at least eight dimensions), and the MLP is 4× it.
    """
    from sca.config import ModelConfig

    base = ex223._make_config(vocab_size, seed, epochs)
    base.model = ModelConfig.model_validate(
        base.model.model_dump()
        | {"n_embd": n_embd, "n_head": 8, "n_head_dim": n_embd // 8, "n_ff": 4 * n_embd, "n_layer": n_layer}
    )
    return base


_load = ex223._load
_answer_logprobs = ex223._answer_logprobs
line_margin = ex223.line_margin
pooled_margin = ex223.pooled_margin
r2_sim = ex223.r2_sim
group_weights = ex223.group_weights
top_quantile = ex223.top_quantile
schedules = ex223.schedules
probe_one = ex223.probe_one

METRICS_REF = "reports/m2/ex-2.2.9/metrics"
CALIBRATION_REF = "reports/m2/ex-2.2.9/calibration"
ARRAYS_REF = "reports/m2/ex-2.2.9/arrays"
TRAJ_REF = "reports/m2/ex-2.2.9/trajectories"
PROBE_REF = "reports/m2/ex-2.2.9/probes"
GEOMETRY_REF = "reports/m2/ex-2.2.9/geometry"
CHECKPOINT_REF = "reports/m2/ex-2.2.9/checkpoints/{label}"
RUN_ARRAYS_REF = "reports/m2/ex-2.2.9/arrays/{label}/{kind}"
"""Every run's full per-line tables (`eval`: the alignment of every line; `score`: every reading under every
operator), one ref each. `ARRAYS_REF` stacks the smaller ones across runs for the report."""

EX223_METRICS_REF = ex223.METRICS_REF
EX223_REFERENCE = "recipe-short"
"""The reference: production's adopted point on the six-op grammar, twenty seeds with the addendum. Every
placement statistic here is read beside its value there, and the removal gates are the ones it was read on."""

EX227_METRICS_REF = "reports/m2/ex-2.2.7/metrics"
EX227_CEILING = "rows-clean"
"""Ex-2.2.7's condition with the syntax embeddings hard-zeroed every step: the in-grammar ceiling for how clean
the syntax embeddings can come, which the untied conditions are read against (H4)."""

# --- Table A+ ------------------------------------------------------------------------------

KEPT = ("mix", "screen", "multiply", "lighten", "darken")
"""Five of ex-2.2.3's six ops. `add` is dropped: a fifth of its pairs go to white, and it is the least
sensitive op to its red operand by either of ex-2.2.4's rules."""

ADDED = ("difference", "exclusion", "hsvmix")
"""Three commutative ops whose answers spread through the cube and depend on both operands."""

ORDER_SENSITIVE = ("hue-hsv", "sat-hsv", "value-hsv")
"""The marked subset: each takes one attribute of op2 at the rest of op1, so these are the first ops in the
grammar whose answer depends on which operand is which. Their reads are reported as a subset, beside the
table, so that anything of their own can be set aside without touching the rest."""

OP_NAMES: tuple[str, ...] = (*KEPT, *ADDED, *ORDER_SENSITIVE)
TABLE: tuple[Op, ...] = tuple(OP_BY_NAME[n] if n in OP_BY_NAME else CANDIDATE_BY_NAME[n] for n in OP_NAMES)
"""Table A+ of ex-2.2.4, in the order the report prints it. Every op is total on the grid."""
N_OPS = len(TABLE)
assert N_OPS == 11

PRIMARY_OP = "mix"
"""Every gated statistic is read on `mix`'s probe lines, as in ex-2.2.3, so the gates mean what they meant."""
SECONDARY_OP = "hsvmix"
"""Reported beside `mix` on every gated read, so a later experiment can make it the reference op if it
behaves. The switch, if it comes, is after the handover; what it would change is set out in the report's
method (the reference op): a stronger removal read, on a probe set that rounds on nearly every line."""

ROUNDING: Rounding = "stochastic"
"""An answer between grid levels rounds to the upper level with probability equal to how far up it sits,
drawn once per line at corpus build (`sca.data.ops.Rounding`). The answer of a line is then a distribution
over the candidate colors, and every exact-match read becomes an expectation over it (ex-2.2.5)."""

# --- The corpus ------------------------------------------------------------------------------

N_LINES = 300_000
"""Three times D2.1's 100k lines: the point the pre-freeze calibration look settled on. At 100k lines the control
missed the calibration bar on `difference` and `hsvmix` at every epoch count and depth looked at; what closed
the gap was more distinct lines, since `hsvmix` rounds on nearly every line and the model has to learn a
distribution it sees one draw of per pair. Ops are drawn uniformly, so each has about 27k lines, more than the
16.7k each op had at six ops. Lines per op is the confound E4 of ex-2.2.3 named; the `narrow` condition below
reads it. Coverage is rendered in the report's method."""

CORPUS_SEED = ex223.CORPUS_SEED
HOLDOUT_FRAC = ex223.HOLDOUT_FRAC
"""Of the distinct unordered pairs of each op, held out. A held-out pair is out in both orders, for every op,
so the held-out share of an op's lines is the same 20% whether or not the op reads operand order."""

EPOCHS = ex223.EPOCHS_SHORT
"""The adopted point's epoch count, 50, now over 300k lines: 4,950 steps, three times the reference's 1,650. The
calibration look found that more passes over the same lines push `hsvmix` away from its ceiling again (the
model memorises the drawn labels), so the extra steps come from the larger corpus rather than more epochs."""

N_LINES_NARROW = round(ex223.N_LINES * N_OPS / len(ex223.OP_NAMES))
EPOCHS_NARROW = round(EPOCHS * N_LINES / N_LINES_NARROW)
"""The `narrow` condition holds lines per op at ex-2.2.3's count (about 16.7k), so its corpus is eleven sixths
of D2.1's 100k and about three fifths of the main corpus, and it runs for more epochs so that the step count
matches. What differs from the handover condition is then how many distinct lines of each op the model sees,
and how often it sees each."""
assert abs(EPOCHS_NARROW * steps_per_epoch(N_LINES_NARROW) - EPOCHS * steps_per_epoch(N_LINES)) <= 60

# --- The recipe --------------------------------------------------------------------------------

LAM = ex223.SCORING_LAMBDA
TAU = ex223.TAU_REF
ANTI_PEAK_RATIO = ex223.ANTI_PEAK_RATIO_REF
ANTI_ANNEAL_END_FRAC = ex223.ANTI_ANNEAL_END_FRAC_REF
"""The adopted point of ex-2.2.3 (`recipe-short`): λ_a = 0.1, annealed over the last tenth of training to a
0.1 floor as ex-2.1.10 did; τ = 0.1; the anti-subspace weight from 2.5× the anchor weight to 0.3× by 90% of
training. Nothing here retunes it."""

ANCHOR_AXIS = ex223.ANCHOR_AXIS
"""*Red* is anchored to e₁ at every slice, the embedding included (ex-2.2.7 settled the Prep C question)."""

Keying = Literal["either", "line"]
PROMPT_SPAN = ex223.SPAN
WHOLE_SPAN = 6
"""The two labellers. `either`, prompt span (ex-2.2.3's): each operand draws at redness⁸ × 0.04, and the pull
covers op1, op, op2, and `=`. `line`, whole span (ex-2.2.6's): the answer draws at its redness rate too, and
the pull covers the answer and the newline as well."""
PER_SLOT_RATE = ex223.PER_SLOT_RATE

TIE_EMBEDDINGS = False
"""The readout is a table of its own, initialised as a copy of the embedding (ex-2.2.7's `untied`). The tied
condition sets this back."""

# --- Conditions ---------------------------------------------------------------------------------

Role = Literal["candidate", "reference", "exploratory"]


@dataclass(frozen=True)
class Cond:
    """One training condition. Everything not named here is the adopted point on table A+ with stochastic
    rounding.
    """

    name: str
    seeds: int
    title: str
    role: Role
    lam: float = LAM
    keying: Keying = "line"
    span: int = WHOLE_SPAN
    tie: bool = TIE_EMBEDDINGS
    n_lines: int = N_LINES
    epochs: int = EPOCHS
    n_embd: int = N_EMBD
    n_layer: int = N_LAYER

    @property
    def condition(self) -> Condition:
        return Condition(
            self.name, self.seeds, self.title, lam=self.lam, epochs=self.epochs, ops=OP_NAMES, n_lines=self.n_lines
        )

    @property
    def steps(self) -> int:
        return self.epochs * steps_per_epoch(self.n_lines)

    @property
    def lines_per_op(self) -> int:
        return self.n_lines // N_OPS


CONTROL = Cond("control", 5, "un-anchored", "reference", lam=0.0)
"""Nothing placed on the axis, on the new grammar with the untied readout: the task reference for H1, and the
calibration reference for the stochastic reads."""

HANDOVER = Cond("handover", 20, "the recipe, untied readout, whole-line labeller", "candidate")
"""Every proposal adopted at once. The one candidate for the grammar of record, at the reference's seed count.
The gates (H1 to H3) are read on this condition alone."""

SLOT = Cond(
    "handover-slot",
    20,
    "as handover, with ex-2.2.3's either-slot labeller",
    "reference",
    keying="either",
    span=PROMPT_SPAN,
)
"""The labeller changed back. Beside `handover` this is the selectivity check ex-2.2.7 asked for (H5): what the
whole-line labeller costs, if anything. Twenty seeds, so the comparison has the same resolution as the gates.
It is not a fallback: the either-slot labeller needs to know which tokens are the operands, and a method that
needs that is not one we can carry to M3."""

TIED = Cond("handover-tied", 9, "as handover, with the tied readout", "reference", tie=True)
"""The readout changed back. Beside `handover` this reads what untying does on the new grammar (H4): the
component on the syntax embeddings, and the non-red cost of the full-position projection."""

NARROW = Cond(
    "handover-narrow",
    5,
    "as handover, lines per op held at ex-2.2.3's count",
    "exploratory",
    n_lines=N_LINES_NARROW,
    epochs=EPOCHS_NARROW,
)
"""Exploratory: the lines-per-op confound of E4 (ex-2.2.3), read on the operand-cube probe scan and on the
task. Not gated. (The prereg draft called this `handover-wide`, when the main corpus was D2.1's 100k lines and
holding lines per op at the six-op count meant a larger corpus; the calibration look made the main corpus
the larger one.)"""

CONDS: tuple[Cond, ...] = (CONTROL, HANDOVER, SLOT, TIED, NARROW)
ANCHORED: tuple[Cond, ...] = tuple(c for c in CONDS if c.lam > 0)
N_RUNS = sum(c.seeds for c in CONDS)
assert N_RUNS == 59
assert len({c.name for c in CONDS}) == len(CONDS)

# --- The probe set -------------------------------------------------------------------------------

N_PROBE = ex223.N_PROBE
PROBE_SEED = ex223.PROBE_SEED
"""As ex-2.2.3: for `mix`, its 27 on-grid partners per color (D2.1's 5,832 lines with the op word changed);
for every other op, 27 partners per color drawn once with a fixed seed and shared across ops."""

PROBE_BOTH_SLOTS = ORDER_SENSITIVE
"""For the order-sensitive subset the probe set also walks every color as op2, against the same partners, so
each of those ops has 11,664 probe lines and its reads can be split by which slot the red operand is in."""

RED_DOSE = ex223.RED_DOSE
NONRED_DOSE = ex223.NONRED_DOSE
"""Dose is the redness of the redder operand. Red lines have dose ≥ 0.8; non-red lines have dose ≤ 0.2. As
ex-2.2.3."""

# REVIEW: the author asked whether setting R to one grid level rather than zero would keep the hue
# and so count the red-op2 lines under `hue-hsv` and `sat-hsv`. It does not: a one-level R is a
# dark red at full saturation, so `hue-hsv` moves as little and `sat-hsv` stops moving. The method
# says so with the counts (`eps_check` in the report); the rule stays at zero.
FAR_MOVE = 0.4
"""The removal lines: red lines on which zeroing the red operand's R moves the true answer by at least 0.4
(distance in the unit cube, ex-2.2.4's *to-zero* read). On `mix` that is every red line, since the move is
half the red operand's R; on `lighten` it drops the lines whose partner is about as red. The count per op is
rendered in the method."""

# REVIEW: the draft set the red-answer lines aside from the selectivity read. The author's review
# asked for the gate to cover them too, with the separate count kept. `mix` has none, so the gated
# read is the same either way; on the ops that have them the deficit is reported both ways. Verify:
# the H3 selectivity read and the glossary say "in the deficit and counted on their own".
RED_ANSWER_LINES_BOTH_WAYS = True
"""A non-red line whose true answer is red (white minus cyan is red, under `difference`) stays in the non-red
deficit and is also counted on its own, and the deficit is reported with and without these lines. Projecting
the axis out at `=` removes *red* from the state that has to produce the answer, so a miss there is removal
on the output side; the separate count says how much of a deficit is that. `mix` has no such lines, so the
gated read is unchanged; the order-sensitive ops have a few dozen each (counts in the method)."""

# --- Gates: the task (H1) -----------------------------------------------------------------------

CALIBRATION_FLOOR = 0.05
"""Before the freeze: the one control seed has learned the grammar when its expected exact match on every kept
and added op is within 0.05 of the ceiling the drawn answers allow on that op. The order-sensitive subset is
recorded and does not block."""

TASK_GATE = ex223.TASK_GATE
TASK_PARTIAL = ex223.TASK_PARTIAL
"""H1: for each op, `handover`'s seed-mean expected exact match on the held-out lines is within 0.02 of the
control's. Partial: every op within 0.05. A comparison that misses by less than a band is reported as
unresolved rather than as a miss. The reference conditions are read on the same table without a gate."""

# --- Gates: placement (H2) --------------------------------------------------------------------------

REF_M_LINE = ex223.REF_M_LINE
MARGIN_RATIO = ex223.MARGIN_RATIO
MARGIN_PARTIAL = ex223.MARGIN_PARTIAL
"""H2 (margin): `handover`'s seed-mean m_line on the `mix` lines is at least 80% of ex-2.1.10's 0.4202, the
same bar ex-2.2.3 set; partial from 60%. Read under the condition's own labeller, as ex-2.2.6 did."""

REF_R2_SIM = ex223.REF_R2_SIM
GRADE_R2_RATIO = 0.9
"""H2 (grading): grading r² at least 90% of ex-2.1.11's reference value. Ex-2.2.3 asked for no more than 0.10
below it, which on 0.782 is 87%; the ratio is a shade tighter and reads like the margin gate."""

MEAN_ALIGN_REF = ex223.MEAN_ALIGN_GATE
"""H2 (containment): ᾱ, the mean alignment over all 216 colors at op1 on the `mix` lines. Every experiment
since ex-2.1.8 gated it at 0.1; here it is a prediction, read beside H3's selectivity. Ex-2.2.7 read 0.13 on
`untied` and 0.23 on `untied-line` against 0.08 on the reference, at nine seeds, with the pull on the right
operand (lead, contrast, and latch all at the reference's values), so we expect `handover` above 0.1. What
the rise costs, if anything, is what H3 measures; a gate we expect to miss on a condition we would still adopt
is not a gate."""

RETENTION_FLOOR = ex223.RETENTION_FLOOR
RETENTION_GATE = ex223.RETENTION_GATE
LEAD_GATE = ex223.LEAD_GATE
CONTRAST_GATE = ex223.CONTRAST_GATE
CONTRAST_PARTIAL = ex223.CONTRAST_PARTIAL
LATCH_PI = ex223.LATCH_PI
"""H2 (retention, concentration, attribution, latch): as ex-2.2.3 H2. Retention: every run whose peak m_line
reaches 0.2 ends, after the anchor weight's anneal, at 0.8 of that peak. Lead: the red group's leading softmin
weight at the embedding is at least 0.4. Contrast: the between-group contrast in op2 weight, mean over the
four post-attention slices, at least 0.2, partial from 0.1. Latch: no run has a non-red-group softmin weight
above 0.5 at op1; ex-2.2.3 held this at twenty seeds."""

# --- Gates: removal and selectivity (H3) --------------------------------------------------------

OPERATORS = ("projection", "operands", "shaped-a0.4-p0")
"""The operators ex-2.2.8 proposed. `projection`: the axis projected out at every slice and position, at full
strength; the gated operator. `operands`: the same at the two operand positions only; the selective
reference. `shaped-a0.4-p0`: a thresholded projection that leaves states below alignment 0.4 alone; the
syntax-free candidate, reported without a gate."""
SHAPED = dict(a=0.4, b=1.0, p=0.0)

RED_KEPT_GATE = ex223.RED_ACC_GATE
"""H3 (removal): under `projection`, on the removal lines of every op, the share of the clean expected exact
match the model keeps is at most 20%, seed mean. This is ex-2.2.3's red-accuracy gate read as a ratio, since
under stochastic rounding the clean value sits below 1 on the ops that round (ex-2.2.5)."""

NONRED_DEFICIT_GATE = ex223.NONRED_DEFICIT_GATE
NONRED_DEFICIT_PARTIAL = ex223.NONRED_DEFICIT_PARTIAL
"""H3 (selectivity): under `projection`, the seed-mean drop in expected exact match on the non-red `mix` lines
is at most 0.05, partial to 0.10. Reported on every op beside it."""

TAIL = 0.07
"""H5: a seed loses selectivity when its non-red `mix` deficit under `projection` is above 0.07, the level
ex-2.2.7 read its tail at (three of nine `untied-line` seeds; one of twenty reference seeds)."""

RESOLUTION_SD = ex223.RESOLUTION_SD
"""A difference between two seed means smaller than 2σ√(1/n_a + 1/n_b), with σ the per-run spread, is reported
as unresolved, as ex-2.2.3 and ex-2.2.8 report it. The seed counts enter through n_a and n_b, so a twenty-seed
condition read against a nine-seed one has a wider band than two twenty-seed conditions."""

NOISE_RUN = ex223.NOISE_RUN
"""Per-run σ of each placement statistic, frozen at the reference. The bands in H2 use these; the handover
re-measures them on the new grammar and the report prints both, as ex-2.2.3's E5 does."""

# REVIEW: named a σ source for every banded read. H4 and H5 both turn on "more than a band" for
# statistics that are not in NOISE_RUN, and without a source the person running it picks one.
# Neither carries a gate now (below), so both take the σ named here and report it. Verify: ex-2.2.8
# tabulates the per-run spread of the non-red deficit under `projection` per op at the reference's
# twenty seeds.
DEFICIT_NOISE = "ex-2.2.8, `projection` on `recipe-short`, twenty seeds, per op"
"""Where σ for the non-red deficit band comes from (H3's reporting and H5): the per-run spread ex-2.2.8
measured at the reference, on `mix`. Frozen, so the candidate's own spread does not move its verdict."""

COMPONENT_NOISE = "pooled within-condition, over the two conditions compared"
"""Where σ for the syntax-embedding component band comes from (H4): ex-2.2.7 reported seed means and ranges
rather than a per-run spread, so the band is built from the two conditions' own seeds and the σ is reported
beside it. H4 carries no gate, so nothing is adopted on this band."""

# --- The decision rule ---------------------------------------------------------------------------

# REVIEW: the first draft chose between two candidates, `handover` and `handover-slot`, with H5 as the
# tie-break, and made `handover-slot` the fallback if the whole-line labeller failed. The author's
# review took the fallback out: the either-slot labeller needs the operand positions, which M3 will
# not have, so it cannot be a grammar we carry forward. `handover-slot` is now a reference that
# attributes a selectivity cost to the labeller, and the rule names one candidate. Verify: nothing
# in the report adopts `handover-slot`.
DECISION = f"""\
The handover is adopted, and `{HANDOVER.name}` becomes the grammar and recipe of record for the anchored-op \
experiments, when it clears H1, H2 (margin, grading, and contrast in full), and H3 in full; every other partial \
band is a reporting level. Otherwise it is not adopted, and the report says which gate was missed and what the reference \
conditions say about which change is responsible, because that sets what the next round tries: the labeller if `{SLOT.name}` clears what `{HANDOVER.name}` missed, the readout if \
`{TIED.name}` does, and the table or the corpus if none of them does."""

SYNTAX_WORDS: tuple[str, ...] = (*OP_NAMES, "=", "\n")
"""The words whose embedding-axis component H4 reads: every word of the grammar that is not a color."""

OPERATOR_SPEC: dict[str, tuple[str, tuple[int, ...] | None]] = {
    "projection": ("projection", None),
    "operands": ("projection", OPERAND_POSITIONS),
    "shaped-a0.4-p0": ("shaped", None),
}
"""Each operator's family and position mask, as ex-2.2.8 spelled its trials."""
assert tuple(OPERATOR_SPEC) == OPERATORS

CUBE_PROBED: dict[str, list[int]] = {
    CONTROL.name: list(range(CONTROL.seeds)),
    HANDOVER.name: list(range(5)),
    NARROW.name: list(range(NARROW.seeds)),
}
"""Which runs get the operand-cube probe scan (the exploratory lines-per-op read): the control, five seeds of
the candidate, and the narrow condition."""

# =============================================================================================
# The DAG
# =============================================================================================


def corpus_key(n_lines: int) -> str:
    """One corpus per size: `aplus-300k` for the main arms, `aplus-183k` for `handover-narrow`."""
    return f"aplus-{n_lines // 1000}k"


def to_zero_move(op: Op, a, b) -> float:
    """How far the true answer moves in the unit cube when the redder operand's R channel is set to zero:
    ex-2.2.4's *to-zero* read, as the report counts it.
    """
    from sca.data.colors import redness
    from sca.data.ops import TOP

    za, zb = ((0, a[1], a[2]), b) if redness(a) >= redness(b) else (a, (0, b[1], b[2]))
    return float(np.linalg.norm(np.subtract(op(za, zb), op(a, b))) / TOP)


def prepare_corpus(
    ops: tuple[str, ...],
    n_lines: int,
    seed: int,
    holdout_frac: float,
    rounding: Rounding,
    n_probe: int,
    probe_seed: int,
    both_slots: tuple[str, ...],
    per_slot_rate: float,
    red_rate: float,
    red_dose: float,
    nonred_dose: float,
    far_move: float,
) -> dict:
    """Build one corpus on table A+, its eval sets, and the probe set of every op.

    Ex-2.2.3's build with the rounding rule as an argument (ex-2.2.5) and the table from this module. The
    probe arrays carry, per line, both labellers' P(labelled), the answer distribution under stochastic
    rounding (`q_idx`, `q_p`: up to eight colors and their masses), the to-zero move, and which walk the line
    belongs to (`walk`: 0 with the palette color as op1, 1 as op2, for the order-sensitive subset). The
    design constants arrive as arguments so a change to one re-runs this step.
    """
    from collections import Counter

    from sca.colorcube import redness as cube_redness
    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data import ops as grammar
    from sca.data.named_colors import WordTokenizer
    from mini.store import put

    key = corpus_key(n_lines)
    by_name = grammar.OP_BY_NAME | grammar.CANDIDATE_BY_NAME
    table = tuple(by_name[o] for o in ops)
    corpus = grammar.sample_corpus(n_lines, seed, table, holdout_frac, rounding=rounding)

    tokenizer_config = TokenizerConfig(vocabulary=grammar.vocabulary(table))
    tokenizer = WordTokenizer(tokenizer_config)
    tokens = grammar.encode_corpus(corpus, tokenizer.stoi)
    n_chars = sum(len(w) for line in corpus for w in line.words)
    meta = CorpusMetadata(
        tokenizer_config=tokenizer_config,
        total_tokens=len(tokens),
        total_chars=n_chars,
        sources=[DatasetMetadata(title=f"table A+ color corpus ({key})", fixes=[], total_chars=n_chars)],
    )
    corpus_dir = get_data_dir() / "corpora" / key
    save_data(tokens, meta, corpus_dir)
    evals = grammar.eval_sets(N_EVAL, seed, table, holdout_frac, rounding=rounding)

    cs = grammar.colors()
    palette_index = {c: i for i, c in enumerate(cs)}
    rgb = np.asarray(cs, dtype=float) / grammar.TOP
    np.testing.assert_allclose(rgb, GRID_RGB, rtol=0, atol=1e-12)
    color_redness = cube_redness(rgb)
    slot_p = np.zeros(tokenizer.vocab_size)
    for name, sp in zip(grammar.PALETTE, color_redness**8 * per_slot_rate, strict=True):
        slot_p[tokenizer.stoi[name]] = sp
    affinity = color_redness**8 * red_rate
    arrays: dict[str, np.ndarray] = {"slot_p": slot_p, "weights": affinity / affinity.sum(), "redness": color_redness}

    eps = 1e-9
    counts: dict[str, dict[str, int]] = {}
    for op in table:
        probe = grammar.probe_lines(op, n_probe, probe_seed, both_slots=op.name in both_slots)
        n_walk = len(cs) * n_probe
        r1 = cube_redness(np.asarray([ln.lhs for ln in probe], dtype=float) / grammar.TOP)
        r2 = cube_redness(np.asarray([ln.rhs for ln in probe], dtype=float) / grammar.TOP)
        r3 = color_redness[[palette_index[ln.result] for ln in probe]]
        p1, p2, p3 = (r**8 * per_slot_rate for r in (r1, r2, r3))
        dists = [grammar.answer_dist(op, ln.lhs, ln.rhs) for ln in probe]
        q_idx = np.full((len(probe), 8), -1, dtype=np.int16)
        q_p = np.zeros((len(probe), 8), dtype=np.float32)
        for i, d in enumerate(dists):
            for k, (c, pc) in enumerate(d.items()):
                q_idx[i, k], q_p[i, k] = palette_index[c], pc
        move = np.array([to_zero_move(op, ln.lhs, ln.rhs) for ln in probe])
        dose_ = np.maximum(r1, r2)
        red, nonred = dose_ >= red_dose - eps, dose_ <= nonred_dose + eps
        counts[op.name] = {
            "lines": len(probe),
            "red": int(red.sum()),
            "removal": int((red & (move >= far_move - eps)).sum()),
            "nonred": int(nonred.sum()),
            "red_answer": int((nonred & (r3 >= red_dose - eps)).sum()),
            "rounded": int(((q_idx >= 0).sum(1) > 1).sum()),
        }
        arrays |= {
            f"{op.name}/tokens": np.array([tokenizer.encode_words(ln.words) for ln in probe], dtype=np.int32),
            f"{op.name}/r1": r1,
            f"{op.name}/r2": r2,
            f"{op.name}/r3": r3,
            f"{op.name}/line_p_either": p1 + p2 - p1 * p2,
            f"{op.name}/line_p_line": 1.0 - (1.0 - p1) * (1.0 - p2) * (1.0 - p3),
            f"{op.name}/q_idx": q_idx,
            f"{op.name}/q_p": q_p,
            f"{op.name}/move": move,
            f"{op.name}/walk": (np.arange(len(probe)) >= n_walk).astype(np.int8),
        }

    held = grammar.holdout(seed, holdout_frac, table)
    rounded = [not grammar.is_on_grid(by_name[ln.op], ln.lhs, ln.rhs) for ln in corpus]
    stats = {
        "key": key,
        "ops": list(ops),
        "n_lines": n_lines,
        "rounding": rounding,
        "total_tokens": int(len(tokens)),
        "vocab_size": tokenizer.vocab_size,
        "lines_per_op": dict(Counter(ln.op for ln in corpus)),
        "distinct_per_op": {o: len({ln.pair for ln in corpus if ln.op == o}) for o in ops},
        "rounded_per_op": dict(Counter(ln.op for ln, r in zip(corpus, rounded, strict=True) if r)),
        "n_pairs": len(grammar.unordered_pairs()),
        "n_holdout_per_op": len(held) // len(ops),
        "eval_n": N_EVAL,
        "probe_counts": counts,
        "line_label_rate": {k: float(arrays[f"mix/line_p_{k}"].mean()) for k in ("either", "line")},
    }
    return {
        "key": key,
        "meta": meta,
        "stats": stats,
        "corpus": put(corpus_dir, name=f"ex-2.2.9-{key}-corpus"),
        "evals": put(grammar.dump_lines(evals), name=f"ex-2.2.9-{key}-evals.json"),
        "probes": put(_npz(**arrays), name=f"ex-2.2.9-{key}-probes.npz"),
    }


def cells(conds: tuple[Cond, ...], preps: dict[str, dict], seeds: dict[str, list[int]] | None = None) -> list[dict]:
    """One row per run: the config (with the readout's tying), both schedules (with the pull's span), the
    labeller's keying, the corpus it trains on, and its labels.
    """
    from sca.utils import align

    rows = []
    for c in conds:
        prep = preps[corpus_key(c.n_lines)]
        tc = prep["meta"].tokenizer_config
        anchor, anti = schedules(c.condition)
        anchor = anchor | {"span": c.span}
        for seed in range(c.seeds) if seeds is None else seeds.get(c.name, []):
            config = _make_config(align(tc.vocab_size, 64), seed, c.epochs, c.n_embd, c.n_layer)
            config.tokenizer = tc.model_copy()
            config.model = config.model.model_copy(update={"tie_embeddings": c.tie})
            rows.append(
                {
                    "config": config,
                    "anchor": anchor,
                    "anti": anti,
                    "keying": c.keying,
                    "condition": c.name,
                    "seed": seed,
                    "label": f"{c.name}-s{seed}",
                    "prep": prep,
                }
            )
    return rows


def train_one(
    config, anchor: dict, anti: dict | None, corpus, traj_stride: int, probes, keying: Keying, label: str
) -> dict:
    """Train one run under its labeller, recording the m_line trajectory on the `mix` probe lines.

    Ex-2.2.7's step: the readout's tying rides in `config.model`, the pull's span in `anchor`, and the
    trajectory's line weights follow the labeller (`line_p_line` or `line_p_either`).
    """
    from sca.anchoring import AnchorSpec, AntiSpec, LabelSpec
    from sca.compute.training import train_anchored
    from mini.store import get, put

    workdir = get_data_dir() / "cells" / label
    corpus_dir = get(corpus, workdir / "corpus")
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens, slot_p, weights = z["mix/tokens"], z["slot_p"], z["weights"]
        line_p = z[f"mix/line_p_{keying}"]
    stride = len(probe_tokens) // len(weights)
    first_of_color = probe_tokens[::stride]
    line_w = line_p[::stride] / line_p[::stride].sum()
    _, metrics, traj = train_anchored(
        config,
        corpus_dir,
        anchor=AnchorSpec(**anchor),
        anti=AntiSpec(**anti) if anti is not None else None,
        label_p=LabelSpec(p=slot_p, keying=keying, pull="span"),
        probe_tokens=first_of_color,
        probe_weights=weights,
        probe_line_w=line_w,
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
    )
    keep = ("epoch", "m_line", "m_op1", "m_span", "alpha_op1", "val_loss", "weight", "anti_weight")
    return {
        "label": label,
        "val_loss": [m.val_loss for m in metrics],
        "train_loss": [m.train_loss for m in metrics],
        "traj": {k: traj[k].tolist() for k in keep if k in traj},
        "checkpoint": put(workdir / "model", name=f"ex-2.2.9-{label}-ckpt"),
    }


def _probe_ops(z) -> list[str]:
    """The ops a probe file carries, in table order."""
    return [o for o in OP_NAMES if f"{o}/tokens" in z.files]


def _stochastic_reads(p: np.ndarray, q: np.ndarray, mode: np.ndarray, drawn: np.ndarray) -> dict[str, np.ndarray]:
    """Per-line reads of a model's color mass *p* (N × 216, off-vocabulary mass left out) against the answer
    distribution *q*: ex-2.2.5's table with expected exact match first.
    """
    rows = np.arange(len(p))
    guess = p.argmax(axis=1)
    p_norm = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    kl = np.where(q > 0, q * (np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(p_norm, 1e-12))), 0).sum(axis=1)
    return {
        "eem": (p * q).sum(axis=1),
        "ceiling": (q**2).sum(axis=1),
        "em_drawn": (guess == drawn).astype(float),
        "em_mode": (guess == mode).astype(float),
        "expected_em_argmax": q[rows, guess],
        "mode_ceiling": q[rows, mode],
        "p_mode": p[rows, mode],
        "support": (p * (q > 0)).sum(axis=1),
        "kl": kl,
        "offvocab": 1.0 - p.sum(axis=1),
    }


def _margin_roles(alpha_lines: np.ndarray, line_w: np.ndarray, roles: int) -> float:
    """m_line over the first *roles* positions: ex-2.2.3's `line_margin` at 4, the whole line at 6."""
    m = np.einsum("n,lnt->lt", line_w, alpha_lines) - alpha_lines.mean(axis=1)  # (L1, T)
    return float(m[:, :roles].max(axis=1).mean())


def _placement(cos, weights, line_p, r1, r2, tokens, tok2color, tau: float, color_pos: int) -> tuple[dict, dict]:
    """The placement statistics of one walk of one op's probe lines: (statistics, arrays).

    *color_pos* is the position the walked palette color sits at (0 for the op1 walk, 2 for the op2 walk), so
    the per-color contraction and the grading r² read that position.
    """
    from sca.anchoring import softmin_weights

    span = ex223.SPAN
    n_slices, n_lines, n_pos = cos.shape
    n_partners = n_lines // len(weights)
    line_w = line_p / line_p.sum()
    g1, g2 = group_weights(r1, r2)
    alpha = cos.reshape(n_slices, len(weights), n_partners, n_pos).mean(axis=2)  # (L1, C, T)
    w_line = softmin_weights(1.0 - cos[:, :, :span], tau)  # (L1, N, SPAN)
    w_color = w_line.reshape(n_slices, len(weights), n_partners, span).mean(axis=2)
    pi = np.einsum("lct,c->lt", w_color, weights)
    w6 = softmin_weights(1.0 - cos, tau).reshape(n_slices, len(weights), n_partners, n_pos).mean(axis=2)
    pi6 = np.einsum("lct,c->lt", w6, weights)
    w_group = np.stack([np.einsum("lnt,n->lt", w_line, g) for g in (g1, g2)])  # (2, L1, SPAN)
    dose_ = np.maximum(r1, r2)
    redder = REDNESS[tok2color[tokens[:, ANSWER_POS]]] > dose_ + 1e-9
    stats = {
        "m_line": line_margin(cos, line_w),
        "m_line6": _margin_roles(cos, line_w, n_pos),
        "m_span": pooled_margin(alpha, weights),
        "alpha_op1": float(alpha[:, :, 0].mean()),
        "alpha_op2": float(alpha[:, :, 2].mean()),
        "r2_sim": r2_sim(alpha[:, :, color_pos].mean(axis=0)),
        "contrast": float(w_group[1, 1:, 2].mean() - w_group[0, 1:, 2].mean()),
        "latch_pi": float(max(pi[1:, 1].mean(), pi[1:, 3].mean())),
        "lead_emb": float(w_group[0, 0, 0]),
        "pi": pi.tolist(),
        "pi6": pi6.tolist(),
        "alpha_pos": cos.mean(axis=1).tolist(),
        "alpha_abs_pos": np.abs(cos).mean(axis=1).tolist(),
        "n_redder": int(redder.sum()),
        "color_pos": color_pos,
    }
    arrays = {
        "alpha": alpha.astype(np.float32),
        "alpha_lines": cos.astype(np.float16),
        "pi": pi.astype(np.float32),
        "pi6": pi6.astype(np.float32),
        "w_group": w_group.astype(np.float32),
        "dose": dose_.astype(np.float32),
        "redder": redder,
    }
    return stats, arrays


def eval_one(trained: dict, evals, probes, tau: float, keying: Keying, condition: str, seed: int, label: str) -> dict:
    """The eval: behavior per op under stochastic rounding, placement per op's probe lines under the run's own
    labeller, and the embedding-component table.

    The gated statistics are the `mix` lines' (`PRIMARY_OP`), lifted to the top level; every op's are under
    `per_op`. For the order-sensitive subset the op2 walk is read on its own under `op2_walk`. Per-line
    alignment goes to the store in half precision.
    """
    from sca.anchoring import alignment
    from sca.data import ops as grammar
    from mini.store import get, put

    workdir = get_data_dir() / "eval" / label
    model, tokenizer, color_ids, tok2color = _load(trained, workdir)
    by_name = grammar.OP_BY_NAME | grammar.CANDIDATE_BY_NAME
    palette_index = {c: i for i, c in enumerate(grammar.colors())}

    # --- Behavior: one teacher-forced pass per (op, split), read at the pre-answer position, against the
    #     answer distribution of every line (ex-2.2.5's reads, with expected exact match first).
    sets: dict[str, dict[str, dict]] = {}
    for op, splits in grammar.load_lines(get(evals, workdir / "evals.json").read_bytes()).items():
        sets[op] = {}
        for split, lns in splits.items():
            lp = _answer_logprobs(model, tokenizer, [ln.prompt for ln in lns])
            p = np.exp(lp[:, color_ids])
            q = np.zeros_like(p)
            for i, ln in enumerate(lns):
                for c, pc in grammar.answer_dist(by_name[op], ln.lhs, ln.rhs).items():
                    q[i, palette_index[c]] = pc
            mode = np.array([palette_index[by_name[op](ln.lhs, ln.rhs)] for ln in lns])
            drawn = np.array([palette_index[ln.result] for ln in lns])
            line = _stochastic_reads(p, q, mode, drawn)
            r = (q > 0).sum(axis=1) > 1
            sets[op][split] = {
                "n": len(lns),
                "n_rounded": int(r.sum()),
                "nll": float(-lp[np.arange(len(lns)), color_ids[drawn]].mean()),
                **{k: float(v.mean()) for k, v in line.items()},
                **{f"{k}_rounded": float(v[r].mean()) if r.any() else float("nan") for k, v in line.items()},
            }

    # --- Placement per op, under the run's own labeller.
    per_op: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    with np.load(get(probes, workdir / "probes.npz")) as z:
        weights = z["weights"]
        probe = {
            o: (z[f"{o}/tokens"], z[f"{o}/r1"], z[f"{o}/r2"], z[f"{o}/line_p_{keying}"], z[f"{o}/walk"])
            for o in _probe_ops(z)
        }
    for op, (tokens, r1, r2, line_p, walk) in probe.items():
        cos = alignment(model, tokens)  # (L1, N, T)
        for w, color_pos in ((0, 0), (1, 2)):
            m = walk == w
            if not m.any():
                continue
            stats, arr = _placement(cos[:, m], weights, line_p[m], r1[m], r2[m], tokens[m], tok2color, tau, color_pos)
            if w == 0:
                per_op[op] = stats
                arrays |= {f"{op}/{k}": v for k, v in arr.items()}
            else:
                per_op[op]["op2_walk"] = stats
                arrays |= {f"{op}/op2_walk/{k}": v for k, v in arr.items()}

    # --- The embedding-component table (H4), as ex-2.2.7 read it: the axis component of every word's
    #     embedding and, when the model has a table of its own, of its readout row.
    def row_table(table) -> dict[str, float]:
        comp = np.asarray(table[:, ANCHOR_AXIS])
        return {w: float(comp[i]) for i in range(len(comp)) if (w := tokenizer.itos.get(i, ""))}

    head = model.transformer.lm_head
    rows = row_table(model.transformer.wte)
    rows_readout = None if head is None else row_table(head)

    traj = np.asarray(trained["traj"]["m_line"], dtype=float)
    peak = float(np.maximum.accumulate(traj).max())
    primary = per_op[PRIMARY_OP]
    return {
        "label": label,
        "condition": condition,
        "seed": seed,
        "tau": tau,
        "keying": keying,
        "tied": head is None,
        "sets": sets,
        "holdout_eem": {op: s["holdout"]["eem"] for op, s in sets.items()},
        "holdout_ceiling": {op: s["holdout"]["ceiling"] for op, s in sets.items()},
        "holdout_em": {op: s["holdout"]["em_drawn"] for op, s in sets.items()},
        **{k: primary[k] for k in ("m_line", "m_span", "alpha_op1", "r2_sim", "contrast", "latch_pi", "lead_emb")},
        "m_line_peak": peak,
        "retention": float(traj[-1] / peak) if peak > 0 else float("nan"),
        "per_op": per_op,
        "rows": rows,
        "rows_readout": rows_readout,
        "syntax_component": {
            "wte": float(np.mean([abs(rows[w]) for w in SYNTAX_WORDS])),
            "readout": None if rows_readout is None else float(np.mean([abs(rows_readout[w]) for w in SYNTAX_WORDS])),
        },
        "arrays": put(_npz(**arrays), name=f"ex-2.2.9-{label}-arrays.npz"),
    }


# --- The eval contract ------------------------------------------------------------------------


class Readout:
    """Per-line readings of one answer distribution on one op's probe lines, under stochastic rounding.

    Ex-2.2.3's `_Readout` with expected exact match (the model's color mass against the line's answer
    distribution) beside P(mode answer), and the groups this experiment reads: `all`, `red`, `nonred`, the
    `removal` lines (red, and the to-zero move at least `FAR_MOVE`), the `red_answer` lines (non-red, with a
    red answer), `nonred_excl` (non-red without them), and the red and removal lines split by the slot the
    red operand sits in.
    """

    def __init__(self, tokens, r1, r2, q_idx, q_p, move, color_ids, tok2color):
        self.tokens, self.color_ids, self.tok2color = tokens, color_ids, tok2color
        self.rows = np.arange(len(tokens))
        lc = tok2color[tokens]  # (N, T) palette index, −1 at the syntax positions
        assert (lc[:, [0, 2, ANSWER_POS]] >= 0).all() and (lc[:, [1, 3, 5]] < 0).all()
        self.answer = tokens[:, ANSWER_POS]
        self.ans_idx = lc[:, ANSWER_POS]
        self.q_idx, self.q_p = np.maximum(q_idx, 0), q_p
        self.move = move
        self.dose = np.maximum(r1, r2)
        eps = 1e-9
        red, nonred = self.dose >= RED_DOSE - eps, self.dose <= NONRED_DOSE + eps
        red_answer = nonred & (REDNESS[self.ans_idx] >= RED_DOSE - eps)
        removal = red & (move >= FAR_MOVE - eps)
        op1 = r1 >= r2
        self.groups = {
            "all": np.ones(len(tokens), bool),
            "red": red,
            "nonred": nonred,
            "removal": removal,
            "red_answer": red_answer,
            "nonred_excl": nonred & ~red_answer,
            "red_op1": red & op1,
            "red_op2": red & ~op1,
            "removal_op1": removal & op1,
            "removal_op2": removal & ~op1,
        }
        self.dists = np.linalg.norm(GRID_RGB[None] - GRID_RGB[self.ans_idx][:, None], axis=2)  # (N, 216)
        self.red_operand = np.where(op1, 0, 2)
        self.cube = np.stack(np.unravel_index(np.arange(len(PALETTE)), (6, 6, 6)), axis=1)

    def by_group(self, v: np.ndarray) -> dict[str, float]:
        return {g: float(np.nanmean(v[m])) if m.any() else float("nan") for g, m in self.groups.items()}

    def ratio_by_group(self, num: np.ndarray, den: np.ndarray) -> dict[str, float]:
        """The ratio of group means: what share of *den* the group keeps in *num*."""
        return {
            g: float(num[m].mean() / den[m].mean()) if m.any() and den[m].mean() > 0 else float("nan")
            for g, m in self.groups.items()
        }

    def __call__(self, logits) -> dict[str, np.ndarray]:
        from sca.intervention import answer_logprobs

        lp = answer_logprobs(logits, ANSWER_POS)
        p_color = np.exp(lp[:, self.color_ids])
        guess = lp.argmax(1)
        g = self.tok2color[guess]
        step = np.abs(self.cube[np.maximum(g, 0)] - self.cube[self.ans_idx]).max(1)
        tests = [
            guess == self.answer,
            guess == self.tokens[self.rows, self.red_operand],
            guess == self.tokens[self.rows, 2 - self.red_operand],
            (g >= 0) & (step == 1),
        ]
        return {
            "eem": (np.take_along_axis(p_color, self.q_idx, axis=1) * self.q_p).sum(1),
            "p_ans": np.exp(lp[self.rows, self.answer]),
            "guess": guess,
            "argmax_dist": np.where(g >= 0, self.dists[self.rows, np.maximum(g, 0)], np.nan),
            "expected_dist": (p_color / p_color.sum(1, keepdims=True) * self.dists).sum(1),
            "offvocab": 1.0 - p_color.sum(1),
            "composition": np.select(tests, [0, 1, 2, 3], default=4),
        }


def _operator(name: str, sub):
    """One of `OPERATORS`: (the operator, its position mask or None)."""
    from sca.intervention import projection, shaped_suppression

    family, positions = OPERATOR_SPEC[name]
    match family:
        case "projection":
            return projection(sub), positions
        case "shaped":
            return shaped_suppression(sub, a=SHAPED["a"], b=SHAPED["b"], p=SHAPED["p"]), positions
        case _:
            raise ValueError(family)


def _score_op(model, sub, read: Readout, operators: tuple[str, ...]) -> tuple[dict, dict[str, np.ndarray]]:
    """One op's probe lines through the clean pass and every operator: (statistics, per-line arrays)."""
    from sca.intervention import angle_between, apply, projection, write_angle

    tokens = read.tokens
    n_pos = tokens.shape[1]
    nonred = read.groups["nonred"]

    # --- The clean pass: the scorer's own control, and the reference every operator is read against.
    clean = apply(model, tokens, projection(sub), slices=())
    alpha_clean = clean.pre[..., ANCHOR_AXIS]  # (L1, N, T)
    c = read(clean.logits)
    stats: dict[str, Any] = {
        "n": {g: int(m.sum()) for g, m in read.groups.items()},
        "move": read.by_group(read.move),
        "clean": {
            "eem": read.by_group(c["eem"]),
            "acc": read.by_group(c["guess"] == read.answer),
            "p_ans": read.by_group(c["p_ans"]),
            "argmax_dist": read.by_group(c["argmax_dist"]),
            "expected_dist": read.by_group(c["expected_dist"]),
            "offvocab": read.by_group(c["offvocab"]),
            "alpha_q99_nonred": top_quantile(np.abs(alpha_clean[:, nonred]), axis=1).tolist(),
        },
        "operators": {},
    }
    arrays: dict[str, np.ndarray] = {
        "clean/eem": c["eem"].astype(np.float32),
        "clean/p_ans": c["p_ans"].astype(np.float32),
        "clean/guess": c["guess"].astype(np.int16),
        "clean/expected_dist": c["expected_dist"].astype(np.float32),
    }

    # --- Each operator: apply, check the contract, read, contract per group.
    for name in operators:
        operator, mask = _operator(name, sub)
        positions = None if mask is None else np.isin(np.arange(n_pos), mask).astype(np.float32)
        out = apply(model, tokens, operator, slices=SLICES, positions=positions)
        theta = angle_between(out.pre, out.post)  # (L1, N, T): the write at every site
        alpha_pre = out.pre[..., ANCHOR_AXIS]
        np.testing.assert_allclose(alpha_pre[0], alpha_clean[0], rtol=0, atol=0)
        on = np.ones(n_pos, bool) if positions is None else positions > 0
        np.testing.assert_allclose(theta[:, :, ~on], 0.0, rtol=0, atol=1e-6)
        if OPERATOR_SPEC[name][0] == "projection":
            np.testing.assert_allclose(theta[:, :, on], write_angle(alpha_pre[:, :, on]), rtol=0, atol=2e-3)
        r = read(out.logits)
        disp = angle_between(out.post[-1, :, DECODE_POS], clean.post[-1, :, DECODE_POS])
        stats["operators"][name] = {
            "eem": read.by_group(r["eem"]),
            "kept": read.ratio_by_group(r["eem"], c["eem"]),
            "deficit": read.by_group(c["eem"] - r["eem"]),
            "acc": read.by_group(r["guess"] == read.answer),
            "p_ans": read.by_group(r["p_ans"]),
            "deficit_p": read.by_group(c["p_ans"] - r["p_ans"]),
            "argmax_dist": read.by_group(r["argmax_dist"]),
            "expected_dist": read.by_group(r["expected_dist"]),
            "offvocab": read.by_group(r["offvocab"]),
            "disp": read.by_group(disp),
            "q99_write_nonred": top_quantile(theta[:, nonred], axis=1).tolist(),
            "mean_write_nonred": theta[:, nonred].mean(1).tolist(),
            "alpha_red_operand": [
                float(alpha_pre[s, read.rows[m], read.red_operand[m]].mean())
                if (m := read.groups["red"]).any()
                else float("nan")
                for s in range(alpha_pre.shape[0])
            ],
            "composition_red": np.bincount(r["composition"][read.groups["red"]], minlength=len(COMPOSITION)).tolist(),
            "composition_removal": np.bincount(
                r["composition"][read.groups["removal"]], minlength=len(COMPOSITION)
            ).tolist(),
        }
        arrays |= {
            f"{name}/eem": r["eem"].astype(np.float32),
            f"{name}/p_ans": r["p_ans"].astype(np.float32),
            f"{name}/guess": r["guess"].astype(np.int16),
            f"{name}/expected_dist": r["expected_dist"].astype(np.float32),
            f"{name}/composition": r["composition"].astype(np.int8),
        }
    return stats, arrays


def score_one(trained: dict, probes, operators: tuple[str, ...], condition: str, seed: int, label: str) -> dict:
    """Score one checkpoint on every op's probe lines under every operator, one op at a time.

    Per-run statistics return as the result; per-line arrays go to the store. The ops are scored one at a
    time rather than concatenated (ex-2.2.8's shortcut) since eleven probe sets, three of them doubled, are
    too many lines to keep the whole stream of at once.
    """
    from sca.intervention import Subspace
    from mini.store import get, put

    workdir = get_data_dir() / "score" / label
    model, _, color_ids, tok2color = _load(trained, workdir)
    sub = Subspace.axis(model.transformer.wte.shape[1], ANCHOR_AXIS)
    keys = ("tokens", "r1", "r2", "q_idx", "q_p", "move")
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe = {o: {k: z[f"{o}/{k}"] for k in keys} for o in _probe_ops(z)}

    ops: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    for op, f in probe.items():
        read = Readout(f["tokens"], f["r1"], f["r2"], f["q_idx"], f["q_p"], f["move"], color_ids, tok2color)
        stats, per_line = _score_op(model, sub, read, operators)
        ops[op] = stats
        arrays |= {f"{op}/{k}": v for k, v in per_line.items()}
        arrays[f"{op}/dose"] = read.dose.astype(np.float32)
    return {
        "label": label,
        "condition": condition,
        "seed": seed,
        "operators": list(operators),
        "ops": ops,
        "arrays": put(_npz(**arrays), name=f"ex-2.2.9-{label}-score.npz"),
    }


# --- Publishing ------------------------------------------------------------------------------


def design() -> dict[str, Any]:
    """The design constants the report checks its rendered values against."""
    return {
        "n_runs": N_RUNS,
        "conditions": [asdict(c) | {"steps": c.steps, "lines_per_op": c.lines_per_op} for c in CONDS],
        "candidate": HANDOVER.name,
        "ops": list(OP_NAMES),
        "kept": list(KEPT),
        "added": list(ADDED),
        "order_sensitive": list(ORDER_SENSITIVE),
        "primary_op": PRIMARY_OP,
        "secondary_op": SECONDARY_OP,
        "rounding": ROUNDING,
        "n_probe": N_PROBE,
        "probe_seed": PROBE_SEED,
        "probe_both_slots": list(PROBE_BOTH_SLOTS),
        "spans": {"prompt": PROMPT_SPAN, "whole": WHOLE_SPAN},
        "epochs": {"main": EPOCHS, "narrow": EPOCHS_NARROW},
        "model": {"n_embd": N_EMBD, "n_layer": N_LAYER},
        "n_lines": {"main": N_LINES, "narrow": N_LINES_NARROW},
        "steps_per_epoch": {"main": steps_per_epoch(N_LINES), "narrow": steps_per_epoch(N_LINES_NARROW)},
        "noise_run": NOISE_RUN,
        "gates": {
            "calibration_floor": CALIBRATION_FLOOR,
            "task": TASK_GATE,
            "task_partial": TASK_PARTIAL,
            "ref_m_line": REF_M_LINE,
            "margin_ratio": MARGIN_RATIO,
            "margin_partial": MARGIN_PARTIAL,
            "ref_r2_sim": REF_R2_SIM,
            "grade_r2_ratio": GRADE_R2_RATIO,
            "mean_align_ref": MEAN_ALIGN_REF,
            "retention_floor": RETENTION_FLOOR,
            "retention": RETENTION_GATE,
            "lead": LEAD_GATE,
            "contrast": CONTRAST_GATE,
            "contrast_partial": CONTRAST_PARTIAL,
            "latch": LATCH_PI,
            "red_kept": RED_KEPT_GATE,
            "nonred_deficit": NONRED_DEFICIT_GATE,
            "nonred_deficit_partial": NONRED_DEFICIT_PARTIAL,
            "tail": TAIL,
            "resolution_sd": RESOLUTION_SD,
        },
        "operators": {k: {"family": f, "positions": p} for k, (f, p) in OPERATOR_SPEC.items()},
        "shaped": SHAPED,
        "dose": {"red": RED_DOSE, "nonred": NONRED_DOSE, "far_move": FAR_MOVE},
        "composition": list(COMPOSITION),
        "syntax_words": list(SYNTAX_WORDS),
        "cube_probed": CUBE_PROBED,
    }


def _slim(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("arrays", "traj", "val_loss", "train_loss")}


def _published_calibration() -> dict:
    """The calibration payload published so far, or an empty one."""
    import json
    import tempfile
    from pathlib import Path

    from mini.store import project_store

    store = project_store()
    art = store.get_refs([CALIBRATION_REF]).get(CALIBRATION_REF)
    if art is None:
        return {}
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "calibration.json")])
        return json.loads(path.read_text())


def publish_calibration(evaled: list[dict], corpora: list[dict], shapes: dict[str, dict] | None = None) -> dict:
    """The pre-freeze calibration: one control seed per epoch count and model shape looked at, as plain metrics
    under `CALIBRATION_REF`, with the read against `CALIBRATION_FLOOR` on every kept and added op.
    """
    import json

    from mini.store import put, set_ref

    shapes = shapes or {CONTROL.name: {"epochs": EPOCHS, "n_lines": N_LINES, "n_embd": N_EMBD, "n_layer": N_LAYER}}
    runs = {r["label"]: _slim(r) | shapes[r["condition"]] for r in evaled}
    # Merge with the looks published before this one, so the report shows every point looked at; a run
    # with the same label (a repeat of the same point) replaces the earlier record.
    before = _published_calibration()
    runs = {r["label"]: r for r in before.get("runs", [])} | runs
    by_key = {c["key"]: c for c in [*before.get("corpora", []), *corpora]}
    payload = {"runs": list(runs.values()), "corpora": list(by_key.values()), "design": design()}
    set_ref(CALIBRATION_REF, put(json.dumps(payload, indent=2).encode(), name="ex-2.2.9-calibration.json"))
    gaps = {r["label"]: {op: r["holdout_ceiling"][op] - r["holdout_eem"][op] for op in OP_NAMES} for r in evaled}
    return {
        "stage": "calibration",
        "holdout_eem": {r["label"]: r["holdout_eem"] for r in evaled},
        "gap_to_ceiling": gaps,
        "clears": {lb: all(g[op] <= CALIBRATION_FLOOR for op in (*KEPT, *ADDED)) for lb, g in gaps.items()},
    }


STACKED = {
    "eval": ("alpha", "pi", "pi6", "w_group", "dose", "redder"),
    "score": ("dose", "clean/eem", "clean/guess", "*/eem", "*/guess", "*/composition"),
}
"""Which per-run arrays the stacked file carries (per op; `*` is any operator). The per-line alignment
stays in each run's own file under `RUN_ARRAYS_REF`."""


def _stacked(name: str, kind: str) -> bool:
    import fnmatch

    tail = name.split("/", 1)[1] if "/" in name else name
    if "op2_walk/" in tail:
        tail = tail.replace("op2_walk/", "")
    return any(fnmatch.fnmatch(tail, pat) for pat in STACKED[kind])


def publish_results(
    trained: list[dict], evaled: list[dict], scored: list[dict], probed: list[dict], corpora: list[dict], probes
) -> dict:
    """Publish the frozen stage: metrics (JSON), trajectories (JSON), the stacked per-run arrays (npz), each
    run's full arrays, the probe set, the cube probes (JSON), and every checkpoint under its own ref.
    """
    import json

    from mini.store import get, put, set_ref

    metrics = {
        "runs": [_slim(r) for r in evaled],
        "scores": [_slim(r) for r in scored],
        "corpora": corpora,
        "design": design(),
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.2.9-metrics.json"))
    traj = {t["label"]: {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.9-trajectories.json"))
    set_ref(GEOMETRY_REF, put(json.dumps({"runs": probed}).encode(), name="ex-2.2.9-geometry.json"))
    set_ref(PROBE_REF, probes)
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])

    arrays = {}
    for r in evaled + scored:
        kind = "eval" if "per_op" in r else "score"
        set_ref(RUN_ARRAYS_REF.format(label=r["label"], kind=kind), r["arrays"])
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}-{kind}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{kind}/{name}": z[name] for name in z.files if _stacked(name, kind)}
    set_ref(ARRAYS_REF, put(_npz(**arrays), name="ex-2.2.9-arrays.npz"))
    return {
        "stage": "full",
        "n_runs": len(evaled),
        "n_scored": len(scored),
        "n_probed": len(probed),
        "holdout_eem": {r["label"]: r["holdout_eem"] for r in evaled},
    }


# --- Orchestration ----------------------------------------------------------------------------


def _stages() -> set[str]:
    """Which stage this wake may launch: `EX229_STAGES`, `full` by default.

    `calibration` trains and evaluates one control seed and stops, so the corpus can be checked for
    learnability before the design freezes. The full stage re-uses that run.
    """
    import os

    return {s.strip() for s in os.environ.get("EX229_STAGES", "full").split(",")}


def _calibration_conds() -> tuple[Cond, ...]:
    """The control at each point of the look: every corpus size in `EX229_CALIBRATION_LINES` × every epoch
    count in `EX229_CALIBRATION_EPOCHS` × every width×depth in `EX229_CALIBRATION_MODELS` (each defaulting to
    the design's). The prereg sends the corpus size and epoch count back for a look when a commutative op
    misses the calibration bar; each extra point is a fresh control seed under its own name, and the design's
    own point keeps the `control` name so the full stage re-uses that run.
    """
    import os
    from dataclasses import replace

    def shape(m: str) -> tuple[int, int]:
        d, n_layer = (int(x) for x in m.split("x"))
        return d, n_layer

    if points := os.environ.get("EX229_CALIBRATION_POINTS"):
        grid = []
        for spec in points.split(","):
            n_lines, e, m = spec.split("/")
            grid.append((int(n_lines), int(e), *shape(m)))
    else:
        counts = [int(e) for e in os.environ.get("EX229_CALIBRATION_EPOCHS", str(EPOCHS)).split(",")]
        sizes = [int(n) for n in os.environ.get("EX229_CALIBRATION_LINES", str(N_LINES)).split(",")]
        models = [shape(m) for m in os.environ.get("EX229_CALIBRATION_MODELS", f"{N_EMBD}x{N_LAYER}").split(",")]
        grid = [(n_lines, e, d, n_layer) for d, n_layer in models for n_lines in sizes for e in counts]
    out = []
    for n_lines, e, d, n_layer in grid:
        if (d, n_layer, n_lines, e) == (N_EMBD, N_LAYER, N_LINES, EPOCHS):
            out.append(CONTROL)
            continue
        model = "" if (d, n_layer) == (N_EMBD, N_LAYER) else f"-d{d}L{n_layer}"
        size = "" if n_lines == N_LINES else f"-{n_lines // 1000}k"
        out.append(
            replace(CONTROL, name=f"control{model}{size}-e{e}", epochs=e, n_lines=n_lines, n_embd=d, n_layer=n_layer)
        )
    return tuple(out)


def _train_and_eval(ctx: Ctx, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    n = len(rows)
    trained = ctx.map(
        train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [r["prep"]["corpus"] for r in rows],
        [TRAJ_STRIDE] * n,
        [r["prep"]["probes"] for r in rows],
        [r["keying"] for r in rows],
        [r["label"] for r in rows],
        role="train",
    )
    evaled = ctx.map(
        eval_one,
        trained,
        [r["prep"]["evals"] for r in rows],
        [r["prep"]["probes"] for r in rows],
        [r["anchor"]["tau"] for r in rows],
        [r["keying"] for r in rows],
        [r["condition"] for r in rows],
        [r["seed"] for r in rows],
        [r["label"] for r in rows],
        role="eval",
    )
    return trained, evaled


def main(ctx: Ctx) -> dict:
    calibrating = _stages() == {"calibration"}
    sizes = {corpus_key(c.n_lines): c.n_lines for c in (_calibration_conds() if calibrating else CONDS)}
    keys = list(sizes)
    n = len(keys)
    prepped = ctx.map(
        prepare_corpus,
        [OP_NAMES] * n,
        [sizes[k] for k in keys],
        [CORPUS_SEED] * n,
        [HOLDOUT_FRAC] * n,
        [ROUNDING] * n,
        [N_PROBE] * n,
        [PROBE_SEED] * n,
        [PROBE_BOTH_SLOTS] * n,
        [PER_SLOT_RATE] * n,
        [RED_RATE] * n,
        [RED_DOSE] * n,
        [NONRED_DOSE] * n,
        [FAR_MOVE] * n,
        role="prep",
    )
    preps = dict(zip(keys, prepped, strict=True))
    corpora = [preps[k]["stats"] for k in keys]

    if calibrating:
        conds = _calibration_conds()
        rows = cells(conds, preps, seeds={c.name: [0] for c in conds})
        _, evaled = _train_and_eval(ctx, rows)
        shapes = {
            c.name: {"epochs": c.epochs, "n_lines": c.n_lines, "n_embd": c.n_embd, "n_layer": c.n_layer} for c in conds
        }
        return ctx.run(publish_calibration, evaled, corpora, shapes, role="prep")

    rows = cells(CONDS, preps)
    trained, evaled = _train_and_eval(ctx, rows)
    scored = ctx.map(
        score_one,
        trained,
        [r["prep"]["probes"] for r in rows],
        [OPERATORS] * len(rows),
        [r["condition"] for r in rows],
        [r["seed"] for r in rows],
        [r["label"] for r in rows],
        role="score",
    )
    cube = [(r, t) for r, t in zip(rows, trained, strict=True) if r["seed"] in CUBE_PROBED.get(r["condition"], [])]
    probed = ctx.map(
        probe_one,
        [t for _, t in cube],
        [r["prep"]["probes"] for r, _ in cube],
        [r["condition"] for r, _ in cube],
        [r["seed"] for r, _ in cube],
        [r["label"] for r, _ in cube],
        role="probe",
    )
    main_key = corpus_key(HANDOVER.n_lines)
    return ctx.run(publish_results, trained, evaled, scored, probed, corpora, preps[main_key]["probes"], role="prep")


experiment = Experiment(
    name="ex-2.2.9",
    main=main,
    roles={
        # Corpus sampling over 100k or 183k lines, then eleven probe sets (three doubled) with an answer
        # distribution and a to-zero move per line: plain Python, a few minutes.
        "prep": dict(cpu=2, timeout=1800),
        # 1,650 steps of 64×64 tokens, as ex-2.2.3's short arms. The watchdog is sized for the gap after
        # the last step: the checkpoint upload emits no step progress.
        "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
        # 22 teacher-forced eval sets and eleven alignment passes over up to 11,664 lines each.
        "eval": dict(gpu="L4", timeout=1800),
        # Four passes (clean and three operators) per op, each keeping the whole stream of up to 11,664 lines.
        "score": dict(gpu="L4", timeout=2400),
        # One stream capture and 270 grouped ridge fits on 5,832 × 64 activations: CPU work.
        "probe": dict(cpu=4, timeout=1800),
    },
)
