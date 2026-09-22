"""
The handover re-run: ex-2.2.9's conditions at fresh seeds, scored with the three measurements ex-2.2.10 proposed.

Ex-2.2.9 put every proposal of the scouting round together (table A+, drawn answers, the whole-line
labeller, the untied readout) and did not adopt the result: the removal gate missed on the three ops that
take one HSV attribute from their second operand, and one seed in twenty ended under the retention gate.
Ex-2.2.10 then went back over the stored runs and found both misses in the measurements rather than in the model. The
removal miss is the line-picking rule's: the projection acts on the red operand like a change of hue, and
the to-zero rule counts lines whose answer takes only red's saturation or value. The retention drop happens
under the constant anchor weight, before the anneal the measurement was meant to check. And the op1 alignment
rises under either half of the handover, so its old reference belongs to the old grammar.

Reads chosen after looking at the data cannot score the same data, so this is the same experiment again,
at seeds ex-2.2.9 never trained, with the measurements fixed in advance:

1. Removal lines are chosen by hue: a red line is a removal line when some channel permutation of its red
   operand moves the true answer by at least `FAR_MOVE`. The slots the rule drops (the ones whose answer
   takes only red's saturation or value) become a second measurement, with no gate.
2. Retention is the final alignment over the alignment at the start of the anneal, gated at 0.8, and the
   level at the end of training is reported beside the references.
3. ᾱ at op1 is a report line with `handover-slot` and `handover-tied` as its references, and gates nothing.

The DAG is ex-2.2.9's with the seed offset, the hue move in the probe arrays, the reference conditions
scored under the operators too, and checkpoints kept on the trajectory stride for a few seeds. Where a
task is unchanged it calls ex-2.2.9's function; where it adds a column it wraps it.

    bin/mini run docs/m2/ex-2.2.11/experiment.py --app modal --max-containers 8 --budget 3h
    bin/mini status ex-2.2.11
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from dataclasses import asdict

from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir


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
same corpus at the same seed, so the held-out pairs are the same and the task measurement is on the same lines."""

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
"""Every gate ex-2.2.9 set, at the same value. The re-run changes which lines the removal gate is scored on,
what the retention ratio divides by, and whether ᾱ is gated; it changes no threshold."""

# REVIEW: bound RESOLUTION_SD, NOISE_RUN, and COMPONENT_NOISE above as well. Every "within a band" or
# "more than a band" comparison in H1, H2, and H4 needs a frozen σ source, and the draft named one only for
# the deficit. Verify: ex-2.2.9's module carries the same three names, with the same sources.

# --- Conditions -------------------------------------------------------------------------------------

SEED_OFFSET = 100
"""Every run trains at a seed ex-2.2.9 never used: condition seed *i* here is model seed `SEED_OFFSET + i`.
Ex-2.2.10 chose the measurements on ex-2.2.9's seeds 0..19, so those checkpoints cannot score them. The corpus
seed is unchanged, so the task is the same; only the initialization and the batch order are fresh."""

CONTROL = Cond("control", 5, "un-anchored", "reference", lam=0.0)
"""As ex-2.2.9: the task reference for H1."""

HANDOVER = Cond("handover", 20, "the recipe, untied readout, whole-line labeller", "candidate")
"""The one candidate, at the reference's seed count. The gates are scored on this condition alone."""

SLOT = ex229.SLOT
"""Ex-2.2.3's either-slot labeller, twenty seeds. Reference for the ᾱ line (it undid half the rise in
ex-2.2.9), for the whole-line label's selectivity cost (H4), and for the saturation-and-value measurement, where it
says whether the shortfall is the operator's or this checkpoint's. Not a fallback: the labeller needs the
operand positions, which M3 will not have."""

TIED = ex229.TIED
"""The tied readout, nine seeds. Reference for the ᾱ line (it undid the other half), for the syntax-token
comparison (H4), and for the saturation-and-value measurement. Nine seeds are enough for a line with no gate on it."""

CONDS: tuple[Cond, ...] = (CONTROL, HANDOVER, SLOT, TIED)
ANCHORED: tuple[Cond, ...] = tuple(c for c in CONDS if c.lam > 0)
N_RUNS = sum(c.seeds for c in CONDS)
assert N_RUNS == 54
"""`handover-narrow` does not return: its lines-per-op question was exploratory and ex-2.2.9 answered it."""

SCORED_UNDER_PROJECTION: tuple[Cond, ...] = (HANDOVER, SLOT, TIED)
"""Every anchored condition is scored under the operators, so the saturation-and-value measurement and the
selectivity comparisons have the references beside the candidate. Ex-2.2.9 scored them too; ex-2.2.10's
answer-mass measurement did not, which is the gap it asked the re-run to close."""

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
disagree. If they disagree on more than `HUE_ROTATION_TOLERANCE` of some op's red lines, the finer rule is the
one to keep, and the report says so before the freeze. This constant is a reminder that the check is part of
the method, and its result is not a gate."""

HUE_ROTATION_TOLERANCE = 0.01
"""The share of an op's red lines the two rules may disagree on before the finer rule replaces the permutation
rule. A change to one line in a hundred moves a kept share by at most a hundredth, under the resolution of
every gate here, so a disagreement that small is not worth a second rule."""

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
    These are the saturation-and-value lines, reported without a gate.
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
before that, as ex-2.2.10 measured it. The old ratio divided by the run's peak, which on `handover` is the high
point of a noisy plateau reached thirty epochs before the anneal."""

LEVEL_REFS: tuple[Cond, ...] = (SLOT, TIED)
"""The level line: `handover`'s final alignment, seed mean and range, beside the same on these references and
on ex-2.2.3's adopted point. No gate. Ex-2.2.10 measured 0.66 against 0.72 and 0.70; the drift that produced the
gap happens under the constant anchor weight and is the [training-dynamics
item](/todo/science/training-dynamics-under-the-retention-drift.md)'s question."""

# --- Containment, as a line -----------------------------------------------------------------------------

ALPHA_REFS: tuple[Cond, ...] = (SLOT, TIED)
"""ᾱ at op1 on `handover`, beside the same on these two references and on ex-2.2.3's adopted point. No gate:
ex-2.2.9 measured 0.28 against 0.18 and 0.16, each reference undoing about half the rise, and no mechanism is
named for either half. A gate returns once one is."""

EOL_ROW = "\n"
"""The `⏎` embedding row's axis component, on every anchored condition, as a line. Ex-2.2.9 measured 0.17 on
`handover` and about zero on `handover-slot`, so the whole-line labeller is what puts it there."""

# --- Checkpoints on the trajectory stride --------------------------------------------------------------

TRAJ_CHECKPOINT_SEEDS = 3
"""The first three seeds of each anchored condition keep a checkpoint at every trajectory point, so a
training-dynamics measurement (the local learning coefficient, the whole-geometry statistics) can be run on the
plateau after the fact. About `TRAJ_STRIDE` checkpoints per run, a few MB each. That measurement is not part of
this experiment."""

# --- Refs -------------------------------------------------------------------------------------------

METRICS_REF = "reports/m2/ex-2.2.11/metrics"
CALIBRATION_REF = ex229.CALIBRATION_REF
"""Ex-2.2.9's calibration stands: same corpus, same point, same control. It is not repeated."""
ARRAYS_REF = "reports/m2/ex-2.2.11/arrays"
TRAJ_REF = "reports/m2/ex-2.2.11/trajectories"
PROBE_REF = "reports/m2/ex-2.2.11/probes"
CHECKPOINT_REF = "reports/m2/ex-2.2.11/checkpoints/{label}"
TRAJ_CHECKPOINT_REF = "reports/m2/ex-2.2.11/checkpoints/{label}/trajectory"
"""One tree per run that keeps them: `<point>/model/checkpoint.eqx` for every trajectory point, in record
order (the last is the end of training). One ref per run rather than one per point, since a ref is a round
trip on the bucket and there are a hundred points."""
RUN_ARRAYS_REF = "reports/m2/ex-2.2.11/arrays/{label}/{kind}"

EX229_METRICS_REF = ex229.METRICS_REF
EX229_TRAJ_REF = ex229.TRAJ_REF
"""Ex-2.2.9's numbers, printed beside every measurement as the before column."""

# --- The decision rule --------------------------------------------------------------------------------

DECISION = f"""\
The handover is adopted, and `{HANDOVER.name}` becomes the grammar and recipe of record for the anchored-op \
experiments, when it clears H1, H2 (margin, grading, and contrast, all in full), and H3 in full on the removal lines chosen by hue; every partial band is a reporting level. \
Otherwise it is not adopted, and the report says which gate was missed and what the references say about \
which change is responsible."""

# =============================================================================================
# The DAG
# =============================================================================================

# Bound by name so a task body never references ex-2.2.9's module object itself.
_npz = ex229._npz
_load = ex229._load
_probe_ops = ex229._probe_ops
_make_config = ex229._make_config
_score_op = ex229._score_op
_slim = ex229._slim
_stacked = ex229._stacked
Readout = ex229.Readout
Keying = ex229.Keying
schedules = ex229.schedules
corpus_key = ex229.corpus_key
ex229_prepare_corpus = ex229.prepare_corpus
ex229_eval_one = ex229.eval_one
ex229_design = ex229.design
ROUNDING = ex229.ROUNDING
PER_SLOT_RATE = ex229.PER_SLOT_RATE
RED_RATE = ex229.RED_RATE
ANCHOR_AXIS = ex229.ANCHOR_AXIS
SLICES = ex229.SLICES


def prepare_corpus(
    ops: tuple[str, ...],
    n_lines: int,
    seed: int,
    holdout_frac: float,
    rounding,
    n_probe: int,
    probe_seed: int,
    both_slots: tuple[str, ...],
    per_slot_rate: float,
    red_rate: float,
    red_dose: float,
    nonred_dose: float,
    far_move: float,
) -> dict:
    """Ex-2.2.9's corpus, eval sets, and probe set, with one column added to every op's probe arrays: the
    hue move (`{op}/hue_move`, the furthest any channel permutation of the red operand moves the true answer).
    The to-zero move stays under `{op}/move`, so both rules can be read. The probe counts gain the removal
    lines under the hue rule and the saturation-and-value lines it sets aside.
    """
    from sca.data import ops as grammar
    from sca.data.colors import redness
    from mini.store import get, put

    prep = ex229_prepare_corpus(
        ops,
        n_lines,
        seed,
        holdout_frac,
        rounding,
        n_probe,
        probe_seed,
        both_slots,
        per_slot_rate,
        red_rate,
        red_dose,
        nonred_dose,
        far_move,
    )
    by_name = grammar.OP_BY_NAME | grammar.CANDIDATE_BY_NAME
    eps = 1e-9
    with np.load(get(prep["probes"], get_data_dir() / "corpora" / prep["key"] / "probes.npz")) as z:
        arrays = {k: z[k] for k in z.files}
    counts = prep["stats"]["probe_counts"]
    for name in ops:
        op = by_name[name]
        probe = grammar.probe_lines(op, n_probe, probe_seed, both_slots=name in both_slots)
        hue = np.array([hue_move(op, ln.lhs, ln.rhs) for ln in probe])
        red = np.array([max(redness(ln.lhs), redness(ln.rhs)) >= red_dose - eps for ln in probe])
        removal = red & (hue >= far_move - eps)
        arrays[f"{name}/hue_move"] = hue
        counts[name] |= {
            "removal_zero": counts[name]["removal"],
            "removal": int(removal.sum()),
            "sv": int((red & ~removal).sum()),
        }
    return prep | {"probes": put(_npz(**arrays), name=f"ex-2.2.11-{prep['key']}-probes.npz")}


def cells(conds: tuple[Cond, ...], prep: dict) -> list[dict]:
    """One row per run, as ex-2.2.9's `cells`, with the model seed offset by `SEED_OFFSET`: condition seed
    *i* keeps label `<cond>-s<i>` and trains at model seed `SEED_OFFSET + i`.
    """
    from sca.utils import align

    tc = prep["meta"].tokenizer_config
    rows = []
    for c in conds:
        anchor, anti = schedules(c.condition)
        anchor = anchor | {"span": c.span}
        for seed in range(c.seeds):
            config = _make_config(align(tc.vocab_size, 64), SEED_OFFSET + seed, c.epochs, c.n_embd, c.n_layer)
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
                    "model_seed": SEED_OFFSET + seed,
                    "label": f"{c.name}-s{seed}",
                    "keep_traj": c.lam > 0 and seed < TRAJ_CHECKPOINT_SEEDS,
                }
            )
    return rows


def train_one(
    config,
    anchor: dict,
    anti: dict | None,
    corpus,
    traj_stride: int,
    probes,
    keying: Keying,
    keep_traj: bool,
    label: str,
) -> dict:
    """Train one run under its labeller, recording the m_line trajectory on the `mix` probe lines.

    Ex-2.2.9's step with one addition: with *keep_traj*, a checkpoint at every trajectory point, stored as one
    tree under `traj_checkpoints` (`<point>/model/checkpoint.eqx`, in record order).
    """
    from sca.anchoring import AnchorSpec, AntiSpec, LabelSpec
    from sca.compute.model import save_checkpoint
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
    traj_dir = workdir / "trajectory"

    def keep(point: int, model) -> None:
        save_checkpoint(model, config, None, traj_dir / f"{point:03d}")

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
        on_record=keep if keep_traj else None,
    )
    out = {
        "label": label,
        "val_loss": [m.val_loss for m in metrics],
        "train_loss": [m.train_loss for m in metrics],
        "traj": {
            k: traj[k].tolist()
            for k in ("epoch", "m_line", "m_op1", "m_span", "alpha_op1", "val_loss", "weight", "anti_weight")
            if k in traj
        },
        "checkpoint": put(workdir / "model", name=f"ex-2.2.11-{label}-ckpt"),
    }
    if keep_traj:
        out["traj_checkpoints"] = put(traj_dir, name=f"ex-2.2.11-{label}-trajectory")
    return out


def anneal_retention(traj: dict, weight_ratio: float) -> dict[str, float]:
    """Retention across the anneal, from one run's trajectory: the anneal starts at the first point after
    the anchor weight's peak where the weight is under *weight_ratio* of it, the alignment at the start is
    the last point before that, and retention is the final alignment over it.
    """
    w, ml, ep = (np.asarray(traj[k], float) for k in ("weight", "m_line", "epoch"))
    below = np.flatnonzero(w < weight_ratio * w.max())
    below = below[below > int(np.argmax(w))]
    i = int(below[0]) if len(below) else len(w) - 1
    at = float(ml[i - 1]) if i > 0 else float(ml[0])
    return {
        "anneal_epoch": float(ep[i]),
        "m_line_at_anneal": at,
        "m_line_final": float(ml[-1]),
        "retention_anneal": float(ml[-1] / at) if at > 0 else float("nan"),
    }


def eval_one(trained: dict, evals, probes, tau: float, keying: Keying, condition: str, seed: int, label: str) -> dict:
    """Ex-2.2.9's eval, with the retention across the anneal (H2's line) added beside the old peak ratio."""
    out = ex229_eval_one(trained, evals, probes, tau, keying, condition, seed, label)
    return out | anneal_retention(trained["traj"], ANNEAL_WEIGHT_RATIO)


class HueReadout(Readout):
    """Ex-2.2.9's readout with the removal lines chosen by hue. `removal` (and its two slots) is red with a
    hue move of at least `FAR_MOVE`; `removal_zero` is the old to-zero set, kept so the report can print both;
    `sv` (and its two slots) is the saturation-and-value lines, the red lines the hue rule sets aside.
    """

    def __init__(self, tokens, r1, r2, q_idx, q_p, hue_move, zero_move, color_ids, tok2color):
        super().__init__(tokens, r1, r2, q_idx, q_p, hue_move, color_ids, tok2color)
        eps = 1e-9
        red = self.groups["red"]
        zero = red & (zero_move >= FAR_MOVE - eps)
        sv = red & ~self.groups["removal"]
        op1 = r1 >= r2
        self.zero_move = zero_move
        self.groups |= {
            "removal_zero": zero,
            "removal_zero_op1": zero & op1,
            "removal_zero_op2": zero & ~op1,
            "sv": sv,
            "sv_op1": sv & op1,
            "sv_op2": sv & ~op1,
        }


def score_one(trained: dict, probes, operators: tuple[str, ...], condition: str, seed: int, label: str) -> dict:
    """Score one checkpoint on every op's probe lines under every operator, as ex-2.2.9 did, with the
    groups of `HueReadout`: the per-group tables carry the hue-rule removal lines, the to-zero ones, and the
    saturation-and-value lines side by side.
    """
    from sca.intervention import Subspace
    from mini.store import get, put

    workdir = get_data_dir() / "score" / label
    model, _, color_ids, tok2color = _load(trained, workdir)
    sub = Subspace.axis(model.transformer.wte.shape[1], ANCHOR_AXIS)
    keys = ("tokens", "r1", "r2", "q_idx", "q_p", "move", "hue_move")
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe = {o: {k: z[f"{o}/{k}"] for k in keys} for o in _probe_ops(z)}

    ops: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    for op, f in probe.items():
        read = HueReadout(
            f["tokens"], f["r1"], f["r2"], f["q_idx"], f["q_p"], f["hue_move"], f["move"], color_ids, tok2color
        )
        stats, per_line = _score_op(model, sub, read, operators)
        stats["zero_move"] = read.by_group(read.zero_move)
        ops[op] = stats
        arrays |= {f"{op}/{k}": v for k, v in per_line.items()}
        arrays[f"{op}/dose"] = read.dose.astype(np.float32)
        arrays[f"{op}/removal"] = read.groups["removal"]
        arrays[f"{op}/removal_zero"] = read.groups["removal_zero"]
    return {
        "label": label,
        "condition": condition,
        "seed": seed,
        "operators": list(operators),
        "ops": ops,
        "arrays": put(_npz(**arrays), name=f"ex-2.2.11-{label}-score.npz"),
    }


# --- Publishing ------------------------------------------------------------------------------


def design() -> dict[str, Any]:
    """Ex-2.2.9's design record with what this re-run changes."""
    return ex229_design() | {
        "n_runs": N_RUNS,
        "conditions": [asdict(c) | {"steps": c.steps, "lines_per_op": c.lines_per_op} for c in CONDS],
        "seed_offset": SEED_OFFSET,
        "scored_under_projection": [c.name for c in SCORED_UNDER_PROJECTION],
        "removal_rule": HUE_RULE,
        "hue_rotation": {"steps": HUE_ROTATION_STEPS, "tolerance": HUE_ROTATION_TOLERANCE},
        "anneal_weight_ratio": ANNEAL_WEIGHT_RATIO,
        "traj_checkpoint_seeds": TRAJ_CHECKPOINT_SEEDS,
        "cube_probed": {},
    }


def publish_results(trained: list[dict], evaled: list[dict], scored: list[dict], corpus: dict) -> dict:
    """Metrics (JSON), trajectories (JSON), the stacked per-run arrays (npz), each run's full arrays, the probe
    set, every end checkpoint, and the trajectory checkpoints of the runs that kept them, each under its ref.
    """
    import json

    from mini.store import get, put, set_ref

    metrics = {
        "runs": [_slim(r) for r in evaled],
        "scores": [_slim(r) for r in scored],
        "corpora": [corpus["stats"]],
        "design": design(),
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.2.11-metrics.json"))
    traj = {t["label"]: {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.11-trajectories.json"))
    set_ref(PROBE_REF, corpus["probes"])
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
        if "traj_checkpoints" in t:
            set_ref(TRAJ_CHECKPOINT_REF.format(label=t["label"]), t["traj_checkpoints"])

    arrays = {}
    for r in evaled + scored:
        kind = "eval" if "per_op" in r else "score"
        set_ref(RUN_ARRAYS_REF.format(label=r["label"], kind=kind), r["arrays"])
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}-{kind}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{kind}/{name}": z[name] for name in z.files if _stacked(name, kind)}
    set_ref(ARRAYS_REF, put(_npz(**arrays), name="ex-2.2.11-arrays.npz"))
    return {
        "n_runs": len(evaled),
        "n_scored": len(scored),
        "n_traj_checkpoints": sum("traj_checkpoints" in t for t in trained),
        "holdout_eem": {r["label"]: r["holdout_eem"] for r in evaled},
    }


# --- Orchestration ----------------------------------------------------------------------------


def main(ctx: Ctx) -> dict:
    prep = ctx.run(
        prepare_corpus,
        OP_NAMES,
        N_LINES,
        CORPUS_SEED,
        HOLDOUT_FRAC,
        ROUNDING,
        N_PROBE,
        PROBE_SEED,
        PROBE_BOTH_SLOTS,
        PER_SLOT_RATE,
        RED_RATE,
        RED_DOSE,
        NONRED_DOSE,
        FAR_MOVE,
        role="prep",
    )
    rows = cells(CONDS, prep)
    n = len(rows)
    trained = ctx.map(
        train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [prep["corpus"]] * n,
        [TRAJ_STRIDE] * n,
        [prep["probes"]] * n,
        [r["keying"] for r in rows],
        [r["keep_traj"] for r in rows],
        [r["label"] for r in rows],
        role="train",
    )
    evaled = ctx.map(
        eval_one,
        trained,
        [prep["evals"]] * n,
        [prep["probes"]] * n,
        [r["anchor"]["tau"] for r in rows],
        [r["keying"] for r in rows],
        [r["condition"] for r in rows],
        [r["seed"] for r in rows],
        [r["label"] for r in rows],
        role="eval",
    )
    anchored = [
        (r, t)
        for r, t in zip(rows, trained, strict=True)
        if r["condition"] in {c.name for c in SCORED_UNDER_PROJECTION}
    ]
    scored = ctx.map(
        score_one,
        [t for _, t in anchored],
        [prep["probes"]] * len(anchored),
        [OPERATORS] * len(anchored),
        [r["condition"] for r, _ in anchored],
        [r["seed"] for r, _ in anchored],
        [r["label"] for r, _ in anchored],
        role="score",
    )
    return ctx.run(publish_results, trained, evaled, scored, prep, role="prep")


experiment = Experiment(
    name="ex-2.2.11",
    main=main,
    roles={
        # Ex-2.2.9's corpus build, then the hue move on every probe line (up to 720 permutations per op).
        "prep": dict(cpu=2, timeout=1800),
        # 4,950 steps; the watchdog covers the checkpoint upload after the last step, and the trajectory
        # tree upload on the runs that keep one.
        "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
        "eval": dict(gpu="L4", timeout=1800),
        "score": dict(gpu="L4", timeout=2400),
    },
)
