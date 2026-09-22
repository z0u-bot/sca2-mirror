"""Ex-2.2.12: what the stream holds on `hue-hsv`, and a small recipe sweep.

Two parts, independent of each other: a scoring-only pass
over ex-2.2.11's stored `handover` checkpoints (part 1), and a training sweep of eight conditions at five
seeds on the handover setup (part 2), compared with ex-2.2.11's twenty `handover` seeds as the reference.
Nothing here is scored; the report proposes, and the handover re-run that follows adopts at fresh seeds.

The design constants come first; the DAG follows, binding everything it does not change from ex-2.2.11's module.

    bin/mini run docs/m2/ex-2.2.12/experiment.py --app modal --max-containers 8 --budget 3h
    bin/mini status ex-2.2.12
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir

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

# =============================================================================================
# The DAG
# =============================================================================================


def _load_ex2211():
    """Ex-2.2.11's module (which loads ex-2.2.9's and ex-2.2.3's the same way), by path and left out of
    `sys.modules`, so the task bodies here still cloudpickle by value for a remote worker.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "ex-2.2.11" / "experiment.py"
    spec = importlib.util.spec_from_file_location("ex2211", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex2211 = _load_ex2211()
ex229 = ex2211.ex229

# --- What stays as ex-2.2.11 had it -------------------------------------------------------------
# Bound by name so a task body never references the module objects themselves.

OP_NAMES = ex2211.OP_NAMES
PRIMARY_OP = ex2211.PRIMARY_OP
N_LINES = ex2211.N_LINES
EPOCHS = ex2211.EPOCHS
CORPUS_SEED = ex2211.CORPUS_SEED
HOLDOUT_FRAC = ex2211.HOLDOUT_FRAC
N_PROBE = ex2211.N_PROBE
PROBE_SEED = ex2211.PROBE_SEED
PROBE_BOTH_SLOTS = ex2211.PROBE_BOTH_SLOTS
RED_DOSE = ex2211.RED_DOSE
NONRED_DOSE = ex2211.NONRED_DOSE
FAR_MOVE = ex2211.FAR_MOVE
TRAJ_STRIDE = ex2211.TRAJ_STRIDE
OPERATORS = ex2211.OPERATORS
OPERATOR_SPEC = ex2211.OPERATOR_SPEC
SEED_OFFSET = ex2211.SEED_OFFSET
ROUNDING = ex2211.ROUNDING
PER_SLOT_RATE = ex2211.PER_SLOT_RATE
RED_RATE = ex2211.RED_RATE
ANCHOR_AXIS = ex2211.ANCHOR_AXIS
ANNEAL_WEIGHT_RATIO = ex2211.ANNEAL_WEIGHT_RATIO
WHOLE_SPAN = ex229.WHOLE_SPAN
N_EMBD = ex229.N_EMBD
ANSWER_POS = ex229.ANSWER_POS
DECODE_POS = ex229.DECODE_POS
COMPOSITION = ex229.COMPOSITION
SYNTAX_WORDS = ex229.SYNTAX_WORDS
"""The grammar, the corpus, the probe sets, the operators, and the seed offset: as ex-2.2.11 froze them. The
sweep's condition seed *i* trains at model seed `SEED_OFFSET + i`, the seed ex-2.2.11's `handover-s<i>` trained at."""

TASK_GATE = ex2211.TASK_GATE
REF_M_LINE = ex2211.REF_M_LINE
MARGIN_RATIO = ex2211.MARGIN_RATIO
LEAD_GATE = ex2211.LEAD_GATE
CONTRAST_GATE = ex2211.CONTRAST_GATE
RED_KEPT_GATE = ex2211.RED_KEPT_GATE
NONRED_DEFICIT_GATE = ex2211.NONRED_DEFICIT_GATE
"""Every gate the promotion rule names, at ex-2.2.11's value."""

_npz = ex229._npz
_load = ex229._load
_probe_ops = ex229._probe_ops
_make_config = ex229._make_config
_placement = ex229._placement
_slim = ex229._slim
_stacked = ex229._stacked
top_quantile = ex229.top_quantile
schedules = ex229.schedules
Condition223 = ex229.Condition
HueReadout = ex2211.HueReadout
ex2211_prepare_corpus = ex2211.prepare_corpus
ex2211_train_one = ex2211.train_one
ex2211_eval_one = ex2211.eval_one
ex2211_design = ex2211.design

# --- Ex-2.2.11's stored runs ----------------------------------------------------------------------

EX2211_PROBE_REF = ex2211.PROBE_REF
EX2211_TRAJ_REF = ex2211.TRAJ_REF
EX2211_METRICS_REF = ex2211.METRICS_REF
EX2211_CHECKPOINT_REF = ex2211.CHECKPOINT_REF
EX2211_RUN_ARRAYS_REF = ex2211.RUN_ARRAYS_REF
assert EX2211_CHECKPOINT_REF == HANDOVER_CHECKPOINT_REF

REFERENCE_SEEDS = {"handover": 20, "handover-slot": 20, "handover-tied": 9, "control": 5}
"""Ex-2.2.11's conditions and their seed counts, as its `CONDS` names them."""

SIDE_CONDITIONS = ("handover", "handover-slot", "handover-tied")
"""Part 1's first measurement runs over every anchored condition ex-2.2.11 scored: the references say whether
the side effect is the handover's or the axis's."""

LABEL_CONDITIONS = ("handover", "handover-slot")
"""The third: `handover-slot` labels through the operands only, so it is the condition with no answer-labelled
lines, and the split there is the same lines without the pull."""

BYPASS_CONDITIONS = ("handover",)
BYPASS_OPS = (MISSED_OP, PRIMARY_OP)
"""The second: `handover` only, on the missed op and on `mix`, the op where the full projection removes cleanly."""

BYPASS_SPEC: dict[str, tuple[tuple[int, ...] | str | None, tuple[int, ...] | None]] = {
    "all": (None, None),
    "embedding": ((0,), None),
    "blocks": ("blocks", None),
    "op2-all": (None, (2,)),
    "op2-embedding": ((0,), (2,)),
}
"""Each edit as (slices, positions): `None` slices is every slice of the model, `"blocks"` is every slice after
the embedding, and `None` positions is every position. `all` is ex-2.2.11's `projection` operator and must
reproduce its kept share."""
assert tuple(BYPASS_SPEC) == BYPASS_EDITS

LABEL_GROUPS = ("neither", "labelled by operand", "labelled by answer", "both", "red op1")
"""The split of the op1 walk's probe lines by which color could earn the whole-line label, at the red dose:
the first four have a non-red op1 (the question is whether the pull reaches a position that is not red), by
whether op2 is red, the answer is red, both, or neither; the last has a red op1, and is the containment line's
usual reading. `handover-slot`'s labeller never reads the answer, so its `labelled by answer` lines are
unlabelled there."""
assert LABEL_GROUPS[1:3] == CONTAINMENT_SPLIT

CONTROL_PLANE = "control-plane"
"""Ex-2.2.11's `control` checkpoints, scored on the plane: the comparison for every plane measurement."""

# --- Refs -------------------------------------------------------------------------------------------

METRICS_REF = "reports/m2/ex-2.2.12/metrics"
ARRAYS_REF = "reports/m2/ex-2.2.12/arrays"
TRAJ_REF = "reports/m2/ex-2.2.12/trajectories"
PART1_REF = "reports/m2/ex-2.2.12/part1"
"""Per-line arrays of part 1: the kept share per removal line by side, and the bypass edits' expected exact
match per line, keyed `<label>/<op>/<name>`."""
CHECKPOINT_REF = "reports/m2/ex-2.2.12/checkpoints/{label}"
RUN_ARRAYS_REF = "reports/m2/ex-2.2.12/arrays/{label}/{kind}"


# --- Part 1 ----------------------------------------------------------------------------------------


def line_sides(op_name: str) -> np.ndarray:
    """The side of red of every probe line of *op_name*, in probe order: the index into `SIDE_GROUPS` of the
    sign of G − B on the redder operand (op1 on a tie, as `Readout` picks it).
    """
    from sca.data import ops as grammar
    from sca.data.colors import redness

    op = (grammar.OP_BY_NAME | grammar.CANDIDATE_BY_NAME)[op_name]
    probe = grammar.probe_lines(op, N_PROBE, PROBE_SEED, both_slots=op_name in PROBE_BOTH_SLOTS)
    red = [ln.lhs if redness(ln.lhs) >= redness(ln.rhs) else ln.rhs for ln in probe]
    gb = np.array([c[1] - c[2] for c in red])
    return np.where(gb == 0, 0, np.where(gb > 0, 1, 2)).astype(np.int8)


def red_colors_by_side() -> list[int]:
    """How many grid colors at or above the red dose sit on each side."""
    from sca.data import ops as grammar
    from sca.data.colors import redness

    reds = [c for c in grammar.colors() if redness(c) >= RED_DOSE - 1e-9]
    gb = np.array([c[1] - c[2] for c in reds])
    return [int((gb == 0).sum()), int((gb > 0).sum()), int((gb < 0).sum())]


def side_of_red(scores: dict[str, Any], ops: tuple[str, ...], missed_op: str) -> dict:
    """Kept share under ex-2.2.11's `projection` on every op's removal lines, split by the side of red.

    *scores* maps a run label to its ex-2.2.11 score arrays. Kept share is the ratio of group means, as
    `Readout.ratio_by_group` scores it; the per-line ratio on the missed op goes to the store for the figure.
    """
    from mini.store import get, get_many, put

    workdir = get_data_dir() / "part1" / "side"
    paths = dict(zip(scores, get_many([(a, workdir / f"{lb}.npz") for lb, a in scores.items()]), strict=True))
    sides = {op: line_sides(op) for op in ops}
    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {f"{op}/side": s for op, s in sides.items()}
    for label, path in paths.items():
        cond, seed = label.rsplit("-s", 1)
        with np.load(path) as z:
            for op in ops:
                removal, clean, proj = z[f"{op}/removal"], z[f"{op}/clean/eem"], z[f"{op}/projection/eem"]
                arrays[f"{op}/removal"] = removal
                if op == missed_op:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        arrays[f"{label}/{op}/kept_line"] = np.where(clean > 0, proj / clean, np.nan).astype(np.float32)
                for i, group in enumerate(SIDE_GROUPS):
                    m = removal & (sides[op] == i)
                    rows.append(
                        {
                            "label": label,
                            "condition": cond,
                            "seed": int(seed),
                            "op": op,
                            "side": group,
                            "n": int(m.sum()),
                            "kept": float(proj[m].mean() / clean[m].mean())
                            if m.any() and clean[m].mean() > 0
                            else float("nan"),
                            "eem_clean": float(clean[m].mean()) if m.any() else float("nan"),
                            "eem_projection": float(proj[m].mean()) if m.any() else float("nan"),
                        }
                    )
    del get
    return {
        "rows": rows,
        "red_colors_by_side": red_colors_by_side(),
        "arrays": put(_npz(**arrays), name="ex-2.2.12-side.npz"),
    }


def label_source(evals: dict[str, Any], probes, ops: tuple[str, ...]) -> dict:
    """ᾱ at op1 on the op1 walk's probe lines, per slice, split by `LABEL_GROUPS`.

    *evals* maps a run label to its ex-2.2.11 eval arrays, whose `<op>/alpha_lines` is the per-line alignment
    (slices × lines × positions) of the op1 walk. Red is at the red dose, as every group here is.
    """
    from mini.store import get, get_many

    workdir = get_data_dir() / "part1" / "labels"
    paths = dict(zip(evals, get_many([(a, workdir / f"{lb}.npz") for lb, a in evals.items()]), strict=True))
    eps = 1e-9
    groups: dict[str, dict[str, np.ndarray]] = {}
    with np.load(get(probes, workdir / "probes.npz")) as z:
        for op in ops:
            w0 = z[f"{op}/walk"] == 0
            r1, r2, r3 = (z[f"{op}/{k}"][w0] >= RED_DOSE - eps for k in ("r1", "r2", "r3"))
            groups[op] = {
                "neither": ~r1 & ~r2 & ~r3,
                "labelled by operand": ~r1 & r2 & ~r3,
                "labelled by answer": ~r1 & ~r2 & r3,
                "both": ~r1 & r2 & r3,
                "red op1": r1,
            }
    rows: list[dict] = []
    for label, path in paths.items():
        cond, seed = label.rsplit("-s", 1)
        with np.load(path) as z:
            for op in ops:
                alpha = z[f"{op}/alpha_lines"].astype(np.float32)[:, :, 0]  # (L1, N): op1
                for group in LABEL_GROUPS:
                    m = groups[op][group]
                    per_slice = alpha[:, m].mean(axis=1) if m.any() else np.full(alpha.shape[0], np.nan)
                    rows.append(
                        {
                            "label": label,
                            "condition": cond,
                            "seed": int(seed),
                            "op": op,
                            "group": group,
                            "n": int(m.sum()),
                            "alpha_op1": float(per_slice.mean()),
                            "alpha_op1_slices": per_slice.tolist(),
                        }
                    )
    return {"rows": rows, "n": {op: {g: int(m.sum()) for g, m in gs.items()} for op, gs in groups.items()}}


def bypass_one(
    checkpoint, probes, ops: tuple[str, ...], edits: tuple[str, ...], condition: str, seed: int, label: str
) -> dict:
    """One stored checkpoint under the projection applied at each of `BYPASS_SPEC`'s (slices, positions).

    Expected exact match per group after each edit, and the kept share against the clean pass, with the
    removal lines split by side. Per-line expected exact match goes to the store.
    """
    from sca.intervention import Subspace, apply, projection
    from mini.store import get, put

    workdir = get_data_dir() / "bypass" / label
    model, _, color_ids, tok2color = _load({"checkpoint": checkpoint}, workdir)
    sub = Subspace.axis(model.transformer.wte.shape[1], ANCHOR_AXIS)
    every = tuple(range(len(model.transformer.blocks) + 1))
    keys = ("tokens", "r1", "r2", "q_idx", "q_p", "move", "hue_move")
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe = {o: {k: z[f"{o}/{k}"] for k in keys} for o in ops}

    out_ops: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    for op, f in probe.items():
        read = HueReadout(
            f["tokens"], f["r1"], f["r2"], f["q_idx"], f["q_p"], f["hue_move"], f["move"], color_ids, tok2color
        )
        side = line_sides(op)
        assert len(side) == len(f["tokens"])
        read.groups |= {f"removal_side{i}": read.groups["removal"] & (side == i) for i in range(len(SIDE_GROUPS))}
        n_pos = f["tokens"].shape[1]
        c = read(apply(model, f["tokens"], projection(sub), slices=()).logits)
        arrays[f"{op}/clean/eem"] = c["eem"].astype(np.float32)
        stats: dict[str, Any] = {
            "n": {g: int(m.sum()) for g, m in read.groups.items()},
            "clean": {"eem": read.by_group(c["eem"])},
            "edits": {},
        }
        for edit in edits:
            slices, positions = BYPASS_SPEC[edit]
            slices = every if slices is None else every[1:] if isinstance(slices, str) else slices
            mask = None if positions is None else np.isin(np.arange(n_pos), positions).astype(np.float32)
            r = read(apply(model, f["tokens"], projection(sub), slices=slices, positions=mask).logits)
            stats["edits"][edit] = {
                "slices": list(slices),
                "positions": None if positions is None else list(positions),
                "eem": read.by_group(r["eem"]),
                "kept": read.ratio_by_group(r["eem"], c["eem"]),
                "deficit": read.by_group(c["eem"] - r["eem"]),
                "acc": read.by_group(r["guess"] == read.answer),
            }
            arrays[f"{op}/{edit}/eem"] = r["eem"].astype(np.float32)
        out_ops[op] = stats
    return {
        "label": label,
        "condition": condition,
        "seed": seed,
        "edits": list(edits),
        "ops": out_ops,
        "arrays": put(_npz(**arrays), name=f"ex-2.2.12-{label}-bypass.npz"),
    }


# --- Part 2 ----------------------------------------------------------------------------------------


def axes_of(c: Condition) -> tuple[int, ...]:
    """Where *red* lives under *c*: the axis, or the plane."""
    return PLANE_AXES if c.subspace == "plane" else (ANCHOR_AXIS,)


def prepare_corpus(*args) -> dict:
    """Ex-2.2.11's corpus, eval sets, and probe set, unchanged, under this experiment's memo."""
    return ex2211_prepare_corpus(*args)


def cells(conds: tuple[Condition, ...], prep: dict) -> list[dict]:
    """One row per run of the sweep: ex-2.2.11's `handover` row with the condition's changes applied. The
    plane conditions carry `axes` in their anchor dict, which `AnchorSpec` reads and every term and
    trajectory measurement follows.
    """
    from sca.utils import align

    tc = prep["meta"].tokenizer_config
    rows = []
    for c in conds:
        base = Condition223(
            c.name,
            c.seeds,
            c.title,
            lam=c.lam,
            tau=c.tau,
            anti_peak_ratio=c.anti_peak,
            epochs=EPOCHS,
            ops=OP_NAMES,
            n_lines=N_LINES,
        )
        anchor, anti = schedules(base)
        anchor = anchor | {"span": WHOLE_SPAN, "axes": axes_of(c)}
        for seed in range(c.seeds):
            config = _make_config(align(tc.vocab_size, 64), SEED_OFFSET + seed, EPOCHS, N_EMBD, c.n_layer)
            config.tokenizer = tc.model_copy()
            config.model = config.model.model_copy(update={"tie_embeddings": False})
            rows.append(
                {
                    "config": config,
                    "anchor": anchor,
                    "anti": anti,
                    "keying": "line",
                    "condition": c.name,
                    "seed": seed,
                    "model_seed": SEED_OFFSET + seed,
                    "label": f"{c.name}-s{seed}",
                    "axes": axes_of(c),
                }
            )
    return rows


def train_one(config, anchor: dict, anti: dict | None, corpus, traj_stride: int, probes, keying, label: str) -> dict:
    """Ex-2.2.11's training step, with no trajectory checkpoints: the sweep proposes, and a dynamics measurement
    would run on the re-run's checkpoints.
    """
    return ex2211_train_one(config, anchor, anti, corpus, traj_stride, probes, keying, False, label)


def _plane_placement(trained: dict, probes, tau: float, keying, label: str, axes: tuple[int, ...]) -> dict:
    """Ex-2.2.9's placement and embedding-component measurements with the alignment read on *axes*."""
    from sca.anchoring import alignment, axes_alignment
    from mini.store import get, put

    workdir = get_data_dir() / "eval" / label / "plane"
    model, tokenizer, _, tok2color = _load(trained, workdir)
    per_op: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    with np.load(get(probes, workdir / "probes.npz")) as z:
        weights = z["weights"]
        probe = {
            o: (z[f"{o}/tokens"], z[f"{o}/r1"], z[f"{o}/r2"], z[f"{o}/line_p_{keying}"], z[f"{o}/walk"])
            for o in _probe_ops(z)
        }
    for op, (tokens, r1, r2, line_p, walk) in probe.items():
        cos = alignment(model, tokens, axes=axes)  # (L1, N, T), unsigned on a plane
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

    def row_table(table) -> dict[str, float]:
        comp = np.asarray(axes_alignment(np.asarray(table), axes))
        return {w: float(comp[i]) for i in range(len(comp)) if (w := tokenizer.itos.get(i, ""))}

    head = model.transformer.lm_head
    rows = row_table(model.transformer.wte)
    rows_readout = None if head is None else row_table(head)
    primary = per_op[PRIMARY_OP]
    return {
        **{k: primary[k] for k in ("m_line", "m_span", "alpha_op1", "r2_sim", "contrast", "latch_pi", "lead_emb")},
        "per_op": per_op,
        "rows": rows,
        "rows_readout": rows_readout,
        "syntax_component": {
            "wte": float(np.mean([abs(rows[w]) for w in SYNTAX_WORDS])),
            "readout": None if rows_readout is None else float(np.mean([abs(rows_readout[w]) for w in SYNTAX_WORDS])),
        },
        "arrays": put(_npz(**arrays), name=f"ex-2.2.12-{label}-arrays.npz"),
    }


def eval_one(
    trained: dict, evals, probes, tau: float, keying, condition: str, seed: int, label: str, axes: tuple[int, ...]
) -> dict:
    """Ex-2.2.11's eval (behavior per op, placement, the retention across the anneal), and on a plane the
    placement and embedding-component measurements again with the alignment read on the pair.
    """
    out = ex2211_eval_one(trained, evals, probes, tau, keying, condition, seed, label)
    if axes != (ANCHOR_AXIS,):
        out |= _plane_placement(trained, probes, tau, keying, label, axes)
    return out | {"axes": list(axes)}


def _score_op(
    model, sub, read, operators: tuple[str, ...], axes: tuple[int, ...]
) -> tuple[dict, dict[str, np.ndarray]]:
    """Ex-2.2.9's `_score_op` with the alignment read on *axes* and the operators applied at every slice the
    model has (six blocks give seven). The write-angle check holds on a plane as it does on an axis: the
    projection removes a component of length α and re-normalizes, whatever the component's dimension.
    """
    from sca.anchoring import axes_alignment
    from sca.intervention import angle_between, apply, projection, write_angle

    tokens = read.tokens
    n_pos = tokens.shape[1]
    nonred = read.groups["nonred"]
    every = tuple(range(len(model.transformer.blocks) + 1))

    clean = apply(model, tokens, projection(sub), slices=())
    alpha_clean = np.asarray(axes_alignment(clean.pre, axes))  # (L1, N, T)
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
    for name in operators:
        operator, mask = ex229._operator(name, sub)
        positions = None if mask is None else np.isin(np.arange(n_pos), mask).astype(np.float32)
        out = apply(model, tokens, operator, slices=every, positions=positions)
        theta = angle_between(out.pre, out.post)
        alpha_pre = np.asarray(axes_alignment(out.pre, axes))
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


def score_one(
    trained: dict, probes, operators: tuple[str, ...], condition: str, seed: int, label: str, axes: tuple[int, ...]
) -> dict:
    """Ex-2.2.11's scoring (every op's probe lines under every operator, with the hue-rule groups), with the
    subspace the operators remove being the run's home of *red*: the axis, or the plane.
    """
    from sca.intervention import Subspace
    from mini.store import get, put

    workdir = get_data_dir() / "score" / label
    model, _, color_ids, tok2color = _load(trained, workdir)
    width = model.transformer.wte.shape[1]
    sub = Subspace.axis(width, axes[0]) if len(axes) == 1 else Subspace.axes(width, axes)
    keys = ("tokens", "r1", "r2", "q_idx", "q_p", "move", "hue_move")
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe = {o: {k: z[f"{o}/{k}"] for k in keys} for o in _probe_ops(z)}

    ops: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    for op, f in probe.items():
        read = HueReadout(
            f["tokens"], f["r1"], f["r2"], f["q_idx"], f["q_p"], f["hue_move"], f["move"], color_ids, tok2color
        )
        stats, per_line = _score_op(model, sub, read, operators, axes)
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
        "axes": list(axes),
        "operators": list(operators),
        "ops": ops,
        "arrays": put(_npz(**arrays), name=f"ex-2.2.12-{label}-score.npz"),
    }


# --- Publishing ------------------------------------------------------------------------------


def design() -> dict[str, Any]:
    """Ex-2.2.11's design record with what this experiment adds."""
    return ex2211_design() | {
        "experiment": "ex-2.2.12",
        "reference": {"experiment": REFERENCE_EXPERIMENT, "condition": REFERENCE_CONDITION, "seeds": REFERENCE_SEEDS},
        "n_runs": NEW_RUNS,
        "sweep_seeds": SWEEP_SEEDS,
        "conditions": [asdict(c) | {"changes": c.changes, "axes": list(axes_of(c))} for c in CONDITIONS],
        "plane_axes": list(PLANE_AXES),
        "missed_op": MISSED_OP,
        "side_groups": list(SIDE_GROUPS),
        "bypass_edits": list(BYPASS_EDITS),
        "label_groups": list(LABEL_GROUPS),
        "kept_band": KEPT_BAND,
        "promotion": PROMOTION,
        "tau_proposal": TAU_PROPOSAL,
    }


def publish_results(
    trained: list[dict],
    evaled: list[dict],
    scored: list[dict],
    side: dict,
    labels: dict,
    bypassed: list[dict],
    corpus: dict,
) -> dict:
    """Metrics (JSON, with part 1's rows under `part1`), trajectories, the stacked per-run arrays, each run's
    full arrays, part 1's per-line arrays, and every end checkpoint, each under its ref.
    """
    import json

    from mini.store import get, put, set_ref

    metrics = {
        "runs": [_slim(r) for r in evaled],
        "scores": [_slim(r) for r in scored],
        "part1": {
            "side": side["rows"],
            "red_colors_by_side": side["red_colors_by_side"],
            "label_source": labels["rows"],
            "label_source_n": labels["n"],
            "bypass": [_slim(b) for b in bypassed],
        },
        "corpora": [corpus["stats"]],
        "design": design(),
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.2.12-metrics.json"))
    traj = {t["label"]: {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.12-trajectories.json"))
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])

    arrays: dict[str, np.ndarray] = {}
    for r in evaled + scored:
        kind = "eval" if "per_op" in r else "score"
        set_ref(RUN_ARRAYS_REF.format(label=r["label"], kind=kind), r["arrays"])
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}-{kind}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{kind}/{name}": z[name] for name in z.files if _stacked(name, kind)}
    set_ref(ARRAYS_REF, put(_npz(**arrays), name="ex-2.2.12-arrays.npz"))

    part1: dict[str, np.ndarray] = {}
    with np.load(get(side["arrays"], get_data_dir() / "publish" / "side.npz")) as z:
        part1 |= {name: z[name] for name in z.files}
    for b in bypassed:
        set_ref(RUN_ARRAYS_REF.format(label=b["label"], kind="bypass"), b["arrays"])
        with np.load(get(b["arrays"], get_data_dir() / "publish" / f"{b['label']}-bypass.npz")) as z:
            part1 |= {f"{b['label']}/{name}": z[name] for name in z.files}
    set_ref(PART1_REF, put(_npz(**part1), name="ex-2.2.12-part1.npz"))
    return {
        "n_runs": len(trained),
        "n_evaled": len(evaled),
        "n_scored": len(scored),
        "n_bypassed": len(bypassed),
        "holdout_eem": {r["label"]: r["holdout_eem"] for r in evaled},
    }


# --- Orchestration ----------------------------------------------------------------------------


def resolve_inputs(part1: list[str], bypass: list[str], control: list[str]) -> dict:
    """Ex-2.2.11's probe set, per-run arrays, checkpoints, and trajectories, by ref, so every task here is
    keyed on the content it scores.
    """
    import json

    from mini.store import get, get_refs

    names = [
        EX2211_PROBE_REF,
        EX2211_TRAJ_REF,
        *(EX2211_RUN_ARRAYS_REF.format(label=lb, kind="score") for lb in part1),
        *(EX2211_RUN_ARRAYS_REF.format(label=lb, kind="eval") for lb in part1),
        *(EX2211_CHECKPOINT_REF.format(label=lb) for lb in bypass + control),
    ]
    refs = get_refs(names)
    missing = [k for k, v in refs.items() if v is None]
    assert not missing, f"ex-2.2.11 refs not in this store: {missing}"
    traj_art = refs[EX2211_TRAJ_REF]
    assert traj_art is not None
    traj = json.loads(get(traj_art, get_data_dir() / "part1" / "trajectories.json").read_bytes())
    return {
        "probes": refs[EX2211_PROBE_REF],
        "scores": {lb: refs[EX2211_RUN_ARRAYS_REF.format(label=lb, kind="score")] for lb in part1},
        "evals": {lb: refs[EX2211_RUN_ARRAYS_REF.format(label=lb, kind="eval")] for lb in part1},
        "checkpoints": {lb: refs[EX2211_CHECKPOINT_REF.format(label=lb)] for lb in bypass + control},
        "trajectories": {lb: traj[lb]["traj"] for lb in control},
    }


def _labels(conds: tuple[str, ...]) -> list[str]:
    return [f"{c}-s{s}" for c in conds for s in range(REFERENCE_SEEDS[c])]


def main(ctx: Ctx) -> dict:
    # --- Part 1: ex-2.2.11's stored runs.
    part1_labels = _labels(SIDE_CONDITIONS)
    bypass_labels = _labels(BYPASS_CONDITIONS)
    control_labels = _labels(("control",))
    inputs = ctx.run(resolve_inputs, part1_labels, bypass_labels, control_labels, role="prep")
    side = ctx.run(side_of_red, inputs["scores"], OP_NAMES, MISSED_OP, role="prep")
    labels = ctx.run(
        label_source,
        {lb: inputs["evals"][lb] for lb in _labels(LABEL_CONDITIONS)},
        inputs["probes"],
        OP_NAMES,
        role="prep",
    )
    nb = len(bypass_labels)
    bypassed = ctx.map(
        bypass_one,
        [inputs["checkpoints"][lb] for lb in bypass_labels],
        [inputs["probes"]] * nb,
        [BYPASS_OPS] * nb,
        [BYPASS_EDITS] * nb,
        [lb.rsplit("-s", 1)[0] for lb in bypass_labels],
        [int(lb.rsplit("-s", 1)[1]) for lb in bypass_labels],
        bypass_labels,
        role="score",
    )

    # --- Part 2: the sweep, on ex-2.2.11's corpus.
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
    rows = cells(tuple(c for c in CONDITIONS if c is not REF), prep)
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
        [r["label"] for r in rows],
        role="train",
    )
    # The control checkpoints scored on the plane ride along as five more rows.
    ctrl = [
        {"checkpoint": inputs["checkpoints"][lb], "traj": inputs["trajectories"][lb], "label": lb}
        for lb in control_labels
    ]
    ctrl_rows = [
        {"condition": CONTROL_PLANE, "seed": s, "label": f"{CONTROL_PLANE}-s{s}", "axes": PLANE_AXES, "keying": "line"}
        for s in range(len(control_labels))
    ]
    all_rows, all_trained = rows + ctrl_rows, trained + ctrl
    m = len(all_rows)
    evaled = ctx.map(
        eval_one,
        all_trained,
        [prep["evals"]] * m,
        [prep["probes"]] * m,
        [r["anchor"]["tau"] if "anchor" in r else REF.tau for r in all_rows],
        [r["keying"] for r in all_rows],
        [r["condition"] for r in all_rows],
        [r["seed"] for r in all_rows],
        [r["label"] for r in all_rows],
        [r["axes"] for r in all_rows],
        role="eval",
    )
    scored = ctx.map(
        score_one,
        all_trained,
        [prep["probes"]] * m,
        [OPERATORS] * m,
        [r["condition"] for r in all_rows],
        [r["seed"] for r in all_rows],
        [r["label"] for r in all_rows],
        [r["axes"] for r in all_rows],
        role="score",
    )
    return ctx.run(publish_results, trained, evaled, scored, side, labels, bypassed, prep, role="prep")


experiment = Experiment(
    name="ex-2.2.12",
    main=main,
    roles={
        # Ex-2.2.11's corpus build, and part 1's reductions over its stored arrays.
        "prep": dict(cpu=2, timeout=1800),
        # 4,950 steps at L4, about half again as long at L6; the watchdog covers the checkpoint upload.
        "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
        "eval": dict(gpu="L4", timeout=1800),
        # The sweep under three operators, and part 1's five edits on two ops.
        "score": dict(gpu="L4", timeout=2400),
    },
)
