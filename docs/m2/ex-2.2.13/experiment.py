"""Ex-2.2.13: does a heavier anchor make the leftover predictable?

One ladder: the anchor weight λ_a at four levels, crossed
with the home of *red* (an axis or a plane), at twenty fresh seeds a condition on ex-2.2.11's handover setup.
The question is the spread of the leftover across seeds rather than its size. Every anchored run is new: no
seed here has been trained before. The un-anchored control is the one thing served from the store.

The design constants come first; the DAG follows, binding everything it does not change from ex-2.2.12's
module (the plane, the plane-aware scoring) and ex-2.2.11's (the recipe, the gates, the operators).

    bin/mini run docs/m2/ex-2.2.13/experiment.py --app modal --max-containers 8 --budget 4h
    bin/mini status ex-2.2.13
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir

# --- What is inherited ---------------------------------------------------------------------------------

REFERENCE_EXPERIMENT = "m2/ex-2.2.11"
"""Every gate, the grammar (table A+), the stochastic corpus, the whole-line labeller, the untied readout,
the removal lines chosen by hue, and the recipe apart from λ_a (τ 0.1, the anti-subspace weight from 2.5× the
anchor weight to 0.3× by 90% of training, 50 epochs at d64-L4): all as ex-2.2.11 froze them. This module
names what the ladder changes and nothing else; the implementation binds the rest from ex-2.2.11's module."""

PLANE_EXPERIMENT = "m2/ex-2.2.12"
"""The plane conditions are ex-2.2.12's `plane`: the same anchor, anti-subspace, and removal machinery
taking a pair of axes where they took one. Its scoring fixes and its control-on-the-plane comparison are
inherited too."""

MISSED_OP = "hue-hsv"
"""The one op that missed removal at ex-2.2.11 and at every condition of ex-2.2.12's sweep. Its removal
lines have red at op2, and the answer takes its hue from that operand."""

KEPT_GATE = 0.2
"""Ex-2.2.11's gate on the kept share, unchanged. Under it, the projection is taken to have removed *red*
from an op's red-dependent answers. The implementation binds this from ex-2.2.11 and asserts the value."""

CONTROL = "control"
"""The un-anchored control: ex-2.2.11's `control` checkpoints at seeds 100–104, served from the store rather
than retrained. It is the task reference (ex-2.2.11's task gate is a seed mean within a band of the
control's) and the ᾱ baseline for the axis conditions. Neither role touches the observation this experiment
follows up, which was read off anchored checkpoints, so borrowing it spends nothing the design needs. The
cost is that comparisons against it are unpaired across seed sets, which a comparison of seed means absorbs."""

CONTROL_PLANE = "control-plane"
"""The same control checkpoints scored on the plane: the comparison for every plane measurement, since an
unsigned two-dimensional alignment sits higher than a signed one-dimensional one for every state, anchored
or not. A scoring pass, not a training condition."""

# --- The ladder ----------------------------------------------------------------------------------------

LADDER = (0.1, 0.14, 0.2, 0.28)
"""The anchor weight, four levels on a √2 ladder. 0.1 is ex-2.2.11's recipe and 0.2 the one step ex-2.2.12
took; the ratio is constant so the levels are evenly spaced in log, which is the scale a weight is read on.
The top stops short of where ex-2.1.11's survey first saw task failures (λ_a ≥ 0.38 on the six-op grammar):
the ladder is here to test the region between the recipe and that point, and where the useful range ends is
left open."""

SUBSPACES = ("axis", "plane")
"""The home of *red*: the first axis e₁, or the plane spanned by e₁ and e₂. Ex-2.2.12 settled against the
plane's first rationale (that a single axis cannot hold which side of red a color sits on), but its one plane
condition at a heavier weight was the tightest across seeds in the sweep, and the plane at the recipe's weight
was seen at five seeds only. The cross says whether a narrowing belongs to the weight, to the subspace, or
to the two together."""

SEED_OFFSET = 200
"""Condition seed *i* trains at model seed `SEED_OFFSET + i`. Ex-2.2.11 and ex-2.2.12 trained at offset 100,
and the observation that motivated this experiment was read off those checkpoints, so none of them can test
it. Every seed here is fresh, and the reference is retrained beside the ladder at the same seeds."""

SEEDS = 20
"""Seeds a condition. Twenty is set by the spread question rather than the mean: a one-sided F-test on the
variance ratio between the top and the bottom of the ladder has power 0.90 at twenty seeds a group for a
halved standard deviation, 0.81 at fifteen, and 0.63 at ten. A difference in means of the size we care
about would be resolved by half as many."""

CONDITIONS = tuple(f"{s}-{lam:g}" for s in SUBSPACES for lam in LADDER)
assert len(CONDITIONS) == 8

N_RUNS = len(CONDITIONS) * SEEDS  # the eight rungs; the control is not retrained
assert N_RUNS == 160


@dataclass(frozen=True)
class Condition:
    name: str
    subspace: str
    lam: float

    @property
    def is_reference(self) -> bool:
        return self.name == REFERENCE


REFERENCE = "axis-0.1"
"""Ex-2.2.11's recipe, retrained at this experiment's seeds: the lowest rung of the ladder and the comparison for
every claim about what the ladder moves. Retraining it also replicates ex-2.2.11's `handover` at fresh seeds."""

GRID = tuple(Condition(f"{s}-{lam:g}", s, lam) for s in SUBSPACES for lam in LADDER)

# --- The measurements ----------------------------------------------------------------------------------

SPREAD_STATISTIC = "kept share on the missed op's removal lines"
"""What H1 is about. The anchor term acts on the alignment of a state with the anchored subspace; the kept
share is taken after training, on held-out lines, through a projection the term never sees, so its spread
across seeds is not a quantity the treatment optimizes. The line margin is the manipulation check and is
reported beside it rather than gated."""

SD_RATIO_GATE = 0.5
"""H1 holds when the across-seed standard deviation of the kept share at the top of the ladder is at most
half of its value at the lowest rung, within a subspace. Half is the effect ex-2.2.12 saw at five seeds
(`plane-lam-0.2` at 0.036 against `plane` at 0.125, a ratio near a third) rounded back toward no effect. A
standard deviation from five draws has a 95% interval of about 0.6× to 2.9× its true value, so the gate
asks for less than the observation that motivated it."""

TREND_ALPHA = 0.05
"""The significance check on H1. Each run's absolute deviation from its own condition's median is regressed
on log λ_a within a subspace, and H1 wants a negative slope at this level. A trend contrast asks the
question H1 actually poses — does the spread shrink as the weight rises — where Levene's test would only
say the four levels differ somehow, and would count a spread that rose and fell as a pass."""

UCB_LEVEL = 0.95
"""The adoption rule's confidence level. A condition's kept share is summarized by the one-sided upper
confidence bound on its seed mean, Student's t at n − 1 degrees of freedom."""

MEAN_BAND = 0.05
"""H2's band: the seed-mean kept share on the missed op is called flat across the ladder when no level
differs from the reference by more than this. It is about the paired-difference resolution at twenty seeds
at the reference's spread, so H2 is a statement about what we can see rather than about what is there."""

WORST_OP_MARGIN = 0.03
"""H3 wants the worst of the ten other ops to fall by at least this much from its value at the reference,
measured in this experiment at twenty seeds rather than quoted from ex-2.2.12. The margin is about the
binomial noise floor on a bounded proportion at these line counts, so a drop smaller than it is not a drop
we can resolve."""

COST_STATISTICS = (
    "the task",
    "the line margin",
    "the lead at the embedding",
    "the contrast between red and non-red",
    "the grading r²",
    "the deficit on the non-red lines",
)
"""What a heavier anchor might spend, each against ex-2.2.11's gate: held-out expected exact match on the
worst op (`task`), the line margin (`m_line`), the lead at the embedding (`lead_emb`), the separation of the
red and non-red groups along the anchored subspace (`contrast`), the squared correlation of the α response
with redness (`r2_sim`), and the deficit on the non-red lines under the projection (`nonred_deficit`)."""

COST_LIST = ", ".join(COST_STATISTICS[:-1]) + f", and {COST_STATISTICS[-1]}"
"""The cost statistics as a prose list."""

ADOPTION = f"""\
A condition qualifies when the one-sided {UCB_LEVEL:.0%} upper confidence bound on its seed-mean kept share \
sits under the {KEPT_GATE:g} gate on `{MISSED_OP}` and on each of the ten other ops, {COST_LIST} are inside \
ex-2.2.11's gates on the seed mean, and its ᾱ at op1 stands in no higher ratio to the un-anchored baseline for \
its own subspace than the reference stands to the axis one. Among those that qualify, the one with the lowest \
upper bound on `{MISSED_OP}` is adopted, and the smaller λ_a breaks a tie. The rule sees the measurements it \
names; if a qualifying condition looks harmful on something it does not name, the report says what and keeps \
the reference, and that override is marked as a decision made after the data. As in ex-2.2.12, this experiment \
proposes: the adopted recipe is confirmed at fresh seeds by the anchored-op experiment that inherits it, and \
no number quoted here for it is free of the selection. If none qualifies, the recipe stays at `{REFERENCE}` \
and the report carries the ladder's best characterization of the leftover — its level, its spread, and which \
lines it is made of — for the anchored-op preregistration to treat as a bounded confound."""

OLD_BAND = 0.06
"""Ex-2.2.12's fixed band: the seed spread its reference showed at twenty seeds."""

OLD_RULE = f"""\
Ex-2.2.12 asked instead for a seed mean under the gate by at least a fixed band of {OLD_BAND:g}, the spread its \
reference showed at twenty seeds. That band is a statement about one condition's resolution, and applying it \
to a condition three times tighter charges it for a spread it does not have. The upper bound above asks the \
same question — is this condition's leftover under the gate, or does the seed spread leave that unsettled — \
of each condition at its own precision. It is the stricter rule on a wide condition and the looser one on a \
tight one, and it is fixed here before the runs exist. Both verdicts are reported."""


# =============================================================================================
# The DAG
# =============================================================================================


def _load_ex2212():
    """Ex-2.2.12's module (which loads ex-2.2.11's, ex-2.2.9's, and ex-2.2.3's the same way), by path and
    left out of `sys.modules`, so the task bodies here still cloudpickle by value for a remote worker.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "ex-2.2.12" / "experiment.py"
    spec = importlib.util.spec_from_file_location("ex2212", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex2212 = _load_ex2212()
ex2211 = ex2212.ex2211
ex229 = ex2211.ex229

# --- What stays as the earlier experiments had it ----------------------------------------------
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
ROUNDING = ex2211.ROUNDING
PER_SLOT_RATE = ex2211.PER_SLOT_RATE
RED_RATE = ex2211.RED_RATE
ANCHOR_AXIS = ex2211.ANCHOR_AXIS
TAU = ex2211.TAU
N_EMBD = ex229.N_EMBD
N_LAYER = ex229.N_LAYER
WHOLE_SPAN = ex229.WHOLE_SPAN
ANSWER_POS = ex229.ANSWER_POS
OPERAND_POSITIONS = ex229.OPERAND_POSITIONS
GRID_RGB = ex229.GRID_RGB
REDNESS = ex229.REDNESS
"""The grammar, the corpus, the probe sets, the operators, and everything in the recipe but λ_a: as
ex-2.2.11 froze them. Only the anchor weight, the home of *red*, and the seed offset move here."""

TASK_GATE = ex2211.TASK_GATE
REF_M_LINE = ex2211.REF_M_LINE
MARGIN_RATIO = ex2211.MARGIN_RATIO
REF_R2_SIM = ex2211.REF_R2_SIM
GRADE_R2_RATIO = ex2211.GRADE_R2_RATIO
LEAD_GATE = ex2211.LEAD_GATE
CONTRAST_GATE = ex2211.CONTRAST_GATE
RED_KEPT_GATE = ex2211.RED_KEPT_GATE
NONRED_DEFICIT_GATE = ex2211.NONRED_DEFICIT_GATE
"""Every gate the hypotheses and the adoption rule name, at ex-2.2.11's value."""

assert RED_KEPT_GATE == KEPT_GATE, f"ex-2.2.11's kept gate moved: {RED_KEPT_GATE} != {KEPT_GATE}"
assert PLANE_EXPERIMENT.endswith("ex-2.2.12") and REFERENCE_EXPERIMENT.endswith("ex-2.2.11")
assert MISSED_OP in OP_NAMES

PLANE_AXES = ex2212.PLANE_AXES
"""The home of *red* under the plane conditions, as ex-2.2.12 defined it: e₁ and e₂ together."""

_npz = ex229._npz
_load = ex229._load
_probe_ops = ex229._probe_ops
_make_config = ex229._make_config
_slim = ex229._slim
schedules = ex229.schedules
Condition229 = ex229.Condition
HueReadout = ex2211.HueReadout
ex2211_prepare_corpus = ex2211.prepare_corpus
ex2211_train_one = ex2211.train_one
ex2212_eval_one = ex2212.eval_one
ex2212_score_one = ex2212.score_one

# --- Ex-2.2.11's stored control -----------------------------------------------------------------

EX2211_PROBE_REF = ex2211.PROBE_REF
EX2211_TRAJ_REF = ex2211.TRAJ_REF
EX2211_METRICS_REF = ex2211.METRICS_REF
EX2211_CHECKPOINT_REF = ex2211.CHECKPOINT_REF
EX2212_METRICS_REF = ex2212.METRICS_REF
"""Ex-2.2.12's numbers, printed beside the four conditions of the ladder its sweep already saw."""

CONTROL_SEEDS = 5
"""How many `control` runs ex-2.2.11 trained, at model seeds 100–104. Both of the control's roles here — the
task reference and the ᾱ baseline — compare seed means, which is what absorbs the unpaired seed sets."""

# --- Refs ---------------------------------------------------------------------------------------

METRICS_REF = "reports/m2/ex-2.2.13/metrics"
TRAJ_REF = "reports/m2/ex-2.2.13/trajectories"
CHECKPOINT_REF = "reports/m2/ex-2.2.13/checkpoints/{label}"
RUN_ARRAYS_REF = "reports/m2/ex-2.2.13/arrays/{label}/{kind}"
"""Each run's own per-line arrays, keyed by label and by `eval` or `score`. There is no stacked file: what
the report reads per line is small enough to ride in the metrics, and a stacked copy of 165 runs would be a
few hundred megabytes for a report to fetch on every render."""


# --- The ladder -----------------------------------------------------------------------------------


def axes_of(c: Condition) -> tuple[int, ...]:
    """Where *red* lives under *c*: the first axis, or the plane."""
    return PLANE_AXES if c.subspace == "plane" else (ANCHOR_AXIS,)


def prepare_corpus(*args) -> dict:
    """Ex-2.2.11's corpus, eval sets, and probe set, unchanged, under this experiment's memo."""
    return ex2211_prepare_corpus(*args)


def cells(conds: tuple[Condition, ...], prep: dict) -> list[dict]:
    """One row per run of the ladder: ex-2.2.11's handover row with the rung's λ_a and home of *red*.

    Everything the rung does not name comes from ex-2.2.9's `Condition` defaults, so the anti-subspace
    schedule keeps its ratios to λ_a and climbs the ladder with the pull.
    """
    from sca.utils import align

    tc = prep["meta"].tokenizer_config
    rows = []
    for c in conds:
        base = Condition229(
            c.name,
            SEEDS,
            f"λ_a {c.lam:g}, red anchored to {'a plane' if c.subspace == 'plane' else 'an axis'}",
            lam=c.lam,
            tau=TAU,
            epochs=EPOCHS,
            ops=OP_NAMES,
            n_lines=N_LINES,
        )
        anchor, anti = schedules(base)
        anchor = anchor | {"span": WHOLE_SPAN, "axes": axes_of(c)}
        for seed in range(SEEDS):
            config = _make_config(align(tc.vocab_size, 64), SEED_OFFSET + seed, EPOCHS, N_EMBD, N_LAYER)
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
    """Ex-2.2.11's training step, with no trajectory checkpoints: the ladder proposes, and a dynamics
    measurement would run on the checkpoints of whatever inherits the recipe.
    """
    return ex2211_train_one(config, anchor, anti, corpus, traj_stride, probes, keying, False, label)


def eval_one(trained: dict, evals, probes, tau: float, keying, condition: str, seed: int, label: str, axes) -> dict:
    """Ex-2.2.12's eval, plus ᾱ at op1 per palette color for the exploratory grading clouds.

    The per-color vector is the alignment at the first operand, averaged over slices, one value per grid
    color. It rides in the record rather than in an array file so the report reads it from the metrics.
    """
    from mini.store import get

    out = ex2212_eval_one(trained, evals, probes, tau, keying, condition, seed, label, tuple(axes))
    workdir = get_data_dir() / "eval" / label
    with np.load(get(out["arrays"], workdir / "alpha.npz")) as z:
        alpha = z[f"{PRIMARY_OP}/alpha"]  # (slices, colors, positions)
    return out | {"alpha_op1_by_color": alpha[:, :, 0].mean(axis=0).astype(float).tolist()}


# --- The achromatic candidate ---------------------------------------------------------------------


def gray_of() -> np.ndarray:
    """Per grid color, the index of the gray with the same value (the largest of its three channels).

    The grid's levels are the same set on every channel, so the gray of a color's value is itself a grid
    color and the map is exact.
    """
    value = GRID_RGB.max(axis=1)
    grays = {tuple(np.round(c, 6)): i for i, c in enumerate(GRID_RGB) if c[0] == c[1] == c[2]}
    return np.array([grays[(round(v, 6),) * 3] for v in value], dtype=np.int64)


def achromatic_one(model, read, sub, color_ids, tok2color) -> dict:
    """The missed op's removal lines with the second operand replaced by a gray of the same value.

    Four passes, all read against the unedited line's own answer: clean, the projection at the blocks,
    the gray substitution on its own, and the two together. If the surviving answers are read off how
    achromatic the second operand looks, graying it keeps them; if they need the operand's own color, it
    does not. Exploratory, so no gate is scored on it.
    """
    from sca.intervention import apply, projection

    tokens = read.tokens
    every = tuple(range(len(model.transformer.blocks) + 1))
    blocks = every[1:]
    op2 = OPERAND_POSITIONS[1]
    gray = np.asarray(color_ids)[gray_of()[tok2color[tokens[:, op2]]]]
    edited = tokens.copy()
    edited[:, op2] = gray

    passes = {
        "clean": (tokens, ()),
        "blocks": (tokens, blocks),
        "gray": (edited, ()),
        "gray-blocks": (edited, blocks),
    }
    eem = {k: read(apply(model, t, projection(sub), slices=s).logits)["eem"] for k, (t, s) in passes.items()}
    base = eem["clean"]
    return {
        "n": {g: int(m.sum()) for g, m in read.groups.items()},
        "eem": {k: read.by_group(v) for k, v in eem.items()},
        "kept": {k: read.ratio_by_group(v, base) for k, v in eem.items()},
        "gray_is_self": float((gray == tokens[:, op2]).mean()),
    }


def score_one(trained: dict, probes, operators: tuple[str, ...], condition: str, seed: int, label: str, axes) -> dict:
    """Ex-2.2.12's scoring, plus the two exploratory passes on the missed op.

    The first is per-line: the kept share of every one of the missed op's removal lines, which says whether a
    narrow seed spread means the same lines survive each time. The second is the achromatic edit. Both ride
    in the record; the per-line vectors are a few hundred numbers a run.
    """
    from sca.intervention import Subspace
    from mini.store import get

    axes = tuple(axes)
    out = ex2212_score_one(trained, probes, operators, condition, seed, label, axes)
    workdir = get_data_dir() / "score" / label
    model, _, color_ids, tok2color = _load(trained, workdir)
    width = model.transformer.wte.shape[1]
    sub = Subspace.axis(width, axes[0]) if len(axes) == 1 else Subspace.axes(width, axes)
    keys = ("tokens", "r1", "r2", "q_idx", "q_p", "move", "hue_move")
    with np.load(get(probes, workdir / "probes.npz")) as z:
        f = {k: z[f"{MISSED_OP}/{k}"] for k in keys}
    read = HueReadout(
        f["tokens"], f["r1"], f["r2"], f["q_idx"], f["q_p"], f["hue_move"], f["move"], color_ids, tok2color
    )
    removal = read.groups["removal"]

    with np.load(get(out["arrays"], workdir / "scored.npz")) as z:
        clean, proj = z[f"{MISSED_OP}/clean/eem"], z[f"{MISSED_OP}/projection/eem"]
    with np.errstate(divide="ignore", invalid="ignore"):
        kept_line = np.where(clean > 0, proj / clean, np.nan)
    return out | {
        "missed_lines": {
            "index": np.flatnonzero(removal).tolist(),
            "op2_color": tok2color[f["tokens"][removal, OPERAND_POSITIONS[1]]].tolist(),
            "kept": np.round(kept_line[removal], 5).tolist(),
            "clean": np.round(clean[removal], 5).tolist(),
        },
        "achromatic": achromatic_one(model, read, sub, color_ids, tok2color),
    }


# --- Publishing ------------------------------------------------------------------------------


def design() -> dict[str, Any]:
    """Ex-2.2.11's design record with what the ladder adds."""
    return ex2211.design() | {
        "experiment": "ex-2.2.13",
        "reference": {"experiment": REFERENCE_EXPERIMENT, "plane": PLANE_EXPERIMENT, "condition": REFERENCE},
        "n_runs": N_RUNS,
        "seeds": SEEDS,
        "seed_offset": SEED_OFFSET,
        "ladder": list(LADDER),
        "subspaces": list(SUBSPACES),
        "conditions": [asdict(c) | {"axes": list(axes_of(c))} for c in GRID],
        "plane_axes": list(PLANE_AXES),
        "control": {"condition": CONTROL, "plane": CONTROL_PLANE, "seeds": CONTROL_SEEDS},
        "missed_op": MISSED_OP,
        "kept_gate": KEPT_GATE,
        "sd_ratio_gate": SD_RATIO_GATE,
        "trend_alpha": TREND_ALPHA,
        "ucb_level": UCB_LEVEL,
        "mean_band": MEAN_BAND,
        "worst_op_margin": WORST_OP_MARGIN,
        "old_band": OLD_BAND,
        "adoption": ADOPTION,
        "old_rule": OLD_RULE,
    }


def publish_results(trained: list[dict], evaled: list[dict], scored: list[dict], corpus: dict) -> dict:
    """Metrics (JSON), trajectories, each run's own arrays, and every end checkpoint, each under its ref."""
    import json

    from mini.store import put, set_ref

    metrics = {
        "runs": [_slim(r) for r in evaled],
        "scores": [_slim(r) for r in scored],
        "corpora": [corpus["stats"]],
        "design": design(),
    }
    set_ref(METRICS_REF, put(json.dumps(metrics).encode(), name="ex-2.2.13-metrics.json"))
    traj = {t["label"]: {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.13-trajectories.json"))
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    for r in evaled + scored:
        kind = "eval" if "per_op" in r else "score"
        set_ref(RUN_ARRAYS_REF.format(label=r["label"], kind=kind), r["arrays"])
    return {
        "n_runs": len(trained),
        "n_evaled": len(evaled),
        "n_scored": len(scored),
        "holdout_eem": {r["label"]: r["holdout_eem"] for r in evaled},
    }


# --- Orchestration ----------------------------------------------------------------------------


def resolve_inputs(control: list[str]) -> dict:
    """Ex-2.2.11's control checkpoints and trajectories, by ref, so the scoring pass on them is keyed on
    the content it scores.
    """
    import json

    from mini.store import get, get_refs

    names = [EX2211_TRAJ_REF, *(EX2211_CHECKPOINT_REF.format(label=lb) for lb in control)]
    refs = get_refs(names)
    missing = [k for k, v in refs.items() if v is None]
    assert not missing, f"ex-2.2.11 refs not in this store: {missing}"
    traj_art = refs[EX2211_TRAJ_REF]
    assert traj_art is not None
    traj = json.loads(get(traj_art, get_data_dir() / "control" / "trajectories.json").read_bytes())
    return {
        "checkpoints": {lb: refs[EX2211_CHECKPOINT_REF.format(label=lb)] for lb in control},
        "trajectories": {lb: traj[lb]["traj"] for lb in control},
    }


def main(ctx: Ctx) -> dict:
    control_labels = [f"{CONTROL}-s{s}" for s in range(CONTROL_SEEDS)]
    inputs = ctx.run(resolve_inputs, control_labels, role="prep")
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
    rows = cells(GRID, prep)
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
    # The stored control checkpoints, scored on the plane, ride along as five more rows.
    ctrl = [
        {"checkpoint": inputs["checkpoints"][lb], "traj": inputs["trajectories"][lb], "label": lb}
        for lb in control_labels
    ]
    ctrl_rows = [
        {"condition": CONTROL_PLANE, "seed": s, "label": f"{CONTROL_PLANE}-s{s}", "axes": PLANE_AXES, "keying": "line"}
        for s in range(CONTROL_SEEDS)
    ]
    all_rows, all_trained = rows + ctrl_rows, trained + ctrl
    m = len(all_rows)
    evaled = ctx.map(
        eval_one,
        all_trained,
        [prep["evals"]] * m,
        [prep["probes"]] * m,
        [TAU] * m,
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
    return ctx.run(publish_results, trained, evaled, scored, prep, role="prep")


experiment = Experiment(
    name="ex-2.2.13",
    main=main,
    roles={
        # Ex-2.2.11's corpus build, and the fan-in that writes the refs of 165 runs.
        "prep": dict(cpu=2, timeout=5400),
        # 4,950 steps at L4; the watchdog covers the checkpoint upload.
        "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
        "eval": dict(gpu="L4", timeout=1800),
        # Three operators on eleven ops, plus the per-line pass and the achromatic edit on the missed op.
        "score": dict(gpu="L4", timeout=2400),
    },
)
