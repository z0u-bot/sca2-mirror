"""Ex-2.2.14: anchoring an operation — the smoke test.

The first anchored-op experiment of D2.2. One operation of table A+ is anchored to e₁ on the handover setup,
with no *red* anchor, at a few seeds, and scored against the alignment and task gates alone. It says whether
anchoring an op works at all before the many-seed equivalence read spends its budget, and it fixes which op
that experiment anchors. The other ten ops ride along at fewer seeds as a description, not a test.

The design constants come first; the DAG follows, binding everything it does not change from ex-2.2.11's
module (the grammar, the corpus, the recipe, the probe sets, the retention read).

    bin/mini run docs/m2/ex-2.2.14/experiment.py --app modal --max-containers 8 --budget 3h
    bin/mini status ex-2.2.14
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
"""The grammar (table A+), the stochastic corpus at 300k lines, the whole-line span, the untied readout, the
recipe (λ_a 0.1 annealed to a 0.1 floor over the last tenth of training, τ 0.1, the anti-subspace weight from
2.5× the anchor weight to 0.3× by 90% of training), 50 epochs at d64-L4, and the task gate: all as ex-2.2.11
froze them and ex-2.2.13 confirmed at fresh seeds. Ex-2.2.13's ladder left the weight where it was, so the
recipe of record is `axis-0.1`, which is ex-2.2.11's `handover`."""

CONTROL_EXPERIMENT = "m2/ex-2.2.11"
CONTROL = "control"
CONTROL_SEEDS = 5
"""The un-anchored control: ex-2.2.11's `control` checkpoints at model seeds 100–104, served from the store.
It is the task reference and the baseline for every alignment measurement. Its labeller differs from this
experiment's (the control drew red labels it never used), so the comparison carries corpus-draw noise on
top of seed noise, as every cross-labeller comparison since ex-2.1.10 has."""

# --- The anchored op -----------------------------------------------------------------------------------

ANCHORED_OP = "difference"
"""The op anchored to e₁. Chosen on paper from table A+ before any run, by three criteria the design named
and one the calibration look added. Op-relevance: on 75% of its lines its answer names it alone, the most
of any commutative op in the table (`mix` 63%, `hsvmix` 59%). Rounding: it is total on the grid, so every
line has one answer and the task score is not capped by stochastic rounding (the control learns it to 0.97
held-out expected exact match against 0.43 on `mix`). Order: it is commutative, so nothing in its measurement
depends on which operand is which. The sweep below reads the other ten ops the same way so the choice can
be revisited on data rather than on paper."""

SHORT_NAMES = {
    "difference": "diff",
    "multiply": "mult",
    "exclusion": "excl",
    "hue-hsv": "hue",
    "sat-hsv": "sat",
    "value-hsv": "val",
}
"""Condition names carry the op's short name, so that `anchor-diff-opword` fits a table cell."""


def short(op: str) -> str:
    return SHORT_NAMES.get(op, op)


ANCHOR_AXIS = 0
"""e₁. The op takes the axis *red* had; there is no *red* anchor in this experiment. A model that carries
both is the follow-up once anchoring an op alone is understood."""

KEYING = "op"
"""A new labeller keying: a line draws a label when its op word is the anchored op, at `LABEL_RATE`. The
pull covers the whole line, as the handover's `line` keying does, so the label never says which position
carried it (the position-free mechanism of ex-2.1.10). The library today keys on the operands and the
answer only, so this keying is the one engineering change the experiment needs."""

LABEL_RATE = 0.02
"""The anchored op's lines draw at this rate, so about `RED_LABEL_SHARE` of the corpus is labelled, red's
share, and each labelled line gets the pull a red line got (the anchor term normalizes by the mask's own
weight). The rate is the primary rather than a pull on every line because the labels the method will have
further up the ladder — an abstract concept in natural language — will be scarce, so the scarce case is the
one the hypotheses should be scored on. The every-line arm below brackets it (REVIEW: the two swapped
places at the human's call; the rate is the more realistic labeller)."""

RED_LABEL_SHARE = 0.0018
"""The share of lines the red labeller labels under `line` keying: three draws per line at redness⁸ × 0.04
per token, averaged over the grid. Computed from the palette, not measured."""

FULL_RATE = 1.0
"""The arm in which every line of the anchored op draws: about a tenth of the corpus, fifty times red's
share. The term's size per batch is unchanged and each labelled line gets a proportionally weaker pull. If
it and the primary agree on the margin, the label share is not what sets it; if they part, the ratio of
their margins is what pulling every line of a categorical concept buys."""

LABEL_PRECISION = 0.8
NOISY_TRUE_RATE = LABEL_RATE * LABEL_PRECISION
FALSE_RATE = LABEL_RATE * (1 - LABEL_PRECISION) / 10
"""The arm with erroneous labels: the anchored op's lines draw at `NOISY_TRUE_RATE` and every other op's
lines at `FALSE_RATE`, so the labelled share of the corpus is the primary's and a fifth of the labels land
on lines that do not carry the op (REVIEW: was a fifth added on top of the primary's labels, which would
have raised the total pull as well; the wrong labels now replace true ones). A labeller for an abstract
concept in natural language will be wrong some of the time, and this arm says what a fifth of wrong labels
costs the margin and the task. The precision is a round number, not a measurement of any labeller."""

OPWORD_SPAN = 2
"""The arm that pulls the op word alone, at the primary's rate: the pull covers position 1 only (a span of
two tokens minus the first operand, in the library's role terms, or a `slot` pull at role 1 once the keying
exists). Under it the use sites are outside the mask, so a contrast at `=` or the answer at the final slice
is carried there by the blocks rather than asked for by the pull. H3 is scored on this arm for that reason;
the primary's use-site contrast is optimized by construction and is reported beside it as a manipulation
check. One position instead of six also makes each pull about six times stronger under the mask normalizer
(the ex-2.1.7 lesson), which the arm's op-position contrast beside the primary's will show."""

WHOLE_SPAN = 6
"""The pull covers all six positions of a labelled line."""

# --- The runs ------------------------------------------------------------------------------------------

SEED_OFFSET = 300
"""Condition seed *i* trains at model seed `SEED_OFFSET + i`. Ex-2.2.13 trained at offset 200 and ex-2.2.11
and ex-2.2.12 at 100; every seed here is fresh."""

SEEDS = 5
"""Seeds for the primary condition. The smoke test asks whether the op lands and the task survives, which
five seeds resolve at the spreads ex-2.2.11 saw (the line margin's standard deviation across seeds was
0.016, the task's 0.005 on the worst op). The equivalence read that follows sets its own count."""

SWEEP_SEEDS = 3
"""Seeds for each of the other ten ops. The sweep describes; it decides nothing unless the primary misses."""

PRIMARY = f"anchor-{short(ANCHORED_OP)}"
FULL_ARM = f"{PRIMARY}-full"
OPWORD_ARM = f"{PRIMARY}-opword"
NOISY_ARM = f"{PRIMARY}-noisy"
ARMS = (FULL_ARM, OPWORD_ARM, NOISY_ARM)
"""Three arms at the primary's five seeds. H3 is scored on the op-word arm; the other two are outside the
hypotheses and the rule."""
SWEEP = tuple(
    f"anchor-{short(op)}"
    for op in (
        "mix",
        "screen",
        "multiply",
        "lighten",
        "darken",
        "exclusion",
        "hsvmix",
        "hue-hsv",
        "sat-hsv",
        "value-hsv",
    )
)
CONDITIONS = (PRIMARY, *ARMS, *SWEEP)
N_RUNS = SEEDS * (1 + len(ARMS)) + SWEEP_SEEDS * len(SWEEP)
assert N_RUNS == 50

# --- The measurements ----------------------------------------------------------------------------------

OP_POSITION = 1
EQUALS_POSITION = 3
ANSWER_POSITION = 4
"""Where in the six-token line the op word, `=`, and the answer sit."""

TASK_GATE = 0.02
TASK_PARTIAL = 0.05
"""H1: for each op, the primary's seed-mean held-out expected exact match is within `TASK_GATE` of the
control's seed mean; partial when every op is within `TASK_PARTIAL`. Ex-2.2.11's gate, unchanged."""

REF_M_LINE = 0.4286
"""Ex-2.2.11's `handover` line margin on the `mix` lines at twenty seeds: what *red* achieved under the same
recipe. The op margin is read against it."""

MARGIN_RATIO = 0.8
MARGIN_PARTIAL = 0.6
"""H2: the op margin — ex-2.2.3's `line_margin` with uniform line weights summing to one over the anchored op's
lines and zero elsewhere (the einsum needs a normalized weight): per slice, the labelled mean alignment minus the mean over all lines at the largest span role,
averaged over every slice — is at least `MARGIN_RATIO` of `REF_M_LINE` on the seed mean; partial from
`MARGIN_PARTIAL`. The same bar the handover set against ex-2.1.10."""

RETENTION_FLOOR = 0.2
RETENTION_GATE = 0.8
"""H2 (retention): every primary run whose op margin reaches `RETENTION_FLOOR` at the start of the anchor
weight's anneal ends training at `RETENTION_GATE` of that value. Ex-2.2.11's rule and denominator (REVIEW:
the draft said "of its peak", which is the rule ex-2.2.10 replaced; a noisy plateau's high point is not a
level to hold). The learning rate is low over the anneal, so a tighter share would be defensible; one
`handover` seed in twenty ended under 0.8 in ex-2.2.11, so the bar stays where a known failure sits and
the per-seed values are reported for the equivalence experiment to set its own."""

USE_CONTRAST_FLOOR = 0.05
"""H3, scored on the op-word arm, a prediction with no decision hanging on it: some of the op reaches the
use sites. At `=` and at the answer position the embedding is the same token on every line, so any
difference in alignment between the anchored op's lines and the others at slices 1..L came through the
blocks, and under the op-word pull nothing asked for it. H3 holds when the arm's seed-mean contrast (mean
cosine on the anchored op's lines minus mean cosine on the other ops' lines) at the final slice, at each of
the two use sites, is at least `USE_CONTRAST_FLOOR` above the control's. The ratio to the contrast at the op
position is reported with no bar (REVIEW: the draft gated the ratio at 0.5, a number with no observation
behind it; the model has so far been seen to put the answer color at the use sites, not a copy of the op).
The primary's use-site contrast is reported beside it as a manipulation check."""

CONTAINMENT_REF = 0.264
"""Ex-2.2.11's ᾱ at op1 on `handover` at twenty seeds: the mean alignment over every color at the first
operand, the containment statistic that rose under the untied readout and has no named mechanism. Its
categorical analogue here — the mean alignment at the op position over the other ten ops' lines — is
reported beside it and beside the control with no gate, as the open item asks."""

PROBE_RIDGE = 1e-2
"""The ridge strength of every probe scan since ex-2.1.5. The op-identity scan fits a one-hot ridge probe
per site and reports its held-out R², anchored against control. Exploratory here: it sizes the margin the
equivalence read declares."""

# --- The rule for the follow-up ------------------------------------------------------------------------

# REVIEW: the sweep fallback checked only the task gate and ranked by margin, so it could adopt an op that
# misses the H2 bar or retention; it now carries every gate the primary does, and "passes" excludes partials.
ADOPTION = f"""\
The equivalence experiment anchors `{ANCHORED_OP}` if the primary passes H1 and H2 in full (a partial \
counts as a miss), margin and retention both. If it does not, it anchors the op in the sweep with the \
largest seed-mean op margin among those that pass the same gates at their three seeds: every op inside the \
task gate, the margin at or above the H2 bar, and every run clearing the retention rule, and that op is \
confirmed at the equivalence experiment's own seeds before any number is quoted for it. If no op qualifies, \
the anchored-op line stops here and the report says what gave way. H3, the arms, and the containment \
measure do not enter the rule: they describe what the anchor did, and the equivalence experiment measures \
them at its own seeds whichever way they came out."""


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

OP_NAMES: tuple[str, ...] = ex2211.OP_NAMES
ORDER_SENSITIVE: tuple[str, ...] = ex2211.ORDER_SENSITIVE
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
ROUNDING = ex2211.ROUNDING
PER_SLOT_RATE = ex2211.PER_SLOT_RATE
RED_RATE = ex2211.RED_RATE
TRAJ_STRIDE = ex2211.TRAJ_STRIDE
LAM = ex2211.LAM
TAU = ex2211.TAU
ANNEAL_WEIGHT_RATIO = ex2211.ANNEAL_WEIGHT_RATIO
N_EMBD = ex229.N_EMBD
N_LAYER = ex229.N_LAYER
"""The grammar, the corpus, the probe sets, and the recipe: as ex-2.2.11 froze them. Only the labeller, the
anchored concept, and the seed offset move here."""

assert ex2211.TASK_GATE == TASK_GATE and ex2211.TASK_PARTIAL == TASK_PARTIAL, "ex-2.2.11's task gate moved"
assert ex2211.RETENTION_FLOOR == RETENTION_FLOOR and ex2211.RETENTION_GATE == RETENTION_GATE
assert ex2211.MARGIN_RATIO == MARGIN_RATIO and ex2211.MARGIN_PARTIAL == MARGIN_PARTIAL
assert ex229.WHOLE_SPAN == WHOLE_SPAN and ex2211.ANCHOR_AXIS == ANCHOR_AXIS
assert ex229.ANSWER_POS == ANSWER_POSITION
assert REFERENCE_EXPERIMENT.endswith("ex-2.2.11") and CONTROL_EXPERIMENT.endswith("ex-2.2.11")

SWEEP_OPS: tuple[str, ...] = tuple(o for o in OP_NAMES if o != ANCHORED_OP)
"""The ten other ops of table A+, in table order."""

assert ANCHORED_OP in OP_NAMES and len(SWEEP_OPS) == 10
assert set(SWEEP) == {f"anchor-{short(o)}" for o in SWEEP_OPS}, "the sweep's names and the table disagree"

_npz = ex229._npz
_load = ex229._load
_make_config = ex229._make_config
_answer_logprobs = ex229._answer_logprobs
_stochastic_reads = ex229._stochastic_reads
_slim = ex229._slim
line_margin = ex229.line_margin
schedules = ex229.schedules
Condition229 = ex229.Condition
anneal_retention = ex2211.anneal_retention
ex229_prepare_corpus = ex229.prepare_corpus

# --- Conditions -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One training condition: which op is anchored, the labeller's rate table, and the pull."""

    name: str
    op: str
    seeds: int
    rate: float
    """The anchored op's per-line label rate."""
    false_rate: float = 0.0
    """Each other op's per-line label rate: zero but on the noisy arm."""
    pull: str = "span"
    """`span` covers the whole line; `slot` the op word alone (the op-word arm)."""
    span: int = WHOLE_SPAN


GRID: tuple[Arm, ...] = (
    Arm(PRIMARY, ANCHORED_OP, SEEDS, LABEL_RATE),
    Arm(FULL_ARM, ANCHORED_OP, SEEDS, FULL_RATE),
    Arm(OPWORD_ARM, ANCHORED_OP, SEEDS, LABEL_RATE, pull="slot", span=OPWORD_SPAN),
    Arm(NOISY_ARM, ANCHORED_OP, SEEDS, NOISY_TRUE_RATE, false_rate=FALSE_RATE),
    *(Arm(f"anchor-{short(o)}", o, SWEEP_SEEDS, LABEL_RATE) for o in SWEEP_OPS),
)
assert tuple(a.name for a in GRID) == CONDITIONS and sum(a.seeds for a in GRID) == N_RUNS


N_SCAN = 512
"""Probe lines per op for the op-identity scan, drawn once with `SCAN_SEED`; a fifth of them (`SCAN_HOLDOUT`),
split by line, are held out for the R²."""
SCAN_SEED = 14
SCAN_HOLDOUT = 0.2

# --- Refs -----------------------------------------------------------------------------------------

EX2211_CHECKPOINT_REF = ex2211.CHECKPOINT_REF
METRICS_REF = "reports/m2/ex-2.2.14/metrics"
TRAJ_REF = "reports/m2/ex-2.2.14/trajectories"
CHECKPOINT_REF = "reports/m2/ex-2.2.14/checkpoints/{label}"


def rates_of(arm: Arm) -> dict[str, float]:
    """The labeller's per-op rate table for *arm*."""
    return {o: (arm.rate if o == arm.op else arm.false_rate) for o in OP_NAMES}


def prepare_corpus(*args) -> dict:
    """Ex-2.2.9's corpus, eval sets, and probe set (the corpus ex-2.2.11 trained on), under this memo."""
    return ex229_prepare_corpus(*args)


def cells(arms: tuple[Arm, ...], prep: dict, epochs: int = EPOCHS) -> list[dict]:
    """One row per run: ex-2.2.11's handover row with the op labeller and this experiment's seeds."""
    from sca.utils import align

    tc = prep["meta"].tokenizer_config
    rows = []
    for a in arms:
        base = Condition229(a.name, a.seeds, a.name, lam=LAM, tau=TAU, epochs=epochs, ops=OP_NAMES, n_lines=N_LINES)
        anchor, anti = schedules(base)
        anchor = anchor | {"span": a.span}
        for seed in range(a.seeds):
            config = _make_config(align(tc.vocab_size, 64), SEED_OFFSET + seed, epochs, N_EMBD, N_LAYER)
            config.tokenizer = tc.model_copy()
            config.model = config.model.model_copy(update={"tie_embeddings": False})
            rows.append(
                {
                    "config": config,
                    "anchor": anchor,
                    "anti": anti,
                    "op": a.op,
                    "rates": rates_of(a),
                    "pull": a.pull,
                    "condition": a.name,
                    "seed": seed,
                    "model_seed": SEED_OFFSET + seed,
                    "label": f"{a.name}-s{seed}",
                }
            )
    return rows


def probe_walk(z, op: str) -> np.ndarray:
    """The op's probe lines with the walked color as op1: every op contributes the same count, so the pool
    over ops weighs each op equally and the anchored op is one eleventh of it.
    """
    tokens, walk = z[f"{op}/tokens"], z[f"{op}/walk"]
    return tokens[walk == 0]


def traj_probe(z, op: str) -> tuple[np.ndarray, np.ndarray]:
    """The trajectory's probe lines (one per palette color per op, every op: 216 lines an op, 2,376 in all, as
    ex-2.2.11's trajectory read one per color on `mix`; the end-of-training reads take every probe line) and
    the op margin's line weights on them: uniform over *op*'s lines, summing to one, zero elsewhere.
    """
    blocks = [probe_walk(z, o)[::N_PROBE] for o in OP_NAMES]
    w = np.concatenate([np.full(len(b), 1.0 if o == op else 0.0) for o, b in zip(OP_NAMES, blocks, strict=True)])
    return np.concatenate(blocks), w / w.sum()


def label_share(train_data, config, spec, span: int) -> dict[str, float]:
    """The share of line visits labelled, and of live positions pulled, over the first epoch's crops.

    Replays the sampler on the training run's own stream: `train_anchored` draws nothing from it before the
    first epoch's batches, so these are the draws that run saw.
    """
    from sca.anchoring import sample_anchored_batches
    from sca.data.batches import batches_per_epoch

    n = batches_per_epoch(len(train_data), config.data, config.model)
    rng = np.random.default_rng(config.seed)
    lines = labelled = pulled = live = 0
    for x, _, mask, local in sample_anchored_batches(
        train_data, config.data, config.model, n, rng, spec, span, lines=True
    ):
        seen = np.zeros((len(x), int(local.max()) + 1), bool)
        hit = np.zeros_like(seen)
        rows = np.arange(len(x))[:, None].repeat(x.shape[1], 1)
        seen[rows[x != 0], local[x != 0]] = True
        hit[rows[mask > 0], local[mask > 0]] = True
        lines += int(seen.sum())
        labelled += int(hit.sum())
        pulled += int((mask > 0).sum())
        live += int((x != 0).sum())
    return {"lines": labelled / lines, "positions": pulled / live}


def train_one(config, anchor: dict, anti: dict, corpus, traj_stride: int, probes, op: str, rates, pull, label):
    """Train one run under the op labeller, recording the op margin on the trajectory probe lines."""
    from sca.anchoring import AnchorSpec, AntiSpec, LabelSpec
    from sca.compute.data_pipelines import load_data
    from sca.compute.training import train_anchored
    from sca.data.batches import split_data
    from sca.data.named_colors import WordTokenizer
    from mini.store import get, put

    workdir = get_data_dir() / "cells" / label
    corpus_dir = get(corpus, workdir / "corpus")
    tokenizer = WordTokenizer(config.tokenizer)
    p = np.zeros(config.model.vocab_size)
    for o, r in rates.items():
        p[tokenizer.stoi[o]] = r
    spec = LabelSpec(p=p, keying="op", pull=pull)
    with np.load(get(probes, workdir / "probes.npz")) as z:
        tokens, line_w = traj_probe(z, op)

    _, metrics, traj = train_anchored(
        config,
        corpus_dir,
        anchor=AnchorSpec(**anchor),
        anti=AntiSpec(**anti),
        label_p=spec,
        probe_tokens=tokens,
        probe_weights=line_w,
        probe_line_w=line_w,
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
    )
    data, _ = load_data(corpus_dir)
    train_data, _ = split_data(data, config.data.train_split)
    return {
        "label": label,
        "val_loss": [m.val_loss for m in metrics],
        "train_loss": [m.train_loss for m in metrics],
        "traj": {
            k: traj[k].tolist()
            for k in ("epoch", "m_line", "alpha_op1", "val_loss", "weight", "anti_weight")
            if k in traj
        },
        "label_share": label_share(train_data, config, spec, anchor["span"]),
        "checkpoint": put(workdir / "model", name=f"ex-2.2.14-{label}-ckpt"),
    }


def _behavior(model, tokenizer, lines_by_op: dict) -> dict[str, dict[str, dict]]:
    """Ex-2.2.9's behavior reads, per op and split: one teacher-forced pass read at the pre-answer position,
    against the answer distribution of every line.
    """
    from sca.data import ops as grammar

    by_name = grammar.OP_BY_NAME | grammar.CANDIDATE_BY_NAME
    palette_index = {c: i for i, c in enumerate(grammar.colors())}
    color_ids = np.array([tokenizer.stoi[n] for n in grammar.PALETTE])
    sets: dict[str, dict[str, dict]] = {}
    for op, splits in lines_by_op.items():
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
            sets[op][split] = {"n": len(lns), **{k: float(v.mean()) for k, v in line.items()}}
    return sets


def op_scan(model, lines: dict[str, np.ndarray], ridge: float, seed: int, holdout: float) -> np.ndarray:
    """Op-identity R² per (slice, position): a ridge probe from the state to the one-hot op word, fit on
    four fifths of the lines and scored on the rest, split by line.
    """
    import jax.numpy as jnp

    from sca.compute.evaluation import ridge_probe

    rng = np.random.default_rng(seed)
    tokens = np.concatenate(list(lines.values()))
    y = np.concatenate([np.full(len(v), i) for i, v in enumerate(lines.values())])
    onehot = np.eye(len(lines))[y]
    test = rng.random(len(tokens)) < holdout
    states = np.concatenate(
        [np.asarray(model.residual_stream(jnp.asarray(tokens[i : i + 512]))) for i in range(0, len(tokens), 512)],
        axis=1,
    )  # (L1, N, T, C)
    n_slices, _, n_pos, _ = states.shape
    r2 = np.zeros((n_slices, n_pos))
    for s in range(n_slices):
        for t in range(n_pos):
            x = states[s, :, t].astype(np.float64)
            r2[s, t] = ridge_probe(x[~test], onehot[~test], x[test], onehot[test], ridge)[2]
    return r2


def eval_one(trained: dict, evals, probes, condition: str, op: str | None, seed: int, label: str) -> dict:
    """The eval: behavior per op, the alignment per op's probe lines, the op margin for every op, retention
    across the anneal, and the op-identity scan.

    The alignment is kept as each op's mean cosine with e₁ per (slice, position): the op margin, the
    contrast, containment, and the alignment map are all contractions of it, since every op contributes the
    same number of probe lines. The op margin is also computed directly, with `line_margin` on the pooled lines.
    """
    from sca.anchoring import alignment
    from sca.data import ops as grammar
    from mini.store import get

    workdir = get_data_dir() / "eval" / label
    model, tokenizer, _, _ = _load(trained, workdir)
    sets = _behavior(model, tokenizer, grammar.load_lines(get(evals, workdir / "evals.json").read_bytes()))

    with np.load(get(probes, workdir / "probes.npz")) as z:
        lines = {o: probe_walk(z, o) for o in OP_NAMES}
    cos = {o: alignment(model, t) for o, t in lines.items()}  # each (L1, N, T)
    pooled = np.concatenate([cos[o] for o in OP_NAMES], axis=1)
    counts = [cos[o].shape[1] for o in OP_NAMES]
    assert len(set(counts)) == 1, f"probe walks differ in size: {counts}"

    def weights(anchored: str) -> np.ndarray:
        w = np.concatenate([np.full(n, 1.0 if o == anchored else 0.0) for o, n in zip(OP_NAMES, counts, strict=True)])
        return w / w.sum()

    op_margin = {o: line_margin(pooled, weights(o)) for o in OP_NAMES}
    rng = np.random.default_rng(SCAN_SEED)
    scan_lines = {o: t[np.sort(rng.choice(len(t), N_SCAN, replace=False))] for o, t in lines.items()}
    out = {
        "label": label,
        "condition": condition,
        "op": op,
        "seed": seed,
        "sets": sets,
        "holdout_eem": {o: s["holdout"]["eem"] for o, s in sets.items()},
        "cos_mean": {o: cos[o].mean(axis=1).tolist() for o in OP_NAMES},  # (L1, T) per op
        "op_margin": op_margin,
        "probe_r2": op_scan(model, scan_lines, PROBE_RIDGE, SCAN_SEED, SCAN_HOLDOUT).tolist(),
        "n_probe_lines": counts[0],
    }
    if "traj" in trained:
        out |= anneal_retention(trained["traj"], ANNEAL_WEIGHT_RATIO)
        out["label_share"] = trained["label_share"]
    return out


# --- Publishing ------------------------------------------------------------------------------


def design() -> dict[str, Any]:
    """The design constants the report reads beside the results."""
    return {
        "experiment": "ex-2.2.14",
        "reference": REFERENCE_EXPERIMENT,
        "control": {"experiment": CONTROL_EXPERIMENT, "condition": CONTROL, "seeds": CONTROL_SEEDS},
        "anchored_op": ANCHORED_OP,
        "keying": KEYING,
        "arms": [asdict(a) for a in GRID],
        "seed_offset": SEED_OFFSET,
        "n_runs": N_RUNS,
        "ops": list(OP_NAMES),
        "order_sensitive": list(ORDER_SENSITIVE),
        "task_gate": TASK_GATE,
        "ref_m_line": REF_M_LINE,
        "retention": {"floor": RETENTION_FLOOR, "gate": RETENTION_GATE, "weight_ratio": ANNEAL_WEIGHT_RATIO},
        "use_contrast_floor": USE_CONTRAST_FLOOR,
        "containment_ref": CONTAINMENT_REF,
        "probe_ridge": PROBE_RIDGE,
        "adoption": ADOPTION,
    }


def publish_results(trained: list[dict], evaled: list[dict], corpus_stats: dict) -> dict:
    """Metrics (JSON), trajectories (JSON), and every end checkpoint, each under its ref."""
    import json

    from mini.store import put, set_ref

    metrics = {"runs": [_slim(r) for r in evaled], "corpus": corpus_stats, "design": design()}
    set_ref(METRICS_REF, put(json.dumps(metrics).encode(), name="ex-2.2.14-metrics.json"))
    traj = {t["label"]: {k: t[k] for k in ("traj", "val_loss", "train_loss", "label_share")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.14-trajectories.json"))
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    return {"n_runs": len(trained), "n_evaled": len(evaled)}


# --- Orchestration ----------------------------------------------------------------------------


def resolve_control(labels: list[str], ref: str) -> dict:
    """Ex-2.2.11's control checkpoints, by ref, so the eval on them is keyed on the content it reads."""
    from mini.store import get_refs

    names = [ref.format(label=lb) for lb in labels]
    refs = get_refs(names)
    missing = [k for k, v in refs.items() if v is None]
    assert not missing, f"ex-2.2.11 refs not in this store: {missing}"
    return {lb: refs[n] for lb, n in zip(labels, names, strict=True)}


def run(ctx: Ctx, arms: tuple[Arm, ...], epochs: int, control_seeds: int) -> tuple[list[dict], list[dict], dict]:
    """Train *arms*, then evaluate them beside the served control: the whole DAG, with the grid and the length
    as arguments so a short prototype runs the same code.
    """
    control_labels = [f"{CONTROL}-s{s}" for s in range(control_seeds)]
    control = ctx.run(resolve_control, control_labels, EX2211_CHECKPOINT_REF, role="prep")
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
    rows = cells(arms, prep, epochs)
    n = len(rows)
    trained = ctx.map(
        train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [prep["corpus"]] * n,
        [TRAJ_STRIDE] * n,
        [prep["probes"]] * n,
        [r["op"] for r in rows],
        [r["rates"] for r in rows],
        [r["pull"] for r in rows],
        [r["label"] for r in rows],
        role="train",
    )
    ctrl = [{"checkpoint": control[lb], "label": lb} for lb in control_labels]
    evaled = ctx.map(
        eval_one,
        trained + ctrl,
        [prep["evals"]] * (n + len(ctrl)),
        [prep["probes"]] * (n + len(ctrl)),
        [r["condition"] for r in rows] + [CONTROL] * len(ctrl),
        [r["op"] for r in rows] + [None] * len(ctrl),
        [r["seed"] for r in rows] + list(range(len(ctrl))),
        [r["label"] for r in rows] + control_labels,
        role="eval",
    )
    return trained, evaled, prep


def main(ctx: Ctx) -> dict:
    trained, evaled, prep = run(ctx, GRID, EPOCHS, CONTROL_SEEDS)
    return ctx.run(publish_results, trained, evaled, prep["stats"], role="prep")


ROLES = {
    # Ex-2.2.9's corpus build, and the fan-in that writes the refs of fifty runs.
    "prep": dict(cpu=2, timeout=1800),
    # 4,950 steps at L4; the watchdog covers the checkpoint upload and the label-share replay.
    "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
    "eval": dict(gpu="L4", timeout=1800),
}

experiment = Experiment(name="ex-2.2.14", main=main, roles=ROLES)
