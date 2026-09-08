"""
Experiment 2.2.2: fallback control for *red* in the anchored transformer.

The experiment adds M1's fallback control (ex-2.9.2) to the D2.1 recipe
(ex-2.1.10 `either-t100`): a training term that teaches the blocks what to
answer once *red* has been removed, at a designed fallback answer, with the
placement of the concept kept out of its gradient (`sca.fallback`). Every
checkpoint is scored through the eval contract (`sca.intervention`) on the
5,832-line probe set the D2.1 experiments share, beside the stored ex-2.1.10
checkpoints ex-2.2.1 scored.

The design constants sit above the DAG so the report can import them and the
prose and the gates cannot drift apart. The DAG: rebuild the corpus (the same
lines ex-2.1.10 trained on, checked against its published statistics), fit the
E6 matrices on the stored no-fallback checkpoints, score the twelve stored
runs, train the twenty-four new ones, score those, and publish.

    bin/mini run docs/m2/ex-2.2.2/experiment.py --app modal --max-containers 8 --budget 6h
    bin/mini status ex-2.2.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sca.anchoring import PROMPT_SPAN
from sca.config import SchedulerConfig
from sca.data.colors import N_LEVELS
from sca.data.named_colors import GRIDS
from mini import Ctx, Experiment, get_data_dir

#: Result refs the report will read.
METRICS_REF = "reports/m2/ex-2.2.2/metrics"
ARRAYS_REF = "reports/m2/ex-2.2.2/arrays"
TRAJ_REF = "reports/m2/ex-2.2.2/trajectories"
CKPT_REF = "reports/m2/ex-2.2.2/checkpoints"  # + f"/{label}"
LUNAR_REF = "reports/m2/ex-2.2.2/lunar"  # + f"/{label}"

#: What this experiment reads from ex-2.1.10: its probe set, its published metrics and alignment maps, and
#: the twelve stored checkpoints.
EX2110 = "ex-2.1.10"
PROBE_REF = f"reports/m2/{EX2110}/probes"
EX2110_METRICS_REF = f"reports/m2/{EX2110}/metrics"
EX2110_ARRAYS_REF = f"reports/m2/{EX2110}/arrays"
EX2110_CKPT_REF = f"reports/m2/{EX2110}/checkpoints"  # + f"/{label}"

# --- The recipe ------------------------------------------------------------------

RECIPE = "ex-2.1.10/either-t100"
"""Everything about training that this experiment does not change: the `v216` corpus (216 colors, one word
each), d64-L4, 100 epochs with a 10-epoch warm-up, the pooled either-operand labeller at τ = 0.1, the anchor at
λ = 0.1 with its anneal, and the anti-subspace term at the ex-2.1.8 operating point. The constants below are
copied from that module rather than imported (an experiment module is registered only while it loads, so a
task function cannot import a sibling), and the prep step checks them against the design block ex-2.1.10
published."""

GRID = "v216"
PEAK_LR = 1e-2
CORPUS_SEED = 0
N_EXAMPLES = 100_000
HOLDOUT_FRAC = 0.2  # of distinct closed pairs
N_PROBE = 27  # probe lines per color: all closed partners, exhaustive
SCHEDULER = SchedulerConfig(epochs=100, warmup_epochs=10, min_lr_factor=0.01)
ANNEAL_START = 90.0  # epoch at which the anchor weight starts coming down
ANNEAL_END = 100.0  # and reaches the floor — the end of training
ANNEAL_FLOOR = 0.1  # fraction of peak it settles at; never zero (M1/ex-2.9.3)
TRAJ_STRIDE = 50  # steps between alignment measurements during training
BLOCK, BATCH = 64, 64
SCORING_LAMBDA = 0.1  # the anchor's peak weight λ
TAU = 0.1  # the mellowmax temperature of the pooled pull
ANTI_PEAK_RATIO = 2.5
ANTI_ANNEAL_END = 90.0
ANTI_HOLD_RATIO = 0.30
RED_RATE = 0.08
PER_SLOT_RATE = RED_RATE / 2  # the either-slot labeller's per-operand rate

ANCHOR = {
    "peak": SCORING_LAMBDA,
    "warmup_epochs": SCHEDULER.warmup_epochs,
    "anneal_start": ANNEAL_START,
    "anneal_end": ANNEAL_END,
    "floor": ANNEAL_FLOOR,
    "span": PROMPT_SPAN,
    "tau": TAU,
}
ANTI = {
    "lam": SCORING_LAMBDA,
    "peak_ratio": ANTI_PEAK_RATIO,
    "hold_ratio": ANTI_HOLD_RATIO,
    "anneal_end": ANTI_ANNEAL_END,
    "anchor_anneal_start": ANNEAL_START,
    "anchor_anneal_end": ANNEAL_END,
    "floor": ANNEAL_FLOOR,
}
LABELLER = {"keying": "either", "pull": "span", "rate": PER_SLOT_RATE}
"""The three dicts ex-2.1.10's `build_sweep` hands its primary cell, verbatim, so every new run's memo key
carries the whole shape of the pull, the repulsion and the labelling."""

ANCHOR_AXIS = 0
"""*Red* sits on e₁ at every slice, as in D2.1."""

# --- The fallback term ---------------------------------------------------------------

FALLBACK_SLICE = 0
"""Where the redirect acts during training: the embedding. Everything after it is the readout the term
trains, which is the transformer form of M1's decoder-only term."""

FALLBACK_POSITIONS = None
"""The redirect reflects every position of the crop. The edit at a position is twice its alignment, so it
singles out nothing and needs no labels: the same operator a model without labelled positions can run."""

FALLBACK_THRESHOLD = 0.5
"""The term scores only lines whose concept operand (the redder one; ties go to op1) has a clean embedding
alignment of at least this much, so a reflection moves the state by at least one unit of alignment. Below it the term is inert, which
is what makes a constant weight safe from step 0: the anchor places the concept inside its warm-up, and the
term switches on as it does."""

FALLBACK_WEIGHT = 0.05
"""w_fb, constant over training. M1's value (ex-2.9.2), on a cross-entropy rather than an MSE."""

FALLBACK_BRACKET = (0.01, 0.25)
"""The weight bracket, a factor of five either side, three seeds each: whether 0.05 sits on a plateau."""

ANTI_ANCHOR_WEIGHT = 0.1
"""The anti-anchor term's constant weight, equal to the anchor's peak. The term is a hinge on negative
alignment, mean(max(−α, 0)) over the same live positions and slices the anti-subspace term reads, so it
keeps the antipode hemisphere empty of clean states without opposing the anchor."""

GRAY = (N_LEVELS - 1) / 2
"""Mid-gray on the 16-level channel scale (7.5): the know-nothing operand the fallback answer mixes with."""

LEVELS = np.asarray(GRIDS["v216"], dtype=float)
"""The six channel levels of the corpus grid: 0, 3, 6, 9, 12, 15."""


def fallback_answer(visible: np.ndarray) -> np.ndarray:
    """The designed answer for a line whose concept operand has been removed: the *visible* operand mixed
    with mid-gray, each channel rounded to the nearest grid level.

    Also the per-channel median of the operand-averaged null (the mix with each of the visible operand's
    three closed partners per channel), so it is the center of the distribution a removed concept leaves.
    The two definitions agree on every level of the grid, and `check_answer()` asserts it.
    """
    v = np.asarray(visible, dtype=float)
    mean = (v + GRAY) / 2
    return LEVELS[np.abs(mean[..., None] - LEVELS).argmin(-1)]


def closed_partners(level: float) -> np.ndarray:
    """The grid levels *c* for which the corpus mix, ⌊(v + c + 1) / 2⌋, lands back on the grid."""
    mixes = np.floor((level + LEVELS + 1) / 2)
    return LEVELS[np.isin(mixes, LEVELS)]


def check_answer() -> None:
    """The gray fallback answer coincides with the median closed-partner mix at every level, and the null has no mode."""
    for v in LEVELS:
        partners = closed_partners(v)
        assert len(partners) == 3, (v, partners)
        mixes = np.sort(np.floor((v + partners + 1) / 2))
        assert len(set(mixes)) == 3, "the null per channel is uniform over three distinct answers"
        assert fallback_answer(np.array([v]))[0] == mixes[1], (v, mixes)


check_answer()

# --- The lines -------------------------------------------------------------------------

RED_DOSE = 0.8
"""*Red lines* have dose ≥ 0.8 (the larger of the two operands' rednesses): 365 probe lines. The fallback
term applies to the corpus lines that meet the same threshold."""

NONRED_DOSE = 0.2
"""*Non-red lines* have dose ≤ 0.2: 1,689 probe lines."""

OPERAND_POSITIONS = (0, 2)
DECODE_POS = 3
"""The `=` position: the state the answer is decoded from."""

SLICES = (0, 1, 2, 3, 4)

# --- The interventions --------------------------------------------------------------------


@dataclass(frozen=True)
class Intervention:
    """One row of the results: an operator, where it acts, and which gate or row it serves."""

    name: str
    kind: str
    """`projection`, `shaped`, `ablate`, or `linear`: a matrix fitted after training (E6)."""
    slices: tuple[int, ...] = SLICES
    positions: tuple[int, ...] | None = None
    gamma: float = 1.0


REDIRECT = Intervention("redirect", "projection", slices=(FALLBACK_SLICE,), positions=FALLBACK_POSITIONS, gamma=2.0)
"""The trained state, at eval: reflect every embedding state through the axis (γ = 2 in the projection
operator). This is the edit training applied, so H1 reads the readout the term trained, with nothing else
changed. H1, H3."""

PROJECTION = Intervention("projection", "projection")
"""Ex-2.2.1's primary intervention: full-strength projection at every slice and position. The fallback was
never trained under it, so this row says whether a trained fallback needs the intervention it was trained
at. H3, H4."""

GAMMAS = (0.5, 1.0, 1.5, 2.0)
"""The transfer sweep (H4): the projection operator at the embedding, every position, from half removal through
zero to the reflection. γ = 2 is `redirect`; γ = 1 is ex-2.2.1's `embedding` arm."""

SWEEP = tuple(
    Intervention(f"gamma-{g}", "projection", slices=(FALLBACK_SLICE,), positions=FALLBACK_POSITIONS, gamma=g)
    for g in GAMMAS
)

RIDE_ALONG = (
    Intervention("operands", "projection", positions=OPERAND_POSITIONS),
    Intervention("shaped", "shaped"),
    Intervention("ablate", "ablate", slices=()),
)
"""Ex-2.2.1's arms, re-run on the fallback condition without gates, so the intervention-tuning pass the design
schedules has the fallback rows to hand. Same definitions as there: `shaped` at a = 0.5, b = 1, p = 1. The
`embedding` arm is the sweep's γ = 1 point. `operands` is the one row that needs position labels."""

SHAPED = dict(a=0.5, b=1.0, p=1.0)

LUNAR = Intervention("lunar", "linear", slices=(FALLBACK_SLICE,), positions=FALLBACK_POSITIONS)
"""E6: a LUNAR-style redirect fitted after training, on each no-fallback checkpoint with the checkpoint frozen.
One 64×64 matrix at the embedding, applied at every position, fitted on training crops with the fallback
cross-entropy over the same qualifying red lines the fallback term uses, plus an identity term on every other
state. Scored through the contract like every other row, beside the fallback condition under `redirect`."""

LUNAR_RETAIN_WEIGHT = 1.0
"""Weight of the identity term (mean squared distance from the unedited state, over non-qualifying positions)
relative to the fallback cross-entropy in the E6 fit."""

LUNAR_STEPS = 2000
"""Adam steps of the E6 fit, at the recipe's batch size; a 64×64 matrix, so CPU minutes per seed."""

# --- Conditions --------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """A set of runs: stored (ex-2.1.10) or trained here."""

    exp: str
    name: str
    seeds: int
    title: str
    w_fb: float = 0.0
    w_aa: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.exp}/{self.name}"

    @property
    def stored(self) -> bool:
        return self.exp != "ex-2.2.2"


CONTROL = Condition("ex-2.1.10", "lam0", 3, "un-anchored")
"""Nothing placed on the axis: the anchor weight at zero."""

NO_FALLBACK = Condition("ex-2.1.10", "either-t100", 9, "no-fallback")
"""The D2.1 recipe as ex-2.2.1 scored it: the reference every hypothesis compares against, re-scored here
through the same code path as the new runs."""

FALLBACK = Condition("ex-2.2.2", "fallback", 9, "fallback", w_fb=FALLBACK_WEIGHT, w_aa=ANTI_ANCHOR_WEIGHT)
"""The recipe plus both new terms. The primary condition, and the one every hypothesis scores."""

ARMS = (
    Condition("ex-2.2.2", "fb-only", 3, "fallback, no anti-anchor", w_fb=FALLBACK_WEIGHT),
    Condition("ex-2.2.2", "anti-only", 3, "anti-anchor, no fallback", w_aa=ANTI_ANCHOR_WEIGHT),
    Condition("ex-2.2.2", "fb-w0.01", 3, "fallback at w_fb = 0.01", w_fb=FALLBACK_BRACKET[0], w_aa=ANTI_ANCHOR_WEIGHT),
    Condition("ex-2.2.2", "fb-w0.25", 3, "fallback at w_fb = 0.25", w_fb=FALLBACK_BRACKET[1], w_aa=ANTI_ANCHOR_WEIGHT),
    Condition("ex-2.2.2", "recipe", 3, "the recipe through the new code path"),
)
"""Each changes one thing from the primary condition; reported at the same statistics without gates."""

N_NEW_RUNS = FALLBACK.seeds + sum(a.seeds for a in ARMS)
assert N_NEW_RUNS == 24

# --- Gates ---------------------------------------------------------------------------------

FALLBACK_ACC_GATE = 0.8
"""H1: "seed-mean fallback accuracy (the argmax at `=` is the fallback answer) on the red lines with a clean
visible operand, under `redirect`, is at least 0.8"."""

FALLBACK_ACC_PARTIAL = 0.5
"""H1 partial: "between 0.5 and 0.8"."""

AGREE_SEEDS = 5
"""E1: a red line "agrees" when at least this many of the nine seeds decode the same answer. Ex-2.2.1's
definition, whose 13% under the projection is the reference. Reported, not gated."""

TASK_GATE = 0.02
"""H2 and H3: the width every D2.1 task gate used, on exact-match accuracy. H2: "clean exact-match accuracy
on all probe lines within 0.02 of the no-fallback condition's". H3: "the seed-mean non-red deficit under
`redirect` stays at or below 0.02" — the deficit, clean accuracy minus intervened accuracy, which is the
statistic ex-2.2.1's H2 gated at this width and reported at 0.024."""

TASK_PARTIAL = 0.05
"""H2 and H3 partial: "within 0.05". Ex-2.2.1's H2 landed partial in this band."""

VISIBLE_RED_DOSE = 0.5
"""Red lines whose *visible* operand also reaches this redness (69 of the 365) have both operand states reflected
and no defined fallback answer. They take no fallback loss, sit outside H1's gate, and are reported beside it."""

MARGIN_RATIO = 0.8
"""H2: "the seed-mean alignment margin at the end of training (ex-2.1.10's m_span) is at least 0.8 of the
no-fallback condition's"."""

GRADE_DIP = 0.02
"""H4: "fallback accuracy is non-decreasing along the sweep, allowing a dip between adjacent strengths of at
most 0.02"."""

TRANSFER_FRAC = 0.5
"""H4: "fallback accuracy at γ = 1 is at least half of its value at γ = 2"."""

RESOLUTION_SD = 2.0
"""Differences between conditions or arms smaller than this many pooled between-seed standard deviations of
the statistic are reported as not resolved. H1's and H3's comparisons against no-fallback use it."""

#: What a red line decodes to under intervention, in the order the composition is stored. Ex-2.2.1's five
#: categories plus the fallback answer; a fallback answer that is also a one-step neighbor of the true answer
#: counts as fallback.
COMPOSITION = ("fallback", "true", "neighbor", "visible_operand", "red_operand", "other")

#: Ridge strength of the off-axis recoverability probe (E3), matching ex-2.1.12 and ex-2.2.1.
DECODE_L2 = 1e-2

#: Groups the per-line statistics are contracted over. `red_clean` is H1's group: red lines whose visible
#: operand is below `VISIBLE_RED_DOSE`; `red_both` is the rest of the red lines.
GROUPS = ("all", "red", "nonred", "red_clean", "red_both")

#: The group sizes the design states; `read_lines` asserts them.
#: REVIEW: the prereg counted 298 clean / 67 both. That count read `redness < 0.5` without roundoff slack, and
#: one color on the 0.5 contour (cf96: 1 · (1 − 0.3 − 0.2) evaluates to 0.49999999999999994) fell on the clean
#: side while its five siblings at the same redness fell on the red side. The design's rule, "≥ 0.5 is red",
#: puts all six together, which is what the slack below does: 296 clean, 69 both. Verify: `read_lines`.
N_LINES = {"all": 5832, "red": 365, "nonred": 1689, "red_clean": 296, "red_both": 69}

LUNAR_LR = 1e-3
"""Adam learning rate of the E6 fit."""

#: Tolerance on the recomputed clean alignment map against the one ex-2.1.10 published, per line, and on the
#: recomputed m_span against its published value: the ex-2.2.1 figure for the same check.
ALPHA_ATOL = 2e-3

#: Token positions of a probe line, in sequence order; the answer's log-prob is read one position earlier.
POSITIONS = ("op1", "+", "op2", "=", "ans", "\n")
ANSWER_POS = 4
PROMPT_POSITIONS = (0, 1, 2, 3)

#: The quantile, over non-red lines, that the write bound is read at (ex-2.2.1's H4 map, drawn for H3).
WRITE_QUANTILE = 99


def interventions_for(cond: Condition) -> tuple[Intervention, ...]:
    """Which rows each condition is scored on.

    The control takes the two gated operators as calibration rows (nothing on the axis). The no-fallback
    condition takes everything the fallback condition does, plus the E6 fit; the sweep's γ = 2 point repeats
    `redirect` and its γ = 1 point repeats ex-2.2.1's `embedding` arm, which the scorer asserts.
    """
    if cond == CONTROL:
        return (REDIRECT, PROJECTION)
    rows = (REDIRECT, PROJECTION, *SWEEP, *RIDE_ALONG)
    return (*rows, LUNAR) if cond == NO_FALLBACK else rows


# --- The DAG ---------------------------------------------------------------------


def _npz(**arrays) -> bytes:
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def fallback_tables(tokenizer, palette: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-token redness and fallback answer over the tokenizer's vocabulary.

    Syntax and padding read zero redness (so they never carry a dose) and answer 0 (never consulted, since a
    line's visible operand is a color). A color's fallback answer is `fallback_answer` of its levels, which
    lands on the grid by construction, looked up as a token.
    """
    from sca.colorcube import redness as colorcube_redness

    names = {v: k for k, v in palette.items()}
    redness = np.zeros(tokenizer.vocab_size)
    answer = np.zeros(tokenizer.vocab_size, dtype=np.int32)
    for name, rgb in palette.items():
        redness[tokenizer.stoi[name]] = colorcube_redness(np.asarray(rgb, dtype=float)[None] / (N_LEVELS - 1))[0]
        answer[tokenizer.stoi[name]] = tokenizer.stoi[names[tuple(int(v) for v in fallback_answer(np.array(rgb)))]]
    return redness, answer


def prepare_corpus(grid: str, n_examples: int, holdout_frac: float, corpus_seed: int) -> dict:
    """Rebuild ex-2.1.10's corpus on this experiment's volume, and the fallback term's tables.

    The corpus is a deterministic function of its seed, so the new runs train on the lines the stored ones
    did; the token count is checked against the statistics ex-2.1.10 published, and the recipe constants
    above against its design block. The probe set is not rebuilt: every scorer reads ex-2.1.10's by ref.
    """
    import json

    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data import named_colors as nc
    from mini.store import get, get_ref, put

    workdir = get_data_dir() / "prep"
    levels = GRIDS[grid]
    palette = nc.grid_palette(levels)
    corpus = nc.sample_corpus(n_examples, corpus_seed, levels, holdout_frac)
    words = [w for ex in corpus for w in nc.as_words(ex)]
    tokenizer_config = TokenizerConfig(vocabulary=[*nc.SYNTAX, *palette])
    tokenizer = nc.WordTokenizer(tokenizer_config)
    tokens = np.asarray(tokenizer.encode_words(words), dtype=np.int32)
    meta = CorpusMetadata(
        tokenizer_config=tokenizer_config,
        total_tokens=len(tokens),
        total_chars=sum(map(len, words)),
        sources=[DatasetMetadata(title=f"named-only color corpus ({grid})", fixes=[], total_chars=len(words))],
    )
    save_data(tokens, meta, get_data_dir() / "corpora" / grid)

    ref = get_ref(EX2110_METRICS_REF)
    assert ref is not None, f"{EX2110} has not published its metrics"
    published = json.loads(get(ref, workdir / "ex2110-metrics.json").read_text())
    assert published["corpus_stats"]["total_tokens"] == len(tokens), "the rebuilt corpus differs from ex-2.1.10's"
    design = published["design"]
    assert design["scoring_lambda"] == SCORING_LAMBDA and design["span"] == PROMPT_SPAN
    assert design["anneal"] == {"start": ANNEAL_START, "end": ANNEAL_END, "floor": ANNEAL_FLOOR}
    assert design["anti"] == {
        "peak_ratio": ANTI_PEAK_RATIO,
        "anneal_end": ANTI_ANNEAL_END,
        "hold_ratio": ANTI_HOLD_RATIO,
    }
    assert design["labelling"] == {"red_rate": RED_RATE, "per_slot_rate": PER_SLOT_RATE}
    assert design["traj_stride"] == TRAJ_STRIDE and design["primary"] == NO_FALLBACK.name
    primary = next(c for c in published["conditions"] if c["name"] == NO_FALLBACK.name)
    assert primary["tau"] == TAU and primary["seeds"] == NO_FALLBACK.seeds and primary["labels"] == LABELLER["keying"]

    redness, answer = fallback_tables(tokenizer, palette)
    n_colors = len(palette)
    assert (redness > 0).sum() == n_colors - sum(1 for rgb in palette.values() if rgb[0] == 0)
    assert set(answer[answer > 0]) <= {tokenizer.stoi[n] for n in palette}
    return {
        "meta": meta,
        "stats": {"n_colors": n_colors, "total_tokens": int(len(tokens)), "vocab_size": tokenizer.vocab_size},
        "tables": put(
            _npz(redness=redness, answer=answer, eq_token=np.int32(tokenizer.stoi["="])),
            name=f"ex-2.2.2-{grid}-tables.npz",
        ),
    }


def _make_config(vocab_size: int, seed: int):
    """The d64-L4 config from ex-2.1.3, unchanged; ex-2.1.10's `_make_config` verbatim."""
    from sca.config import DataConfig, ModelConfig, OptimizerConfig, TokenizerConfig, TrainingConfig

    return TrainingConfig(
        model=ModelConfig(
            vocab_size=vocab_size,
            block_size=BLOCK,
            n_embd=64,
            n_head=8,
            n_head_dim=8,
            n_ff=256,
            n_layer=4,
        ),
        tokenizer=TokenizerConfig(vocabulary=[]),
        data=DataConfig(batch_size=BATCH, oversample=16, train_split=0.9, padding_chance=0.1),
        optimizer=OptimizerConfig(weight_decay=0, learning_rate=PEAK_LR, betas=(0.9, 0.95)),
        scheduler=SCHEDULER,
        seed=seed,
    )


def build_cells(prep: dict) -> list[tuple]:
    """(config, fallback, condition, seed, label) per new run; cheap and deterministic per wake.

    The fallback dict carries the two weights and the thresholds, so a change to any of them re-runs the
    cell the way a schedule change would. Seeds are range(seeds): the primary shares ex-2.1.10's nine.
    """
    from sca.utils import align

    tc = prep["meta"].tokenizer_config
    cells = []
    for c in (FALLBACK, *ARMS):
        fallback = {
            "weight": c.w_fb,
            "anti_anchor_weight": c.w_aa,
            "dose_min": RED_DOSE,
            "visible_max": VISIBLE_RED_DOSE,
            "alignment_min": FALLBACK_THRESHOLD,
            "slice": FALLBACK_SLICE,
        }
        for seed in range(c.seeds):
            config = _make_config(align(tc.vocab_size, 64), seed)
            config.tokenizer = tc.model_copy()
            cells.append((config, fallback, c.name, seed, f"{c.name}-s{seed}"))
    return cells


def train_one(
    config, anchor: dict, anti: dict, labeller: dict, fallback: dict, grid: str, traj_stride: int, tables, label: str
) -> dict:
    """Train one cell: ex-2.1.10's `train_one` with the fallback spec added.

    The labeller, the probe lines and the trajectory instrument are ex-2.1.10's, read off its published probe
    set, so the batches and label draws at a given seed are the ones the stored runs saw.
    """
    from sca.anchoring import AnchorSpec, AntiSpec, LabelSpec
    from sca.compute.training import train_anchored
    from sca.fallback import FallbackSpec
    from mini.store import get, get_ref, put

    workdir = get_data_dir() / "cells" / label
    probes = get_ref(PROBE_REF)
    assert probes is not None, f"{EX2110} has not published its probe set"
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens, slot_p, weights, line_p = z["probe_tokens"], z["slot_p"], z["weights"], z["line_p"]
    with np.load(get(tables, workdir / "tables.npz")) as z:
        spec = FallbackSpec(
            redness=z["redness"],
            answer=z["answer"],
            eq_token=int(z["eq_token"]),
            dose_min=fallback["dose_min"],
            visible_max=fallback["visible_max"],
            alignment_min=fallback["alignment_min"],
            slice=fallback["slice"],
        )

    stride = len(probe_tokens) // len(weights)
    first_of_color = probe_tokens[::stride]
    line_w = line_p[::stride] / line_p[::stride].sum()
    _, metrics, traj = train_anchored(
        config,
        get_data_dir() / "corpora" / grid,
        anchor=AnchorSpec(**anchor),
        anti=AntiSpec(**anti),
        label_p=LabelSpec(p=slot_p, keying=labeller["keying"], pull=labeller["pull"]),
        probe_tokens=first_of_color,
        probe_weights=weights,
        probe_line_w=line_w,
        fallback=spec,
        fallback_weight=fallback["weight"],
        anti_anchor_weight=fallback["anti_anchor_weight"],
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
    )
    return {
        "label": label,
        "val_loss": [m.val_loss for m in metrics],
        "train_loss": [m.train_loss for m in metrics],
        "traj": {k: v.tolist() for k, v in traj.items()},
        "checkpoint": put(workdir / "model", name=f"ex-2.2.2-{label}-ckpt"),
    }


def fit_lunar(seed: int, tables, grid: str, steps: int, retain_weight: float, lr: float) -> dict:
    """E6: fit one matrix at the embedding of a frozen no-fallback checkpoint, on training crops.

    The loss is the fallback cross-entropy on the qualifying lines the term uses, plus *retain_weight* times
    the mean squared distance from the unedited state over every other live position. The matrix starts at
    the identity and is applied as `normalize(h @ W)` at every position, which is how the scorer runs it.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    from sca.compute.data_pipelines import load_data
    from sca.compute.model import load_checkpoint
    from sca.data.batches import sample_batches, split_data
    from sca.fallback import FallbackSpec, fallback_term, qualifying_lines
    from sca.model import NGPT
    from sca.model._shared import normalize
    from mini.progress import emit_metrics, emit_progress, expect_metrics
    from mini.store import get, get_ref, put

    label = f"{NO_FALLBACK.name}-s{seed}"
    workdir = get_data_dir() / "lunar" / label
    ckpt = get_ref(f"{EX2110_CKPT_REF}/{label}")
    assert ckpt is not None, f"{EX2110} has not published {label}"
    get(ckpt, workdir / "model")
    model, config, _ = load_checkpoint(workdir)
    assert isinstance(model, NGPT), "the reflection and the readout act on nGPT's between-block stream"
    with np.load(get(tables, workdir / "tables.npz")) as z:
        spec = FallbackSpec(
            redness=z["redness"],
            answer=z["answer"],
            eq_token=int(z["eq_token"]),
            dose_min=RED_DOSE,
            visible_max=VISIBLE_RED_DOSE,
            alignment_min=FALLBACK_THRESHOLD,
            slice=FALLBACK_SLICE,
        )
    assert spec.slice == 0, "the E6 fit edits the embedding"
    data, _ = load_data(get_data_dir() / "corpora" / grid)
    train_data, _ = split_data(data, config.data.train_split)
    rng = np.random.default_rng(seed + 20_000)

    wte, enc = model.transformer.wte, model.transformer.rotary_enc
    blocks = model.transformer.blocks

    def loss_fn(w, x):
        h0 = normalize(wte[x])
        live = (x != 0).astype(h0.dtype)
        q = qualifying_lines(x, h0[..., ANCHOR_AXIS], spec)
        h = normalize(h0 @ w)
        keep = live * (1.0 - q.concept)
        retain = jnp.sum(jnp.sum((h - h0) ** 2, axis=-1) * keep) / (jnp.sum(keep) + 1e-8)
        for block in blocks:
            h = block(h, enc)
        fb = fallback_term((h @ wte.T) * model.s_z(), q.target, q.active)
        return fb + retain_weight * retain, (fb, retain, q.active.sum())

    optimizer = optax.adam(lr)
    w = jnp.eye(config.model.n_embd, dtype=jnp.float32)
    opt_state = optimizer.init(w)

    @eqx.filter_jit
    def step(w, opt_state, x):
        (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(w, x)
        updates, opt_state = optimizer.update(grads, opt_state, w)
        return optax.apply_updates(w, updates), opt_state, aux

    expect_metrics(fallback="down")
    curve = []
    for i, (x, _) in enumerate(sample_batches(train_data, config.data, config.model, steps, rng), start=1):
        w, opt_state, (fb, retain, n_active) = step(w, opt_state, jnp.asarray(x))
        emit_metrics(fallback=float(fb), retain=float(retain))
        emit_progress(i, steps)
        if i % 10 == 0 or i == steps:
            curve.append((i, float(fb), float(retain), float(n_active)))
    return {
        "label": label,
        "seed": seed,
        "curve": curve,
        "matrix": put(_npz(w=np.asarray(w)), name=f"ex-2.2.2-lunar-{label}.npz"),
    }


# --- The scorer: ex-2.2.1's, with the fallback statistics added --------------------------


def top_quantile(a, q: float = WRITE_QUANTILE, axis: int = -1):
    """The *q*-th percentile as an order statistic: the ⌈n(1 − q/100)⌉-th largest value along *axis*."""
    a = np.asarray(a)
    n = a.shape[axis]
    k = int(np.ceil(n * (1 - q / 100)))
    return np.take(np.sort(a, axis=axis), n - k, axis=axis)


def ridge_readout(x, y, l2: float = DECODE_L2):
    """A ridge probe fitted on *x* → *y* as a closure that decodes new states, centered as the fit was."""
    x64, y64 = np.asarray(x, np.float64), np.asarray(y, np.float64)
    mx, my = x64.mean(0), y64.mean(0)
    xc = x64 - mx
    w = np.linalg.solve(xc.T @ xc + l2 * np.eye(x64.shape[1]), xc.T @ (y64 - my))
    return lambda h: (np.asarray(h, np.float64) - mx) @ w + my


def cv_ridge_r2(x, y, l2: float = DECODE_L2, n_folds: int = 5) -> float:
    """Held-out R² of a ridge probe at a fixed penalty, over interleaved folds, so the estimate has no seed."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    fold = np.arange(len(x)) % n_folds
    preds = np.empty_like(y)
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        preds[te] = ridge_readout(x[tr], y[tr], l2)(x[te])
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def pooled_margin(alpha: np.ndarray, weights: np.ndarray) -> float:
    """m_span, as ex-2.1.10 scores it: mean over slices of the max-over-span margin on (slices, colors, positions)."""
    m = np.einsum("c,lct->lt", weights, alpha) - alpha.mean(axis=1)  # (L1, T)
    return float(m[:, :PROMPT_SPAN].max(axis=1).mean())


@dataclass(frozen=True)
class Lines:
    """The probe set as the statistics see it: groups, the color of every token, and each line's fallback answer."""

    tokens: np.ndarray
    """(N, T) token ids."""
    red: np.ndarray
    nonred: np.ndarray
    red_clean: np.ndarray
    red_both: np.ndarray
    color_ids: np.ndarray
    """Token id of each palette color, in palette order."""
    tok2color: np.ndarray
    """Token id → palette index, −1 off the color vocabulary."""
    red_operand: np.ndarray
    """Per line, the position of the operand carrying the dose (op1 on a tie): the concept operand."""
    dose: np.ndarray
    fb_token: np.ndarray
    """Per line, the fallback answer's token id, or −1 where the visible operand is itself red."""

    @property
    def n(self) -> int:
        return len(self.tokens)

    @property
    def n_pos(self) -> int:
        return self.tokens.shape[1]

    @property
    def groups(self) -> dict[str, np.ndarray]:
        return {
            "all": np.ones(self.n, bool),
            "red": self.red,
            "nonred": self.nonred,
            "red_clean": self.red_clean,
            "red_both": self.red_both,
        }

    @property
    def line_colors(self) -> np.ndarray:
        """(N, T) palette index per token; the syntax positions read −1."""
        return self.tok2color[self.tokens]

    @property
    def answer(self) -> np.ndarray:
        return self.tokens[:, ANSWER_POS]

    @property
    def visible(self) -> np.ndarray:
        """Per line, the token at the visible operand: the other one."""
        return self.tokens[np.arange(self.n), 2 - self.red_operand]

    def by_group(self, v: np.ndarray) -> dict[str, float]:
        def mean(a: np.ndarray) -> float:
            return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")

        return {g: mean(v[m]) for g, m in self.groups.items()}


def read_lines(
    tokens: np.ndarray, r1: np.ndarray, r2: np.ndarray, color_redness: np.ndarray, tokenizer, n_logits: int
) -> Lines:
    """Group the probe lines by dose and visibility, and check the set is the one the design describes."""
    from sca.colorcube import redness as colorcube_redness
    from sca.data.named_colors import grid_palette
    from sca.vis_grading import GRID_RGB

    n_colors = len(color_redness)
    dose, visible_redness = np.maximum(r1, r2), np.minimum(r1, r2)
    eps = 1e-9
    red, nonred = dose >= RED_DOSE - eps, dose <= NONRED_DOSE + eps
    clean_visible = visible_redness < VISIBLE_RED_DOSE - eps
    red_clean, red_both = red & clean_visible, red & ~clean_visible
    palette = grid_palette(GRIDS[GRID])
    color_ids = np.array([tokenizer.stoi[n] for n in palette])
    np.testing.assert_allclose(colorcube_redness(GRID_RGB), color_redness, rtol=0, atol=1e-12)
    assert n_logits >= tokenizer.vocab_size
    tok2color = np.full(n_logits, -1)
    tok2color[color_ids] = np.arange(n_colors)
    colors = tok2color[tokens]
    assert (colors[:, [0, 2, ANSWER_POS]] >= 0).all() and (colors[:, [1, 3, 5]] < 0).all()
    n_partners = len(tokens) // n_colors
    assert n_partners == N_PROBE
    np.testing.assert_array_equal(colors[:, 0], np.repeat(np.arange(n_colors), n_partners))
    red_operand = np.where(r1 >= r2, 0, 2)
    _, answer_table = fallback_tables(tokenizer, palette)
    fb_token = np.where(clean_visible, answer_table[tokens[np.arange(len(tokens)), 2 - red_operand]], -1)
    lines = Lines(tokens, red, nonred, red_clean, red_both, color_ids, tok2color, red_operand, dose, fb_token)
    counts = {g: int(m.sum()) for g, m in lines.groups.items()}
    assert counts == N_LINES, counts
    return lines


def composition(lines: Lines, guess: np.ndarray) -> np.ndarray:
    """Per line, what the decoded answer is, as an index into `COMPOSITION`.

    The fallback answer is tested first: it can coincide with the true answer (the middle partner's mix, one
    line in 27 per visible operand) and with the visible operand (levels 6 and 9 map to themselves), and the
    design counts both as fallback. The rest keep ex-2.2.1's precedence: true, red operand, visible operand,
    then a one-step neighbor of the true mix (one level away in at least one channel, no more than one in any).
    """
    g = lines.tok2color[guess]
    rows = np.arange(lines.n)
    cube = np.stack(np.unravel_index(np.arange(len(lines.color_ids)), (6, 6, 6)), axis=1)
    step = np.abs(cube[np.maximum(g, 0)] - cube[lines.line_colors[:, ANSWER_POS]]).max(1)
    tests = [
        (lines.fb_token >= 0) & (guess == lines.fb_token),
        guess == lines.answer,
        guess == lines.tokens[rows, lines.red_operand],
        guess == lines.visible,
        (g >= 0) & (step == 1),
    ]
    order = [COMPOSITION.index(k) for k in ("fallback", "true", "red_operand", "visible_operand", "neighbor")]
    return np.select(tests, order, default=COMPOSITION.index("other"))


def guess_distance(lines: Lines, guess: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Unit-cube distance from the decoded answer to *target* (token ids); NaN off the color vocabulary or where the target is undefined."""
    from sca.vis_grading import GRID_RGB

    g, t = lines.tok2color[guess], lines.tok2color[np.maximum(target, 0)]
    d = np.linalg.norm(GRID_RGB[np.maximum(g, 0)] - GRID_RGB[np.maximum(t, 0)], axis=1)
    return np.where((g >= 0) & (target >= 0), d, np.nan)


def offaxis_r2(states: np.ndarray, lines: Lines) -> list[float]:
    """E3: held-out R² for the concept operand's redness, read off its state with the axis deleted, per slice."""
    from sca.anchoring import ANCHOR_AXIS

    rows = np.arange(lines.n)
    return [cv_ridge_r2(np.delete(states[s, rows, lines.red_operand], ANCHOR_AXIS, axis=1), lines.dose) for s in SLICES]


def build_triple(iv: Intervention, model, lunar: np.ndarray | None):
    """The (model, subspace, operator) triple for one intervention."""
    from sca.anchoring import ANCHOR_AXIS
    from sca.intervention import Subspace, ablate_weights, projection, shaped_suppression
    from sca.model._shared import normalize

    sub = Subspace.axis(model.transformer.wte.shape[1], ANCHOR_AXIS)
    match iv.kind:
        case "projection":
            return model, sub, projection(sub, gamma=iv.gamma)
        case "shaped":
            return model, sub, shaped_suppression(sub, a=SHAPED["a"], b=SHAPED["b"], p=SHAPED["p"])
        case "ablate":
            return ablate_weights(model, sub), sub, projection(sub)
        case "linear":
            assert lunar is not None, "the linear row needs a fitted matrix"
            import jax.numpy as jnp

            w = jnp.asarray(lunar)
            return model, sub, lambda h: normalize(h @ w)
    raise ValueError(iv.kind)


def check_contract(iv: Intervention, out, alpha_clean: np.ndarray, theta: np.ndarray, positions) -> None:
    """The contract's promises, checked on every run (ex-2.2.1's checks, with the linear row exempt from the closed form)."""
    from sca.anchoring import ANCHOR_AXIS
    from sca.intervention import write_angle

    alpha_pre = out.pre[..., ANCHOR_AXIS]
    if iv.kind != "ablate":
        np.testing.assert_allclose(alpha_pre[0], alpha_clean[0], rtol=0, atol=0)
    skipped = np.ones((len(SLICES), theta.shape[-1]), bool)
    for s in iv.slices:
        skipped[s] = False if positions is None else positions == 0
    off = skipped.nonzero()
    np.testing.assert_allclose(theta[off[0], :, off[1]], 0.0, rtol=0, atol=1e-6)
    if iv.kind == "projection":
        on = (~skipped).nonzero()
        expected = write_angle(alpha_pre[on[0], :, on[1]], iv.gamma)
        np.testing.assert_allclose(theta[on[0], :, on[1]], expected, rtol=0, atol=2e-3)
    if iv.kind == "ablate":
        assert np.abs(alpha_pre).max() < 1e-6, "an ablated model never carries the axis"


def score_run(exp: str, cond: str, seed: int, interventions: tuple[Intervention, ...], ckpt, lunar) -> dict:
    """Score one checkpoint: the clean pass, then each intervention through the eval contract.

    *ckpt* is the checkpoint artifact for a run trained here, or None for a stored run (read by ref from
    *exp*); *lunar* is the E6 matrix artifact for the rows that need one. Per-run statistics return as the
    result; per-line arrays go to the store as one npz.
    """
    from sca.anchoring import ANCHOR_AXIS
    from sca.compute.model import load_checkpoint
    from sca.data.named_colors import WordTokenizer
    from sca.intervention import Subspace, angle_between, answer_logprobs, apply, projection
    from sca.model import NGPT
    from mini.store import get_many, get_ref, put

    label = f"{cond}-s{seed}"
    workdir = get_data_dir() / "score" / f"{exp}-{label}"
    probes_ref = get_ref(PROBE_REF)
    if ckpt is None:
        ckpt = get_ref(f"reports/m2/{exp}/checkpoints/{label}")
    assert probes_ref is not None and ckpt is not None, f"{exp} has not published {label} and the probe set"
    fetch = [(probes_ref, workdir / "probes.npz"), (ckpt, workdir / "model")]
    if lunar is not None:
        fetch.append((lunar, workdir / "lunar.npz"))
    paths = get_many(fetch)
    with np.load(paths[0]) as z:
        tokens, r1, r2, color_redness, weights = z["probe_tokens"], z["r1"], z["r2"], z["redness"], z["weights"]
    matrix = None
    if lunar is not None:
        with np.load(paths[2]) as z:
            matrix = z["w"]
    model, config, _ = load_checkpoint(workdir)
    assert isinstance(model, NGPT), "the contract's operators act on nGPT's between-block stream"
    lines = read_lines(tokens, r1, r2, color_redness, WordTokenizer(config.tokenizer), model.transformer.wte.shape[0])
    rows = np.arange(lines.n)

    # --- The clean pass: the scorer's own control, and the reference every intervention is read against.
    sub = Subspace.axis(model.transformer.wte.shape[1], ANCHOR_AXIS)
    clean = apply(model, tokens, projection(sub), slices=())
    alpha_clean = clean.pre[..., ANCHOR_AXIS]  # (L1, N, T)
    lp_clean = answer_logprobs(clean.logits, ANSWER_POS)
    p_clean = np.exp(lp_clean[rows, lines.answer])
    p_fb_clean = np.where(lines.fb_token >= 0, np.exp(lp_clean[rows, np.maximum(lines.fb_token, 0)]), np.nan)
    guess_clean = lp_clean.argmax(1)
    acc_clean = guess_clean == lines.answer
    alpha_q99 = top_quantile(np.abs(alpha_clean[:, lines.nonred]), axis=1)  # (L1, T)
    alpha_colors = alpha_clean.reshape(alpha_clean.shape[0], len(weights), N_PROBE, -1).mean(axis=2)
    negative = (alpha_clean < 0).mean(axis=1)  # (L1, T): E2's fraction of clean states past the antipode plane

    arrays: dict[str, Any] = {
        "clean/alpha": alpha_clean.astype(np.float32),
        "clean/p_ans": p_clean.astype(np.float32),
        "clean/p_fb": p_fb_clean.astype(np.float32),
        "clean/guess": guess_clean.astype(np.int16),
    }
    stats: dict[str, Any] = {
        "acc": lines.by_group(acc_clean),
        "p_ans": lines.by_group(p_clean),
        "fb_acc": lines.by_group(guess_clean == lines.fb_token),
        "p_fb": lines.by_group(p_fb_clean),
        "bound": np.arcsin(alpha_q99).tolist(),
        "alpha_q99_nonred": alpha_q99.tolist(),
        "m_span": pooled_margin(alpha_colors, weights),
        "alpha_mean_op1": float(alpha_colors[:, :, 0].mean()),
        "negative_frac": negative.tolist(),
        "offaxis_r2": offaxis_r2(clean.pre, lines),
    }
    scored: dict[str, Any] = {}

    # --- Each intervention: build the triple, score it, contract the per-line quantities.
    for iv in interventions:
        positions = None if iv.positions is None else np.isin(np.arange(lines.n_pos), iv.positions).astype(np.float32)
        target, edited, op = build_triple(iv, model, matrix)
        out = apply(target, tokens, op, slices=iv.slices, positions=positions)

        lp = answer_logprobs(out.logits, ANSWER_POS)
        p_ans = np.exp(lp[rows, lines.answer])
        p_fb = np.where(lines.fb_token >= 0, np.exp(lp[rows, np.maximum(lines.fb_token, 0)]), np.nan)
        guess = lp.argmax(1)
        acc = guess == lines.answer
        damage = p_clean - p_ans
        offvocab = 1.0 - np.exp(lp[:, lines.color_ids]).sum(1)
        theta = angle_between(out.pre, out.post)  # (L1, N, T): the write at every site
        alpha_pre = out.pre[..., ANCHOR_AXIS]
        disp = angle_between(out.post[-1, :, DECODE_POS], clean.post[-1, :, DECODE_POS])
        comp = composition(lines, guess)
        check_contract(iv, out, alpha_clean, theta, positions)

        scored[iv.name] = {
            "kind": iv.kind,
            "slices": list(iv.slices),
            "positions": None if iv.positions is None else list(iv.positions),
            "gamma": iv.gamma,
            "acc": lines.by_group(acc),
            "deficit": lines.by_group(acc_clean.astype(float) - acc),
            "p_ans": lines.by_group(p_ans),
            "damage": lines.by_group(damage),
            "fb_acc": lines.by_group(guess == lines.fb_token),
            "p_fb": lines.by_group(p_fb),
            "offvocab": lines.by_group(offvocab),
            "disp": lines.by_group(disp),
            "write_prompt_sum": lines.by_group(theta[:, :, list(PROMPT_POSITIONS)].sum(axis=(0, 2))),
            "q99_write_nonred": top_quantile(theta[:, lines.nonred], axis=1).tolist(),
            "q99_alpha_nonred": top_quantile(np.abs(alpha_pre[:, lines.nonred]), axis=1).tolist(),
            "composition_red": np.bincount(comp[lines.red], minlength=len(COMPOSITION)).tolist(),
            "composition_red_clean": np.bincount(comp[lines.red_clean], minlength=len(COMPOSITION)).tolist(),
            "guess_dist_true_red": float(np.nanmean(guess_distance(lines, guess, lines.answer)[lines.red])),
            "guess_dist_fb_red_clean": float(np.nanmean(guess_distance(lines, guess, lines.fb_token)[lines.red_clean])),
            "offaxis_r2": offaxis_r2(out.post, lines),
            "cos_to_axis": float(abs(edited.basis[0] @ sub.basis[0])),
        }
        arrays |= {
            f"{iv.name}/p_ans": p_ans.astype(np.float32),
            f"{iv.name}/p_fb": p_fb.astype(np.float32),
            f"{iv.name}/guess": guess.astype(np.int16),
            f"{iv.name}/offvocab": offvocab.astype(np.float32),
            f"{iv.name}/disp": disp.astype(np.float32),
            f"{iv.name}/composition": comp.astype(np.int8),
            f"{iv.name}/alpha_pre": alpha_pre.astype(np.float16),
            f"{iv.name}/theta": theta.astype(np.float16),
        }

    # The sweep's endpoint is `redirect` by another name: one operator, two rows.
    if REDIRECT.name in scored and f"gamma-{REDIRECT.gamma}" in scored:
        np.testing.assert_array_equal(arrays[f"{REDIRECT.name}/guess"], arrays[f"gamma-{REDIRECT.gamma}/guess"])

    return {
        "label": label,
        "exp": exp,
        "condition": cond,
        "seed": seed,
        "n": {g: int(m.sum()) for g, m in lines.groups.items()},
        "fb_overlap": {
            "true": int((lines.fb_token == lines.answer)[lines.red_clean].sum()),
            "visible": int((lines.fb_token == lines.visible)[lines.red_clean].sum()),
        },
        "clean": stats,
        "interventions": scored,
        "arrays": put(_npz(**arrays), name=f"ex-2.2.2-{exp}-{label}-arrays.npz"),
    }


def publish_results(results: list[dict], trained: list[dict], lunar: list[dict]) -> dict:
    """Stack every run's arrays into one npz and its statistics into one JSON, under the refs above.

    Three things need every run at once: the recomputed clean alignment maps and m_span of the stored runs
    are checked against what ex-2.1.10 published, the seed agreement of the response is read across each
    condition's seeds, and the new checkpoints and E6 matrices get their refs.
    """
    import io
    import json

    from sca.vis_grading import GRID_RGB, REDNESS
    from mini.store import get, get_ref, put, set_ref

    workdir = get_data_dir() / "publish"
    arrays: dict[str, Any] = {}
    for r in results:
        with np.load(get(r["arrays"], workdir / f"{r['exp']}-{r['label']}.npz")) as z:
            arrays |= {f"{r['exp']}/{r['label']}/{k}": z[k] for k in z.files}

    # The instrument check: the stored runs' clean maps and margins against ex-2.1.10's published values.
    published_ref, metrics_ref = get_ref(EX2110_ARRAYS_REF), get_ref(EX2110_METRICS_REF)
    assert published_ref is not None and metrics_ref is not None
    ex2110 = json.loads(get(metrics_ref, workdir / "ex2110-metrics.json").read_text())
    cells = {c["label"]: c for c in ex2110["cells"]}
    stored = [r for r in results if r["exp"] == EX2110]
    alpha_diff, m_span_diff = {}, {}
    with np.load(get(published_ref, workdir / "published.npz")) as z:
        for r in stored:
            ours = arrays[f"{r['exp']}/{r['label']}/clean/alpha"].astype(np.float32)
            alpha_diff[r["label"]] = float(np.abs(ours - z[f"{r['label']}/alpha_lines"]).max())
            m_span_diff[r["label"]] = float(abs(r["clean"]["m_span"] - cells[r["label"]]["m_span"]))
    assert max(alpha_diff.values()) < ALPHA_ATOL, alpha_diff
    assert max(m_span_diff.values()) < ALPHA_ATOL, m_span_diff

    probes_ref = get_ref(PROBE_REF)
    assert probes_ref is not None
    with np.load(get(probes_ref, workdir / "probes.npz")) as z:
        tokens, r1, r2 = z["probe_tokens"], z["r1"], z["r2"]
    dose, visible = np.maximum(r1, r2), np.minimum(r1, r2)
    arrays |= {"probe/tokens": tokens, "probe/dose": dose, "probe/r1": r1, "probe/r2": r2}
    eps = 1e-9
    groups = {"red": dose >= RED_DOSE - eps, "red_clean": (dose >= RED_DOSE - eps) & (visible < VISIBLE_RED_DOSE - eps)}

    # Seed agreement: per condition and intervention, over red and red-clean lines. `agree` is the fraction
    # of lines on which at least AGREE_SEEDS seeds decode the same answer (defined for the nine-seed
    # conditions); `plurality` is the mean over lines of the plurality fraction, defined for every condition.
    agreement: dict[str, dict] = {}
    for cond in (CONTROL, NO_FALLBACK, FALLBACK, *ARMS):
        runs = [r for r in results if r["condition"] == cond.name and r["exp"] == cond.exp]
        agreement[cond.key] = {}
        for iv in interventions_for(cond):
            for g, mask in groups.items():
                guesses = np.stack([arrays[f"{r['exp']}/{r['label']}/{iv.name}/guess"][mask] for r in runs])
                modal = np.array([np.bincount(col).max() for col in guesses.T])
                agreement[cond.key][f"{iv.name}/{g}"] = {
                    "agree": float((modal >= AGREE_SEEDS).mean()) if len(runs) >= AGREE_SEEDS else None,
                    "plurality": float((modal / len(runs)).mean()),
                }

    for t in trained:
        set_ref(f"{CKPT_REF}/{t['label']}", t["checkpoint"])
    for fit in lunar:
        set_ref(f"{LUNAR_REF}/{fit['label']}", fit["matrix"])

    metrics = {
        "runs": [{k: v for k, v in r.items() if k != "arrays"} for r in results],
        "training": [{k: v for k, v in t.items() if k not in ("traj", "checkpoint")} for t in trained],
        "lunar": [{k: v for k, v in fit.items() if k != "matrix"} for fit in lunar],
        "agreement": agreement,
        "alpha_max_diff": alpha_diff,
        "m_span_max_diff": m_span_diff,
        "rgb_floor": cv_ridge_r2(GRID_RGB, REDNESS),
        "design": {
            "recipe": RECIPE,
            "conditions": [c.__dict__ for c in (CONTROL, NO_FALLBACK, FALLBACK, *ARMS)],
            "interventions": {
                iv.name: iv.__dict__ for c in (CONTROL, NO_FALLBACK, FALLBACK) for iv in interventions_for(c)
            },
            "shaped": SHAPED,
            "gammas": GAMMAS,
            "fallback": {
                "slice": FALLBACK_SLICE,
                "positions": FALLBACK_POSITIONS,
                "threshold": FALLBACK_THRESHOLD,
                "weight": FALLBACK_WEIGHT,
                "bracket": FALLBACK_BRACKET,
                "anti_anchor_weight": ANTI_ANCHOR_WEIGHT,
                "gray": GRAY,
            },
            "lunar": {"retain_weight": LUNAR_RETAIN_WEIGHT, "steps": LUNAR_STEPS, "lr": LUNAR_LR},
            "groups": GROUPS,
            "n_lines": N_LINES,
            "composition": COMPOSITION,
            "positions": POSITIONS,
            "slices": SLICES,
            "dose": {"red": RED_DOSE, "nonred": NONRED_DOSE, "visible_red": VISIBLE_RED_DOSE},
            "traj_stride": TRAJ_STRIDE,
            "decode_l2": DECODE_L2,
            "gates": {
                "fallback_acc": FALLBACK_ACC_GATE,
                "fallback_acc_partial": FALLBACK_ACC_PARTIAL,
                "task": TASK_GATE,
                "task_partial": TASK_PARTIAL,
                "margin_ratio": MARGIN_RATIO,
                "grade_dip": GRADE_DIP,
                "transfer_frac": TRANSFER_FRAC,
                "resolution_sd": RESOLUTION_SD,
                "agree_seeds": AGREE_SEEDS,
            },
        },
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.2.2-metrics.json"))
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    set_ref(ARRAYS_REF, put(buf.getvalue(), name="ex-2.2.2-arrays.npz"))
    traj = {f"{t['label']}/{k}": np.asarray(v, dtype=np.float32) for t in trained for k, v in t["traj"].items()}
    set_ref(TRAJ_REF, put(_npz(**traj), name="ex-2.2.2-trajectories.npz"))
    return {"n_runs": len(results), "n_arrays": len(arrays)}


def main(ctx: Ctx) -> dict:
    prep = ctx.run(prepare_corpus, GRID, N_EXAMPLES, HOLDOUT_FRAC, CORPUS_SEED, role="prep")

    # The cheap stages first: the E6 fits and the stored runs' scores are ready, and the scorer proven,
    # before the GPU stage starts.
    seeds = list(range(NO_FALLBACK.seeds))
    n = len(seeds)
    lunar = ctx.map(
        fit_lunar, seeds, [prep["tables"]] * n, [GRID] * n, [LUNAR_STEPS] * n, [LUNAR_RETAIN_WEIGHT] * n, [LUNAR_LR] * n,
        role="lunar",
    )  # fmt: skip
    by_seed = {fit["seed"]: fit["matrix"] for fit in lunar}
    stored = [(c, s) for c in (CONTROL, NO_FALLBACK) for s in range(c.seeds)]
    scored_stored = ctx.map(
        score_run,
        [c.exp for c, _ in stored],
        [c.name for c, _ in stored],
        [s for _, s in stored],
        [interventions_for(c) for c, _ in stored],
        [None] * len(stored),
        [by_seed[s] if c == NO_FALLBACK else None for c, s in stored],
        role="score",
    )

    cells = build_cells(prep)
    configs, fallbacks, conds, seeds_new, labels = zip(*cells, strict=True)
    m = len(cells)
    trained = ctx.map(
        train_one,
        configs,
        [ANCHOR] * m,
        [ANTI] * m,
        [LABELLER] * m,
        fallbacks,
        [GRID] * m,
        [TRAJ_STRIDE] * m,
        [prep["tables"]] * m,
        labels,
        role="train",
    )
    conditions = {c.name: c for c in (FALLBACK, *ARMS)}
    scored_new = ctx.map(
        score_run,
        ["ex-2.2.2"] * m,
        list(conds),
        list(seeds_new),
        [interventions_for(conditions[c]) for c in conds],
        [t["checkpoint"] for t in trained],
        [None] * m,
        role="score",
    )
    scored = [*scored_stored, *scored_new]
    summary = ctx.run(publish_results, scored, trained, lunar, role="prep")

    def seed_mean(cond: Condition, iv: str, stat: str, group: str) -> float:
        values = [r["interventions"][iv][stat][group] for r in scored if r["condition"] == cond.name]
        return round(float(np.mean(values)), 3)

    conds_all = (CONTROL, NO_FALLBACK, FALLBACK, *ARMS)
    return {
        **summary,
        "fb_acc_redirect": {c.title: seed_mean(c, REDIRECT.name, "fb_acc", "red_clean") for c in conds_all},
        "acc_clean": {
            c.title: round(float(np.mean([r["clean"]["acc"]["all"] for r in scored if r["condition"] == c.name])), 3)
            for c in conds_all
        },
        "nonred_deficit_redirect": {c.title: seed_mean(c, REDIRECT.name, "deficit", "nonred") for c in conds_all},
    }


experiment = Experiment(
    name="ex-2.2.2",
    main=main,
    roles={
        # Corpus sampling is a plain-numpy loop over 100k lines; the publish step stacks 36 npz files.
        "prep": dict(cpu=2, timeout=900),
        # ~3.3k steps of 64×64 tokens as ex-2.1.10, each carrying a second forward and backward through the
        # blocks, so about twice its cost; ex-2.1.10 ran at 3600. The watchdog is sized for the gap after the
        # last step (the checkpoint upload), as there.
        "train": dict(gpu="L4", timeout=7200, watchdog=900, watchdog_grace=900),
        # 2,000 Adam steps on a 64×64 matrix through a frozen d64-L4 model at the recipe's batch: CPU minutes.
        "lunar": dict(cpu=4, timeout=3600, watchdog=600, watchdog_grace=900),
        # One checkpoint per task: a dozen forward passes over 5,832 six-token lines and ~60 small ridge fits.
        "score": dict(cpu=4, timeout=3600, watchdog_grace=1800),
    },
    deps=[EX2110],
)
