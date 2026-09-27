"""
The op-word pass on ex-2.2.14: round 1 of the D2.2 quick route, scoring only.

Ex-2.2.14 put `difference` on e₁ and found the anchor concentrated on the state of the op word: on the op's
lines that position sits at a cosine near 1 with e₁ at every slice, and holds little else. This pass edits that
state on the stored checkpoints (the primary `anchor-diff`, the op-word arm, and the control of ex-2.2.11) and asks
what the edit removes. Two things follow from the geometry, and the pass is built around them.

The projection has no defined landing at the op word: removing e₁ from a state that is almost all e₁ leaves a
small remainder, which the re-normalization scales up, so the edited state is whatever the remainder happened
to be. The pass uses two operators with a landing instead: the reflection (`projection` at γ = 2, a state at α
lands at −α) and a repulsion onto the antipode (`repulsion` with a linear mapper to b = −1, where the state's
off-axis part drops out and every edited state lands on −e₁ itself). And an edit at the op word removes the
token as well as the concept, so every edit is compared with a token-mask row: the state at the op word
replaced, at the same slices, by the mean over the eleven op words of that state given the same first operand.
One row edits
the use site `=` with the plain projection, which is where the whole-line pull put about 0.1 of alignment. The
answer position is not edited: the answer is read from the logits at `=`, so an edit there cannot move it.

Every row runs at all five slices over every probe line of all eleven ops. Per line the pass stores expected
exact match, the distance of the model's answer from the line's, and which color the model names; on the
`difference` lines it also stores the mass on the answer of each other op, against a designed null (the uniform
mixture of the other ten ops) and the line's op-relevance.

    bin/mini run docs/m2/ex-2.2.17/experiment.py --app modal --max-containers 5 --budget 1h
    bin/mini status ex-2.2.17
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir


def _load_ex2214():
    """Ex-2.2.14's module (which loads ex-2.2.11, ex-2.2.9 and ex-2.2.3 the same way), by path and left out of
    `sys.modules`, so the task bodies here still cloudpickle by value for a remote worker.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "ex-2.2.14" / "experiment.py"
    spec = importlib.util.spec_from_file_location("ex2214", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex2214 = _load_ex2214()
ex229 = ex2214.ex229

# --- What ex-2.2.14 fixed, bound by name so a task body never references the module objects -------

PRIMARY: str = ex2214.PRIMARY
OPWORD_ARM: str = ex2214.OPWORD_ARM
CONTROL: str = ex2214.CONTROL
SEEDS: int = ex2214.SEEDS
CONTROL_SEEDS: int = ex2214.CONTROL_SEEDS
ANCHORED_OP: str = ex2214.ANCHORED_OP
OP_NAMES: tuple[str, ...] = ex2214.OP_NAMES
OTHER_OPS: tuple[str, ...] = tuple(o for o in OP_NAMES if o != ANCHORED_OP)
ANCHOR_AXIS: int = ex2214.ANCHOR_AXIS
N_LAYER: int = ex2214.N_LAYER
SLICES: tuple[int, ...] = tuple(range(N_LAYER + 1))
OP_POSITION: int = ex2214.OP_POSITION
EQUALS_POSITION: int = ex2214.EQUALS_POSITION
ANSWER_POSITION: int = ex2214.ANSWER_POSITION
TASK_GATE: float = ex2214.TASK_GATE
TASK_PARTIAL: float = ex2214.TASK_PARTIAL
"""The conditions, the op table, the model's depth, the roles' positions, and the task gate, as ex-2.2.14 had
them. `TASK_GATE` is the largest drop in expected exact match that counts as no cost (0.02), with a partial band
to `TASK_PARTIAL`."""

_npz = ex229._npz
_load = ex229._load
_probe_ops = ex229._probe_ops

assert len(OTHER_OPS) == 10 and ANCHORED_OP == "difference"

# --- What this pass reads ---------------------------------------------------------------------------

EX2214_CHECKPOINT_REF: str = ex2214.CHECKPOINT_REF
EX2211_CHECKPOINT_REF: str = ex2214.EX2211_CHECKPOINT_REF
PROBE_REF: str = ex229.PROBE_REF
"""Ex-2.2.14's checkpoints, ex-2.2.11's control checkpoints, and ex-2.2.9's probe lines (the ones every
experiment since has scored: 5,832 lines per op, 11,664 for the three order-sensitive ops)."""


@dataclass(frozen=True)
class Source:
    """One stored condition: its name, the ref its checkpoints are under, and how many seeds it has."""

    condition: str
    ref: str
    seeds: int
    role: str

    def labels(self) -> list[str]:
        return [f"{self.condition}-s{s}" for s in range(self.seeds)]


SOURCES: tuple[Source, ...] = (
    Source(PRIMARY, EX2214_CHECKPOINT_REF, SEEDS, "the whole-line pull on `difference`; the gated condition"),
    Source(OPWORD_ARM, EX2214_CHECKPOINT_REF, SEEDS, "the pull on the op word alone; a reference for the use site"),
    Source(
        CONTROL,
        EX2211_CHECKPOINT_REF,
        CONTROL_SEEDS,
        "un-anchored; what each operator does to a state that never held the op",
    ),
)
N_RUNS = sum(s.seeds for s in SOURCES)

# --- The rows ---------------------------------------------------------------------------------------

POLE_THRESHOLD = 0.5
"""The threshold *a* of the repulsion: a state below this alignment is left where it is. The alignment map of ex-2.2.14 puts
the anchored op word near 1 and every other op word under 0.1 on the primary, so the threshold separates them
with room on both sides, and the repulsion is untouched on the other ten ops and on the control by construction.
REVIEW: placed midway between the two populations; a ladder of landings (b ∈ {−0.25, −0.5, −1}) is the todo
item `repulsion-onto-the-fallback` and waits for a trained fallback, so one landing here."""

POLE_LANDING = -1.0
"""The landing *b* of the repulsion: the antipode of the anchor. At b = −1 the √(1 − b²) factor of the mapper is zero, so
the off-axis part of the arriving state drops out and every edited state lands on −e₁ exactly. It is the one
landing the "declared landing state" of the design can mean without choosing a direction off the axis.
REVIEW: the other reading is a landing *at* the mean state of the token mask, which is not a rotation in the
plane of e₁ and would need an operator of its own; the mask row covers that state as a replacement."""

REFLECT_GAMMA = 2.0
"""The reflection: `projection` at γ = 2 sends a state at alignment α to −α, keeping its off-axis part."""


@dataclass(frozen=True)
class Row:
    """One edit: where it acts (a position, at every slice), and which operator."""

    name: str
    position: int
    kind: str
    """`projection` (with `gamma`), `repulsion` (with `a`, `b`, linear mapper), or `mask` (the token mask)."""
    gamma: float = 1.0
    a: float = 0.0
    b: float = 0.0
    role: str = ""


ROWS: tuple[Row, ...] = (
    Row(
        "reflect",
        OP_POSITION,
        "projection",
        gamma=REFLECT_GAMMA,
        role="the op word reflected through the equator: α → −α",
    ),
    Row(
        "pole",
        OP_POSITION,
        "repulsion",
        a=POLE_THRESHOLD,
        b=POLE_LANDING,
        role="the op word sent to −e₁ when its alignment is at least the threshold",
    ),
    Row(
        "mask",
        OP_POSITION,
        "mask",
        role="the state at the op word replaced by the mean over the eleven op words, given the same first operand",
    ),
    Row(
        "equals",
        EQUALS_POSITION,
        "projection",
        gamma=1.0,
        role="e₁ removed at `=`, the use site the whole-line pull reached",
    ),
)
OP_WORD_ROWS: tuple[str, ...] = tuple(r.name for r in ROWS if r.position == OP_POSITION)
PASSES: tuple[str, ...] = ("clean", *(r.name for r in ROWS))
"""The clean forward pass first, then every row. Each per-line array is stored under `{op}/{pass}/{stat}`."""

# --- Gates ------------------------------------------------------------------------------------------

REMOVAL_GATE = 0.25
REMOVAL_PARTIAL = 0.5
"""H1: on the named `difference` lines (no other op shares the answer), the seed-mean expected exact match under
an op-word edit on the primary, which is about 0.97 clean, falls to at most `REMOVAL_GATE`; between the two
levels is partial. REVIEW: gated on expected exact match rather than the answer distance the answer-distance
re-score adopted, because the fallback here is a mixture of other ops whose answers straddle `difference`'s, so
the mean of the model's distribution can sit near the right answer while the mass on it is gone. The
normalized distances are quoted beside, and the threshold is a proposal."""

NULL_TOL = 0.1
NULL_MIN_LINES = 100
"""H2: per op-relevance bin (lines grouped by how many other ops share their answer), the seed-mean expected
exact match under an op-word edit is within `NULL_TOL` of what the designed null predicts, 1 − r̄, on every bin
with at least `NULL_MIN_LINES` lines. REVIEW: the tolerance is a proposal."""

SELECTIVITY_GATE = TASK_GATE
SELECTIVITY_PARTIAL = TASK_PARTIAL
"""H3: on each of the other ten ops, the seed-mean drop in expected exact match under an op-word edit on the
primary is at most the task gate; the control under the same row is the calibration."""

MARKER_MAX = 0.1
CARRIER_MIN = 0.5
"""H4: the drop in expected exact match on the `difference` lines under the `=` edit on the primary. At most
`MARKER_MAX` the axis at `=` marks the op without carrying it; at least `CARRIER_MIN` it carries it; between,
unresolved. REVIEW: both levels are proposals; the drop under the mask row bounds what "carrying" could mean."""

# --- Refs -------------------------------------------------------------------------------------------

METRICS_REF = "reports/m2/ex-2.2.17/metrics"
ARRAYS_REF = "reports/m2/ex-2.2.17/arrays"
LINES_REF = "reports/m2/ex-2.2.17/lines"
"""The run summaries (JSON), every run's per-line arrays stacked under one ref, and the per-line constants
(raw answers, floors, chance, op-relevance, the answer distributions of the other ops) under another."""

BATCH = 2048
"""Lines per forward pass."""

# =============================================================================================
# Line constants
# =============================================================================================


def load_probes(probes, workdir) -> dict[str, dict[str, np.ndarray]]:
    from mini.store import get

    keys = ("tokens", "q_idx", "q_p")
    with np.load(get(probes, workdir / "probes.npz")) as z:
        return {o: {k: z[f"{o}/{k}"] for k in keys} for o in _probe_ops(z)}


def other_answers(op1: np.ndarray, op2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The answer distribution of every other op on each line, `(N, 10, 8)` as palette indices (−1 padded) and
    probabilities, in `OTHER_OPS` order.
    """
    from sca.data.ops import CANDIDATE_BY_NAME, OP_BY_NAME, answer_dist, colors

    cs = colors()
    index = {c: i for i, c in enumerate(cs)}
    ops = [(OP_BY_NAME | CANDIDATE_BY_NAME)[o] for o in OTHER_OPS]
    q_idx = np.full((len(op1), len(ops), 8), -1, np.int16)
    q_p = np.zeros((len(op1), len(ops), 8), np.float32)
    for n, (a, b) in enumerate(zip(op1, op2, strict=True)):
        for j, op in enumerate(ops):
            for k, (color, p) in enumerate(answer_dist(op, cs[a], cs[b]).items()):
                q_idx[n, j, k], q_p[n, j, k] = index[color], p
    return q_idx, q_p


def line_constants(probes, checkpoint) -> dict:
    """What is fixed per line and shared by every run: the raw answer, its floor, chance, and the operand colors
    for every op; and on the anchored op, the answer distributions of the other ten ops, the designed null they make,
    the line's op-relevance *r*, and *k*, how many of them name the same answer.
    """
    from mini.store import put
    from sca import answer_distance as ad

    workdir = get_data_dir() / "lines"
    probe = load_probes(probes, workdir)
    _, _, _, tok2color = _load({"checkpoint": checkpoint}, workdir)
    arrays: dict[str, np.ndarray] = {}
    counts: dict[str, Any] = {}
    for op, f in probe.items():
        raw = ad.raw_answer(f["q_idx"], f["q_p"])
        arrays |= {
            f"{op}/raw": raw.astype(np.float32),
            f"{op}/floor_mode": ad.floors(f["q_idx"], f["q_p"])["mode"].astype(np.float32),
            f"{op}/chance": ad.chance(raw).astype(np.float32),
            f"{op}/operands": tok2color[f["tokens"][:, [0, 2]]].astype(np.int16),
        }
        counts[op] = int(len(raw))
    f = probe[ANCHORED_OP]
    op1, op2 = arrays[f"{ANCHORED_OP}/operands"].T
    assert (f["q_p"].max(axis=1) >= 1.0 - 1e-9).all(), "the answers of the anchored op are on the grid"
    answer = f["q_idx"][np.arange(len(op1)), f["q_p"].argmax(axis=1)]
    q_idx, q_p = other_answers(op1, op2)
    # The probability each other op gives the answer of the anchored op, then the null (their uniform mixture) over the grid.
    agree = np.where(q_idx == answer[:, None, None], q_p, 0.0).sum(axis=2)  # (N, 10)
    null = np.zeros((len(op1), len(ad.GRID)), np.float32)
    for j in range(q_idx.shape[1]):
        np.add.at(null, (np.arange(len(op1))[:, None], np.maximum(q_idx[:, j], 0)), q_p[:, j] / q_idx.shape[1])
    modes = np.take_along_axis(q_idx, q_p.argmax(axis=2)[..., None], axis=2)[..., 0]  # (N, 10)
    k = (modes == answer[:, None]).sum(axis=1)
    r = 1.0 - agree.mean(axis=1)
    arrays |= {
        f"{ANCHORED_OP}/other_q_idx": q_idx,
        f"{ANCHORED_OP}/other_q_p": q_p,
        f"{ANCHORED_OP}/k": k.astype(np.int8),
        f"{ANCHORED_OP}/r": r.astype(np.float32),
        f"{ANCHORED_OP}/null_eem": (1.0 - r).astype(np.float32),
        f"{ANCHORED_OP}/null_mean": ad.distance(ad.mean(null), arrays[f"{ANCHORED_OP}/raw"]).astype(np.float32),
    }
    counts["k"] = {int(kk): int(c) for kk, c in zip(*np.unique(k, return_counts=True), strict=True)}
    return {"n_lines": counts, "arrays": put(_npz(**arrays), name="ex-2.2.17-lines.npz")}


# =============================================================================================
# The scoring task
# =============================================================================================


def op_word_targets(model, op_ids: np.ndarray, color_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """The landing states of the token mask: for each first operand, the mean over the eleven op words of the clean
    state at the op position, per slice, re-normalized; `(L1, 216, C)` in palette order. Under causal attention
    that state depends on the first two tokens alone, so it comes from the 2,376 two-token prefixes. Also returns
    the per-word states `(L1, 216, 11, C)` the mean was taken over, and summary statistics of the mean.
    """
    from sca.intervention import Subspace, apply, projection

    prefixes = np.stack(np.meshgrid(color_ids, op_ids, indexing="ij"), axis=-1).reshape(-1, 2)
    sub = Subspace.axis(model.transformer.wte.shape[1], ANCHOR_AXIS)
    clean = apply(model, prefixes, projection(sub), slices=(), batch_size=BATCH)
    states = clean.pre[:, :, OP_POSITION].reshape(clean.pre.shape[0], len(color_ids), len(op_ids), -1)
    mean = states.mean(axis=2)
    norm = np.linalg.norm(mean, axis=-1)
    stats = {
        # How far apart the eleven op words sit: 1 when they share a state, small when they spread out.
        "norm": norm.mean(axis=1).tolist(),
        "alpha": (mean[..., ANCHOR_AXIS] / norm).mean(axis=1).tolist(),
        "alpha_by_op": states[..., ANCHOR_AXIS].mean(axis=1).T.tolist(),  # (11, L1)
    }
    return (mean / norm[..., None]).astype(np.float32), states, stats


def mask_apply(model, tokens: np.ndarray, targets: np.ndarray, tok2color: np.ndarray, slices: tuple[int, ...]):
    """`sca.intervention.apply` with the token mask spliced in: at each of *slices*, the state at the op position
    is replaced by the target for the line's first operand. Returns the same `Applied` triple.
    """
    import equinox as eqx
    import jax.numpy as jnp

    from sca.intervention import Applied
    from sca.model._shared import normalize

    tgt_table = jnp.asarray(targets)
    color_of = jnp.asarray(tok2color)

    def forward(m, idx):
        tgt = tgt_table[:, color_of[idx[:, 0]]]  # (L1, B, C)

        def edit(x, ell):
            return x.at[:, OP_POSITION, :].set(tgt[ell])

        x = normalize(m.transformer.wte[idx])
        pre, post = [x], [edit(x, 0) if 0 in slices else x]
        x = post[-1]
        for i, block in enumerate(m.transformer.blocks, start=1):
            x = block(x, m.transformer.rotary_enc)
            pre.append(x)
            post.append(edit(x, i) if i in slices else x)
            x = post[-1]
        return jnp.stack(pre), jnp.stack(post), (x @ m.transformer.readout.T) * m.s_z()

    run = eqx.filter_jit(forward)
    pres, posts, logits = [], [], []
    for i in range(0, len(tokens), BATCH):
        a, b, c = run(model, jnp.asarray(tokens[i : i + BATCH]))
        pres.append(np.asarray(a))
        posts.append(np.asarray(b))
        logits.append(np.asarray(c))
    return Applied(np.concatenate(pres, axis=1), np.concatenate(posts, axis=1), np.concatenate(logits))


def run_row(row: Row, model, sub, tokens: np.ndarray, targets: np.ndarray, tok2color: np.ndarray):
    from sca.intervention import apply, projection, repulsion

    if row.kind == "mask":
        return mask_apply(model, tokens, targets, tok2color, SLICES)
    positions = (np.arange(tokens.shape[1]) == row.position).astype(np.float32)
    operator = projection(sub, row.gamma) if row.kind == "projection" else repulsion(sub, row.a, row.b, kind="linear")
    return apply(model, tokens, operator, slices=SLICES, positions=positions, batch_size=BATCH)


def coefficient(sub, states: np.ndarray) -> np.ndarray:
    """`Subspace.coefficients` on numpy states: how much of the anchored direction each state carries."""
    return ((states - sub.mean) @ sub.dual.T)[..., 0]


def check_row(row: Row, out, sub, targets: np.ndarray, colors: np.ndarray) -> dict[str, float]:
    """The contract checks: nothing moves but the row's position, and the write at that position is what the
    closed form of the operator says. Returns the largest deviation of each check, and how many states the
    repulsion left where they were because they arrived fully aligned (their off-axis part undefined).
    """
    from sca.intervention import angle_between, repulsion_mapper, write_angle

    theta = angle_between(out.pre, out.post)  # (L1, N, T)
    others = np.arange(theta.shape[2]) != row.position
    np.testing.assert_allclose(theta[:, :, others], 0.0, atol=1e-5)
    alpha_pre = coefficient(sub, out.pre[:, :, row.position])  # (L1, N)
    alpha_post = coefficient(sub, out.post[:, :, row.position])
    measured = theta[:, :, row.position]
    stuck = 0
    if row.kind == "projection":
        expected = write_angle(alpha_pre, row.gamma)
    elif row.kind == "repulsion":
        a = np.maximum(alpha_pre, 0.0)
        m = repulsion_mapper(a, row.a, row.b, "linear")
        rest = np.linalg.norm(out.pre[:, :, row.position] - a[..., None] * sub.basis[0], axis=-1)
        moved = (m != a) & (rest > 1e-6)
        stuck = int(((m != a) & ~moved).sum())
        expected = np.where(moved, np.abs(np.arccos(np.clip(m, -1, 1)) - np.arccos(np.clip(a, -1, 1))), 0.0)
        np.testing.assert_allclose(alpha_post[moved], m[moved], atol=2e-3)
    else:
        expected = measured
        landed = targets[:, colors]  # (L1, N, C)
        np.testing.assert_allclose(out.post[:, :, row.position], landed, atol=1e-4)
    np.testing.assert_allclose(measured, expected, atol=2e-3)
    return {"write_dev": float(np.abs(measured - expected).max()), "stuck": stuck}


def measure(p: np.ndarray, q_idx: np.ndarray, q_p: np.ndarray, raw: np.ndarray, floor: np.ndarray, chance: np.ndarray):
    """Per-line statistics from color mass *p* `(N, 216)` in palette order: expected exact match; the distance
    of the mean of the distribution and of the greedy guess from the raw answer, in grid steps and normalized
    so that 0 is a perfect answer and 1 is chance; the guess; and the mass off the color vocabulary.
    """
    from sca import answer_distance as ad

    guess = p.argmax(axis=1)
    greedy = ad.distance(ad.GRID[guess], raw)
    mean = ad.distance(ad.mean(p), raw)
    return {
        "eem": (np.take_along_axis(p, np.maximum(q_idx, 0), axis=1) * q_p).sum(1),
        "mean": mean,
        "mean_norm": mean / chance,
        "greedy": greedy,
        "greedy_norm": (greedy - floor) / (chance - floor),
        "guess": guess,
        "offvocab": 1.0 - p.sum(axis=1),
    }


def score_one(checkpoint, probes, lines: dict, condition: str, seed: int, label: str) -> dict:
    """One stored checkpoint under the clean pass and every row, over every probe line of every op.

    Per op and pass: the means of `measure`'s statistics, the alignment at the edited position before and after
    the edit per slice, and the write per slice. On the anchored op: the same by op-relevance (named lines, shared
    lines, each *k*), and the mass on the answer of each other op. Per-line arrays go to the store.
    """
    from mini.progress import emit_progress
    from mini.store import get, put
    from sca.intervention import Subspace, angle_between, answer_logprobs

    workdir = get_data_dir() / "score" / label
    model, tokenizer, color_ids, tok2color = _load({"checkpoint": checkpoint}, workdir)
    sub = Subspace.axis(model.transformer.wte.shape[1], ANCHOR_AXIS)
    probe = load_probes(probes, workdir)
    op_ids = np.array([tokenizer.stoi[o] for o in OP_NAMES])
    targets, word_states, target_stats = op_word_targets(model, op_ids, color_ids)
    consts = np.load(get(lines["arrays"], workdir / "lines.npz"))

    def coeff(states: np.ndarray, position: int) -> np.ndarray:
        return coefficient(sub, states[:, :, position])  # (L1, N)

    def summarize(stats: dict[str, np.ndarray], m: np.ndarray) -> dict[str, float]:
        keys = ("eem", "mean", "mean_norm", "greedy", "greedy_norm", "offvocab")
        return {k: float(stats[k][m].mean()) for k in keys} | {"n": int(m.sum())}

    summary: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    contract: dict[str, dict[str, float]] = {}
    for i, (op, f) in enumerate(probe.items()):
        emit_progress(i, len(probe), op)
        tokens = f["tokens"]
        colors = consts[f"{op}/operands"][:, 0]
        raw, floor, chance = consts[f"{op}/raw"], consts[f"{op}/floor_mode"], consts[f"{op}/chance"]
        every = np.ones(len(tokens), bool)
        groups = {"all": every}
        if op == ANCHORED_OP:
            k = consts[f"{op}/k"]
            groups |= {"named": k == 0, "shared": k > 0} | {f"k{kk}": k == kk for kk in np.unique(k)}
            other_idx, other_p = consts[f"{op}/other_q_idx"], consts[f"{op}/other_q_p"]

        clean = run_row(Row("clean", OP_POSITION, "projection", gamma=0.0), model, sub, tokens, targets, tok2color)
        # The causal claim behind the prefix table of the mask: the clean state at the op word is the state of the prefix.
        np.testing.assert_allclose(clean.pre[:, :, OP_POSITION], word_states[:, colors, OP_NAMES.index(op)], atol=1e-4)
        outs = {"clean": clean} | {row.name: run_row(row, model, sub, tokens, targets, tok2color) for row in ROWS}
        per_op: dict[str, Any] = {}
        for name, out in outs.items():
            p = np.exp(answer_logprobs(out.logits, ANSWER_POSITION)[:, color_ids])  # (N, 216), palette order
            stats = measure(p, f["q_idx"], f["q_p"], raw, floor, chance)
            entry: dict[str, Any] = {g: summarize(stats, m) for g, m in groups.items()}
            row = next((r for r in ROWS if r.name == name), None)
            site = OP_POSITION if row is None else row.position
            entry["alpha_pre"] = coeff(out.pre, site).mean(axis=1).tolist()
            entry["alpha_post"] = coeff(out.post, site).mean(axis=1).tolist()
            entry["write"] = angle_between(out.pre, out.post)[:, :, site].mean(axis=1).tolist()
            if row is None:
                # The clean alignment at both edited sites, per slice: the reference for every row's `alpha_pre`.
                entry["alpha_equals"] = coeff(out.pre, EQUALS_POSITION).mean(axis=1).tolist()
            if row is not None:
                contract[f"{op}/{name}"] = check_row(row, out, sub, targets, colors)
            arrays |= {
                f"{op}/{name}/eem": stats["eem"].astype(np.float16),
                f"{op}/{name}/mean_norm": stats["mean_norm"].astype(np.float16),
                f"{op}/{name}/guess": stats["guess"].astype(np.int16),
            }
            if op == ANCHORED_OP:
                # The mass on the answer of each other op: where the removed answers go, op by op.
                comp = (np.take_along_axis(p[:, None, :], np.maximum(other_idx, 0), axis=2) * other_p).sum(2)  # (N, 10)
                entry["composition"] = {
                    g: dict(zip(OTHER_OPS, comp[m].mean(axis=0).tolist(), strict=True)) for g, m in groups.items()
                }
                arrays[f"{op}/{name}/composition"] = comp.astype(np.float16)
                if site == OP_POSITION:
                    arrays[f"{op}/{name}/alpha_pre"] = coeff(out.pre, site).astype(np.float16)
                    arrays[f"{op}/{name}/alpha_post"] = coeff(out.post, site).astype(np.float16)
            per_op[name] = entry
        summary[op] = per_op
        del outs, clean
    emit_progress(len(probe), len(probe), "done")

    return {
        "label": label,
        "condition": condition,
        "seed": seed,
        "ops": summary,
        "mask_target": target_stats,
        "contract": contract,
        "arrays": put(_npz(**arrays), name=f"ex-2.2.17-{label}-arrays.npz"),
    }


# --- Publishing ------------------------------------------------------------------------------------


def design() -> dict[str, Any]:
    return {
        "sources": [{"condition": s.condition, "seeds": s.seeds, "ref": s.ref, "role": s.role} for s in SOURCES],
        "rows": [r.__dict__ for r in ROWS],
        "passes": list(PASSES),
        "ops": list(OP_NAMES),
        "other_ops": list(OTHER_OPS),
        "anchored_op": ANCHORED_OP,
        "slices": list(SLICES),
        "positions": {"op": OP_POSITION, "equals": EQUALS_POSITION, "answer": ANSWER_POSITION},
        "gates": {
            "removal": [REMOVAL_GATE, REMOVAL_PARTIAL],
            "null_tol": NULL_TOL,
            "null_min_lines": NULL_MIN_LINES,
            "selectivity": [SELECTIVITY_GATE, SELECTIVITY_PARTIAL],
            "marker_max": MARKER_MAX,
            "carrier_min": CARRIER_MIN,
        },
    }


def publish_results(lines: dict, scored: list[dict]) -> dict:
    """Metrics (JSON) with the design, the line constants under `LINES_REF`, and every run's per-line arrays
    stacked under `ARRAYS_REF`.
    """
    import json

    from mini.store import get_many, put, set_ref

    metrics = {
        "runs": [{k: v for k, v in r.items() if k != "arrays"} for r in scored],
        "n_lines": lines["n_lines"],
        "design": design(),
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.2.17-metrics.json"))
    set_ref(LINES_REF, lines["arrays"])
    workdir = get_data_dir() / "publish"
    paths = get_many([(r["arrays"], workdir / f"{r['label']}-arrays.npz") for r in scored])
    stacked: dict[str, np.ndarray] = {}
    for r, path in zip(scored, paths, strict=True):
        with np.load(path) as z:
            stacked |= {f"{r['label']}/{name}": z[name] for name in z.files}
    set_ref(ARRAYS_REF, put(_npz(**stacked), name="ex-2.2.17-arrays.npz"))
    return {"n_runs": len(scored), "n_arrays": len(stacked)}


# --- Orchestration ----------------------------------------------------------------------------------


def resolve_inputs() -> dict:
    """The stored checkpoints and the probe set, by ref, so each scoring task is keyed on what it reads."""
    from mini.store import get_refs

    names = {(s.condition, lb): s.ref.format(label=lb) for s in SOURCES for lb in s.labels()}
    refs = get_refs([PROBE_REF, *names.values()])
    missing = [n for n, v in refs.items() if v is None]
    assert not missing, f"refs not in this store: {missing}"
    return {"probes": refs[PROBE_REF], "checkpoints": {lb: refs[n] for (_, lb), n in names.items()}}


def main(ctx: Ctx) -> dict:
    inputs = ctx.run(resolve_inputs, role="prep")
    runs = [(s.condition, seed, lb) for s in SOURCES for seed, lb in enumerate(s.labels())]
    lines = ctx.run(line_constants, inputs["probes"], inputs["checkpoints"][runs[0][2]], role="prep")
    scored = ctx.map(
        score_one,
        [inputs["checkpoints"][lb] for _, _, lb in runs],
        [inputs["probes"]] * len(runs),
        [lines] * len(runs),
        [c for c, _, _ in runs],
        [s for _, s, _ in runs],
        [lb for _, _, lb in runs],
        role="score",
    )
    return ctx.run(publish_results, lines, scored, role="prep")


# The score task also runs on CPU (`--app local`): five passes over 81,648 six-token lines is a few minutes a
# checkpoint. The thread caps keep concurrent local JAX processes from each claiming every core.
THREAD_ENV = {
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2",
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
}

experiment = Experiment(
    name="ex-2.2.17",
    main=main,
    roles={
        "prep": dict(cpu=2, timeout=900, env=THREAD_ENV),
        # Five passes per op, keeping the stream for the contract checks; the largest op is 11,664 lines.
        "score": dict(gpu="L4", timeout=1800, env=THREAD_ENV),
    },
)
