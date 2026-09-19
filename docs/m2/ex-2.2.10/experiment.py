"""
A second scouting round, on the runs ex-2.2.9 stored: what the model answers on the order-sensitive
ops once *red* is projected out, read as answer mass and as decoded hidden states.

Scouting, not scored: no gates, no verdicts. Ex-2.2.9 missed its removal gate on `hue-hsv`, `sat-hsv`
and `value-hsv` alone, and the miss was one-sided by slot. The report's reading, made after the data,
was that the axis carries the *redness* of a color while a red color's saturation and value are read
from elsewhere. This pass gives that reading something to be checked against. For every red line of
those three ops (and of `mix`, as the op where removal is clean), on `handover`'s twenty seeds, it
stores the model's whole answer distribution over the grid, clean and under the projection, and the
residual stream at every site, decoded through linear RGB probes fit on the non-red lines. The report
derives, from the ops' own code, what a model that had lost only the hue of *red* would answer, and
reads the stored answers against that.

The other two questions ex-2.2.9 left (retention through the anneal, containment under the untied
readout) are reads of what ex-2.2.9 already published, and need nothing from this module but its refs.

    bin/mini run docs/m2/ex-2.2.10/experiment.py --app modal --max-containers 5
    bin/mini status ex-2.2.10
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir


def _load_ex229():
    """Ex-2.2.9's module, loaded by path and left out of `sys.modules`, as it loads ex-2.2.3 itself, so
    the task bodies here still cloudpickle by value for a remote worker.
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

# Bound by name so a task body never references the module object itself.
Readout = ex229.Readout
_load = ex229._load
_probe_ops = ex229._probe_ops
_npz = ex229._npz
GRID_RGB = ex229.GRID_RGB
PALETTE = ex229.PALETTE
REDNESS = ex229.REDNESS
ANCHOR_AXIS = ex229.ANCHOR_AXIS
SLICES = ex229.SLICES
ANSWER_POS = ex229.ANSWER_POS
DECODE_POS = ex229.DECODE_POS
OPERAND_POSITIONS = ex229.OPERAND_POSITIONS
ORDER_SENSITIVE = ex229.ORDER_SENSITIVE
RED_DOSE = ex229.RED_DOSE
FAR_MOVE = ex229.FAR_MOVE
TABLE = ex229.TABLE

# --- What this pass reads ------------------------------------------------------------------

EX229_METRICS_REF = ex229.METRICS_REF
EX229_ARRAYS_REF = ex229.ARRAYS_REF
EX229_TRAJ_REF = ex229.TRAJ_REF
EX229_PROBE_REF = ex229.PROBE_REF
EX229_CHECKPOINT_REF = ex229.CHECKPOINT_REF
"""Ex-2.2.9's published refs. The metrics, the stacked arrays and the trajectories are what the report's
pure reads use; the checkpoints and the probe set are what the scoring task here runs on."""

CONDITION = ex229.HANDOVER.name
SEEDS = tuple(range(ex229.HANDOVER.seeds))
"""Every seed of the candidate condition: the answer masses are cheap, and twenty seeds are what the removal
read was made on."""

STATE_SEEDS = tuple(range(5))
"""The seeds whose raw residual states are kept beside the decoded ones (about 3 MB per op per seed, against
a few hundred kB for the decoded coordinates), for a probe the report may want to fit on its own terms."""

OPS_READ: tuple[str, ...] = (ex229.PRIMARY_OP, *ORDER_SENSITIVE)
"""The three ops that missed the removal gate, and `mix` as the op where removal is clean, read the same way
so the figures have a reference panel."""

OPERATOR = "projection"
"""The removal operator ex-2.2.9 gated: the plain projection off the axis, at every slice and position."""

PROBE_L2 = 1e-2
"""The ridge on the probes, as `sca.compute.geometry` fits them."""

PROBE_TARGETS: dict[str, int] = {"op1": 0, "op2": 2, "ans": ANSWER_POS}
"""The three colors a line carries, each read from every site: op1, op2, and the answer. The answer target
is the rule's raw (unrounded) color, since under stochastic rounding the drawn token is one sample of it."""

METRICS_REF = "reports/m2/ex-2.2.10/metrics"
ARRAYS_REF = "reports/m2/ex-2.2.10/arrays"
STATES_REF = "reports/m2/ex-2.2.10/states/{label}"
"""The stacked per-line arrays (answer masses, decoded coordinates, probe weights) for every run under one
ref, and the raw states of the `STATE_SEEDS` runs under one ref each."""

# =============================================================================================
# The scoring task
# =============================================================================================


def ridge_fit(x: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge fit → (weights (C, K), bias (K,)): `sca.compute.geometry`'s fit, on the full data."""
    x64, y64 = np.asarray(x, np.float64), np.asarray(y, np.float64)
    mx, my = x64.mean(0), y64.mean(0)
    xc = x64 - mx
    w = np.linalg.solve(xc.T @ xc + l2 * np.eye(x64.shape[1]), xc.T @ (y64 - my))
    return w, my - mx @ w


def r2_cols(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """R² per target column."""
    ss = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ((y - pred) ** 2).sum(0) / np.maximum(ss, 1e-12)


def raw_answers(op_name: str, tokens: np.ndarray, tok2color: np.ndarray) -> np.ndarray:
    """The rule's unrounded answer of every line, in the unit cube: the probe target for `ans`."""
    from sca.data.ops import CANDIDATE_BY_NAME, OP_BY_NAME, TOP, colors

    op = (OP_BY_NAME | CANDIDATE_BY_NAME)[op_name]
    cs = colors()
    idx = tok2color[tokens]
    return np.array([op.raw(cs[a], cs[b]) for a, b in zip(idx[:, 0], idx[:, 2], strict=True)], float) / TOP


def score_one(
    checkpoint, probes, ops: tuple[str, ...], keep_states: bool, condition: str, seed: int, label: str
) -> dict:
    """One stored checkpoint, one op at a time: the whole answer distribution of every red line, clean and
    under the projection; RGB probes fit at every site on the clean stream of the non-red lines; and the red
    lines' states decoded through them under both passes.

    Per-run summaries return as the result (the probes' fit on the red lines, and the kept share per group,
    which should reproduce ex-2.2.9's). Per-line arrays go to the store.
    """
    from sca.intervention import Subspace, answer_logprobs, apply, projection
    from mini.progress import emit_progress
    from mini.store import get, put

    workdir = get_data_dir() / "score" / label
    model, _, color_ids, tok2color = _load({"checkpoint": checkpoint}, workdir)
    sub = Subspace.axis(model.transformer.wte.shape[1], ANCHOR_AXIS)
    keys = ("tokens", "r1", "r2", "q_idx", "q_p", "move")
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe = {o: {k: z[f"{o}/{k}"] for k in keys} for o in _probe_ops(z) if o in ops}

    summary: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    states: dict[str, np.ndarray] = {}
    for i, (op, f) in enumerate(probe.items()):
        emit_progress(i, len(probe), op)
        read = Readout(f["tokens"], f["r1"], f["r2"], f["q_idx"], f["q_p"], f["move"], color_ids, tok2color)
        red = read.groups["red"]
        rows = np.flatnonzero(red)
        tokens = read.tokens
        n_pos = tokens.shape[1]

        # --- Two passes over every line: clean, and the projection at every slice and position.
        clean = apply(model, tokens, projection(sub), slices=())
        edited = apply(model, tokens, projection(sub), slices=SLICES)
        c, e = read(clean.logits), read(edited.logits)
        summary[op] = {
            "n": {g: int(m.sum()) for g, m in read.groups.items()},
            "kept": read.ratio_by_group(e["eem"], c["eem"]),
            "eem_clean": read.by_group(c["eem"]),
        }

        # --- The answer distribution over the grid, for the red lines under both passes.
        for name, out in (("clean", clean), ("projection", edited)):
            lp = answer_logprobs(out.logits[rows], ANSWER_POS)
            arrays[f"{op}/{name}/mass"] = np.exp(lp[:, color_ids]).astype(np.float16)  # (N_red, 216)
        for k in ("eem", "guess", "expected_dist"):
            arrays[f"{op}/clean/{k}"] = c[k][rows]
            arrays[f"{op}/projection/{k}"] = e[k][rows]
        arrays[f"{op}/rows"] = rows.astype(np.int32)  # which probe lines, in the file's order
        arrays[f"{op}/removal"] = read.groups["removal"][rows]
        arrays[f"{op}/move"] = read.move[rows].astype(np.float32)
        arrays[f"{op}/red_operand"] = read.red_operand[rows].astype(np.int8)
        arrays[f"{op}/ans_idx"] = read.ans_idx[rows].astype(np.int16)
        arrays[f"{op}/tokens"] = tok2color[tokens[rows]].astype(np.int16)  # palette index, −1 at syntax

        # --- Probes: one ridge fit per (slice, position, target) on the clean stream of the non-red lines,
        #     then the red lines decoded through them, clean and edited. `pre` at slice s is the state as it
        #     arrived there: the clean stream on the clean pass, and downstream of the earlier edits on the
        #     other, which is what the next block would have read.
        targets = {
            "op1": GRID_RGB[tok2color[tokens[:, 0]]],
            "op2": GRID_RGB[tok2color[tokens[:, 2]]],
            "ans": raw_answers(op, tokens, tok2color),
        }
        fit_rows = np.flatnonzero(~red)
        n_slices = clean.pre.shape[0]
        width = clean.pre.shape[-1]
        weights = np.zeros((n_slices, n_pos, len(targets), width, 3), np.float32)
        bias = np.zeros((n_slices, n_pos, len(targets), 3), np.float32)
        r2_fit = np.zeros((n_slices, n_pos, len(targets), 3), np.float32)
        r2_red = np.zeros((n_slices, n_pos, len(targets), 3), np.float32)
        decoded = {
            name: np.zeros((n_slices, n_pos, len(targets), len(rows), 3), np.float16)
            for name in ("clean", "projection")
        }
        for s in range(n_slices):
            for p in range(n_pos):
                x_fit = clean.pre[s, fit_rows, p]
                for t, y in enumerate(targets.values()):
                    w, b = ridge_fit(x_fit, y[fit_rows], PROBE_L2)
                    weights[s, p, t], bias[s, p, t] = w, b
                    r2_fit[s, p, t] = r2_cols(x_fit @ w + b, y[fit_rows])
                    dc = clean.pre[s, rows, p] @ w + b
                    r2_red[s, p, t] = r2_cols(dc, y[rows])
                    decoded["clean"][s, p, t] = dc
                    decoded["projection"][s, p, t] = edited.pre[s, rows, p] @ w + b
        arrays[f"{op}/probe/weights"] = weights
        arrays[f"{op}/probe/bias"] = bias
        arrays[f"{op}/probe/r2_fit"] = r2_fit
        arrays[f"{op}/probe/r2_red"] = r2_red
        for name, d in decoded.items():
            arrays[f"{op}/{name}/decoded"] = d
        summary[op]["probe"] = {
            "targets": list(targets),
            # The fit on the red lines at the sites the report reads first: each operand at its own
            # position, and the answer at `=`, per slice.
            "r2_red": {
                "op1@op1": r2_red[:, 0, 0].mean(-1).tolist(),
                "op2@op2": r2_red[:, 2, 1].mean(-1).tolist(),
                "ans@=": r2_red[:, DECODE_POS, 2].mean(-1).tolist(),
            },
        }
        if keep_states:
            states[f"{op}/clean"] = clean.pre[:, rows].astype(np.float16)  # (L1, N_red, T, C)
            states[f"{op}/projection"] = edited.pre[:, rows].astype(np.float16)
    emit_progress(len(probe), len(probe), "done")

    result: dict[str, Any] = {
        "label": label,
        "condition": condition,
        "seed": seed,
        "ops": summary,
        "arrays": put(_npz(**arrays), name=f"ex-2.2.10-{label}-arrays.npz"),
    }
    if keep_states:
        result["states"] = put(_npz(**states), name=f"ex-2.2.10-{label}-states.npz")
    return result


# --- Publishing ------------------------------------------------------------------------------


def design() -> dict[str, Any]:
    return {
        "condition": CONDITION,
        "seeds": list(SEEDS),
        "state_seeds": list(STATE_SEEDS),
        "ops": list(OPS_READ),
        "operator": OPERATOR,
        "probe_l2": PROBE_L2,
        "probe_targets": PROBE_TARGETS,
        "slices": list(SLICES),
        "answer_pos": ANSWER_POS,
        "decode_pos": DECODE_POS,
        "red_dose": RED_DOSE,
        "far_move": FAR_MOVE,
    }


def publish_results(scored: list[dict]) -> dict:
    """Metrics (JSON) and the per-line arrays of every run stacked under one ref; each run's raw states
    under a ref of its own.
    """
    import json

    from mini.store import get, put, set_ref

    metrics = {
        "scores": [{k: v for k, v in r.items() if k not in ("arrays", "states")} for r in scored],
        "design": design(),
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.2.10-metrics.json"))
    arrays = {}
    for r in scored:
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}-arrays.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{name}": z[name] for name in z.files}
        if "states" in r:
            set_ref(STATES_REF.format(label=r["label"]), r["states"])
    set_ref(ARRAYS_REF, put(_npz(**arrays), name="ex-2.2.10-arrays.npz"))
    return {"n_runs": len(scored), "ops": list(OPS_READ)}


# --- Orchestration ----------------------------------------------------------------------------


def resolve_inputs(labels: list[str]) -> dict:
    """Ex-2.2.9's checkpoints and probe set, by ref, so each scoring task is keyed on the checkpoint's content."""
    from mini.store import get_refs

    refs = get_refs([EX229_PROBE_REF, *(EX229_CHECKPOINT_REF.format(label=lb) for lb in labels)])
    missing = [k for k, v in refs.items() if v is None]
    assert not missing, f"ex-2.2.9 refs not in this store: {missing}"
    return {
        "probes": refs[EX229_PROBE_REF],
        "checkpoints": {lb: refs[EX229_CHECKPOINT_REF.format(label=lb)] for lb in labels},
    }


def main(ctx: Ctx) -> dict:
    labels = [f"{CONDITION}-s{s}" for s in SEEDS]
    inputs = ctx.run(resolve_inputs, labels, role="prep")
    scored = ctx.map(
        score_one,
        [inputs["checkpoints"][lb] for lb in labels],
        [inputs["probes"]] * len(labels),
        [OPS_READ] * len(labels),
        [s in STATE_SEEDS for s in SEEDS],
        [CONDITION] * len(labels),
        list(SEEDS),
        labels,
        role="score",
    )
    return ctx.run(publish_results, scored, role="prep")


experiment = Experiment(
    name="ex-2.2.10",
    main=main,
    roles={
        "prep": dict(cpu=2, timeout=900),
        # Two passes per op over up to 11,664 lines, keeping the stream, then 90 ridge fits on ~10k × 64.
        "score": dict(gpu="L4", timeout=1800),
    },
)
