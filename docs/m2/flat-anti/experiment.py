"""Flat anti: the recipe of record with an anti-subspace weight that is the same at every slice.

The recipe of record (`τ_ℓ fixed`) anneals the anti weight from 0.25 on every slice to a hold that falls off with
depth, in a straight line from 0.2 at the embedding to 0.03 at slice 4 (`falloff`). The only runs with a hold that is
the same at every slice (split-anti-seeds' `deep, hold 0.2` and `deep, hold 0.1`, late-pull's `late` at 0.03) used an
older pull: a step on slices 2 to 4, a sharp τ_p from the start, and no pool over slices. This experiment trains the
recipe at eight seeds under five anti weights:

- `recipe`: `τ_ℓ fixed` as it is, with the falloff, retrained beside the others so that every condition shares one card and one regime;
- `flat 0.115`: the anneal from 0.25 to a hold of 0.115 at every slice, the mean of the falloff's five holds, so the
  term carries the same total weight spread evenly;
- `flat 0.2`: the same anneal to a hold of 0.2 at every slice, the falloff's embedding value everywhere;
- `level 0.115` and `level 0.2`: the value at every slice from epoch 0, with no high start. ex-2.1.11 found the high
  start kept the pull off a syntax token; these ask whether the soft τ_p start now does that job.
- `level 0.14` and `level 0.17`: two more constant weights between those two, added after the first round, where
  `level 0.115` let block 1 spill and `level 0.2` cost removal;
- `from 0.17`: the anneal to a hold of 0.115 at every slice from a lower start of 0.17, added with them, to ask
  whether a small start keeps block 1 clean at the low hold.

Every condition keeps the end anneal it shares with the anchor (epochs 360 to 400). Everything else, and the scoring,
is the fixed confirmation's, and a control (no pull, no anti term) trains at each seed for the task change. Conditions are compared as two samples (no pairing by seed). Training and scoring run on
L40S cards only, so the card is the same for every condition.

    bin/mini run docs/m2/flat-anti/experiment.py --app modal --max-containers 16 --budget 2h --keep-stale-done
    bin/mini status flat-anti
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from typing import Any

from mini import Ctx, Experiment


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as the ex-2.2 experiments do."""
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


fc = _load_sibling("fixed-confirm", "flat_anti_fc")
ac, sps, lr, rb, lean, ex2222 = fc.ac, fc.sps, fc.lr, fc.rb, fc.lean, fc.ex2222

OP_NAMES = fc.OP_NAMES
ANCHORED_OP = fc.ANCHORED_OP
K = fc.K
EPOCHS = fc.EPOCHS
NAME = "flat-anti"

FALLOFF_HOLDS: tuple[float, ...] = sps.FALLOFF_HOLDS
"""The recipe's hold by slice: 0.2 at the embedding to 0.03 at slice 4, in a straight line."""
N_SLICES = len(FALLOFF_HOLDS)
MEAN_HOLD = sum(FALLOFF_HOLDS) / N_SLICES
"""0.115: the falloff's mean hold, so a flat hold at this value has the same total weight over the five slices."""
EMB_HOLD = FALLOFF_HOLDS[0]
"""0.2: the falloff's hold at the embedding."""


@dataclass(frozen=True)
class Condition:
    """An anti weight: the hold at every slice (`None` keeps the falloff) and the weight at epoch 0 (`None` keeps the
    recipe's 0.25, annealing to the hold by epoch 360).
    """

    name: str
    hold: float | None = None
    peak: float | None = None

    def anti(self, anti: dict) -> dict:
        out = anti | {"hold": FALLOFF_HOLDS if self.hold is None else (self.hold,) * N_SLICES}
        return out if self.peak is None else out | {"peak": self.peak}


CONDITIONS: tuple[Condition, ...] = (
    Condition("recipe"),
    Condition("flat-0.115", MEAN_HOLD),
    Condition("flat-0.2", EMB_HOLD),
    Condition("level-0.115", MEAN_HOLD, MEAN_HOLD),
    Condition("level-0.2", EMB_HOLD, EMB_HOLD),
    Condition("level-0.14", 0.14, 0.14),
    Condition("level-0.17", 0.17, 0.17),
    Condition("from-0.17", MEAN_HOLD, 0.17),
)

SEEDS: tuple[int, ...] = tuple(range(720, 728))
"""Eight seeds no finished run of the recipe has used (the loss-rate counts stopped them at epoch 64)."""
KEEP = tuple(sorted({*fc.STOP_KEEP, *fc.FINISH_KEEP}))
"""Checkpoints kept along training: 16, 32, 64, then every 32 epochs, as the fixed confirmation's runs."""

BUDGET_USD = 14
"""From the fixed-labels crossed runs on L40S cards (sixteen runs to epoch 400 for about \\$1 of training) and the
fixed confirmation's scoring: 48 runs to 400, controls included (about \\$3), their end measurements and axis scoring (\\$2), the
axis scoring of thirteen kept checkpoints on each (\\$2), and room for a retry; then \\$6 for the three rungs of the second round."""


def label_of(c: Condition, model_seed: int) -> str:
    return f"{NAME}-{c.name}-s{model_seed}"


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/flat-anti"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
KEPT_REF = PREFIX + "/kept/{label}"
TRAJ_REF = PREFIX + "/trajectories"
RESULTS_REF = PREFIX + "/results"
AXES_REF = PREFIX + "/axes"
RESTRICTED_REF = PREFIX + "/restricted-axes"
ALONG_REF = PREFIX + "/along"


def rows_of(meta, conditions: tuple[Condition, ...], seeds: tuple[int, ...], epochs: int = EPOCHS) -> list[dict]:
    """One row per run: the fixed confirmation's row of `τ_ℓ fixed` at the seed, with the condition's anti weight."""
    base = {r["model_seed"]: r for r in fc.rows_of(meta, seeds, None, epochs)}
    return [
        base[s] | {"anti": c.anti(base[s]["anti"]), "kind": c.name, "flat": asdict(c), "label": label_of(c, s)}
        for c in conditions
        for s in seeds
    ]


def design() -> dict[str, Any]:
    anchor, anti = sps.st.schedules()
    return {
        "experiment": NAME,
        "base": fc.BASE,
        "anchor": next(c for c in sps.CONDITIONS if c.name == fc.BASE).anchor(anchor),
        "anti": {c.name: c.anti(anti) for c in CONDITIONS},
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "seeds": list(SEEDS),
        "keep": list(KEEP),
        "window": list(lr.WINDOW),
        "track_slice": lr.TRACK_SLICE,
        "traj_stride_epochs": sps.TRAJ_STRIDE_EPOCHS,
    }


def publish(rows: list[dict], trained: list[dict], measured: list[dict], axes: list[dict],
            restricted: list[dict], along: list[list[dict]]) -> dict:  # fmt: skip
    """Trajectories (with the loss-rate call), checkpoints, end measurements, the axis scoring, the restricted edits,
    and the axis scoring of the kept checkpoints, under refs laid out as the fixed confirmation's.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"kind": r["kind"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
        if t.get("kept_epochs"):
            set_ref(KEPT_REF.format(label=t["label"]), t["kept_checkpoints"])
    traj = {
        t["label"]: meta[t["label"]]
        | {k: t[k] for k in ("traj", "val_loss", "train_loss")}
        | {"window_mean": (w := lr.window_mean(t["traj"])), "call": lr.call_of(w)}
        for t in trained
    }
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name=f"{NAME}-trajectories.json"))

    def body(runs: list[dict], **extra) -> bytes:
        return json.dumps({"design": design(), **extra, "runs": [meta[r["label"]] | r for r in runs]}).encode()

    set_ref(RESULTS_REF, put(body(measured), name=f"{NAME}-results.json"))
    set_ref(AXES_REF, put(body(axes), name=f"{NAME}-axes.json"))
    sets = {k: list(v) for k, v in sps.RESTRICTED_SETS.items()}
    set_ref(RESTRICTED_REF, put(body(restricted, slice_sets=sets), name=f"{NAME}-restricted-axes.json"))
    set_ref(ALONG_REF, put(body([r for a in along for r in a]), name=f"{NAME}-along.json"))
    return {"n_runs": len(trained)}


def run(ctx: Ctx, epochs: int = EPOCHS, conditions: tuple[Condition, ...] = CONDITIONS,
        seeds: tuple[int, ...] = SEEDS, keep: tuple[int, ...] = KEEP, full: bool = True) -> dict:  # fmt: skip
    """Train every condition and a control (no pull, no anti term) at every seed to the end, and score the finished
    runs and the anchored runs' kept checkpoints. The
    arguments let a smoke run take the same path at a fraction of the size, and leave the refs alone.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    anchored = rows_of(resolved["meta"], conditions, seeds, epochs)
    controls = ac.control_rows(resolved["meta"], seeds, epochs)
    rows = anchored + controls
    n = len(rows)
    done = ac.train(ctx, resolved, rows, [None] * n, [keep] * len(anchored) + [()] * len(controls))

    labels = [r["label"] for r in rows]
    holdout = [resolved["holdout"]] * n
    checkpoints = [t["checkpoint"] for t in done]
    measured = ctx.map(lean.measure_one, checkpoints, holdout, labels, [True] * n, role="measure")
    a = range(len(anchored))
    along = ctx.map(
        ac.kept_axes_one,
        [done[i]["kept_checkpoints"] for i in a],
        [checkpoints[i] for i in a],
        [resolved["holdout"]] * len(a),
        [labels[i] for i in a],
        [tuple(done[i]["kept_epochs"]) for i in a],
        [epochs] * len(a),
        role="measure",
    )
    if not full:  # a smoke run leaves out the slow passes and the refs
        return {"finished": labels, "along": [len(x) for x in along]}
    axes = ctx.map(rb.axes_one, checkpoints, holdout, labels, [None] * n, role="measure")
    restricted = ctx.map(sps.restricted_axes_one, checkpoints[: len(a)], holdout[: len(a)], labels[: len(a)], role="measure")  # fmt: skip
    return ctx.run(publish, rows, done, measured, axes, restricted, along, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """Two conditions at one seed for eight epochs, measured."""
    return run(ctx, epochs=8, conditions=CONDITIONS[2:], seeds=SEEDS[:1], keep=(2, 4), full=False)


COMPUTE = fc.COMPUTE | {role: fc.COMPUTE[role] | {"gpu": "L40S"} for role in ("train", "measure")}
"""The fixed confirmation's roles, with training and scoring on L40S cards only, so every condition shares a card."""

experiment = Experiment(name=NAME, main=main, roles=COMPUTE)
