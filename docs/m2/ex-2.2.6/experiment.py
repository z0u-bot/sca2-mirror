"""
Pilot: the whole-span labeller, with the answer in the label's span.

A scouting run, in the sense of the science skill: no gates, no verdicts, a record of what
we ran and what we saw. Ex-2.2.3's labeller reads the operands (either slot draws at its
redness rate) and pulls the prompt span, so a line whose answer is redder than both operands
draws no label for that answer, and its answer position is never pulled: the blind span E3
reads. A document-level label, the M3 shape, says nothing about position. This pilot runs
the adopted point (`recipe-short` of ex-2.2.3) under labellers that move toward that shape
along two axes: the answer draws too (`line` keying, `sca.anchoring.LabelSpec`), and the pull
covers the whole line, answer and newline included (`span=6`).

Three arms, each a corner of the (keying, span) square whose fourth corner is production's
`recipe-short`: both changes together, the answer drawing with the prompt-span pull, and the
whole-line pull under the operand-only labeller. Every task function but the training step
and a probe-table patch is ex-2.2.3's, loaded from its module unchanged; the report reads
the production seeds for the fourth corner.

    bin/mini run docs/m2/ex-2.2.6/experiment.py --app modal --max-containers 6 --budget 2h
    bin/mini status ex-2.2.6
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from mini import Ctx, Experiment, get_data_dir


def _load_ex223():
    """Ex-2.2.3's module, loaded by path and left out of `sys.modules`, as `mini.load_experiment` does,
    so its task functions still cloudpickle by value for a remote worker.
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
REDNESS = ex223.REDNESS
PALETTE = ex223.PALETTE
ANSWER_POS = ex223.ANSWER_POS
TRAJ_STRIDE = ex223.TRAJ_STRIDE
_npz = ex223._npz
_make_config = ex223._make_config
prepare_corpus = ex223.prepare_corpus
eval_one = ex223.eval_one
score_one = ex223.score_one

METRICS_REF = "reports/m2/ex-2.2.6/metrics"
ARRAYS_REF = "reports/m2/ex-2.2.6/arrays"
TRAJ_REF = "reports/m2/ex-2.2.6/trajectories"
PROBE_REF = "reports/m2/ex-2.2.6/probes"
CHECKPOINT_REF = "reports/m2/ex-2.2.6/checkpoints/{label}"

EX223_METRICS_REF = ex223.METRICS_REF
EX223_ARRAYS_REF = ex223.ARRAYS_REF
"""Production's `recipe-short` (twenty seeds with the addendum): the fourth corner of the square."""

# --- Conditions ---------------------------------------------------------------------------

Keying = Literal["either", "line"]
EPOCHS = ex223.EPOCHS_SHORT
PROMPT = ex223.SPAN
WHOLE = 6
"""The whole line: op1, op, op2, `=`, the answer, and the newline."""


@dataclass(frozen=True)
class Arm:
    """One labeller: which slots draw, and how many roles the pull covers. Everything else is the recipe."""

    name: str
    seeds: int
    title: str
    keying: Keying
    span: int

    @property
    def condition(self) -> Condition:
        return Condition(self.name, self.seeds, self.title, lam=ex223.SCORING_LAMBDA, epochs=EPOCHS)


LINE_WHOLE = Arm("line-whole", 3, "the answer draws; the pull covers the whole line", "line", WHOLE)
LINE_PROMPT = Arm("line-prompt", 2, "the answer draws; the pull covers the prompt span", "line", PROMPT)
EITHER_WHOLE = Arm("either-whole", 2, "operands draw, as production; the pull covers the whole line", "either", WHOLE)
ARMS = (LINE_WHOLE, LINE_PROMPT, EITHER_WHOLE)
REFERENCE = "recipe-short"
"""Production's arm at the fourth corner: operands draw, prompt-span pull."""
N_RUNS = sum(a.seeds for a in ARMS)

# --- The probe table under line keying -------------------------------------------------------


def line_keyed_probes(probes, vocabulary: list[str], per_slot_rate: float) -> dict:
    """Ex-2.2.3's probe table with `line_p` recomputed for a labeller whose answer slot draws too.

    P(labelled) becomes 1 − (1 − p1)(1 − p2)(1 − p3), with p3 the answer's rate. The operand-only version
    stays beside it as `line_p_either`, so the report can weight m_line either way on any arm.
    """
    from mini.store import get, put

    workdir = get_data_dir() / "probes"
    stoi = {w: i for i, w in enumerate(vocabulary)}
    tok2color = np.full(len(vocabulary), -1)
    for i, name in enumerate(PALETTE):
        tok2color[stoi[name]] = i
    with np.load(get(probes, workdir / "probes.npz")) as z:
        arrays = {k: z[k] for k in z.files}
    for op in [k[: -len("/tokens")] for k in arrays if k.endswith("/tokens")]:
        r3 = REDNESS[tok2color[arrays[f"{op}/tokens"][:, ANSWER_POS]]]
        p1, p2, p3 = (r**8 * per_slot_rate for r in (arrays[f"{op}/r1"], arrays[f"{op}/r2"], r3))
        arrays[f"{op}/line_p_either"] = arrays[f"{op}/line_p"]
        arrays[f"{op}/line_p"] = 1.0 - (1.0 - p1) * (1.0 - p2) * (1.0 - p3)
        arrays[f"{op}/r3"] = r3
    return {"probes": put(_npz(**arrays), name="ex-2.2.6-probes-line.npz")}


# --- Training ---------------------------------------------------------------------------------


def cells(arms: tuple[Arm, ...], prep: dict, probes_by_keying: dict[str, object]) -> list[dict]:
    from sca.utils import align

    rows = []
    tc = prep["meta"].tokenizer_config
    for arm in arms:
        anchor, anti = ex223.schedules(arm.condition)
        anchor = anchor | {"span": arm.span}
        for seed in range(arm.seeds):
            config = _make_config(align(tc.vocab_size, 64), seed, EPOCHS)
            config.tokenizer = tc.model_copy()
            rows.append(
                {
                    "config": config,
                    "anchor": anchor,
                    "anti": anti,
                    "keying": arm.keying,
                    "condition": arm.name,
                    "seed": seed,
                    "label": f"{arm.name}-s{seed}",
                    "corpus": prep["corpus"],
                    "evals": prep["evals"],
                    "probes": probes_by_keying[arm.keying],
                }
            )
    return rows


def train_one(
    config, anchor: dict, anti: dict | None, corpus, traj_stride: int, probes, keying: Keying, label: str
) -> dict:
    """Ex-2.2.3's training step with the labeller's keying as an argument; the pull's span rides in `anchor`."""
    from sca.anchoring import AnchorSpec, AntiSpec, LabelSpec
    from sca.compute.training import train_anchored
    from mini.store import get, put

    workdir = get_data_dir() / "cells" / label
    corpus_dir = get(corpus, workdir / "corpus")
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens, slot_p, weights, line_p = z["mix/tokens"], z["slot_p"], z["weights"], z["mix/line_p"]
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
        "checkpoint": put(workdir / "model", name=f"ex-2.2.6-{label}-ckpt"),
    }


# --- Publishing ------------------------------------------------------------------------------


def design() -> dict:
    return {
        "n_runs": N_RUNS,
        "arms": [asdict(a) for a in ARMS],
        "reference": REFERENCE,
        "epochs": EPOCHS,
        "prompt_span": PROMPT,
        "whole_span": WHOLE,
        "red_dose": ex223.RED_DOSE,
        "nonred_dose": ex223.NONRED_DOSE,
    }


def _slim(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("arrays", "traj", "val_loss", "train_loss")}


def publish_results(trained: list[dict], evaled: list[dict], scored: list[dict], corpus_stats: dict, probes) -> dict:
    import json

    from mini.store import get, put, set_ref

    metrics = {
        "runs": [_slim(r) for r in evaled],
        "scores": [_slim(r) for r in scored],
        "corpus": corpus_stats,
        "design": design(),
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.2.6-metrics.json"))
    traj = {t["label"]: {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.6-trajectories.json"))
    set_ref(PROBE_REF, probes)
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    arrays = {}
    for r in evaled + scored:
        kind = "eval" if "per_op" in r else "score"
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}-{kind}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{kind}/{name}": z[name] for name in z.files}
    set_ref(ARRAYS_REF, put(_npz(**arrays), name="ex-2.2.6-arrays.npz"))
    return {"n_runs": len(evaled), "holdout_em": {r["label"]: r["holdout_em"] for r in evaled}}


# --- Orchestration ----------------------------------------------------------------------------


def main(ctx: Ctx) -> dict:
    prep = ctx.run(
        prepare_corpus,
        ex223.OP_NAMES,
        ex223.N_LINES,
        ex223.CORPUS_SEED,
        ex223.HOLDOUT_FRAC,
        ex223.N_PROBE,
        ex223.PROBE_SEED,
        ex223.PER_SLOT_RATE,
        ex223.RED_RATE,
        role="prep",
    )
    lined = ctx.run(
        line_keyed_probes,
        prep["probes"],
        list(prep["meta"].tokenizer_config.vocabulary),
        ex223.PER_SLOT_RATE,
        role="prep",
    )
    rows = cells(ARMS, prep, {"either": prep["probes"], "line": lined["probes"]})
    n = len(rows)
    trained = ctx.map(
        train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [r["corpus"] for r in rows],
        [TRAJ_STRIDE] * n,
        [r["probes"] for r in rows],
        [r["keying"] for r in rows],
        [r["label"] for r in rows],
        role="train",
    )
    evaled = ctx.map(
        eval_one,
        trained,
        [r["evals"] for r in rows],
        [r["probes"] for r in rows],
        [r["anchor"]["tau"] for r in rows],
        [r["condition"] for r in rows],
        [r["seed"] for r in rows],
        [r["label"] for r in rows],
        role="eval",
    )
    scored = ctx.map(
        score_one,
        trained,
        [r["probes"] for r in rows],
        [r["condition"] for r in rows],
        [r["seed"] for r in rows],
        [r["label"] for r in rows],
        role="score",
    )
    return ctx.run(publish_results, trained, evaled, scored, prep["stats"], lined["probes"], role="prep")


experiment = Experiment(
    name="ex-2.2.6",
    main=main,
    roles={
        "prep": dict(cpu=2, timeout=900),
        "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
        "eval": dict(gpu="L4", timeout=1200),
        "score": dict(gpu="L4", timeout=1800),
    },
)
