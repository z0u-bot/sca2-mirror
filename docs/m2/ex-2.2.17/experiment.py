"""Ex-2.2.17: a scout on the plateau in ex-2.2.16's center control.

Ex-2.2.16's d64-L4 control at the center corpus condition (`k3-r0.3`, unanchored) scored 0.27 held-out expected
exact match against a Bayes ceiling of 0.52, its validation loss plateauing near 2.40 through the first half of
training and falling again only as the cosine schedule decayed its learning rate. Two readings compete: the
peak learning rate (0.01, ex-2.2.14's `_make_config`) is too high for the task, or the model sits on a
pre-in-context-learning plateau and needs more steps. This scout trains three arms of the same unanchored
control at fresh seeds — `long` (three times the length at the same peak learning rate), `long-low` (three
times the length at a lower peak learning rate), and `low` (the same length at the lower peak learning rate) —
and records the held-out expected exact match and calibration KL against training step, not just at the end,
so the two readings can be told apart.

    bin/mini run docs/m2/ex-2.2.17/experiment.py --app modal --max-containers 9 --budget 2h
    bin/mini status ex-2.2.17
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir
from sca.data.incontext import context_length


def _load_ex2216():
    """Ex-2.2.16's module, by path and left out of `sys.modules` (the `_load_ex2214` pattern that module itself
    uses), so this module's task bodies still cloudpickle by value for a remote worker. Reused for the grammar,
    the corpus condition, the recipe (`Condition229`, `schedules`, `_make_config`), and the end-of-training
    measurements (`eval_one` and its helpers). Loaded once here, at module import time, rather than inside a
    task body: ex-2.2.16 hit a bug loading a sibling by path from inside one.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "ex-2.2.16" / "experiment.py"
    spec = importlib.util.spec_from_file_location("ex2217_ex2216", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex2216 = _load_ex2216()

# --- What is inherited, unchanged --------------------------------------------------------------------------

CENTRE = ex2216.CENTRE
"""The corpus condition every arm trains on: ex-2.2.16's center, `k3-r0.3`, reused rather than regenerated."""

CORPUS_KEY = ex2216.cond_key(*CENTRE)

EPOCHS = ex2216.EPOCHS
"""The length ex-2.2.16 trained the center control at (`low` and the base of `long`/`long-low`)."""

PEAK_LR = 0.01
"""The peak learning rate `_make_config` sets by default (Adam), which `long` keeps and `low`/`long-low` cut."""

LOW_LR = 0.003
"""The lower peak learning rate `low` and `long-low` train at."""

_baseline_config = ex2216._make_config(64, 0, 1, *ex2216.model_dims(ex2216.MODEL))
assert _baseline_config.optimizer.learning_rate == PEAK_LR, "PEAK_LR should match _make_config's own default"

SEED_OFFSET = 600
"""Ex-2.2.16 trained the center control at model seeds 500-502; every model seed here is fresh."""

SEEDS = 3

N_TRAJ_POINTS = 50
"""Target trajectory points per run (about every 2% of training): the stride is sized per arm so a run three
times as long still gets about this many records, rather than three times as many."""

N_TRAJ_EEM_PER_OP = 200
"""Held-out contexts per op the trajectory's expected-exact-match and calibration reads are taken on (2,200 in
all): the first this many of ex-2.2.16's `HOLDOUT_CONTEXTS` per op, a fixed subset held constant across every
record and every run."""


@dataclass(frozen=True)
class Arm:
    """One arm of the scout: how long it trains and at what peak learning rate."""

    name: str
    epoch_mult: int
    """The multiple of `EPOCHS` this arm trains for."""
    peak_lr: float
    note: str = ""
    seeds: int = SEEDS


ARMS: tuple[Arm, ...] = (
    Arm("long", 3, PEAK_LR, "three times ex-2.2.16's steps at its peak LR: the training-length reading alone"),
    Arm("long-low", 3, LOW_LR, "three times the steps at the lower peak LR: both readings at once"),
    Arm("low", 1, LOW_LR, "ex-2.2.16's own length at the lower peak LR: the learning-rate reading alone"),
)
N_RUNS = sum(a.seeds for a in ARMS)
assert N_RUNS == 9


def arm(name: str) -> Arm:
    return next(a for a in ARMS if a.name == name)


# =============================================================================================
# The DAG
# =============================================================================================


def resolve_refs(key: str) -> dict:
    """Ex-2.2.16's published corpus condition *key*: the corpus, labels, holdout, and probes artifacts a
    training run needs, and the corpus's tokenizer and token count (the latter sizes the trajectory stride).
    Resolving these by ref rather than rebuilding the corpus records ex-2.2.16 as this experiment's lineage.
    """
    from sca.compute.data_pipelines import load_data
    from mini.store import get, get_ref

    corpus = get_ref(ex2216.CORPUS_REF.format(key=key))
    labels = get_ref(ex2216.LABELS_REF.format(key=key))
    holdout = get_ref(ex2216.HOLDOUT_REF.format(key=key))
    probes = get_ref(ex2216.PROBES_REF.format(key=key))
    assert corpus and labels and holdout and probes, f"ex-2.2.16 has not published corpus condition {key!r}"
    _, meta = load_data(get(corpus, get_data_dir() / "resolve" / "corpus"))
    return {"corpus": corpus, "labels": labels, "holdout": holdout, "probes": probes, "meta": meta}


def epoch_length_of(total_tokens: int, config) -> int:
    """Optimizer steps per epoch on the resolved corpus, at *config*'s batch size and window: what
    `train_anchored` itself computes internally, so the trajectory stride can be sized in whole epochs without
    loading the corpus a second time.
    """
    from sca.data.batches import batches_per_epoch

    return batches_per_epoch(int(config.data.train_split * total_tokens), config.data, config.model)


def traj_stride_for(epochs: int, epoch_length: int, n_points: int) -> int:
    """The `traj_stride` that gives a run of *epochs* epochs about *n_points* trajectory records."""
    return max(1, round(epochs * epoch_length / n_points))


def cells(arms: tuple[Arm, ...], resolved: dict, seeds: int = SEEDS) -> list[dict]:
    """One row per run: the config (the arm's epoch count and peak learning rate) and the un-anchored schedule,
    over the center control's corpus condition, at fresh model seeds. Reuses ex-2.2.16's `_make_config`,
    `Condition229`, and `schedules`, as its own `cells` does.
    """
    from sca.utils import align

    n_embd, n_layer = ex2216.model_dims(ex2216.MODEL)
    vocab = align(resolved["meta"].tokenizer_config.vocab_size, 64)
    rows = []
    for a in arms:
        epochs = EPOCHS * a.epoch_mult
        for seed in range(min(a.seeds, seeds)):
            model_seed = SEED_OFFSET + seed
            config = ex2216._make_config(vocab, model_seed, epochs, n_embd, n_layer)
            config.tokenizer = resolved["meta"].tokenizer_config.model_copy()
            config.model.block_size = ex2216.BLOCK
            config.model.tie_embeddings = False
            config.optimizer.learning_rate = a.peak_lr
            base = ex2216.Condition229(
                a.name, 1, a.name, lam=0.0, tau=ex2216.TAU, epochs=epochs, ops=ex2216.OP_NAMES, n_lines=ex2216.N_LINES
            )
            anchor, anti = ex2216.schedules(base)
            assert anti is None, "every arm is unanchored"
            rows.append(
                {
                    "config": config,
                    "anchor": anchor,
                    "arm": a.name,
                    "epochs": epochs,
                    "peak_lr": a.peak_lr,
                    "seed": seed,
                    "model_seed": model_seed,
                    "label": f"{a.name}-s{seed}",
                }
            )
    return rows


def train_one(
    config,
    anchor: dict,
    corpus,
    labels,
    probes,
    holdout,
    k: int,
    traj_stride: int,
    n_traj_eem: int,
    label: str,
) -> dict:
    """Train one unanchored run of the center control, recording every *traj_stride* steps, beside the loss
    `train_anchored`'s own trajectory already carries: the held-out expected exact match and the calibration
    KL from the Bayes predictive, overall and per op, on a fixed subset of *n_traj_eem* held-out contexts per
    op. Scored the way ex-2.2.16's `eval_one` scores a final checkpoint (`_color_probs`, `_match`,
    `_post_queries`, `logits_at`), on the model as it stands rather than a saved checkpoint.
    """
    from sca.anchoring import AnchorSpec, LabelSpec
    from sca.compute.training import train_anchored
    from sca.data.named_colors import WordTokenizer
    from sca.data.ops import PALETTE
    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get, put

    workdir = get_data_dir() / "cells" / label
    corpus_dir = get(corpus, workdir / "corpus")
    tokenizer = WordTokenizer(config.tokenizer)
    newline_id = tokenizer.stoi["\n"]
    anchored_op_id = ex2216.OP_NAMES.index(ex2216.ANCHORED_OP)

    with np.load(get(labels, workdir / "labels.npz")) as z:
        spec = LabelSpec(
            p=np.zeros(config.model.vocab_size),
            keying="context",
            context_op=z["op_ids"],
            anchored_op_id=anchored_op_id,
            label_rate=ex2216.LABEL_RATE,
            variant="whole",
            context_len=z["context_len"],
        )

    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens = z["tokens"]  # (n_ops * N_TRAJ_PROBE, context_tokens(k))

    per_op = len(probe_tokens) // ex2216.N_OPS
    weights = np.concatenate([np.full(per_op, 1.0 if o == ex2216.ANCHORED_OP else 0.0) for o in ex2216.OP_NAMES])
    weights = weights / weights.sum()

    # --- The fixed EEM/calibration probe: the first n_traj_eem held-out contexts of every op --------------
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), k)
    sub = np.concatenate([np.flatnonzero(ho.op_ids == o)[:n_traj_eem] for o in range(ex2216.N_OPS)])
    sub_tokens, sub_op_ids, sub_posterior = ho.tokens[sub], ho.op_ids[sub], ho.posterior[sub]
    color_ids = np.array([tokenizer.stoi[n] for n in PALETTE])
    tok2color = np.full(config.model.vocab_size, -1)
    tok2color[color_ids] = np.arange(len(PALETTE))
    sub_holdout = ex2216.Holdout(
        sub_tokens,
        sub_op_ids,
        sub_posterior,
        np.empty(0),
        np.empty((0, 0), dtype=np.int32),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=bool),
    )
    P = ex2216._get_posterior()
    table = P.build_table(ex2216.TABLE)
    probe_ctx = ex2216._post_queries(P, sub_holdout, k, tok2color)
    identity = projection(Subspace.axis(config.model.n_embd), 0.0)
    eq_role = ex2216.query_role(k, ex2216.QUERY_EQ)

    traj_extra: dict[str, list] = {"eem": [], "eem_per_op": [], "kl": [], "kl_per_op": []}

    def on_record(_index: int, model) -> None:
        p = ex2216._color_probs(logits_at(model, sub_tokens, identity, (), eq_role), color_ids)
        eem = ex2216._match(table, probe_ctx, p)
        kl = P.kl(P.predictive(table, probe_ctx, sub_posterior), p)
        traj_extra["eem"].append(float(eem.mean()))
        traj_extra["eem_per_op"].append(ex2216._per_op(eem, sub_op_ids))
        traj_extra["kl"].append(float(kl.mean()))
        traj_extra["kl_per_op"].append(ex2216._per_op(kl, sub_op_ids))

    _, metrics, traj = train_anchored(
        config,
        corpus_dir,
        anchor=AnchorSpec(**anchor),
        anti=None,
        label_p=spec,
        probe_tokens=probe_tokens,
        probe_weights=weights,
        crop=ex2216.CROP_POLICY,
        anchor_slices=None,
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
        newline_id=newline_id,
        min_line_tokens=context_length(k),
        on_record=on_record,
    )

    epoch_train_loss = np.asarray([m.train_loss for m in metrics], dtype=np.float64)
    epoch_axis = np.arange(1, len(epoch_train_loss) + 1, dtype=np.float64)
    train_loss = np.interp(np.asarray(traj["epoch"], dtype=np.float64), epoch_axis, epoch_train_loss).tolist()

    return {
        "label": label,
        "traj": {
            "step": traj["step"].tolist(),
            "epoch": traj["epoch"].tolist(),
            "lr": traj["lr"].tolist(),
            "val_loss": traj["val_loss"].tolist(),
            "train_loss": train_loss,
        }
        | traj_extra,
        "checkpoint": put(workdir / "model", name=f"ex-2.2.17-{label}-ckpt"),
    }


# --- Publishing ------------------------------------------------------------------------------

TRAJ_REF = "reports/m2/ex-2.2.17/trajectories"
METRICS_REF = "reports/m2/ex-2.2.17/metrics"
CHECKPOINT_REF = "reports/m2/ex-2.2.17/checkpoints/{label}"
EVAL_REF = "reports/m2/ex-2.2.17/eval"
EVAL_ARRAYS_REF = "reports/m2/ex-2.2.17/eval-arrays/{label}"


def design() -> dict[str, Any]:
    """The design constants a report (and stage C) reads beside the results."""
    return {
        "experiment": "ex-2.2.17",
        "question": "whether ex-2.2.16's center control plateau reflects too high a peak learning rate, too "
        "little training, or both",
        "corpus_condition": CORPUS_KEY,
        "arms": [asdict(a) for a in ARMS],
        "n_runs": N_RUNS,
        "seed_offset": SEED_OFFSET,
        "epochs_base": EPOCHS,
        "peak_lr": PEAK_LR,
        "low_lr": LOW_LR,
        "n_traj_points": N_TRAJ_POINTS,
        "n_traj_eem_per_op": N_TRAJ_EEM_PER_OP,
        "block": ex2216.BLOCK,
        "model": ex2216.MODEL,
        "ops": list(ex2216.OP_NAMES),
    }


def publish_results(trained: list[dict], rows: list[dict]) -> dict:
    """The trajectories (JSON, one entry per run), the design and every run's row (JSON), and every end
    checkpoint, each under its own ref.
    """
    import json

    from mini.store import put, set_ref

    traj = {t["label"]: t["traj"] for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.17-trajectories.json"))
    row_meta = [{k: v for k, v in r.items() if k not in ("config", "anchor")} for r in rows]
    set_ref(
        METRICS_REF, put(json.dumps({"design": design(), "runs": row_meta}).encode(), name="ex-2.2.17-metrics.json")
    )
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    return {"n_runs": len(trained)}


def publish_evaluation(rows: list[dict], evaled: list[dict]) -> dict:
    """The end-of-training evaluation (JSON, one entry per run), and every run's per-context arrays under its
    own ref. Mirrors ex-2.2.16's `publish_evaluation`, without the suppression pass this scout has no arms for.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {k: v for k, v in r.items() if k not in ("config", "anchor")} for r in rows}

    def slim(r: dict) -> dict:
        return meta[r["label"]] | {k: v for k, v in r.items() if k != "arrays"}

    for r in evaled:
        set_ref(EVAL_ARRAYS_REF.format(label=r["label"]), r["arrays"])
    body = {"design": design(), "runs": [slim(r) for r in evaled]}
    set_ref(EVAL_REF, put(json.dumps(body).encode(), name="ex-2.2.17-eval.json"))
    return {"n_evaled": len(evaled)}


# --- Orchestration ----------------------------------------------------------------------------


def run(
    ctx: Ctx, arms: tuple[Arm, ...], seeds: int, n_traj_points: int, n_traj_eem: int
) -> tuple[list[dict], list[dict], dict]:
    """Resolve the center control's corpus condition, then train *arms*: the whole DAG, with the seeds, the
    trajectory point target, and the EEM probe size as arguments so a short prototype runs the same code (the
    smoke test overrides all four, plus the arm set itself).
    """
    resolved = ctx.run(resolve_refs, CORPUS_KEY, role="prep")
    rows = cells(arms, resolved, seeds)
    epoch_length = epoch_length_of(resolved["meta"].total_tokens, rows[0]["config"])
    n = len(rows)
    trained = ctx.map(
        train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [resolved["corpus"]] * n,
        [resolved["labels"]] * n,
        [resolved["probes"]] * n,
        [resolved["holdout"]] * n,
        [CENTRE[0]] * n,
        [traj_stride_for(r["epochs"], epoch_length, n_traj_points) for r in rows],
        [n_traj_eem] * n,
        [r["label"] for r in rows],
        role="train",
    )
    return trained, rows, resolved


def main(ctx: Ctx) -> dict:
    trained, rows, resolved = run(ctx, ARMS, SEEDS, N_TRAJ_POINTS, N_TRAJ_EEM_PER_OP)
    published = ctx.run(publish_results, trained, rows, role="prep")
    n = len(trained)
    evaled = ctx.map(
        ex2216.eval_one,
        [t["checkpoint"] for t in trained],
        [resolved["holdout"]] * n,
        [CENTRE[0]] * n,
        [r["label"] for r in rows],
        role="eval",
    )
    return published | ctx.run(publish_evaluation, rows, evaled, role="prep")


COMPUTE = {
    # One ref resolution and a small corpus-metadata read.
    "prep": dict(cpu=2, timeout=600),
    # 13,200 steps for `low`, about 39,600 for `long` and `long-low`, at ex-2.2.16's per-step cost; the watchdog
    # covers the checkpoint upload.
    "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
    # Forward passes only, over 22,000 held-out contexts, as ex-2.2.16's eval role.
    "eval": dict(gpu="L4", timeout=900),
}

experiment = Experiment(name="ex-2.2.17", main=main, roles=COMPUTE)
