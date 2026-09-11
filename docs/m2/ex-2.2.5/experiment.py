"""
Pilot: stochastic rounding of off-grid answers.

A scouting run, in the sense of the science skill: no gates, no verdicts, a record of what
we ran and what we saw. The six-op grammar snaps each off-grid channel value to the nearer
grid level, so `screen`, `multiply`, and (off its on-grid pairs) `mix` are step functions of
their raw values. The stochastic variant (`sca.data.ops.Rounding`) draws the upper level with
probability equal to how far up the interval the value sits, once per line at corpus build.
The todo item asks three things of such a variant: what the exact-match ceiling becomes and
whether exact match is then the wrong statistic; whether the answer's representation becomes
more graded; and whether the redder-than-both counts move.

The run is small: the un-anchored control and the adopted recipe (`recipe-short` of
ex-2.2.3) retrained on a stochastic corpus at a few seeds, beside two fresh seeds of the
nearest-rounded control for a same-code comparison. The anchored arm's placement statistics
are read against the production `recipe-short` seeds in the report. Every task function but
the corpus build and the rounding readout is ex-2.2.3's, loaded from its module unchanged.

    bin/mini run docs/m2/ex-2.2.5/experiment.py --app modal --max-containers 6 --budget 2h
    bin/mini status ex-2.2.5
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sca.data.ops import Rounding
from mini import Ctx, Experiment, get_data_dir


def _load_ex223():
    """Ex-2.2.3's module, loaded by path and left out of `sys.modules`, as `mini.load_experiment` does,
    so its task functions still cloudpickle by value for a remote worker.
    """
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

# The pieces of ex-2.2.3 this pilot reuses, bound by name so a task body never references the module
# object itself (cloudpickle would then ship the whole namespace).
Condition = ex223.Condition
GRID_RGB = ex223.GRID_RGB
REDNESS = ex223.REDNESS
PALETTE = ex223.PALETTE
TOP = ex223.TOP
N_EVAL = ex223.N_EVAL
ANSWER_POS = ex223.ANSWER_POS
TRAJ_STRIDE = ex223.TRAJ_STRIDE
cube_redness = ex223.cube_redness
_npz = ex223._npz
_load = ex223._load
_answer_logprobs = ex223._answer_logprobs
_make_config = ex223._make_config
schedules = ex223.schedules
eval_one = ex223.eval_one
score_one = ex223.score_one
probe_one = ex223.probe_one

METRICS_REF = "reports/m2/ex-2.2.5/metrics"
ARRAYS_REF = "reports/m2/ex-2.2.5/arrays"
TRAJ_REF = "reports/m2/ex-2.2.5/trajectories"
GEOMETRY_REF = "reports/m2/ex-2.2.5/geometry"
CHECKPOINT_REF = "reports/m2/ex-2.2.5/checkpoints/{label}"

EX223_METRICS_REF = ex223.METRICS_REF
EX223_GEOMETRY_REF = ex223.GEOMETRY_REF
"""The production results the report reads beside this pilot's: `recipe-short` at twenty seeds, and the
cube probes on the full-length control."""

# --- Conditions ---------------------------------------------------------------------------

EPOCHS = ex223.EPOCHS_SHORT
"""The adopted point's length: 50 epochs, 1,650 steps."""

CONTROL_STOCH = Condition("control-stoch", 2, "un-anchored, stochastic corpus", lam=0.0, epochs=EPOCHS)
RECIPE_STOCH = Condition(
    "recipe-stoch", 3, "the adopted recipe, stochastic corpus", lam=ex223.SCORING_LAMBDA, epochs=EPOCHS
)
CONTROL_NEAR = Condition("control-near", 2, "un-anchored, nearest-rounded corpus", lam=0.0, epochs=EPOCHS)
"""Two fresh seeds of ex-2.2.3's `control-short`, trained here so the rounding readout and the cube probes
have a same-code nearest-rounded comparison (production scored neither on that arm)."""

CONDITIONS = (CONTROL_STOCH, RECIPE_STOCH, CONTROL_NEAR)
ROUNDING: dict[str, Rounding] = {
    CONTROL_STOCH.name: "stochastic",
    RECIPE_STOCH.name: "stochastic",
    CONTROL_NEAR.name: "nearest",
}
STOCHASTIC = (CONTROL_STOCH.name, RECIPE_STOCH.name)
N_RUNS = sum(c.seeds for c in CONDITIONS)

# --- The corpus -------------------------------------------------------------------------


def corpus_key(rounding: Rounding) -> str:
    return f"ops6-100k-{rounding}"


def prepare_corpus(
    rounding: Rounding,
    n_lines: int,
    seed: int,
    holdout_frac: float,
    n_probe: int,
    probe_seed: int,
    per_slot_rate: float,
    red_rate: float,
) -> dict:
    """Ex-2.2.3's corpus build with the rounding rule as an argument.

    Same (op, pair) sequence and eval lines as the production corpus at this seed; only the answers of the
    rounded lines differ. The probe lines keep the nearest answer, since they read the prompt's geometry.
    """
    from collections import Counter

    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data import ops as grammar
    from sca.data.named_colors import WordTokenizer
    from mini.store import put

    key = corpus_key(rounding)
    table = grammar.OPS
    corpus = grammar.sample_corpus(n_lines, seed, table, holdout_frac, rounding=rounding)

    tokenizer_config = TokenizerConfig(vocabulary=grammar.vocabulary())
    tokenizer = WordTokenizer(tokenizer_config)
    tokens = grammar.encode_corpus(corpus, tokenizer.stoi)
    n_chars = sum(len(w) for line in corpus for w in line.words)
    meta = CorpusMetadata(
        tokenizer_config=tokenizer_config,
        total_tokens=len(tokens),
        total_chars=n_chars,
        sources=[DatasetMetadata(title=f"multi-op color corpus ({key})", fixes=[], total_chars=n_chars)],
    )
    corpus_dir = get_data_dir() / "corpora" / key
    save_data(tokens, meta, corpus_dir)
    evals = grammar.eval_sets(N_EVAL, seed, table, holdout_frac, rounding=rounding)

    rgb = np.asarray(grammar.colors(), dtype=float) / TOP
    color_redness = cube_redness(rgb)
    slot_p = np.zeros(tokenizer.vocab_size)
    for name, sp in zip(grammar.PALETTE, color_redness**8 * per_slot_rate, strict=True):
        slot_p[tokenizer.stoi[name]] = sp
    affinity = color_redness**8 * red_rate
    arrays: dict[str, np.ndarray] = {"slot_p": slot_p, "weights": affinity / affinity.sum(), "redness": color_redness}
    for op in table:
        probe = grammar.probe_lines(op, n_probe, probe_seed)
        r1 = cube_redness(np.asarray([ln.lhs for ln in probe], dtype=float) / TOP)
        r2 = cube_redness(np.asarray([ln.rhs for ln in probe], dtype=float) / TOP)
        p1, p2 = r1**8 * per_slot_rate, r2**8 * per_slot_rate
        arrays |= {
            f"{op.name}/tokens": np.array([tokenizer.encode_words(ln.words) for ln in probe], dtype=np.int32),
            f"{op.name}/r1": r1,
            f"{op.name}/r2": r2,
            f"{op.name}/line_p": p1 + p2 - p1 * p2,
        }

    ops = [op.name for op in table]
    rounded = [not grammar.is_on_grid(grammar.OP_BY_NAME[ln.op], ln.lhs, ln.rhs) for ln in corpus]
    stats = {
        "key": key,
        "rounding": rounding,
        "n_lines": n_lines,
        "total_tokens": int(len(tokens)),
        "vocab_size": tokenizer.vocab_size,
        "lines_per_op": dict(Counter(ln.op for ln in corpus)),
        "rounded_per_op": dict(Counter(ln.op for ln, r in zip(corpus, rounded, strict=True) if r)),
        "distinct_answers_per_key": float(np.mean(list(Counter(ln.key for ln in corpus).values()))),
        "eval_n": N_EVAL,
        "ops": ops,
    }
    return {
        "key": key,
        "meta": meta,
        "stats": stats,
        "corpus": put(corpus_dir, name=f"ex-2.2.5-{key}-corpus"),
        "evals": put(grammar.dump_lines(evals), name=f"ex-2.2.5-{key}-evals.json"),
        "probes": put(_npz(**arrays), name=f"ex-2.2.5-{key}-probes.npz"),
    }


def grammar_table() -> dict:
    """What changes about the grammar itself under stochastic rounding, with no model in the loop.

    Per op: the exact-match ceiling (mean over unordered pairs of the mode's probability), the share of
    pairs that round at all, the expected redder-than-both count over the op's lines (an answer counts
    with its probability), and the same count under nearest rounding, for E3's blind span.
    """
    from sca.data import ops as grammar

    pairs = grammar.unordered_pairs()
    out: dict[str, dict] = {}
    for op in grammar.OPS:
        ceiling = np.array([grammar.mode_prob(op, a, b) for a, b in pairs])
        rounded = np.array([not grammar.is_on_grid(op, a, b) for a, b in pairs])
        redder_near = redder_stoch = 0.0
        for a, b in grammar.lines():
            d = grammar.dose(a, b) + 1e-9
            redder_near += grammar.redness(op(a, b)) > d
            redder_stoch += sum(p for c, p in grammar.answer_dist(op, a, b).items() if grammar.redness(c) > d)
        out[op.name] = {
            "ceiling": float(ceiling.mean()),
            "ceiling_rounded": float(ceiling[rounded].mean()) if rounded.any() else 1.0,
            "rounded_frac": float(rounded.mean()),
            "redder_nearest": int(redder_near),
            "redder_stochastic": float(redder_stoch),
            "n_lines": len(grammar.lines()),
        }
    return out


# --- Training and the rounding readout -------------------------------------------------------


def cells(conditions: tuple, preps: dict[str, dict]) -> list[dict]:
    """One row per run, as ex-2.2.3's `cells`, with the corpus chosen by the condition's rounding."""
    from sca.utils import align

    rows = []
    for c in conditions:
        prep = preps[corpus_key(ROUNDING[c.name])]
        tc = prep["meta"].tokenizer_config
        anchor, anti = schedules(c)
        for seed in range(c.seeds):
            config = _make_config(align(tc.vocab_size, 64), seed, c.epochs)
            config.tokenizer = tc.model_copy()
            rows.append(
                {
                    "config": config,
                    "anchor": anchor,
                    "anti": anti,
                    "condition": c.name,
                    "seed": seed,
                    "label": f"{c.name}-s{seed}",
                    "prep": prep,
                }
            )
    return rows


def train_one(config, anchor: dict, anti: dict | None, corpus, traj_stride: int, probes, label: str) -> dict:
    """Ex-2.2.3's training step (the either-slot labeller, span pull), with this pilot's checkpoint names."""
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
        label_p=LabelSpec(p=slot_p, keying="either", pull="span"),
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
        "checkpoint": put(workdir / "model", name=f"ex-2.2.5-{label}-ckpt"),
    }


def eval_rounding(trained: dict, evals, condition: str, seed: int, label: str) -> dict:
    """The answer distribution against the model's, on every eval line.

    Per line: the model's exact match against the drawn answer (what ex-2.2.3's `accuracy` is) and against
    the rule's mode (the nearest answer), the chance a fresh draw of the line would match the model's guess
    (the exact match the model would score on average), the ceiling (the mode's probability), the model's
    mass on the answer's support and on the mode, and the KL from the answer distribution to the model's
    over the palette. Means per (op, split) over all lines and over the rounded lines come back as the
    result; the per-line table goes to the store.
    """
    from sca.data import ops as grammar
    from mini.store import get, put

    workdir = get_data_dir() / "rounding" / label
    model, tokenizer, color_ids, _ = _load(trained, workdir)
    palette_index = {rgb: i for i, rgb in enumerate(PALETTE.values())}
    sets = grammar.load_lines(get(evals, workdir / "evals.json").read_bytes())

    per: dict[str, dict[str, dict]] = {}
    cols: dict[str, list] = {
        k: []
        for k in (
            "op",
            "split",
            "ceiling",
            "p_mode",
            "support",
            "kl",
            "em_drawn",
            "em_mode",
            "expected_em",
            "rounded",
            "dose",
        )
    }
    for op, splits in sets.items():
        table = grammar.OP_BY_NAME[op]
        per[op] = {}
        for split, lns in splits.items():
            logp = _answer_logprobs(model, tokenizer, [ln.prompt for ln in lns])[:, color_ids]
            p = np.exp(logp)
            p_norm = p / p.sum(axis=1, keepdims=True)
            guess = p.argmax(axis=1)
            q = np.zeros_like(p)
            for i, ln in enumerate(lns):
                for c, pc in grammar.answer_dist(table, ln.lhs, ln.rhs).items():
                    q[i, palette_index[c]] = pc
            mode = np.array([palette_index[table(ln.lhs, ln.rhs)] for ln in lns])
            drawn = np.array([palette_index[ln.result] for ln in lns])
            rows = np.arange(len(lns))
            ceiling = q[rows, mode]
            support = (p * (q > 0)).sum(axis=1)
            kl = np.where(q > 0, q * (np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(p_norm, 1e-12))), 0).sum(axis=1)
            line = {
                "ceiling": ceiling,
                "p_mode": p[rows, mode],
                "support": support,
                "kl": kl,
                "em_drawn": (guess == drawn).astype(float),
                "em_mode": (guess == mode).astype(float),
                "expected_em": q[rows, guess],
                "rounded": (q > 0).sum(axis=1) > 1,
                "dose": np.array([grammar.dose(ln.lhs, ln.rhs) for ln in lns]),
            }
            for k, v in line.items():
                cols[k].extend(np.asarray(v).tolist())
            cols["op"].extend([op] * len(lns))
            cols["split"].extend([split] * len(lns))
            r = line["rounded"]
            per[op][split] = {
                "n": len(lns),
                "n_rounded": int(r.sum()),
                **{k: float(v.mean()) for k, v in line.items() if k not in ("rounded", "dose")},
                **{
                    f"{k}_rounded": float(v[r].mean()) if r.any() else float("nan")
                    for k, v in line.items()
                    if k not in ("rounded", "dose")
                },
            }
    arrays = {k: np.asarray(v) for k, v in cols.items()}
    return {
        "label": label,
        "condition": condition,
        "seed": seed,
        "per_op": per,
        "lines": put(_npz(**arrays), name=f"ex-2.2.5-{label}-lines.npz"),
    }


# --- Publishing ------------------------------------------------------------------------------


def design() -> dict:
    return {
        "n_runs": N_RUNS,
        "conditions": [asdict(c) | {"steps": c.steps, "rounding": ROUNDING[c.name]} for c in CONDITIONS],
        "epochs": EPOCHS,
        "span": ex223.SPAN,
        "red_dose": ex223.RED_DOSE,
        "nonred_dose": ex223.NONRED_DOSE,
    }


def _slim(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("arrays", "lines", "traj", "val_loss", "train_loss")}


def publish_results(
    trained: list[dict],
    evaled: list[dict],
    rounded: list[dict],
    scored: list[dict],
    probed: list[dict],
    corpora: list[dict],
    table: dict,
) -> dict:
    import json

    from mini.store import get, put, set_ref

    metrics = {
        "runs": [_slim(r) for r in evaled],
        "rounding": [_slim(r) for r in rounded],
        "scores": [_slim(r) for r in scored],
        "corpora": corpora,
        "grammar": table,
        "design": design(),
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.2.5-metrics.json"))
    traj = {t["label"]: {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.5-trajectories.json"))
    set_ref(GEOMETRY_REF, put(json.dumps({"runs": probed}).encode(), name="ex-2.2.5-geometry.json"))
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    arrays = {}
    for r, kind, key in [
        *((r, "eval", "arrays") for r in evaled),
        *((r, "score", "arrays") for r in scored),
        *((r, "lines", "lines") for r in rounded),
    ]:
        path = get(r[key], get_data_dir() / "publish" / f"{r['label']}-{kind}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{kind}/{name}": z[name] for name in z.files}
    set_ref(ARRAYS_REF, put(_npz(**arrays), name="ex-2.2.5-arrays.npz"))
    return {"n_runs": len(evaled), "holdout_em": {r["label"]: r["holdout_em"] for r in evaled}}


# --- Orchestration ----------------------------------------------------------------------------


def main(ctx: Ctx) -> dict:
    kinds: list[Rounding] = ["stochastic", "nearest"]
    prepped = ctx.map(
        prepare_corpus,
        kinds,
        [ex223.N_LINES] * 2,
        [ex223.CORPUS_SEED] * 2,
        [ex223.HOLDOUT_FRAC] * 2,
        [ex223.N_PROBE] * 2,
        [ex223.PROBE_SEED] * 2,
        [ex223.PER_SLOT_RATE] * 2,
        [ex223.RED_RATE] * 2,
        role="prep",
    )
    preps = {corpus_key(k): p for k, p in zip(kinds, prepped, strict=True)}
    table = ctx.run(grammar_table, role="prep")

    rows = cells(CONDITIONS, preps)
    n = len(rows)
    trained = ctx.map(
        train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [r["prep"]["corpus"] for r in rows],
        [TRAJ_STRIDE] * n,
        [r["prep"]["probes"] for r in rows],
        [r["label"] for r in rows],
        role="train",
    )
    evaled = ctx.map(
        eval_one,
        trained,
        [r["prep"]["evals"] for r in rows],
        [r["prep"]["probes"] for r in rows],
        [r["anchor"]["tau"] for r in rows],
        [r["condition"] for r in rows],
        [r["seed"] for r in rows],
        [r["label"] for r in rows],
        role="eval",
    )
    rounded = ctx.map(
        eval_rounding,
        trained,
        [r["prep"]["evals"] for r in rows],
        [r["condition"] for r in rows],
        [r["seed"] for r in rows],
        [r["label"] for r in rows],
        role="eval",
    )
    scored = ctx.map(
        score_one,
        trained,
        [r["prep"]["probes"] for r in rows],
        [r["condition"] for r in rows],
        [r["seed"] for r in rows],
        [r["label"] for r in rows],
        role="score",
    )
    controls = [i for i, r in enumerate(rows) if r["condition"] in (CONTROL_STOCH.name, CONTROL_NEAR.name)]
    probed = ctx.map(
        probe_one,
        [trained[i] for i in controls],
        [rows[i]["prep"]["probes"] for i in controls],
        [rows[i]["condition"] for i in controls],
        [rows[i]["seed"] for i in controls],
        [rows[i]["label"] for i in controls],
        role="probe",
    )
    return ctx.run(
        publish_results, trained, evaled, rounded, scored, probed, [p["stats"] for p in prepped], table, role="prep"
    )


experiment = Experiment(
    name="ex-2.2.5",
    main=main,
    roles={
        "prep": dict(cpu=2, timeout=1800),
        "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
        "eval": dict(gpu="L4", timeout=1200),
        "score": dict(gpu="L4", timeout=1800),
        "probe": dict(cpu=4, timeout=1800),
    },
)
