"""Equations as images: the in-context grammar drawn with pictures for operands, and the model answering pixel by pixel.

Each pixel of a query image pair gets a context of its own. Its three examples are pixels drawn at random from three
example image pairs, each shown with its answer under the hidden op, and its query is the pixel pair itself. The model
answers at the query `=`, so the answers over every pixel make a picture: the op as the model infers it. The full edit
(the e₁ projection at every slice and position) gives the same picture for the edited model.

The operands are photos from the Kodak Lossless True Color suite and two synthetic images (a gray ramp and a
hue/lightness map), resized and then put on the 216-color grid in two ways: posterized (each channel to its nearest
level) and dithered (Floyd–Steinberg error diffusion over the levels). The runs are `τ_ℓ fixed` at seeds 712 to 719
(fixed-confirm).

    bin/mini run docs/m2/equation-images/experiment.py --app modal
    bin/mini status equation-images
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir


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


fc = _load_sibling("fixed-confirm", "equation_images_fc")
ex2216 = fc.ex2222.ex2216

OP_NAMES: tuple[str, ...] = tuple(fc.OP_NAMES)
ANCHORED_OP: str = fc.ANCHORED_OP
K: int = fc.K
SEEDS: tuple[int, ...] = tuple(fc.SEEDS)

UNIT = 6  # tokens per equation: a ? b = y and a separator
EQ = UNIT * K + 3  # the query `=`, where the answer is read

KODAK_REPO = "danjacobellis/kodak"
KODAK_REVISION = "8ad2123f7cc141390636df3632dda7c77a7914b1"
KODAK_FILE = "data/validation-00000-of-00001.parquet"
# The Kodak photos used, all landscape (768 × 512), by suite number.
PHOTOS = {"parrots": 23, "hats": 3, "barn": 22, "bikes": 5, "rafting": 14}
SYNTHETIC = ("ramp", "huelight")

BASE_SIZE = (192, 128)  # width, height at scale 1: a quarter of the Kodak frame
SCALES = (1,)
RENDERS = ("posterized", "dithered")

# The example pairs (a, b) the three examples of each context are drawn from, and the query pairs.
EXAMPLES = (("hats", "huelight"), ("barn", "ramp"), ("bikes", "rafting"))
QUERIES = (("parrots", "ramp"), ("parrots", "huelight"))
EDITS = ("clean", "full edit")
DRAW_SEED = 2026_11_05  # the example draws and the T = 1 samples; shared by every run, so runs pair pixel by pixel

PREFIX = "reports/m2/equation-images"
OPERANDS_REF = PREFIX + "/operands/x{scale}"
ANSWERS_REF = PREFIX + "/answers/x{scale}/s{seed}"


def size_of(scale: int) -> tuple[int, int]:
    return BASE_SIZE[0] * scale, BASE_SIZE[1] * scale


# =============================================================================================
# Operands
# =============================================================================================


def grid_index(rgb: np.ndarray) -> np.ndarray:
    """Palette indices (`sca.data.ops.colors()` order) of grid colors `(..., 3)` on the 0..15 scale."""
    from sca.data.ops import colors

    lookup = {c: i for i, c in enumerate(colors())}
    flat = rgb.reshape(-1, 3).astype(int)
    return np.array([lookup[tuple(int(v) for v in c)] for c in flat], dtype=np.uint8).reshape(rgb.shape[:-1])


def posterize(x: np.ndarray) -> np.ndarray:
    """Each channel of an image `(H, W, 3)` on the 0..15 scale to its nearest grid level."""
    from sca.data.ops import LEVELS

    lv = np.asarray(LEVELS, float)
    return lv[np.abs(x[..., None] - lv).argmin(-1)]


def dither(x: np.ndarray) -> np.ndarray:
    """Floyd–Steinberg error diffusion over the grid levels, per channel, in raster order."""
    x = x.astype(float).copy()
    out = np.zeros_like(x)
    h, w = x.shape[:2]
    for i in range(h):
        for j in range(w):
            new = posterize(x[i, j])
            out[i, j] = new
            e = x[i, j] - new
            if j + 1 < w:
                x[i, j + 1] += e * 7 / 16
            if i + 1 < h:
                if j > 0:
                    x[i + 1, j - 1] += e * 3 / 16
                x[i + 1, j] += e * 5 / 16
                if j + 1 < w:
                    x[i + 1, j + 1] += e / 16
    return out


def synthetic(name: str, w: int, h: int) -> np.ndarray:
    """The synthetic operands on the 0..15 scale: a gray ramp from black (left) to white (right), and a map of hue
    across and lightness down (white at the top, black at the bottom, fully saturated between).
    """
    import colorsys

    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    match name:
        case "ramp":
            return np.broadcast_to(u[None, :, None] * 15, (h, w, 3)).copy()
        case "huelight":
            return np.array([[colorsys.hls_to_rgb(hu, 1 - li, 1.0) for hu in u] for li in v]) * 15
        case _:
            raise ValueError(name)


def operands(scale: int) -> dict:
    """Every operand at *scale*, in both renders, as palette indices `(H, W)`; also the smooth images, for reference."""
    import io

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    from mini.store import put

    w, h = size_of(scale)
    path = hf_hub_download(KODAK_REPO, KODAK_FILE, repo_type="dataset", revision=KODAK_REVISION)
    rows = {r["path"].rsplit("/", 1)[-1]: r["image"]["bytes"] for r in pq.read_table(path).to_pylist()}
    smooth: dict[str, np.ndarray] = {}
    for name, n in PHOTOS.items():
        im = Image.open(io.BytesIO(rows[f"kodim{n:02d}.png"])).convert("RGB")
        assert im.size == (768, 512), f"kodim{n:02d} is not landscape"
        smooth[name] = np.asarray(im.resize((w, h), Image.LANCZOS), float) / 255 * 15
    for name in SYNTHETIC:
        smooth[name] = synthetic(name, w, h)
    arrays: dict[str, np.ndarray] = {}
    for name, x in smooth.items():
        arrays[f"smooth/{name}"] = (x / 15 * 255).round().astype(np.uint8)
        arrays[f"posterized/{name}"] = grid_index(posterize(x))
        arrays[f"dithered/{name}"] = grid_index(dither(x))
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return {"scale": scale, "operands": put(buf.getvalue(), name=f"equation-images-operands-x{scale}.npz")}


# =============================================================================================
# Answers
# =============================================================================================


def op_tables() -> np.ndarray:
    """Each op as a lookup table `(n_ops, 216, 216)` of palette indices: the answer to (a, b), rounded to the nearest
    grid level (the most likely answer under the stochastic rounding of the corpus).
    """
    from sca.data.ops import CANDIDATE_BY_NAME, OP_BY_NAME, colors

    by_name = OP_BY_NAME | CANDIDATE_BY_NAME
    cs = colors()
    lookup = {c: i for i, c in enumerate(cs)}
    out = np.zeros((len(OP_NAMES), len(cs), len(cs)), dtype=np.uint8)
    for o, name in enumerate(OP_NAMES):
        op = by_name[name]
        for i, a in enumerate(cs):
            for j, b in enumerate(cs):
                out[o, i, j] = lookup[op(a, b)]
    return out


def contexts(
    ops: np.ndarray, imgs: dict[str, np.ndarray], render: str, query: tuple[str, str], o: int, rng
) -> tuple[np.ndarray, np.ndarray]:
    """One context per query pixel, as palette indices `(N, K, 3)` of example (a, b, y), and the query `(N, 2)`.

    Each example is a pixel drawn at random from its example pair, its answer the op applied there.
    """
    qa, qb = imgs[f"{render}/{query[0]}"].reshape(-1), imgs[f"{render}/{query[1]}"].reshape(-1)
    n = len(qa)
    ex = np.zeros((n, K, 3), dtype=np.int64)
    for e, (an, bn) in enumerate(EXAMPLES):
        ea, eb = imgs[f"{render}/{an}"].reshape(-1), imgs[f"{render}/{bn}"].reshape(-1)
        at = rng.integers(0, len(ea), n)
        ex[:, e, 0], ex[:, e, 1] = ea[at], eb[at]
        ex[:, e, 2] = ops[o, ex[:, e, 0], ex[:, e, 1]]
    return ex, np.stack([qa, qb], axis=1).astype(np.int64)


def tokens_of(ex: np.ndarray, q: np.ndarray, color_ids: np.ndarray, stoi: dict[str, int]) -> np.ndarray:
    """The token rows `(N, 6K + 6)` of contexts in the layout of `sca.data.incontext.Context.words`. The query answer
    slot holds the newline token, since the answer is read at the query `=` and nothing after it is seen.
    """
    n = len(q)
    sym, eq, sep, nl = stoi["?"], stoi["="], stoi[","], stoi["\n"]
    rows = np.zeros((n, UNIT * (K + 1)), dtype=np.int32)
    for e in range(K):
        rows[:, UNIT * e : UNIT * (e + 1)] = np.stack(
            [
                color_ids[ex[:, e, 0]],
                np.full(n, sym),
                color_ids[ex[:, e, 1]],
                np.full(n, eq),
                color_ids[ex[:, e, 2]],
                np.full(n, sep),
            ],
            axis=1,
        )
    rows[:, UNIT * K :] = np.stack(
        [color_ids[q[:, 0]], np.full(n, sym), color_ids[q[:, 1]], np.full(n, eq), np.full(n, nl), np.full(n, nl)],
        axis=1,
    )
    return rows


def answers_one(checkpoint, operand_art, scale: int, seed: int) -> dict:
    """Every answer picture of one run at *scale*: per op, query pair, render and edit, the argmax color and one draw
    at T = 1 at each pixel (palette indices), and the probability the model gives the true op's answer.
    """
    import io

    from mini.store import get, put
    from sca.intervention import Subspace, logits_at, projection

    workdir = get_data_dir() / "equation-images" / f"x{scale}-s{seed}"
    model, tokenizer, color_ids, _ = ex2216._load_run(checkpoint, workdir)
    with np.load(get(operand_art, workdir / "operands.npz")) as z:
        imgs = {k: z[k] for k in z.files}
    ops = op_tables()
    w, h = size_of(scale)
    sub = Subspace.axis(model.transformer.wte.shape[1])
    every = tuple(range(len(model.transformer.blocks) + 1))
    operators = {"clean": (projection(sub, 0.0), ()), "full edit": (projection(sub, 1.0), every)}
    out: dict[str, np.ndarray] = {}
    for render in RENDERS:
        for query in QUERIES:
            for o, op in enumerate(OP_NAMES):
                rng = np.random.default_rng([DRAW_SEED, RENDERS.index(render), QUERIES.index(query), o])
                ex, q = contexts(ops, imgs, render, query, o, rng)
                toks = tokens_of(ex, q, color_ids, tokenizer.stoi)
                u = rng.random(len(q))  # one uniform per pixel, shared by both edits
                truth = ops[o, q[:, 0], q[:, 1]]
                for edit in EDITS:
                    operator, slices = operators[edit]
                    p = ex2216._color_probs(logits_at(model, toks, operator, slices, EQ), color_ids)
                    key = f"{render}/{query[1]}/{op}/{edit}"
                    out[f"{key}/argmax"] = p.argmax(1).astype(np.uint8).reshape(h, w)
                    cdf = np.cumsum(p / p.sum(1, keepdims=True), axis=1)
                    out[f"{key}/sample"] = (
                        (cdf < u[:, None]).sum(1).clip(0, p.shape[1] - 1).astype(np.uint8).reshape(h, w)
                    )
                    out[f"{key}/p_truth"] = p[np.arange(len(q)), truth].astype(np.float16).reshape(h, w)
                    out[f"{key}/color_mass"] = p.sum(1).astype(np.float16).reshape(h, w)
    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    return {"scale": scale, "seed": seed, "answers": put(buf.getvalue(), name=f"equation-images-x{scale}-s{seed}.npz")}


# =============================================================================================
# The DAG
# =============================================================================================


def resolve() -> dict:
    """The checkpoint of each run."""
    from mini.store import get_ref

    refs = {s: fc.CHECKPOINT_REF.format(label=f"fixed-s{s}") for s in SEEDS}
    checkpoints = {s: get_ref(r) for s, r in refs.items()}
    missing = [r for s, r in refs.items() if checkpoints[s] is None]
    assert not missing, f"not published: {missing}"
    return checkpoints


def publish(made: list[dict], answered: list[dict]) -> dict:
    from mini.store import set_ref

    for m in made:
        set_ref(OPERANDS_REF.format(scale=m["scale"]), m["operands"])
    for a in answered:
        set_ref(ANSWERS_REF.format(scale=a["scale"], seed=a["seed"]), a["answers"])
    return {"operands": len(made), "answers": len(answered)}


def main(ctx: Ctx) -> dict:
    checkpoints = ctx.run(resolve, role="prep")
    made = ctx.map(operands, list(SCALES), role="prep")
    jobs = [(m, s) for m in made for s in SEEDS]
    answered = ctx.map(
        answers_one,
        [checkpoints[s] for _, s in jobs],
        [m["operands"] for m, _ in jobs],
        [m["scale"] for m, _ in jobs],
        [s for _, s in jobs],
        role="answer",
    )
    return ctx.run(publish, made, answered, role="prep")


def design() -> dict[str, Any]:
    return {"ops": OP_NAMES, "anchored_op": ANCHORED_OP, "examples": EXAMPLES, "queries": QUERIES, "renders": RENDERS}


COMPUTE = {
    "prep": dict(cpu=2, memory=4096, timeout=1800),
    # 56 passes of 24,576 contexts at scale 1 (16× that at scale 4): a few minutes on four cores at scale 1.
    "answer": dict(cpu=4, memory=16384, timeout=3600),
}

experiment = Experiment(name="equation-images", main=main, roles=COMPUTE)
