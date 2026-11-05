# ruff: noqa: B018
# title: Equations as images

# A figure study for the D2.2 post, with no new training: the in-context grammar drawn with pictures for operands, and
# the `τ_ℓ fixed` runs (fixed-confirm, seeds 712 to 719) answering one context per pixel, before and after the full
# edit. `experiment.py` beside this script builds the operands and stores the answer pictures; this script draws them
# and writes each panel out as its own PNG, for an interactive figure.

import io
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import experiment as ex
from mini.lit import memo, stop
from mini.reports import current_publisher
from mini.store import project_store
from mini.vis import light_dark, themed
from sca.data.ops import colors

SCALE = 1
W, H = ex.size_of(SCALE)
PALETTE = np.asarray(colors(), float) / 15
OPS = ex.OP_NAMES
A = ex.ANCHORED_OP


def fetch(refs: list[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}.npz") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def load() -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]] | None:
    refs = [ex.OPERANDS_REF.format(scale=SCALE)] + [ex.ANSWERS_REF.format(scale=SCALE, seed=s) for s in ex.SEEDS]
    with tempfile.TemporaryDirectory() as tmp:
        files = fetch(refs, Path(tmp))
        if any(f is None for f in files.values()):
            return None
        out = []
        for r in refs:
            f = files[r]
            assert f is not None
            with np.load(f) as z:
                out.append({k: z[k] for k in z.files})
    return out[0], dict(zip(ex.SEEDS, out[1:], strict=True))


loaded = load()
if loaded is None:
    stop(
        "No results yet. Run the experiment:\n```bash\nbin/mini run docs/m2/equation-images/experiment.py --app modal\n```"
    )
operands, answers = loaded
OP_TABLES = ex.op_tables()


def truth(render: str, query: tuple[str, str], op: str) -> np.ndarray:
    """The true op answer at each pixel of a query pair, rounded to the nearest grid level."""
    a, b = operands[f"{render}/{query[0]}"], operands[f"{render}/{query[1]}"]
    return OP_TABLES[OPS.index(op), a, b]


def example_answer(render: str, pair: tuple[str, str], op: str) -> np.ndarray:
    return truth(render, pair, op)


def p_truth(seed: int, render: str, query: tuple[str, str], op: str, edit: str) -> float:
    return float(answers[seed][f"{render}/{query[1]}/{op}/{edit}/p_truth"].astype(float).mean())


# The seed shown: the median of the eight in how much of `difference` the full edit takes out, on the first query.
_q0 = ex.QUERIES[0]
_kept = {s: p_truth(s, "posterized", _q0, A, "full edit") / p_truth(s, "posterized", _q0, A, "clean") for s in ex.SEEDS}
SEED = sorted(_kept, key=lambda s: _kept[s])[len(_kept) // 2]

N_CONTEXTS = W * H * len(ex.QUERIES) * len(ex.RENDERS) * len(OPS)

rf"""
# Equations as images

/// tip |
<!-- lede -->
We drew contexts of the in-context grammar with pictures for operands, so the model answers a whole image one pixel at a time. The unedited model paints a recognizable picture under every op, and the full edit turns the `difference` picture into a speckle of the other ops while leaving the others alone.
///

This is a figure study for the D2.2 post, with no new training. It reads the eight `τ_ℓ fixed` runs of the fixed-confirm experiment and asks each of them about {N_CONTEXTS:,} contexts, one per pixel of each query image, op and way of putting the operands on the grid, with and without the edit.
"""

# %%

rf"""
## Scope and method

Each pixel of a query image pair gets a context of its own. Its three examples are pixels drawn at random from three example image pairs, each shown with its answer under the hidden op, and its query is the pixel pair itself. The model answers at the query `=`, and the answers over every pixel make a picture of the op as the model infers it. The examples are the same draws for every run and edit, so pictures from different runs and edits match pixel for pixel.

Operands
:   Photos from the Kodak Lossless True Color suite, which Kodak released for unrestricted use, and two synthetic images: a gray ramp from black to white, and a map of hue across and lightness down. Each is resized to {W} × {H} and put on the grid of 216 colors in one of two ways. Posterized rounds each channel to its nearest level. Dithered spreads the rounding error to the neighboring pixels (Floyd–Steinberg), so a patch of pixels averages to the original color.

Examples and queries
:   The examples are drawn from {", ".join(f"{a} × {b}" for a, b in ex.EXAMPLES)}. The queries are {" and ".join(f"{a} × {b}" for a, b in ex.QUERIES)}, where "huelight" is the hue and lightness map.

True answer
:   The op applied to the pixel pair, rounded to the nearest grid level, which is the most likely answer under the random rounding the corpus uses.

Argmax and T = 1
:   The color the model gives the most probability, and one color drawn from the model's distribution over colors.

Full edit
:   The e₁ component removed from the state at every position and every slice, as in the D2.2 design.

The pictures below are from seed {SEED}, the median of the eight in how much of `difference` the edit takes out. The table in E3 covers all eight.
"""

# %%


def show(ax, idx: np.ndarray) -> None:
    ax.imshow(PALETTE[idx], interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color(light_dark("0.75", "0.35"))


PANEL_IN = W / 96  # one image pixel per CSS pixel at the page's 96 px per inch


def draw_context(seed: int, render: str, query: tuple[str, str]) -> plt.Figure:
    rows = [(*pair, example_answer(render, pair, A)) for pair in ex.EXAMPLES]
    a = answers[seed]
    rows.append((query[0], query[1], a[f"{render}/{query[1]}/{A}/clean/sample"]))
    gap = 0.3
    fig = plt.figure(figsize=(3 * PANEL_IN + 2 * gap + 0.3, len(rows) * (PANEL_IN * H / W + 0.12)))
    gs = fig.add_gridspec(len(rows), 5, width_ratios=[PANEL_IN, gap, PANEL_IN, gap, PANEL_IN], wspace=0.04, hspace=0.12)
    for r, (an, bn, y) in enumerate(rows):
        for c, img in ((0, operands[f"{render}/{an}"]), (2, operands[f"{render}/{bn}"]), (4, y)):
            show(fig.add_subplot(gs[r, c]), img)
        for c, sym in ((1, "?"), (3, "=")):
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")
            ax.text(0.5, 0.5, sym, ha="center", va="center", fontsize=16)
    return fig


@memo
def context_figure(seed: int, render: str, query: tuple[str, str]) -> str:
    alt = (
        f"Four equations drawn as pictures, one per row: three examples of {A}, each an image, a question mark, a "
        f"second image, an equals sign and their combination, then the query {query[0]} ? {query[1]} with the model "
        "answer painted pixel by pixel."
    )
    return themed(draw_context, name=f"context-{render}-{query[1]}", alt_text=alt)(seed, render, query)


r"""
## A context as pictures (E1)

What does one context look like when its operands are pictures? Below are the three examples of `difference`, each with its true answer, and the query with the model's answer, drawn at T = 1.
"""

context_figure(SEED, "posterized", ex.QUERIES[0])

r"""
The examples show the op on very different inputs: a hue and lightness map, a gray ramp, and two photos. The query picture is mostly clean. Under `difference` the parrots darken where the ramp is close to their own brightness and turn to their complementary colors where the ramp is light, as the true op does, with a scatter of single pixels where the drawn examples left the op in doubt or the draw landed on a less likely color.
"""

# %%

COLUMNS = (("truth", "true answer"), ("clean/argmax", "argmax"), ("clean/sample", "T = 1"))
EDITED = (("full edit/argmax", "edited, argmax"), ("full edit/sample", "edited, T = 1"))


def panel(seed: int, render: str, query: tuple[str, str], op: str, col: str) -> np.ndarray:
    if col == "truth":
        return truth(render, query, op)
    return answers[seed][f"{render}/{query[1]}/{op}/{col}"]


def draw_ops(seed: int, render: str, query: tuple[str, str]) -> plt.Figure:
    cols = COLUMNS + EDITED
    fig, axs = plt.subplots(
        len(OPS), len(cols), figsize=(len(cols) * (PANEL_IN + 0.06) + 0.9, len(OPS) * (PANEL_IN * H / W + 0.08) + 0.3)
    )
    for r, op in enumerate(OPS):
        for c, (key, title) in enumerate(cols):
            ax = axs[r, c]
            show(ax, panel(seed, render, query, op, key))
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(op, fontsize=9, rotation=0, ha="right", va="center")
    return fig


@memo
def ops_figure(seed: int, render: str, query: tuple[str, str]) -> str:
    alt = (
        f"A grid of pictures, one row per op and five columns: the true answer for {query[0]} combined with "
        f"{query[1]} ({render}), the unedited model answer by argmax and by a draw at T = 1, and the edited model "
        "answer the same two ways."
    )
    return themed(draw_ops, name=f"ops-{render}-{query[1]}", alt_text=alt)(seed, render, query)


rf"""
## Every op, before and after the edit (E2)

Does the picture carry the op for every op, and what does the edit do to each? Each row is one op, with the true answer, then the unedited model, then the edited model. The query is {ex.QUERIES[0][0]} × {ex.QUERIES[0][1]}, posterized.
"""

ops_figure(SEED, "posterized", ex.QUERIES[0])

r"""
The unedited model paints each op recognizably. Most argmax pictures are close to the true answers, with a light speckle that the T = 1 pictures thicken. `mix` is speckled even at argmax: its answers round at random on most pairs in the corpus, so the model rightly spreads its probability over neighboring colors and the most likely one changes from pixel to pixel. Under the edit only the `difference` row changes. The picture becomes a speckle of the other ops, and on this query mostly of the ones that give the parrots back in their own colors. The argmax picture shows the op the edited model favors at each pixel, and the T = 1 picture shows the whole spread.

The same figure for the hue and lightness query follows. There the edited `difference` picture looks most like the `mix` picture, which fits an edited model that falls back on the ops that best explain the examples.
"""

ops_figure(SEED, "posterized", ex.QUERIES[1])

r"""
And the two queries again with dithered operands, examples included:
"""

ops_figure(SEED, "dithered", ex.QUERIES[0])

# %%

ops_figure(SEED, "dithered", ex.QUERIES[1])

r"""
Dithering makes the operands and the true answers easier to read at this size, since the eye blends the pixels. The model's speckle is harder to tell apart from the dither, but the edit shows as plainly in both.
"""

# %%


def table_rows() -> list[dict]:
    rows = []
    for op in OPS:
        row: dict = {"op": op}
        for edit in ex.EDITS:
            v = np.array([p_truth(s, "posterized", ex.QUERIES[0], op, edit) for s in ex.SEEDS])
            row[edit] = (float(v.mean()), float(v.min()), float(v.max()))
        rows.append(row)
    return rows


TABLE = table_rows()


def fmt(t: tuple[float, float, float]) -> str:
    return f"{t[0]:.2f} ({t[1]:.2f} to {t[2]:.2f})"


_lines = "\n".join(f"| `{r['op']}` | {fmt(r['clean'])} | {fmt(r['full edit'])} |" for r in TABLE)
_diff = next(r for r in TABLE if r["op"] == A)

rf"""
## All eight runs (E3)

Do the other seeds paint the same pictures? The table gives the probability each run puts on the true answer, averaged over the pixels of the {ex.QUERIES[0][0]} × {ex.QUERIES[0][1]} query (posterized), as the seed mean with the seed range in parentheses.

| op | unedited | full edit |
| --- | ---: | ---: |
{_lines}

The edit takes the `difference` picture from {_diff["clean"][0]:.2f} to {_diff["full edit"][0]:.2f} on the seed mean and moves the other ops by a few hundredths at most, so the pictures of seed {SEED} are typical. These probabilities are not the D2.2 scores, which are taken on held-out corpus contexts with replacement op noise and weigh every answer the random rounding can give.
"""

# %%

MANIFEST: dict[str, str] = {}


def write_panels(seed: int) -> None:
    """Each panel of the shown seed as its own PNG at its native size, and a manifest naming them."""
    pub = current_publisher()
    if pub is None:
        return

    def png(idx: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray((PALETTE[idx] * 255).round().astype(np.uint8)).save(buf, format="PNG")
        return buf.getvalue()

    for render in ex.RENDERS:
        for name in sorted({n for pair in ex.EXAMPLES + ex.QUERIES for n in pair}):
            key = f"operand-{render}-{name}"
            MANIFEST[key] = pub.asset_url(png(operands[f"{render}/{name}"]), name=f"{key}.png")
        for pair in ex.EXAMPLES:
            key = f"example-{render}-{pair[0]}-{pair[1]}-{A}"
            MANIFEST[key] = pub.asset_url(png(example_answer(render, pair, A)), name=f"{key}.png")
        for query in ex.QUERIES:
            for op in OPS:
                for col, _ in COLUMNS + EDITED:
                    key = f"answer-{render}-{query[1]}-{op}-{col.replace('/', '-').replace(' ', '-')}"
                    MANIFEST[key] = pub.asset_url(png(panel(seed, render, query, op, col)), name=f"{key}.png")
    MANIFEST["manifest"] = pub.asset_url(json.dumps(MANIFEST, indent=1).encode(), name="panels.json")


write_panels(SEED)

rf"""
## The panels as files

Every panel of seed {SEED} is also a PNG at its native size ({W} × {H}, one pixel per grid color), named in [panels.json]({MANIFEST.get("manifest", "_assets/panels.json")}), for an interactive figure. Scale them up with `image-rendering: pixelated` to keep the pixels sharp.

## References

fixed-confirm
:   [Fixed confirmation](../fixed-confirm/report.py): `τ_ℓ fixed` kept the anchor on all eight fresh seeds, 712 to 719, and became the recipe of record.
"""
