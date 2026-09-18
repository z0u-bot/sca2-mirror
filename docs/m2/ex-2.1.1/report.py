# title: Ex 2.1.1: un-anchored color-mixing transformer

import json
import tempfile
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

# The document's directory is on sys.path while it runs, so the experiment
# definition is importable — refs and sweep constants can't drift.
from experiment import CKPT_REF, CORPUS_SEED, DEPTHS, HOLDOUT_FRAC, METRICS_REF, N_EXAMPLES, SEEDS, WEIGHTS_REF, WIDTHS
from mini.lit import stop
from mini.reports import externalize_html
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed
from sca import baselines as bl
from sca.data import colors, cube
from sca.vis import CUBE_VIEWS, draw_cube_bound, grid_diameter, plot_rgb_cube, project_cube
from subline.series import Series
from subline.subline import Subline

EVAL_SETS = ["named_seen", "named_holdout", "hex_unseen", "cross_unseen"]


def load_results() -> tuple[list[dict], dict[str, np.ndarray]] | None:
    """Resolve the metrics and probe weights from the store, or None if unpublished."""
    store = project_store()
    arts = store.get_refs([METRICS_REF, WEIGHTS_REF])
    m_art, w_art = arts[METRICS_REF], arts[WEIGHTS_REF]
    if m_art is None or w_art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        m_path, w_path = store.get_many([(m_art, Path(d) / "metrics.json"), (w_art, Path(d) / "weights.npz")])
        metrics = json.loads(m_path.read_text())
        with np.load(w_path) as z:
            weights = {k: z[k] for k in z.files}
    return metrics, weights


def label(w: int, d: int, s: int) -> str:
    return f"d{w}-L{d}-s{s}"


def acc(metrics: list[dict], w: int, d: int, s: int, eval_set: str) -> float:
    (r,) = [r for r in metrics if r["label"] == label(w, d, s)]
    return r["accuracy"][eval_set]["accuracy"]


def width_shades() -> dict[int, tuple]:
    stops = light_dark([0.7, 0.45, 0.12], [0.8, 0.55, 0.28])
    return dict(zip(WIDTHS, plt.cm.viridis(stops), strict=True))


def pick_arch(metrics: list[dict]) -> tuple[int, int]:
    """The smallest condition (by params ∝ width²·depth) that saturates the unseen-pair sets."""

    def unseen(w: int, d: int) -> float:
        return float(np.mean([acc(metrics, w, d, s, es) for s in SEEDS for es in ("hex_unseen", "cross_unseen")]))

    cells = sorted(((w, d) for w in WIDTHS for d in DEPTHS), key=lambda c: c[0] ** 2 * c[1])
    return next((c for c in cells if unseen(*c) >= 0.995), max(cells, key=lambda c: unseen(*c)))


r"""
# Ex 2.1.1: un-anchored color-mixing transformer

/// tip |
<!-- tl;dr -->
A small transformer learns a character-level language of color-mixing equations, solving the forms it saw in training and unseen hex pairs. Color turns out to be linearly decodable from its residual stream, with each seed putting *redness* in a different place. Held-out *named* pairs sit at zero accuracy.
///

In M2, we want to see whether Sparse Concept Anchoring carries over from autoencoders to transformers. Before we anchor anything we need a baseline, so this experiment trains a small transformer on a well-defined task.

The model must learn a character-level language of color mixing equations on a 16-level RGB grid. Here are the sample types, which we will refer to throughout:

| Type | Example |
|------|---------|
| Named pairs | `red + blue = purple` |
| Hex pairs   | `#f00 + #00f = #808`  |
| Cross-form  | `red + #00f = #808`   |
| Alias       | `red = #f00`          |

Every operand spans several tokens in both of its spellings, to force the model to perform two tasks simultaneously: it must mix the colors and spell the result.

Mixing (`+`) is the channel-wise round-half-up mean, so each prompt has one correct completion.

We sweep width {16, 32, 64} × depth {2, 4} × 3 seeds ([experiment definition](./experiment.py)), and for each condition we measure:

- Completion accuracy: greedy decoding, scored as an exact string match, over four evaluation sets. Those are named pairs seen in training; held-out named pairs, which never appear as named equations, so the model has to combine the alias dictionary with hex arithmetic to answer them; hex-only equations; and cross-form operand pairs that were never shown together.

- Probe alignment [^probes]: ridge regression from the residual stream at each depth out to the operand color, the result color, and the *redness* of the result.

## Hypotheses

**H1.** A small nGPT should learn the task, with near-perfect accuracy on seen forms and on unseen *hex* pairs; that leaves the anchored runs room to show any degradation later.

**H2.** Color should be linearly readable from the residual stream, more so as depth increases.

**H3.** The *redness* probe directions should vary from seed to seed. This is part of the motivation for this work: searching for a concept after training turns up a different geometry every time, whereas SCA should let us fix the location in advance.

[^probes]: A probe is a small linear model we fit on the internal
activations of the model to read out what those activations carry. Ridge regression is linear regression with a penalty on large weights, which keeps the fit stable.

## Training data

The corpus sampler is deterministic. Regenerating it here with the constants defined in the experiment gives back the same training data the model saw, and these are its first lines:
"""

train_pairs, holdout = colors.split_named_pairs(CORPUS_SEED, HOLDOUT_FRAC)
corpus = colors.sample_corpus(N_EXAMPLES, CORPUS_SEED, train_pairs)


def _form(ex) -> str:
    if ex.rhs is None:
        return "alias"
    return {0: "named", 3: "hex"}.get(ex.prompt.count("#") + ex.answer.count("#"), "cross")


counts = {f: sum(_form(ex) == f for ex in corpus) for f in ("hex", "named", "cross", "alias")}
pairs = {p for ex in corpus if (p := ex.pair) is not None}
grid = colors.N_LEVELS**3
all_pairs = grid * (grid + 1) // 2
head = "".join(ex.text for ex in corpus[:10])
body = f"```\n{head}```"
caption = f"{len(corpus):,} lines in total: {', '.join(f'{n:,} {f}' for f, n in counts.items())}."

rf"""
{figure_html(body, caption=caption, class_="report-figure")}

Between them they cover {len(pairs):,} distinct operand pairs, **{len(pairs) / all_pairs:.2%}** of the {all_pairs / 1e6:.1f}M in the grid. So the unseen-pair eval sets, sampled to steer clear of all of them, test the mixing rule rather than recall.

### The color space

Every color is a point on an RGB grid with 16 levels per channel, so 16³ = 4096 points in all. If we rotate the cube so its black-to-white diagonal stands vertical, *value* runs up the page and hue wraps around it. That is the figure below, seen front-on toward the *red* corner. Hex and cross equations draw their operands from anywhere in this cube.
"""


@themed(
    name="color-space-cube",
    alt_text="An orthographic front view of the RGB grid, rotated so the black-to-white diagonal is vertical: black at the bottom, white at the top, hues fanned around the middle, showing the red, green, and magenta faces. Each of the 4096 grid colors is a filled dot, packed densely enough to read as a smooth solid.",
    caption="The 16³ hex grid",
)
def color_space_cube() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    # The grid fills its own silhouette, so the hexagon bound would only trace what the
    # data already draws; `grid_diameter` sizes the dots to tile it exactly.
    plot_rgb_cube(ax, cube.grid(), diameter=grid_diameter(colors.N_LEVELS), bound=False)
    return fig


color_space_cube()

r"""
The 27 named colors sit only on the {0, 8, 15}³ sub-lattice, the corners and edge midpoints of the cube. Note that no *color* is held out: every point shows up in training, since hex operands are sampled over the whole grid and each name appears in an alias line. What we hold out is operand pairs, both named and hex.
"""

# Split the two-panel lattice into two independent figures so they reflow and shrink
# separately: on a narrow screen the pair stacks instead of shrinking as a block,
# keeping each panel legible. The panels used to share a y-axis; now that they are
# separate figures, an identical figsize and the fixed limits of the cube panel give them the
# same lattice scale — and, since the tight crop is dominated by that shared axes box,
# the same size — without a shared axis. The lettered tags sit just outside those limits
# (annotation_clip=False) and are picked up by the crop.
vals = list(colors.PALETTE.values())
idx = {c: i for i, c in enumerate(vals)}
train_edges = [p for p in train_pairs if p[0] != p[1]]  # self-pairs are just the vertices
named = cube.named()
# A single front view suffices: the lattice is mostly empty, so nothing hides behind it.
# Drawn by hand rather than through `plot_rgb_cube`, because the edges and lettered tags
# need the projected coordinates and the per-vertex depth to order themselves.
xy = project_cube(named)
x, y = xy[:, 0], xy[:, 1]
depth = named @ CUBE_VIEWS["solid"].toward
dmin, dspan = depth.min(), depth.max() - depth.min()
cx, cy = x.mean(), y.mean()


def lattice_panel(bold, other, examples) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 4.4))
    faint, vedge = light_dark("#0001", "#fff1"), light_dark("#0006", "#fff7")
    ink, halo = light_dark("#111", "#eee"), light_dark("#fff", "#111")

    def edge(pair, lw_lo, lw_hi, **kw):
        u, v = idx[pair[0]], idx[pair[1]]
        mid = (depth[u] + depth[v]) / 2  # orders each edge against the vertices for occlusion
        # Taper by depth: front-facing edges read heavy, back/interior ones recede.
        lw = lw_lo + (lw_hi - lw_lo) * (mid - dmin) / dspan
        ax.plot([x[u], x[v]], [y[u], y[v]], lw=lw, zorder=float(mid), solid_capstyle="round", **kw)

    def letter(name, ch):
        i = idx[colors.PALETTE[name]]
        ang = np.arctan2(y[i] - cy, x[i] - cx)  # nudge the tag radially outward, clear of the lattice
        if name == "white":
            ang = np.radians(150)  # apex points straight at the title; send it up-left instead
        ax.annotate(
            ch,
            (x[i], y[i]),
            (x[i] + np.cos(ang) * 0.14, y[i] + np.sin(ang) * 0.14),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            fontstyle="italic",
            color=ink,
            zorder=100,
            annotation_clip=False,
            path_effects=[pe.withStroke(linewidth=3, foreground=halo)],
        )

    for p in other:
        edge(p, 0.4, 2.0, color=faint)
    for p in bold:
        result = tuple(np.array(colors.mix(*p)) / (colors.N_LEVELS - 1))
        edge(p, 1.4, 3.0, color=result)
    # Vertices in true color; +ε on zorder so a vertex wins a depth tie with an edge.
    for i in range(len(vals)):
        ax.scatter(x[i], y[i], c=[named[i]], s=60, edgecolors=vedge, lw=0.6, zorder=float(depth[i]) + 1e-3)
    u, v, chs = examples
    letter(u, chs[0])
    letter(v, chs[1])
    draw_cube_bound(ax)
    return fig


train_alt = "An orthographic front view of the 27 named colors as a lattice in the rotated RGB cube, value vertical with black at the bottom and white at the top. Each named color is a small dot in its true color. The pairs used as named equations in training are bold and colored by the color their two operands mix to; the held-out pairs are drawn faint for context. One training edge is picked out on the cube silhouette with italic letters a and b at its endpoints (white and magenta), the worked example a + b = orchid. The panel has no background fill or axes; front-facing edges are heavier than back and interior ones, so the lattice reads three-dimensionally. Titled 'train'."
holdout_alt = "The same orthographic front view of the 27 named colors in the rotated RGB cube, same orientation and styling as the train panel. Here the pairs held out for the named-holdout evaluation are bold and colored by their mixed result, with the training pairs drawn faint for context. One held-out edge is picked out on the cube silhouette with italic letters c and d at its endpoints (magenta and blue), the worked example c + d = violet. Titled 'held out for eval'."
left = themed(
    lambda: lattice_panel(train_edges, holdout, ("white", "magenta", "ab")),
    name="named-pair-lattice-train",
    alt_text=train_alt,
    caption="Train",
)()
right = themed(
    lambda: lattice_panel(holdout, train_edges, ("magenta", "blue", "cd")),
    name="named-pair-lattice-holdout",
    alt_text=holdout_alt,
    caption="Held out for eval",
)()
# Two sub-figures under one caption: figure_html nests the themed panels in a <figure>
# that the `figure:has(> figure)` rule in report.css reflows to a stack on a narrow screen.
figure_html(
    f"{left}{right}",
    caption="""
Named pairs on the cube.
<b>a-b</b>: white + magenta = orchid.
<b>c-d</b>: magenta + blue = violet.
""",
)

rf"""
Both figures show the same lattice from the front. The vertices are the named colors, and each edge joins the two operands of a named pair. The midpoint of an edge is the answer that equation should produce.

Two edges are labeled as worked examples:

- $\overline{{ab}}$: {colors.swatch("white")} + {colors.swatch("magenta")} = {colors.swatch("orchid")} (train)
- $\overline{{cd}}$: {colors.swatch("magenta")} + {colors.swatch("blue")} = {colors.swatch("violet")} (held out)

Only the connected pairs ever appear as named equations *with a named answer*. Every other operand pair the model sees is written in hex or cross form, and those draw their operands from the full 16³ grid.

A held-out edge like `magenta + blue = violet` can be answered two ways. One is recall, which is ruled out here, since that named rendering never appears in training. The other is composition: recognize both names as colors, mix them as if they had been written in hex, and translate the result back into a name. Composition is what the `named_holdout` eval set measures.

The `hex_unseen` and `cross_unseen` sets are sampled at evaluation time from the full grid, steering clear of every operand pair the corpus used.
"""

loaded = load_results()
if loaded is None:
    stop(
        "No results yet — run the experiment (it publishes metrics and probe weights on completion):\n\n"
        "```bash\nbin/mini run docs/m2/ex-2.1.1/experiment.py --app modal --max-containers 9\n```"
    )
metrics, weights = loaded

hex_accs = [acc(metrics, w, d, s, "hex_unseen") for w in WIDTHS for d in DEPTHS for s in SEEDS]
hold_accs = [acc(metrics, w, d, s, "named_holdout") for w in WIDTHS for d in DEPTHS for s in SEEDS]

rf"""
Results: Accuracy on unseen hex pairs spans **{min(hex_accs):.2f}–{max(hex_accs):.2f}** across the sweep, while held-out named pairs, the compositional test, span **{min(hold_accs):.2f}–{max(hold_accs):.2f}**. The figures below break this down by condition and eval set.

## Completion accuracy across the sweep

The figure below shows accuracy vs. model width, with one panel per eval set. Each panel has one line per model depth (the mean over seeds). Individual seeds are shown as faint points.

The named-holdout panel is interesting. It can only be solved by combining the alias dictionary with the mixing arithmetic, and we find that the model never learns to do that.
"""


@themed(
    name="accuracy-sweep",
    alt_text="Four line charts of completion accuracy (0 to 1) against model width (16, 32, 64), one panel per eval set: named seen, named holdout, hex unseen, and cross unseen. Each panel has one line per depth (2 and 4 layers, darker is deeper), averaged over three seeds, with individual seeds as faint points.",
)
def accuracy_sweep() -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2), sharey=True)
    stops = light_dark([0.6, 0.2], [0.7, 0.4])
    shades = dict(zip(DEPTHS, plt.cm.viridis(stops), strict=True))
    for ax, es in zip(axes, EVAL_SETS, strict=True):
        for d in DEPTHS:
            per_seed = np.array([[acc(metrics, w, d, s, es) for s in SEEDS] for w in WIDTHS])
            for s in range(len(SEEDS)):
                ax.plot(WIDTHS, per_seed[:, s], "o", color=shades[d], alpha=0.3, ms=3)
            ax.plot(WIDTHS, per_seed.mean(axis=1), "o-", color=shades[d], label=f"{d} layers", lw=2)
        ax.set(title=es.replace("_", " "), xlabel="width", xscale="log", ylim=(-0.03, 1.03))
        ax.set_xticks(WIDTHS, labels=[str(w) for w in WIDTHS])
        ax.set_xticks([], minor=True)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("completion accuracy")
    axes[0].legend(fontsize=8)
    return fig


accuracy_sweep()

rf"""
## Watching it answer, character by character

Let's see where in each sequence the model was unsure.

For the d{pick_arch(metrics)[0]}-L{pick_arch(metrics)[1]} model (seed {SEEDS[0]}), we plot one example per eval set and draw two series beneath the text, both as fractions of $\log |V|$, the value a uniform guess over the vocabulary would give. The first is the surprisal of the model at each character: how "surprised" it is by the character that actually comes next. The second is the entropy of its predictive distribution, the surprisal it expected on average before seeing that character.

Operands are unpredictable, so both series should spike at the first characters of each operand and settle as the prefix pins down the rest. Everything after `=` can be computed from the operands, so a model that has learnt color mixing should coast through the answer at near-zero surprisal, even on operand pairs it has never seen. If instead it guesses the answer, surprisal should spike across the answer characters.
"""

arch = pick_arch(metrics)
(cell,) = [r for r in metrics if r["label"] == label(*arch, SEEDS[0])]
# Index 1 of named_holdout is `lime + black = green`, the example "Why the
# named answers fail" walks through below; index 0 is a same-set spare.
target_index = {"named_holdout": 1}
rows = [(es, cell["surprisal"][es][target_index.get(es, 0)]) for es in EVAL_SETS]
log_v = np.log(len(colors.alphabet()))
sub_width = min(max(len(r["text"]) for _, r in rows), 80)
# Match the dark background of the sublines to this notebook, rather than the
# neutral subline default; light mode already matches. `css` overrides the
# `--bg-color` of the library (later rule wins).
sub_css = "svg { --bg-color: light-dark(#fff, #181c1a); }"


def sublines(rows: list[tuple[str, dict]], series, aria_label: str, name: str) -> str:
    """Lay out one captioned subline per eval set; `series(row)` builds its series list.

    The block is inlined (so the SVGs share the page CSS) *and* externalized as
    `_assets/<name>.html` — a plain file for tooling that can't run the frontend.
    """

    def one(tag_name: str, row: dict) -> str:
        svg = Subline(chars_per_line=sub_width, css=sub_css).plot(row["text"], series(row))
        tag = f'<span style="font-size: 11px; font-family: monospace; opacity: 0.65">{tag_name}</span>'
        return figure_html(svg, caption=tag, style="display: inline-block; margin: 0 .5em")

    strip = "".join(one(tag_name, row) for tag_name, row in rows)
    return externalize_html(figure_html(strip, aria_label=aria_label), name=name)


def pad(row: dict, key: str) -> np.ndarray:
    """Scale to fractions of log |V| and align with the text: position 0 has no prediction."""
    return np.concatenate([[np.nan], np.asarray(row[key]) / log_v])


def surprisal_series(row: dict) -> list[Series]:
    return [
        Series(raw=np.clip(pad(row, "nll"), 0, 1), label="surprisal"),
        Series(raw=np.clip(pad(row, "entropy"), 0, 1), label="entropy", dasharray="3 2"),
    ]


sublines(
    rows,
    surprisal_series,
    "Four short mixing equations, one per eval set, each with a sparkline of per-character "
    "surprisal (solid) and predictive entropy (dashed) drawn under the text, on a shared "
    "0-to-log-V scale. The two series track each other, spiking at operand starts and "
    "staying near zero across the answers — except named holdout, where surprisal rises "
    "well above entropy on the answer characters.",
    name="sublines-surprisal",
)

r"""
The gap between those two series is [the surprisal beyond what the model expected](https://www.lesswrong.com/posts/Kjo64rSWkFfc3sre5/detecting-out-of-distribution-text-with-surprisal-and#5__Surprise_surprise__A_new_metric),

$$s_2 = \frac{i - h}{\log |V|}$$

where $i$ is the surprisal and $h$ the entropy. This sits near zero when the model confidence matched the outcome, whether it was confident and right or unsure and appropriately surprised. It goes positive when the model was confidently wrong, and negative when the character was more predictable than its distribution suggested. The sparkline clips at zero, so we draw the negative values as a second, flipped series, $-s_2$.
"""


def s2(row: dict) -> np.ndarray:
    return pad(row, "nll") - pad(row, "entropy")


def s2_series(row: dict) -> list[Series]:
    return [
        Series(raw=np.clip(s2(row), None, 1), label="s₂"),
        Series(raw=np.clip(-s2(row), None, 1), label="−s₂", dasharray="3 2"),
    ]


sublines(
    rows,
    s2_series,
    "The same four equations, now with sparklines of surprise-surprise: surprisal minus "
    "entropy as a fraction of log V. The solid series shows the positive part (more surprised "
    "than expected); the dashed series shows the negative part flipped above zero (less "
    "surprised than expected). Three sets stay close to the baseline; named holdout shows "
    "tall positive spikes across its answer characters.",
    name="sublines-surprise-surprise",
)

r"""
None spike except `named_holdout`, the one set this sweep never solves (accuracy 0 above). The model is confident even on those wrong answers: entropy stays low while the true characters arrive as a surprise. What is the model so sure of?
"""

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from sca.compute.evaluation import greedy_completions  # noqa: E402
from sca.compute.model import load_checkpoint  # noqa: E402
from sca.data.tokenizer import CharTokenizer  # noqa: E402

store = project_store()
refs = {s: f"{CKPT_REF}/{label(*arch, s)}" for s in SEEDS}
resolved = store.get_refs(refs.values())
arts = {s: a for s, r in refs.items() if (a := resolved[r]) is not None}
if len(arts) < len(SEEDS):
    stop("The checkpoints aren't in the store yet — re-run the experiment to publish them.")
models = {}
with tempfile.TemporaryDirectory() as tmp:
    store.get_many([(art, Path(tmp) / str(s) / "model") for s, art in arts.items()])
    for s in arts:
        model, config, _ = load_checkpoint(Path(tmp) / str(s))
        models[s] = (model, CharTokenizer(config.tokenizer))


def complete(seed: int, prompts: list[str]) -> list[str]:
    """Greedy completions from the chosen configuration trained with *seed*."""
    model, tok = models[seed]
    return greedy_completions(model, tok, prompts, 12)


def name_prob(seed: int, prompt: str, word: str) -> float:
    """P(word | prompt), the product of its per-character probabilities."""
    model, tok = models[seed]
    p = 1.0
    for i, ch in enumerate(word):
        logits = model(jnp.array(tok.encode([prompt + word[:i]])[0])[None])[0, -1]
        q = jax.nn.softmax(logits)
        p *= float(q[tok.stoi[ch]])
    return p


p_blue = name_prob(SEEDS[0], "lime + bl", "u")

rf"""
## Why the named answers fail

The sparklines above are teacher-forced: the model is shown the true answer from the validation set, and we watch how much each one surprises it. `lime + black = green` is an interesting case.

Left to choose, this seed opens `lime + black` with `t`, for *teal*, so the true `g` is mildly surprising. Once it is forced onto `g`, the model guesses `r` correctly, since *gray* and *green* share the prefix `gr`. Then the true `e` is very surprising: on the `gr…` branch the model is all but sure the word is *gray*, but `e` rules that out. The spike is the model fluently spelling a different palette name, then being surprised when the truth arrives.

*Teal* is a one-channel neighbor of the true mix, and the tall spike on the `a` of `black` hints at why. After `lime + bl` the model puts {p_blue:.1%} on `u`, so it is all but certain the second operand is *blue*, and `lime + blue = teal` is an equation it trained on.

When it sees that the operand is different, the correction barely reaches the answer. Going from the `lime + blue = ` prompt to `lime + black = ` lifts *gray* by a factor of {name_prob(SEEDS[0], "lime + black = ", "gray") / name_prob(SEEDS[0], "lime + blue = ", "gray"):.0f}, from {name_prob(SEEDS[0], "lime + blue = ", "gray"):.1%} to {name_prob(SEEDS[0], "lime + black = ", "gray"):.0%}. The true answer *green* moves by a similar factor and stays negligible, from {name_prob(SEEDS[0], "lime + blue = ", "green"):.0e} to {name_prob(SEEDS[0], "lime + black = ", "green"):.0e}, while the trained *teal* keeps the top spot at {name_prob(SEEDS[0], "lime + black = ", "teal"):.0%}. So the operand correction redistributes mass among wrong names rather than finding the arithmetic. The model does learn the result-form rule, which says a named answer appears iff both operands are named. The difficulty is choosing which name, and a trained neighbor seems to overrule.

Below is every held-out pair, prompted exactly as in the `named_holdout` eval set, with one column per seed of the chosen architecture.
"""

named_holdout_exs = colors.as_named(holdout, seed=2)  # the eval set, verbatim
by_seed = {s: complete(s, [ex.prompt for ex in named_holdout_exs]) for s in SEEDS}

head_row = (
    f"<tr><th>prompt</th><th>{colors.swatch(None)} expected</th>"
    + "".join(f"<th>{colors.swatch(None)} seed {s}</th>" for s in SEEDS)
    + "</tr>"
)
body_rows = "".join(
    f"<tr><td><code>{ex.prompt}</code></td><td>{colors.swatch(ex.answer)}</td>"
    + "".join(f"<td>{colors.swatch(by_seed[s][i])}</td>" for s in SEEDS)
    + "</tr>"
    for i, ex in enumerate(named_holdout_exs)
)
holdout_table = f'<table class="report-table" style="font-size: 0.9em">{head_row}{body_rows}</table>'

figure_html(
    holdout_table,
    caption=f"Greedy completions of the `named_holdout` prompts, d{arch[0]}-L{arch[1]}, all seeds.",
    class_="report-figure",
)

# %%

rng = np.random.default_rng(9)
form_sets = {
    "named": named_holdout_exs,
    "cross": [colors.make_example("cross", a, b, rng) for a, b in holdout],
    "hex": [colors.make_example("hex", a, b, rng) for a, b in holdout],
}
form_scores = {}
for form, exs in form_sets.items():
    got = complete(SEEDS[0], [ex.prompt for ex in exs])
    form_scores[form] = sum(g == ex.answer for g, ex in zip(got, exs, strict=True))
n_holdout = len(named_holdout_exs)

# "A neighbor of the mix" is only interesting against how many neighbors there
# are: the 27-color palette gives each mix four to six, out of 27 names.
pal = np.array(list(colors.PALETTE.values()))
shell = bl.shell_mask((0, 8, 15), pal, [ex.result for ex in named_holdout_exs])
null_rate = float(shell.mean())
hits = tot = agree = 0
for i, ex in enumerate(named_holdout_exs):
    guesses = [complete(s, [ex.prompt])[0] for s in SEEDS]
    agree += len(set(guesses)) == 1 and guesses[0] != ex.answer
    for g in guesses:
        tot += 1
        hits += g in colors.PALETTE and shell[i][list(colors.PALETTE).index(g)]

rf"""
The model never answers these in hex. It always reaches for a name, and usually one adjacent to the true mix: {hits} of {tot} guesses land in the one-step neighborhood of the mix, against {null_rate:.0%} for a name drawn uniformly from the palette. Sometimes the name is an echo of one operand (`olive + lavender = lavender`), though on this palette an operand is often a neighbor anyway, so that part is weaker evidence than it looks.

The seeds mostly agree: {agree} of the {n_holdout} pairs draw the same wrong answer from all three. Independent guessing inside the neighborhood would produce well under one such pair, so the bias is systematic; perhaps retrieval of the nearest memorized named equation.

The mixing arithmetic itself is fine. Prompted with the same held-out value pairs, seed {SEEDS[0]} solves **{form_scores["hex"]}/{n_holdout}** in hex form and **{form_scores["cross"]}/{n_holdout}** in cross form, against **{form_scores["named"]}/{n_holdout}** as named equations.
"""

reps = round(N_EXAMPLES * colors.FORM_WEIGHTS["named"] / len(train_pairs))

rf"""
The corpus may make this hard for the model. Possible causes:

- Named equations draw on only {len(train_pairs)} distinct pairs, so each one is seen about {reps} times in training. A lookup table is enough, and the model may build one (`named_seen` ≈ 1). Once the loss on that slice reaches zero, nothing nudges the model toward the compositional route.
- The alias dictionary runs one way. Alias lines are always `name = hex`. This may be an instance of the *reversal curse*, where training on `A = B` doesn't teach `B = A`.
- A hex answer can be computed channel-by-channel, whereas the first character of a name depends on all three channels and the inverted dictionary at the same time. The probe section below looks at this more.

A few corpus changes might help: reverse some alias lines (`#f00 = red`); add named operands whose off-palette mix forces a hex answer (`red + navy = #804`), so that `name + name` prompts have to engage the arithmetic instead of the lookup table; and use a denser named palette, so memorization is harder.

## Where color is represented

Here we fit the probes at each residual-stream depth (depth 0 is the embedding) and plot their R² against depth[^rsquare]. The figure has one panel per probe target and one line per width. We test only the deepest models (L{pick_arch(metrics)[1]}) and show the mean over seeds.

[^rsquare]: This R² is the fraction of target variance the probe
recovers, so 1 means the color is fully readable from the stream and 0 means it is not there linearly.
"""

PROBES = ["operand_rgb", "result_rgb", "result_redness"]


@themed(
    name="probe-r2",
    alt_text="Three line charts of probe R-squared against residual-stream depth for the four-layer models, one panel per probe target: operand RGB, result RGB, and result redness. One line per width (16, 32, 64; darker is wider), averaged over seeds. R-squared for the operand rises within the first layers; the result targets rise later in depth.",
)
def probe_r2() -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2), sharey=True)
    shades = width_shades()
    d = max(DEPTHS)
    for ax, probe in zip(axes, PROBES, strict=True):
        for w in WIDTHS:
            probe_rows = [r["probe_r2"][probe] for r in metrics if r["label"].startswith(f"d{w}-L{d}-")]
            ax.plot(np.mean(probe_rows, axis=0), "o-", color=shades[w], label=f"width {w}", lw=2)
        ax.set(title=probe.replace("_", " "), xlabel="residual depth", ylim=(-0.05, 1.05))
        ax.set_xticks(range(max(DEPTHS) + 1), ["emb", *map(str, range(1, max(DEPTHS) + 1))])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("probe R² (held-out half)")
    axes[0].legend(fontsize=8)
    return fig


probe_r2()

r"""
Rising R² for the *result* means the mix becomes partly readable before the answer starts. It plateaus well below the operand R², though, even in conditions whose hex accuracy is perfect. Probing every answer position, per channel, would map that spread-out schedule (to do).

The probes read the residual stream at two positions, marked below. The first is the last character of the first operand, where the whole operand has been read in, so its value can be represented. The second is the space after `=`, the last position before the answer begins, where the result has to be ready.
"""

mark_rng = np.random.default_rng(7)
mark_exs = [
    colors.make_example("named", colors.PALETTE["red"], colors.PALETTE["blue"], mark_rng),
    colors.make_example("cross", colors.PALETTE["orange"], (2, 12, 7), mark_rng),
]


def mark_positions(ex) -> str:
    p = ex.prompt
    i, j = len(p.split(" ")[0]) - 1, len(p) - 1

    def hl(ch, c, t):
        return f'<span style="background: {c}; border-radius: 2px" title="{t}">{ch}</span>'

    chars = [
        hl(c, "#e4572e66", "operand read-out") if k == i else hl("␣", "#4d9de066", "result read-out") if k == j else c
        for k, c in enumerate(p)
    ]
    return "".join(chars) + f'<span style="opacity: 0.55">{ex.answer}</span>'


figure_html(
    '<pre style="line-height: 2.2; font-size: 1.05em">' + "<br>".join(mark_positions(ex) for ex in mark_exs) + "</pre>",
    caption="""
        <span style="background: #e4572e66; border-radius: 2px">&nbsp;operand&nbsp;</span>
        probes read the color of the first operand at this character;
        <span style="background: #4d9de066; border-radius: 2px">&nbsp;result&nbsp;</span>
        probes read the result color and redness at the space just before the answer (shown as ␣).
        The dimmed answer is never probed.
        """,
    class_="report-figure",
)

r"""
## Do seeds agree on where *redness* points?

For each pair of seeds trained with the same architecture, we take their fitted redness-probe directions and measure the absolute cosine similarity between them, layer by layer. Two random directions in n dimensions should have |cos| ≈ 0.8/√n, drawn here as the dashed line.

If this geometry were stable across seeds, there would be little point in anchoring. The spread we see is part of why we want to pin the direction down at training time.
"""


def redness_cosines(w: int, d: int) -> np.ndarray:
    """Pairwise |cos| between redness probe directions across seeds: (n_pairs, depth+1)."""
    vecs = [weights[f"{label(w, d, s)}/result_redness"][:, :, 0] for s in SEEDS]  # (L+1, C) each
    unit = [v / np.linalg.norm(v, axis=1, keepdims=True) for v in vecs]
    return np.array([np.abs((unit[i] * unit[j]).sum(axis=1)) for i in range(3) for j in range(i + 1, 3)])


@themed(
    name="probe-direction-agreement",
    alt_text="Line chart of the absolute cosine similarity between redness probe directions fitted on different seeds, against residual-stream depth, one line per width for the four-layer models. A dashed horizontal line marks the expected similarity of random directions for each width.",
)
def probe_direction_agreement() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    shades = width_shades()
    d = max(DEPTHS)
    for w in WIDTHS:
        cos = redness_cosines(w, d)
        ax.plot(cos.mean(axis=0), "o-", color=shades[w], label=f"width {w}", lw=2)
        ax.axhline(0.8 / np.sqrt(w), color=shades[w], lw=1, ls="--", alpha=0.6)
    ax.set(xlabel="residual depth", ylabel="cross-seed |cos| of redness direction", ylim=(0, 1))
    ax.set_xticks(range(max(DEPTHS) + 1), ["emb", *map(str, range(1, max(DEPTHS) + 1))])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig


probe_direction_agreement()

# %%

best = pick_arch(metrics)

rf"""
## Findings

The smallest condition that saturates the unseen-pair eval sets is **width {best[0]}, {best[1]} layers**. For D2.1, we can take that architecture as the baseline and add the anchor, which pulls sequences labeled *red-ish* (supplied as sparse, noisy labels) toward a chosen direction at chosen layers. Then we can re-run these measurements and compare the anchored and baseline versions.

The held-out named pairs sit at zero validation accuracy, so that set gives the anchored runs no headroom and probably can't help us spot any unintended degradation.
"""
