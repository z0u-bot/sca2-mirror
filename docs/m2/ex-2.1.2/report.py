# title: Ex 2.1.2: making composition necessary

r"""
# Ex 2.1.2: making composition necessary

/// tip |
<!-- tl;dr -->
The previous model never answered a held-out *named* pair, so we changed the grammar to make composition the only route: reverse alias lines, and named equations whose mix falls off the palette. The model picks up both new skills and still won't chain them within one forward pass, leaving held-out named accuracy at zero.
///

[Ex-2.1.1](../ex-2.1.1/) gave us a decent baseline, but no model in the sweep ever solved `named_holdout`. That eval set contains pairs of named colors whose named-answer equation never appears in training, so the expected way to answer is to combine two skills: the alias dictionary, which says which name goes with which hex color, and the mixing arithmetic itself.

That report looked into why the models never learn to combine them. In this experiment, we keep everything from ex-2.1.1 fixed, including the d64-L4 architecture, the split, and the training recipe, and add two new kinds of sequence:

- Reverse alias lines (`#f00 = red`). The base grammar never trained the hex → name readout.
- Named equations that mix to a color that falls *off* the palette, answered in hex (`red + navy = #804`). Now the surface form of the answer depends on the mix rather than on the operand forms alone, so a name + name prompt can't be settled by lookup.

| Type | Example |
|------|---------|
| Named pairs | `red + blue = purple` |
| Hex pairs   | `#f00 + #00f = #808`  |
| Cross-form  | `red + #00f = #808`   |
| Alias       | `red = #f00`          |
| **Reverse** | `#f00 = red`          |
| **Open**    | `red + orange = #f40` |

We run the two grammar changes as a 2 × 2 factorial. Each one replaces the same number of tokens from the hex slice, so the four conditions differ only in what they add on top of the baseline.

This is still an **un-anchored** experiment.
"""

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# The document's directory is on sys.path while it runs, so the experiment
# definition is importable — refs and sweep constants can't drift.
from experiment import (
    CONDITIONS,
    CORPUS_SEED,
    HOLDOUT_FRAC,
    MARGINS_REF,
    METRICS_REF,
    N_EXAMPLES,
    OPEN_HOLDOUT_FRAC,
    SEEDS,
)
from experiment import _classify as classify_form
from experiment import _corpus as corpus_for
from mini.lit import memo, stop
from mini.store import project_store
from mini.vis import figure_html, light_dark, svg_figure, themed
from sca.data import colors
from subline.series import Series
from subline.subline import Subline

EVAL_SETS = [
    "named_seen",
    "named_holdout",
    "hex_unseen",
    "cross_unseen",
    "open_seen",
    "open_holdout",
    "alias_rev",
]
CONDS = list(CONDITIONS)


def load_results() -> tuple[dict, dict[str, np.ndarray]] | None:
    """Resolve the metrics and margin arrays from the store, or None if unpublished."""
    store = project_store()
    arts = store.get_refs([METRICS_REF, MARGINS_REF])
    m_art, g_art = arts[METRICS_REF], arts[MARGINS_REF]
    if m_art is None or g_art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        m_path, g_path = store.get_many([(m_art, Path(d) / "metrics.json"), (g_art, Path(d) / "margins.npz")])
        metrics = json.loads(m_path.read_text())
        with np.load(g_path) as z:
            margins = {k: z[k] for k in z.files}
    return metrics, margins


def label(cond: str, s: int) -> str:
    return f"{cond}-s{s}"


def cell(metrics: dict, cond: str, s: int) -> dict:
    (r,) = [r for r in metrics["cells"] if r["label"] == label(cond, s)]
    return r


def acc(metrics: dict, cond: str, s: int, eval_set: str) -> float:
    return cell(metrics, cond, s)["accuracy"][eval_set]["accuracy"]


def cond_shades() -> dict[str, tuple]:
    stops = light_dark([0.82, 0.55, 0.32, 0.08], [0.88, 0.62, 0.42, 0.2])
    return dict(zip(CONDS, plt.cm.viridis(stops), strict=True))


r"""
## Training data

We regenerate the `both` corpus here. It shows the two new forms in their natural context: reverse aliases and off-palette named equations, sitting among the lines we already had.
"""

corpora = {cond: corpus_for(cond) for cond in CONDS}
corpus_head = "".join(ex.text for ex in corpora["both"][:12])
form_counts = {
    cond: {f: sum(classify_form(ex) == f for ex in corpus) for f in CONDITIONS[cond]}
    for cond, corpus in corpora.items()
}
FORMS = ["hex", "named", "cross", "alias", "alias_rev", "open"]
data_rows = "".join(
    f"<tr><td><code>{cond}</code></td>"
    + "".join(f'<td class="num">{form_counts[cond].get(f, 0) or "—"}</td>' for f in FORMS)
    + "</tr>"
    for cond in CONDS
)
open_train, open_holdout = colors.split_open_pairs(CORPUS_SEED, OPEN_HOLDOUT_FRAC)
named_train, named_holdout = colors.split_named_pairs(CORPUS_SEED, HOLDOUT_FRAC)
data_table = (
    '<table class="report-table"><tr><th>condition</th>'
    + "".join(f'<th class="num"><code>{f}</code></th>' for f in FORMS)
    + f"</tr>{data_rows}</table>"
)
data_caption = f"Lines per form in the {N_EXAMPLES:,}-line corpus of each condition. The named equations draw from the same {len(named_train)} training pairs as ex-2.1.1, with the same {len(named_holdout)} pairs held out. The open equations draw from {len(open_train)} of the 302 off-palette named pairs, and {len(open_holdout)} are held out entirely."

f"""
```
{corpus_head}```
"""

# %%

figure_html(data_table, caption=data_caption, class_="report-figure")

r"""
## Hypotheses

Written before looking at any results.

**H1.** Accuracy: `named_holdout` stays at zero in `control` and is lifted well off zero by `both`. `rev` alone lifts `named_holdout` part of the way: the garden path showed the mix is already half-computed on named prompts, and `rev` supplies the missing readout for whatever computation is there. `open` alone forces the computation, but leaves name emission supervised only through the memorizable named slice.

**H2.** Margins: scoring all 27 names as complete answers gives each held-out pair a *margin*, the log-probability of the true name minus that of the best competitor. In `control` the margins are negative (accuracy is zero) but spread out well above the random floor. The pairs with the least-negative control margins should be the first to flip positive under intervention. The interventions shift the whole margin distribution upward, while `named_seen` margins stay large and positive.

**H3.** The answer schedule: probing each answer position for each RGB channel (on hex prompts, strictly before each digit lands in the context) shows a stair-step: channel k stays low and becomes strongly decodable at the position that emits digit k.

**H4.** Computed but not emitted: a result-color probe fit on open-pair prompts (name + name surface form) at the pre-answer position transfers to the held-out named prompts in proportion to how much the mix is actually computed there: middling R² in `control` (partial computation), rising toward the fit ceiling in the `open` conditions, and tracking `named_holdout` accuracy across conditions.

**H5.** No side effects: `named_seen`, `hex_unseen`, and `cross_unseen` stay saturated in every condition, even though we are displacing up to 15% of the hex data. The `alias_rev` eval set reads ≈ 1 wherever reverse aliases are trained and stays random elsewhere, as in ex-2.1.1.

If the H1 interaction shows up but the single interventions do nothing, then composition needed both ingredients at once. If `rev` alone saturates the set, then the readout was the only missing piece and the pull toward lookup never mattered. If nothing moves, then the diagnosis was wrong somewhere.
"""

loaded = load_results()
if loaded is None:
    stop(
        "No results yet — run the experiment (it publishes metrics, margins, and checkpoints "
        "on completion):\n\n"
        "```bash\nbin/mini run docs/m2/ex-2.1.2/experiment.py --app modal --max-containers 12\n```"
    )
metrics, margins = loaded

headline_hold = {cond: np.mean([acc(metrics, cond, s, "named_holdout") for s in SEEDS]) for cond in CONDS}
headline_sat = {
    cond: np.mean([acc(metrics, cond, s, es) for s in SEEDS for es in ("named_seen", "hex_unseen", "cross_unseen")])
    for cond in CONDS
}

f"""
**Headline numbers.** Mean `named_holdout` accuracy by condition: {", ".join(f"`{cond}` **{v:.2f}**" for cond, v in headline_hold.items())}. The saturated sets (named seen, hex and cross unseen) average {", ".join(f"{v:.3f}" for v in headline_sat.values())} in the same order.

## Completion accuracy across the factorial

Did the changes move `named_holdout` off zero, and did the two new forms train the way we meant them to? The figure below shows exact-match accuracy for all seven eval sets, in each of the four conditions.
"""


@memo
@themed(
    name="accuracy-factorial",
    alt_text="""
        Seven bar-and-dot panels of completion accuracy (0 to 1) against condition (control, rev, open, both), one panel per eval set: named seen, named holdout, hex unseen, cross unseen, open seen, open holdout, and reverse alias. Bars show the mean over three seeds, dots the individual seeds.
    """,
    caption="""
        Each panel is one eval set: the bar is the mean over three seeds and the dots are the individual seeds. `named_holdout` is the set H1 is about; `open_holdout` asks whether the forced computation carries over to off-palette pairs never seen in training; `alias_rev` checks the reverse-alias supervision. In `control` and `rev`, the open-form sets ask for a surface form those corpora never train on (a name + name prompt), so a low score there means the grammar is simply missing from that corpus, rather than a genuine attempt that fell short.
    """,
)
def accuracy_factorial_plot(accs: dict[str, np.ndarray]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(EVAL_SETS), figsize=(12.5, 3.0), sharey=True)
    shades = cond_shades()
    xs = np.arange(len(CONDS))
    for ax, es in zip(axes, EVAL_SETS, strict=True):
        per_seed = accs[es]  # (conds, seeds)
        ax.bar(xs, per_seed.mean(axis=1), color=[shades[c] for c in CONDS], width=0.62)
        for i in range(len(CONDS)):
            ax.plot([xs[i]] * len(SEEDS), per_seed[i], "o", color="#0008", ms=2.5, zorder=3)
        ax.set(title=es.replace("_", " "), ylim=(-0.03, 1.03))
        ax.set_xticks(xs, CONDS, rotation=90, fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("completion accuracy")
    return fig


accuracy_factorial_plot(
    {es: np.array([[acc(metrics, cond, s, es) for s in SEEDS] for cond in CONDS]) for es in EVAL_SETS}
)

# %%

hold_acc = {cond: float(np.mean([acc(metrics, cond, s, "named_holdout") for s in SEEDS])) for cond in CONDS}
h1_interaction = hold_acc["both"] - hold_acc["rev"] - hold_acc["open"] + hold_acc["control"]

rf"""
H1 is refuted: `named_holdout` stays at zero in every condition (its interaction term is a degenerate {h1_interaction:+.2f}). The other panels show that both changes did train, though. The `open` conditions answer held-out off-palette pairs at ≈ 0.9, so the mixing arithmetic runs on name + name prompts and carries over to operand pairs never seen in that surface form. And `alias_rev` reads 1.0 right where reverse aliases were trained, so the hex → name readout exists and works in the frame it was taught in. We supplied both of the missing ingredients, and still the composition does not surface as a named answer.

## Right value, wrong spelling

Let's look at what the model actually writes. Below are the greedy completions of the held-out named prompts, meaning we take the single most likely character at each step and read off the answer. In `control` and `rev` the answers are the familiar retrieval-like wrong names. But in the `open` conditions, some of the pairs the model writes a *hex* answer, and when it does, the value is the correct mix.
"""

# Greedy completions of the named-holdout prompts are precomputed in `eval_one`
# (where the model is already loaded) and ride the metrics JSON — so this reads
# them from the store rather than pulling 12 checkpoints and re-decoding per edit.
if any("holdout_completions" not in cell(metrics, cond, s) for cond in CONDS for s in SEEDS):
    stop("These metrics predate precomputed completions — re-run the experiment to republish.")
completions = {(cond, s): cell(metrics, cond, s)["holdout_completions"] for cond in CONDS for s in SEEDS}


def answer_value(text: str) -> tuple[int, int, int] | None:
    """The color a completion denotes, in either surface form (None if malformed)."""
    if text in colors.PALETTE:
        return colors.PALETTE[text]
    if len(text) == 4 and text.startswith("#") and all(c in "0123456789abcdef" for c in text[1:]):
        r, g, b = (int(c, 16) for c in text[1:])
        return (r, g, b)
    return None


_train_pairs, _holdout_pairs = colors.split_named_pairs(CORPUS_SEED, HOLDOUT_FRAC)
holdout_exs = colors.as_named(_holdout_pairs, seed=2)


def completion_swatch(text: str, want: tuple | None = None) -> str:
    rgb = answer_value(text)
    if rgb is None:
        return f"<code>{text}</code>"
    mark = "" if want is None else (" ✓" if rgb == want else "")
    return (
        f'<span aria-hidden="true" style="background: {colors.to_hex(rgb)}; border: 1px solid #8886; '
        f'border-radius: 2px; display: inline-block; width: 0.8em; height: 0.8em"></span> {text}{mark}'
    )


completions_head = (
    f"<tr><th>prompt</th><th>{colors.swatch(None)} expected</th>"
    + "".join(f"<th>{colors.swatch(None)} {cond}</th>" for cond in CONDS)
    + "</tr>"
)
completions_rows = "".join(
    f"<tr><td><code>{ex.prompt}</code></td><td>{completion_swatch(ex.answer)}</td>"
    + "".join(f"<td>{completion_swatch(completions[cond, SEEDS[0]][i], ex.result)}</td>" for cond in CONDS)
    + "</tr>"
    for i, ex in enumerate(holdout_exs)
)
completions_table = f'<table class="report-table">{completions_head}{completions_rows}</table>'

figure_html(
    completions_table,
    caption=f"Greedy completions of the `named_holdout` prompts, seed {SEEDS[0]}, one column per condition. A ✓ marks an answer whose value equals the true mix, but is given in the wrong form.",
    class_="report-figure",
)

# %%

hexed = {cond: [sum(g.startswith("#") for g in completions[cond, s]) for s in SEEDS] for cond in CONDS}
value_ok = {
    cond: [
        sum(answer_value(g) == ex.result for g, ex in zip(completions[cond, s], holdout_exs, strict=True))
        for s in SEEDS
    ]
    for cond in CONDS
}


def form_margin(cond: str) -> float:
    """Mean log P(correct hex) − log P(true name) on the holdout prompts."""
    vals = []
    for s in SEEDS:
        cands = cell(metrics, cond, s)["margin_candidates"]["named_holdout"]
        lp = margins[f"{label(cond, s)}/margins/named_holdout"]
        for i, ex in enumerate(holdout_exs):
            vals.append(lp[i, cands.index(colors.to_hex(ex.result))] - lp[i, cands.index(ex.answer)])
    return float(np.mean(vals))


n_holdout = len(holdout_exs)

f"""
Across all seeds, the hex-form answers on the {n_holdout} held-out prompts number {", ".join(f"`{c}` **{sum(hexed[c])}/{3 * n_holdout}**" for c in CONDS)}, and the value-correct answers {", ".join(f"`{c}` **{sum(value_ok[c])}/{3 * n_holdout}**" for c in CONDS)}. Every value-correct answer is hex, and every name answer has the wrong value. The form-choice margin agrees at {", ".join(f"`{c}` {form_margin(c):+.1f}" for c in CONDS)} nats[^form].

So the `open` intervention worked, in that the mix is computed on name + name prompts. But the *form rule* ("answer with a name exactly when the mix lands on the palette") did not carry over: the model treats a held-out closed pair like an open one and answers in hex, or else falls back on a name near the mix. Whether that name is the answer of the closest memorized equation is a further claim we haven't tested here.

So the reverse mapping `rev` didn't help.

[^form]: "Form choice margin": the log-probability of the correct hex minus
that of the true name under teacher forcing.

## Error margins

Exact-match accuracy tells us whether the true name came out on top, but how close it was. So let's grade the name-identity question on a continuous scale.
"""


def margin_rows(eval_set: str, exs: list) -> np.ndarray:
    """(len(CONDS), n_seeds, n_examples) margin of the true name over the best *other name*."""
    out = np.empty((len(CONDS), len(SEEDS), len(exs)))
    for ci, cond in enumerate(CONDS):
        for si, s in enumerate(SEEDS):
            cands = cell(metrics, cond, s)["margin_candidates"][eval_set]
            names = [i for i, c in enumerate(cands) if not c.startswith("#")]
            lp = margins[f"{label(cond, s)}/margins/{eval_set}"]
            true_idx = np.array([cands.index(ex.answer) for ex in exs])
            truth = lp[np.arange(len(exs)), true_idx]
            rival = lp[:, names].copy()
            rival[np.arange(len(exs)), [names.index(t) for t in true_idx]] = -np.inf
            out[ci, si] = truth - rival.max(axis=1)
    return out


m_hold = margin_rows("named_holdout", holdout_exs)
m_seen = margin_rows("named_seen", colors.as_named(_train_pairs, seed=1))


@memo
@themed(
    name="margin-trajectories",
    alt_text="""
        Line chart of the answer margin (log-probability of the true name minus the best competitor) against condition (control, rev, open, both), one line per held-out named pair, averaged over seeds. A horizontal line marks zero, where the true answer starts to come out ahead; a shaded band shows the range of margins on the seen named pairs.
    """,
    caption="""
        Margins between predicted and true named colors. Positive means the true name comes out ahead of the other names, and the magnitude says by how much. One line per held-out pair, averaged over seeds and traced across the four conditions; the shaded band is the range of `named_seen` margins, for reference.
    """,
)
def margin_trajectories_plot(m_seen: np.ndarray, m_hold: np.ndarray, names: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    xs = np.arange(len(CONDS))
    seen_mean = m_seen.mean(axis=1)  # (conds, n_seen)
    ax.fill_between(
        xs,
        np.percentile(seen_mean, 5, axis=1),
        np.percentile(seen_mean, 95, axis=1),
        color="#4d9de0",
        alpha=0.3,
        label="named seen (5–95%)",
    )
    mean = m_hold.mean(axis=1)  # (conds, n_pairs)
    order = np.argsort(mean[0])
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, mean.shape[1]))
    # Spread the end-of-line labels vertically so neighbors don't collide.
    label_y = mean[-1].astype(float).copy()
    for prev, nxt in zip(np.argsort(label_y)[:-1], np.argsort(label_y)[1:], strict=True):
        label_y[nxt] = max(label_y[nxt], label_y[prev] + 1.1)
    for rank, p in enumerate(order):
        ax.plot(xs, mean[:, p], "o-", color=cmap[rank], lw=1.4, ms=3.5)
        ax.annotate(
            names[p],
            (xs[-1], mean[-1, p]),
            xytext=(xs[-1] + 0.12, label_y[p]),
            textcoords="data",
            fontsize=6.5,
            va="center",
            color=cmap[rank],
        )
    ax.axhline(0, color="#888", lw=1, ls="--")
    ax.set_xticks(xs, CONDS)
    ax.set(xlabel="condition", ylabel="margin: log P(true) − log P(best competitor)")
    ax.set_xlim(-0.3, len(CONDS) - 0.25 + 1.6)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper left")
    return fig


margin_trajectories_plot(m_seen, m_hold, [f"{ex.prompt}{ex.answer}" for ex in holdout_exs])

# %%

m_hold_mean = m_hold.mean(axis=1)  # (conds, pairs)
m_ctrl, m_both = m_hold_mean[0], m_hold_mean[CONDS.index("both")]
h2_rho = float(np.corrcoef(np.argsort(np.argsort(m_ctrl)), np.argsort(np.argsort(m_both)))[0, 1])
h2_flipped = int((m_both > 0).sum())

f"""
**H2 mostly refuted.** The distribution barely moves: the mean margin is {m_ctrl.mean():+.1f} nats in `control` and {m_both.mean():+.1f} in `both`, with {h2_flipped}/10 pairs ending positive. The true name never gets close among the names; it stays about ten nats off while `named_seen` is far above zero. So the value → name translation is not *almost there* and narrowly behind. It simply never engages, even in the condition whose greedy answers prove the value has been computed. There is some rank structure, but it is weak: the rank correlation between `control` and `both` is {h2_rho:.2f}.

## When each channel becomes readable in-sequence

The ex-2.1.1 probes read one pre-answer position and averaged over R, G, and B, which hides at which token the mix gets computed. So here we fit a separate ridge probe for every combination of position, channel, and layer on the hex prompts. We then watch each channel become readable as the answer is spelled out.

H3 predicts a stair-step: each channel stays low until the position that emits its own digit, then climbs sharply.
"""

sched_offsets = cell(metrics, "control", SEEDS[0])["schedule_offsets"]
# Schedule R² arrays travel in the same npz as the margins (see eval_one).
sched_r2 = np.mean([margins[f"{label('control', s)}/schedule/r2"] for s in SEEDS], axis=0)


@memo
@themed(
    name="answer-schedule",
    alt_text="""
        Line charts of probe R-squared against position offset around the answer, one panel per residual-stream depth, three lines per panel for the R, G, and B channels of the result. Lines are solid before the digit of each channel enters the context and dotted after. In the deeper layers each channel peaks near 1 at its own emission position and falls away on either side, so the three channels form a sequence of staggered peaks rather than a cumulative plateau; at depth 0 the dotted segments jump to 1 as each digit becomes readable from the context.
    """,
    caption="""
        Probe alignment per color channel of the answer. One panel per transformer layer. Offset 0 is the `#`, digit k sits at offset k + 1 and is emitted from offset k. A line is solid where its digit is not yet in the context (so decoding it is computation) and dotted once it has landed (decoding is copying).
    """,
)
def answer_schedule_plot(sched_offsets: list[int], sched_r2: np.ndarray) -> plt.Figure:
    depths = sched_r2.shape[1]
    fig, axes = plt.subplots(1, depths, figsize=(12.5, 2.9), sharey=True)
    chan_colors = ["#e4572e", "#3aa76d", "#4d9de0"]
    offs = np.array(sched_offsets)
    for d, ax in enumerate(axes):
        for k, c in enumerate(chan_colors):
            emit = k + 1  # digit k enters the context at offset k + 1
            pre = offs <= emit
            ax.plot(offs[pre], sched_r2[pre, d, k], "o-", color=c, lw=1.8, ms=3, label=f"channel {'RGB'[k]}")
            ax.plot(offs[emit <= offs], sched_r2[emit <= offs, d, k], "o:", color=c, lw=1, ms=2, alpha=0.6)
        ax.axvline(-0.5, color="#888", lw=0.8, ls="--", alpha=0.6)
        ax.set(title="emb" if d == 0 else f"depth {d}", xlabel="offset from '#'", ylim=(-0.05, 1.05))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("probe R² (held-out half)")
    axes[0].legend(fontsize=7)
    return fig


answer_schedule_plot(sched_offsets, sched_r2)

r"""
At the final layer, channel k is almost perfectly decodable right at its emission offset (R² ≈ 0.97), and *only* there. One position earlier it is much weaker, and once emission moves on to the next digit, the earlier channels mostly fade from the deep residual stream.

So each answer position holds just the one channel it is about to emit, riding on a diffuse trace of the whole mix (around 0.5 R²) that lingers from the pre-answer position. The mix is never fully represented at any single position; the "result" that the pre-answer probe reads is more of a head start than a finished value. This looks like just-in-time computation.

## Is the mix computed on the named prompts?

If the mix really is computed on the named prompts, then a probe trained to read the result somewhere else should carry over to them. So we fit the probe on the open-pair prompts (the name + name surface form) at the pre-answer position, then score it on the named eval sets. Where the mix is never represented, no probe fit elsewhere can recover it.
"""

TRANSFER_SETS = ["fit", "open_holdout", "named_seen", "named_holdout"]


@memo
@themed(
    name="transfer-probe",
    alt_text="""
        Four line charts of probe R-squared against residual-stream depth, one panel per prompt set: the held-back half of the fit set, open holdout, named seen, and named holdout. One line per condition (control, rev, open, both; darker means richer corpus). The probes were fit on open-pair prompts at the pre-answer position.
    """,
    caption="""
        One panel per scored set: the held-back half of the fit set, open holdout, named seen, and named holdout. R² against residual depth, one line per condition (darker means a richer corpus). Depth 0 is left out, since the pre-answer embedding is constant across prompts until attention runs.
    """,
)
def transfer_probe_plot(r2: dict[tuple[str, str], np.ndarray]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(TRANSFER_SETS), figsize=(11.5, 3.0), sharey=True)
    shades = cond_shades()
    for ax, name in zip(axes, TRANSFER_SETS, strict=True):
        for cond in CONDS:
            rows = r2[name, cond]  # (seeds, depths)
            # Skip depth 0: the raw embedding at the pre-answer position is a constant
            # across prompts (attention hasn't run), so its transfer R² only reflects
            # the fit-vs-eval mean gap — no result is decodable there in any set.
            depths = np.arange(rows.shape[1])
            ax.plot(depths[1:], rows.mean(axis=0)[1:], "o-", color=shades[cond], label=cond, lw=2)
        ax.set(title=name.replace("_", " "), xlabel="residual depth", ylim=(-0.05, 1.05))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("probe R² (result RGB)")
    axes[0].legend(fontsize=8)
    return fig


transfer_probe_plot(
    {
        (name, cond): np.array([cell(metrics, cond, s)["transfer_r2"][name] for s in SEEDS])
        for name in TRANSFER_SETS
        for cond in CONDS
    }
)

r"""
Partial computation shows up everywhere, `control` included. The held-out named prompts carry about as much linearly-decodable result as the ceiling of the fit set (≈ 0.6 at the deep layers). The `open` conditions lift the mid-depth transfer a little, which fits the idea that the arithmetic is doing real work on these prompts, and their *final*-layer R² drops on the named sets, where the job of the last layer has become committing to a surface form. The main point, though, is that "computed but not emitted" was already true in `control`: making the computation stronger (`open`) or the readout available (`rev`) still does not join the two up.
"""


def garden_path_guess(cond: str, s: int) -> str:
    """What the model actually completes `lime + black = ` with."""
    fails = cell(metrics, cond, s)["accuracy"]["named_holdout"]["failures"]
    return next((g for p, _, g in fails if p.startswith("lime + black")), "green")


garden_path_seed0 = ", ".join(f"`{c}` {garden_path_guess(c, SEEDS[0])}" for c in CONDS)

rf"""
## The garden path, revisited

Let's return to the walkthrough example from ex-2.1.1. A garden-path sentence is one that leads you into a wrong reading until a later word forces a correction, and that's what happens here. On `lime + black = green`, the model guesses the second operand as *blue*, revises to *black* when it sees the `a`, and then only half-corrects the answer, so a memorized neighbor of the mix comes out ahead.

Which neighbor varies. At seed {SEEDS[0]} the completions are {garden_path_seed0}, and across the other seeds `control` also gives olive and teal. The competitor is not one fixed wrong answer.

With the new corpora, does the correction finally overtake the lookup? It does not: the margin stays several nats negative. The sublines below trace per-character surprisal and entropy for seed {SEEDS[0]}.
"""

gp_idx = next(i for i, ex in enumerate(holdout_exs) if {"lime", "black"} == set(ex.prompt.split(" = ")[0].split(" + ")))
gp_rows = "".join(
    f"<tr><td><code>{cond}</code></td>"
    + "".join(f'<td class="num">{m_hold[ci, si, gp_idx]:+.1f}</td>' for si in range(len(SEEDS)))
    + f'<td class="num"><b>{m_hold[ci, :, gp_idx].mean():+.1f}</b></td></tr>'
    for ci, cond in enumerate(CONDS)
)
gp_table = (
    '<table class="report-table"><tr><th>condition</th>'
    + "".join(f'<th class="num">seed {s}</th>' for s in SEEDS)
    + f'<th class="num">mean</th></tr>{gp_rows}</table>'
)

figure_html(
    gp_table,
    caption="Margin of `green` over its best competitor on the `lime + black` prompt, per seed and condition. A positive value means the arithmetic comes out ahead.",
    class_="report-figure",
)

# %%

log_v = np.log(len(colors.alphabet()))
gp_text = holdout_exs[gp_idx].prompt + holdout_exs[gp_idx].answer
gp_rows_by_cond = [
    (cond, row)
    for cond in CONDS
    for row in cell(metrics, cond, SEEDS[0])["surprisal"]["named_holdout"]
    if row["text"] == gp_text
]
if not gp_rows_by_cond:
    stop("*(The garden-path example fell outside the captured surprisal rows — skipping the sublines.)*")


def gp_pad(row: dict, key: str) -> np.ndarray:
    return np.concatenate([[np.nan], np.asarray(row[key]) / log_v])


def gp_subline(cond: str, row: dict) -> str:
    series = [
        Series(raw=np.clip(gp_pad(row, "nll"), 0, 1), label="surprisal"),
        Series(raw=np.clip(gp_pad(row, "entropy"), 0, 1), label="entropy", dasharray="3 2"),
    ]
    svg = Subline(chars_per_line=len(row["text"])).plot(row["text"], series)
    label = f'<span style="font-size: 11px; font-family: monospace; opacity: 0.65">{cond}</span>'
    return figure_html(svg, caption=label, style="display: inline-block; margin: 0 1em 0 0")


svg_figure(
    "".join(gp_subline(cond, row) for cond, row in gp_rows_by_cond),
    alt_text="""
        The equation lime plus black equals green, repeated once per condition, each with a sparkline of per-character surprisal (solid) and predictive entropy (dashed) under the text on a shared 0-to-log-V scale.
    """,
    name="sublines-garden-path",
)

# %%

s2_values = sorted(
    c["calibration"][es]["s2"]
    for cond in CONDS
    for s in SEEDS
    for c in [cell(metrics, cond, s)]
    for es, a in c["accuracy"].items()
    if es in c["calibration"] and a["accuracy"] == 0.0
)

rf"""
Every set the model fails, it fails confidently rather than hedging, which is what the garden-path sublines illustrate. Across the {len(s2_values)} condition and eval-set pairs at zero accuracy, s₂ averages {np.mean(s2_values):.2f} and spans {s2_values[0]:.2f} to {s2_values[-1]:.2f}.
"""

open_holdout_acc = float(np.mean([acc(metrics, "both", s, "open_holdout") for s in SEEDS]))

f"""
## Findings

Adding the inverse dictionary and the pressure to compute on named prompts did teach the model those two skills: reverse aliases reach 1.0, and off-palette generalization reaches ≈ {open_holdout_acc:.2f}. But the model didn't learn to combine them: `named_holdout` stays at zero everywhere.

The model often gets the value right but writes it as hex instead of as a name, and the value → name translation never runs partway through an equation, even though the model has learned that mapping perfectly well on its own. Nothing in training ever asks the model to chain the two skills inside a single forward pass, and the just-in-time answer schedule suggests it never holds a full intermediate result anywhere as a latent embedding.

So for D2.1, `named_holdout` is not able to identify degradation: it sits at zero, so it has no headroom to lose. `open_holdout` from the new corpus may be a better indicator: it is compositional (unseen pairs, with the form of the answer decided by a computed value), and it sits near the ceiling without quite reaching it.
"""
