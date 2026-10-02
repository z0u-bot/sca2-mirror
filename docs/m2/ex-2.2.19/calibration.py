# title: Ex 2.2.19, follow-up: where the calibration KL sits
# mini:manual-publish

# A post hoc split of the calibration KL of ex-2.2.19 over every held-out context, by true op, by how sure the Bayes
# posterior is of that op, and by the op whose answers hold the mass the model has in excess of the Bayes answer
# distribution. It reads the per-context arrays ex-2.2.19 (and ex-2.2.18, for the 400-epoch run at the scout seed)
# published, and trains nothing.
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import light_dark, themed

X = ex.ex2216
ALL_OPS: tuple[str, ...] = tuple(X.OP_NAMES)
OPS = ex.OP_SET.ops
N_OPS = len(OPS)
SET_IDS = [ALL_OPS.index(o) for o in OPS]
DROPPED_IDS = [i for i in range(len(ALL_OPS)) if i not in SET_IDS]
HSV_CHANNEL = ("hue-hsv", "sat-hsv", "value-hsv")
SEEDS = (ex.SCOUT_SEED, *ex.CONFIRM_SEEDS)
REF_E = ex.REFERENCE_EPOCHS
# The confident threshold of the ex-2.2.17 to ex-2.2.19 op confusion matrices, and the bins of posterior on the true
# op that this note splits the held-out set into; the last bin is the confident set.
CONFIDENT = 0.99
BINS = (0.0, 0.5, 0.9, CONFIDENT, 1.0 + 1e-9)
BIN_NAMES = ("below 0.5", "0.5 to 0.9", "0.9 to 0.99", "above 0.99")
# Columns of the excess split: the seven ops of the set, then colors only a dropped op gives, colors no op gives, and
# mass on tokens that are not colors.
COLUMNS = (*OPS, "dropped", "none", "non-color")


# --- Loading ------------------------------------------------------------------------------------------------


def read(ref: str, into: Path) -> Path:
    store = project_store()
    art = store.get_ref(ref)
    assert art is not None, f"{ref} is not published"
    return store.get(art, into / ref.replace("/", "-"))


with tempfile.TemporaryDirectory() as _tmp:
    EVAL = json.loads(read(ex.EVAL_REF, Path(_tmp)).read_text())
PICK: int = EVAL["selection"]["pick"]
LENGTHS = (PICK, REF_E)


def array_ref(epochs: int, seed: int) -> str:
    """The per-context arrays of the run at *epochs* and *seed*. The 400-epoch run at the scout seed is the ex-2.2.18
    `no-four` run, as in the ex-2.2.19 report.
    """
    if epochs == REF_E and seed == ex.SCOUT_SEED:
        return ex.ex2218.EVAL_ARRAYS_REF.format(label=ex.OP_SET.name)
    rate = ex.PEAK_LR if epochs == REF_E else EVAL["selection"]["rate"][str(epochs)]
    return ex.EVAL_ARRAYS_REF.format(label=ex.label_of(epochs, rate, seed))


@memo
def answer_table():
    """The answers of all eleven ops on every pair, so that the answers of a dropped op stay defined."""
    return X._get_posterior().build_table(X.TABLE)


@memo
def split_run(ref: str) -> dict[str, np.ndarray]:
    """Per held-out context of one run: the true op, the posterior on it, the calibration KL, and that KL split over
    `COLUMNS` in proportion to where the model has mass in excess of the Bayes answer distribution.

    The excess on a color is the model mass beyond the Bayes mass, where it is positive. A color that ops of the set
    give is shared among them by the posterior over ops, renormalized over those that give it. A color that only a
    dropped op gives goes to `dropped`, and one that no op gives to `none`. Mass on tokens that are not colors goes to
    `non-color`. The shares sum to one in every context, so the columns of a context sum to its KL.
    """
    table = answer_table()
    with tempfile.TemporaryDirectory() as tmp, np.load(read(ref, Path(tmp))) as z:
        a = {k: z[k] for k in z.files}
    p, post, pair, true = a["p"].astype(np.float64), a["posterior"].astype(np.float64), a["query_pair"], a["op_ids"]
    n, n_colors = p.shape
    # The answer distribution of each of the eleven ops on each query pair, dense over the grid.
    dense = np.zeros((len(ALL_OPS), n, n_colors + 1))
    for o in range(len(ALL_OPS)):
        idx = table.idx[o, pair]
        np.put_along_axis(dense[o], np.where(idx < 0, n_colors, idx), table.prob[o, pair], axis=1)
    dense = dense[..., :n_colors]
    q = np.einsum("no,ony->ny", post, dense[SET_IDS])
    kl = (q * (np.log(np.where(q > 0, q, 1.0)) - np.log(np.maximum(p, 1e-300)))).sum(1)

    excess = np.maximum(p - q, 0.0)
    gives = dense[SET_IDS] > 0
    weight = post.T[:, :, None] * gives
    total = weight.sum(0)
    share = np.where(total > 0, weight / np.where(total > 0, total, 1.0), 0.0)
    by_op = (share * excess[None]).sum(2).T
    unclaimed = excess * (total == 0)
    by_dropped = (dense[DROPPED_IDS] > 0).any(0)
    cols = np.column_stack([by_op, (unclaimed * by_dropped).sum(1), (unclaimed * ~by_dropped).sum(1), 1.0 - p.sum(1)])
    frac = cols / np.maximum(cols.sum(1, keepdims=True), 1e-12)

    return {
        "true": true,
        "p_true": post[np.arange(n), true],
        "kl": kl,
        "kl_stored": a["kl"].astype(np.float64),
        "split": kl[:, None] * frac,
        "top_gap": p.max(1) - q.max(1),
    }


RUNS = {(e, s): split_run(array_ref(e, s)) for e in LENGTHS for s in SEEDS}
N = len(RUNS[(PICK, ex.SCOUT_SEED)]["kl"])
assert all(len(r["kl"]) == N for r in RUNS.values())
# Answer distributions are stored in float16, so the recomputed KL differs a little from the stored one.
KL_DRIFT = max(abs(r["kl"].mean() - r["kl_stored"].mean()) for r in RUNS.values())
assert KL_DRIFT < 2e-3


# --- Derived measurements -----------------------------------------------------------------------------------


def bin_of(r: dict) -> np.ndarray:
    return np.digitize(r["p_true"], BINS[1:-1])


def kl_all(e: int) -> np.ndarray:
    """(seed,): the calibration KL of each run at length *e*."""
    return np.array([RUNS[(e, s)]["kl"].mean() for s in SEEDS])


def kl_share(e: int, b: int) -> np.ndarray:
    """(seed,): the part of the calibration KL of each run that sits in bin *b*: its sum over the bin, over all contexts."""
    return np.array([RUNS[(e, s)]["kl"][bin_of(RUNS[(e, s)]) == b].sum() / N for s in SEEDS])


def kl_mean(e: int, b: int) -> np.ndarray:
    """(seed,): the mean KL per context in bin *b*."""
    return np.array([RUNS[(e, s)]["kl"][bin_of(RUNS[(e, s)]) == b].mean() for s in SEEDS])


def gap_mean(e: int, b: int) -> np.ndarray:
    """(seed,): the mean, over the contexts of bin *b*, of the mass the model puts on its top answer less the mass
    the Bayes answer distribution puts on its own.
    """
    return np.array([RUNS[(e, s)]["top_gap"][bin_of(RUNS[(e, s)]) == b].mean() for s in SEEDS])


def bin_frac(b: int) -> float:
    return float((bin_of(RUNS[(PICK, ex.SCOUT_SEED)]) == b).mean())


def matrix(e: int) -> np.ndarray:
    """(op, column): the seed mean of each part of the calibration KL at length *e*, over all held-out contexts.
    Rows are the true op. The whole matrix sums to the calibration KL.
    """
    return np.mean(
        [
            np.stack([r["split"][r["true"] == o].sum(0) / N for o in range(N_OPS)])
            for r in (RUNS[(e, s)] for s in SEEDS)
        ],
        axis=0,
    )


def row_kl(e: int, ops: Sequence[str]) -> float:
    return float(sum(matrix(e)[OPS.index(o)].sum() for o in ops))


def op_mean_kl(e: int, op: str) -> float:
    """The seed mean of the mean KL per context of true op *op*."""
    return float(np.mean([r["kl"][r["true"] == OPS.index(op)].mean() for r in (RUNS[(e, s)] for s in SEEDS)]))


def col_share(e: int, cols: Sequence[str], b: int | None = None) -> float:
    """The share of the KL (of bin *b*, or of every context) that the columns *cols* take."""
    rs = [RUNS[(e, s)] for s in SEEDS]
    masks = [np.ones(N, bool) if b is None else bin_of(r) == b for r in rs]
    num = np.mean([r["split"][m][:, [COLUMNS.index(c) for c in cols]].sum() for r, m in zip(rs, masks, strict=True)])
    den = np.mean([r["split"][m].sum() for r, m in zip(rs, masks, strict=True)])
    return float(num / den)


def diag_share(e: int, b: int | None = None) -> float:
    """The share of the KL that goes to the column of the true op."""
    rs = [RUNS[(e, s)] for s in SEEDS]
    masks = [np.ones(N, bool) if b is None else bin_of(r) == b for r in rs]
    num = np.mean([r["split"][m][np.arange(m.sum()), r["true"][m]].sum() for r, m in zip(rs, masks, strict=True)])
    den = np.mean([r["split"][m].sum() for r, m in zip(rs, masks, strict=True)])
    return float(num / den)


LAST = len(BIN_NAMES) - 1
CONF_SHARE = {e: kl_share(e, LAST).mean() / kl_all(e).mean() for e in LENGTHS}
LOW_SHARE = {e: kl_share(e, 0).mean() / kl_all(e).mean() for e in LENGTHS}
D_LOW = kl_mean(REF_E, 0) - kl_mean(PICK, 0)
D_CONF = kl_mean(REF_E, LAST) - kl_mean(PICK, LAST)
OTHER_OPS = tuple(o for o in OPS if o not in HSV_CHANNEL)
assert (D_LOW > 0).all() and (D_CONF < 0).all(), "the prose says the two ends move apart at every seed"
assert all((gap_mean(e, 0) > 0.05).all() for e in LENGTHS), "the prose says the model commits more than Bayes there"
assert (gap_mean(REF_E, 0) > gap_mean(PICK, 0)).all(), "the prose says the longer run commits more, at every seed"
assert all((abs(gap_mean(e, LAST)) < 0.03).all() for e in LENGTHS), "the prose says the top mass is near Bayes there"
assert max(OPS, key=lambda o: op_mean_kl(PICK, o)) == "mix"
_diag = {e: np.diag(matrix(e)[:, :N_OPS]) for e in LENGTHS}
assert (_diag[REF_E] > _diag[PICK]).all(), "the alt text says the diagonal is darker at the longer length in every row"
assert [o for i, o in enumerate(OPS) if matrix(PICK)[i].argmax() != i] == ["difference"], (
    "the alt text names the exception"
)


def fmt_range(v: np.ndarray, f: str = ".3f") -> str:
    return f"{v.min():{f}} to {v.max():{f}}"


# --- Figures ------------------------------------------------------------------------------------------------


def shade(i: int) -> tuple:
    lo, hi = light_dark((0.45, 0.9), (0.4, 0.85))
    return plt.get_cmap(light_dark("Blues", "magma"))(lo + (hi - lo) * i)


def seq_cmap():
    cmap = plt.get_cmap(light_dark("Blues", "magma")).copy()
    cmap.set_bad(light_dark("#fff", "#111"))
    return cmap


def cell_text_color(v: float, vmax: float) -> str:
    dark_cell = (v / vmax > 0.55) == (light_dark(0, 1) == 0)
    return "#fff" if dark_cell else "#000"


@memo
def bins_draw(alt_text: str, caption: str) -> str:
    @themed(name="kl-by-confidence", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), layout="constrained")
        rng = np.random.default_rng(0)
        x = np.arange(len(BIN_NAMES))
        panels = (
            (kl_share, "part of the calibration KL"),
            (kl_mean, "mean KL per context"),
            (gap_mean, "top-answer mass, model − Bayes"),
        )
        for ax, (fn, ylabel) in zip(axes, panels, strict=True):
            for i, e in enumerate(LENGTHS):
                dx = (i - 0.5) * 0.28
                for b in x:
                    v = fn(e, int(b))
                    ax.plot(b + dx + rng.uniform(-0.04, 0.04, len(v)), v, "o", ms=2.6, color=shade(i), alpha=0.6, mew=0)
                    ax.plot(b + dx, v.mean(), "o", ms=5.5, color=shade(i), mec=light_dark("white", "#111"), mew=0.6)
                ax.plot([], [], "o", color=shade(i), label=f"{e} epochs")
            ax.set_xticks(x, BIN_NAMES, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlim(-0.5, len(BIN_NAMES) - 0.5)
        axes[2].axhline(0, color=light_dark("#333", "#ddd"), lw=0.8, ls="--")
        fig.supxlabel("Bayes posterior on the true op", fontsize=9)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=2, frameon=False, fontsize=8)
        return fig

    return _plot()


@memo
def matrix_draw(alt_text: str, caption: str) -> str:
    @themed(name="kl-by-op", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), layout="constrained", sharey=True)
        vmax = max(float(matrix(e).max()) for e in LENGTHS)
        im = None
        for ax, e in zip(axes, LENGTHS, strict=True):
            m = matrix(e)
            im = ax.imshow(m, cmap=seq_cmap(), vmin=0, vmax=vmax, aspect="auto")
            for i, j in np.ndindex(*m.shape):
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.3f}"[1:],
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=cell_text_color(m[i, j], vmax),
                )
            for i in range(N_OPS):
                ax.plot(i, i, "s", ms=17, mfc="none", mec="0.6", mew=0.6)
            ax.set_title(f"{e} epochs (KL {kl_all(e).mean():.3f})", fontsize=9)
            ax.set_xticks(range(len(COLUMNS)), COLUMNS, rotation=90, fontsize=8)
            ax.set_yticks(range(N_OPS), [f"{o}  {m[i].sum():.3f}" for i, o in enumerate(OPS)], fontsize=8)
        fig.supxlabel("where the model has mass in excess of the Bayes answer distribution", fontsize=9)
        fig.supylabel("true op, and its part of the KL", fontsize=9)
        assert im is not None
        fig.colorbar(im, ax=axes, shrink=0.6, label="part of the calibration KL (nats)")
        return fig

    return _plot()


# --- Report -------------------------------------------------------------------------------------------------

rf"""
# Ex 2.2.19, follow-up: where the calibration KL sits

/// tip |
<!-- tl;dr -->
Most of the calibration KL of the seven-op runs sits in contexts whose op the examples leave unclear: the contexts below a posterior of 0.5 on the true op are a quarter of the held-out set and hold {LOW_SHARE[PICK]:.0%} of the KL, and the confident contexts that the op confusion matrices use hold {CONF_SHARE[PICK]:.0%}. By true op, the KL is spread over all seven ops, highest on `mix`. The total is level between {PICK} and {REF_E} epochs, but at every seed the longer run is better calibrated on confident contexts and worse on the least confident, where it puts more mass on its top answer than the Bayes answer distribution does.
///

## Scope

[Ex-2.2.19](./report.py) measured two things about how a model spreads its answers. The calibration KL[^kl] is taken over every held-out context. The op confusion matrix (E3) shows which op answers the model leans toward, but only on confident contexts, where the Bayes posterior on the true op is above {CONFIDENT}. Sandy asked, reviewing it, whether a matrix like the confusion one can be made over the whole held-out set, showing how much each op contributes to the KL ([backlog item](/todo/science/per-op-calibration-kl-over-all-contexts.md)).

[^kl]: The KL divergence from the Bayes answer distribution to that of the model, in nats, averaged over contexts. It is the extra cross-entropy loss of the model over a perfectly calibrated one, so zero is the best a model can do.

This note splits the KL three ways, on the {PICK}- and {REF_E}-epoch runs of the seven-op set at four seeds each ({len(SEEDS)} runs per length). It is post hoc, with no hypotheses or gates, and trains nothing. Each length has {N:,} held-out contexts, the same contexts in every run.

## By confidence

The first split groups contexts by how sure the Bayes posterior is of the true op, and the confident set of the confusion matrices is the last group. The shares of contexts in the four groups are {", ".join(f"{bin_frac(b):.0%}" for b in range(LAST))}, and {bin_frac(LAST):.0%}, from least to most confident.

Most of the KL is in the least confident contexts. The contexts below 0.5 hold {LOW_SHARE[PICK]:.0%} of the KL at {PICK} epochs and {LOW_SHARE[REF_E]:.0%} at {REF_E}; the confident set holds {CONF_SHARE[PICK]:.0%} and {CONF_SHARE[REF_E]:.0%}. The KL per context falls steadily as the posterior sharpens, from about {kl_mean(PICK, 0).mean():.2f} below 0.5 to {kl_mean(PICK, LAST).mean():.2f} above {CONFIDENT}.

The two lengths move apart at the two ends. From {PICK} to {REF_E} epochs, the KL per context below 0.5 rises by {fmt_range(D_LOW)} over the four seeds, and in the confident set it falls by {fmt_range(-D_CONF)}. Every seed moves the same way at both ends. The total barely changes ({kl_all(PICK).mean():.3f} against {kl_all(REF_E).mean():.3f}), which is what ex-2.2.19 reported as calibration not separating the two lengths.

The mass on the top answer suggests why. Below a posterior of 0.5, the model puts {gap_mean(PICK, 0).mean():.2f} more mass on its top answer than the Bayes answer distribution puts on its own at {PICK} epochs, and {gap_mean(REF_E, 0).mean():.2f} more at {REF_E}, larger at every seed. So where the examples allow several ops, the model commits to one answer more than they justify, and longer training commits it further. In the confident set the top mass is within {max(abs(gap_mean(e, LAST)).max() for e in LENGTHS):.2f} of Bayes at both lengths, so the gain there comes from the rest of the distribution: at {REF_E} epochs less of the KL goes to colors no op gives ({col_share(REF_E, ["none"], LAST):.0%} of the KL of the confident set, against {col_share(PICK, ["none"], LAST):.0%} at {PICK}).

"""

bins_draw(
    f"""
        Three panels, each with four groups of contexts on the horizontal axis, from a posterior on the true op below
        0.5 to above 0.99, and two dot columns per group, for {PICK} and {REF_E} epochs. Left: the part of the KL in
        each group, largest below 0.5 (about {kl_share(PICK, 0).mean():.2f}) and smallest above 0.99 (about
        {kl_share(PICK, LAST).mean():.2f}). Middle: the mean KL per context, falling from about
        {kl_mean(PICK, 0).mean():.2f} to {kl_mean(PICK, LAST).mean():.2f}; the {REF_E}-epoch dots sit above the
        {PICK}-epoch ones in the first group and below them in the last. Right: the model less the Bayes
        top-answer mass, well above zero in the first group, higher at {REF_E} epochs, and near zero in the last.
    """,
    f"""
        **The calibration KL by how sure the Bayes posterior is of the true op.** {len(SEEDS)} runs per length; small
        dots are runs and large dots the seed mean. Left: the KL summed over the contexts of the group and divided by
        all contexts, so the four groups add up to the calibration KL. Middle: the mean KL per context in the group.
        Right: the mass the model puts on its top answer less the mass the Bayes answer distribution puts on its
        own; above the dashed line the model commits more than Bayes.
    """,
)

rf"""
## By op

The second and third splits make the matrix Sandy asked for. Rows are the true op, and a row adds up to the part of the KL on the contexts of that op. Columns say where the model has mass in excess of the Bayes answer distribution, and each context divides its KL among the columns in proportion to that excess. Excess on a color several ops of the set give is shared among them by the posterior over ops; excess on a color only a dropped op gives goes to `dropped`, on a color no op gives to `none`, and mass on tokens that are not colors to `non-color`.

The column split is a convention, and a weaker one than the row split. The KL counts the mass the model is short of on the Bayes answers, and the excess says where that mass went; giving the KL to the excess in proportion is one way to join them, and the rows add up either way.

By row, the KL is spread over all seven ops. The three HSV-channel ops hold {row_kl(PICK, HSV_CHANNEL) / kl_all(PICK).mean():.0%} of it at {PICK} epochs, for three ops of seven. The highest KL per context is on `mix` ({op_mean_kl(PICK, "mix"):.2f}), and the lowest on `lighten` and `darken` ({op_mean_kl(PICK, "lighten"):.2f} and {op_mean_kl(PICK, "darken"):.2f}). On confident contexts, the confusion matrix of ex-2.2.19 put most of the extra mass of the shorter run in the HSV-channel rows. Over all contexts, the KL does not gather there to the same degree.

By column, the true op takes {diag_share(PICK):.0%} of the KL, the other six ops of the set {col_share(PICK, OPS) - diag_share(PICK):.0%}, colors no op gives {col_share(PICK, ["none"]):.0%}, and the dropped ops {col_share(PICK, ["dropped"]):.0%}. Mass on tokens that are not colors is negligible. Below a posterior of 0.5 the other ops take {col_share(PICK, OPS, 0) - diag_share(PICK, 0):.0%}, consistent with a model that leans further than Bayes toward one op when the examples allow several. No single other op stands out in any row. From {PICK} to {REF_E} epochs the share of the true op rises to {diag_share(REF_E):.0%} and that of colors no op gives falls to {col_share(REF_E, ["none"]):.0%}: the longer run moves mass off the colors no op gives and onto the answers of the true op.

"""

matrix_draw(
    f"""
        Two heatmaps of the seven true ops by ten columns: the seven ops, then dropped, none, and non-color. The left
        is {PICK} epochs and the right {REF_E}. The diagonal is outlined and is the darkest square in every row but
        difference at {PICK} epochs, where the none column is larger; it is largest for mix, at about
        {matrix(PICK)[0, 0]:.3f}, and darker at {REF_E} epochs in every row. The none column is the next darkest; the
        rest of each row is pale and even, and the non-color column is near zero. Row totals beside the op names run from about
        {min(matrix(PICK).sum(1)):.3f} to {max(matrix(PICK).sum(1)):.3f}.
    """,
    f"""
        **The calibration KL by true op and by where the model has excess mass, at {PICK} and {REF_E} epochs.** Seed
        mean over {len(SEEDS)} runs, over all held-out contexts. Each square is a part of the KL in nats, and the
        whole map sums to the calibration KL in the title. Beside each op name is the sum of its row. The diagonal,
        outlined, is the excess on the answers of the true op.
    """,
)

rf"""
## What it means for what follows

The KL that remains at {PICK} and {REF_E} epochs is mostly in contexts whose op is unclear, which the confusion matrices leave out. So a lower KL would come mainly from better hedging between ops on those contexts, and less from the HSV-channel ops, whose shortfall shows in exact match on confident contexts.

The longer run trades calibration on unclear contexts for calibration on clear ones. That fits a model that keeps sharpening its answers with training, and it could be overfitting to the training corpus; the held-out arrays alone cannot tell the two apart, since there is no training-set KL to set them beside. Either way, a level total KL between two lengths does not mean the two are calibrated alike.

## Method

The runs are the {PICK}- and {REF_E}-epoch runs of the seven-op set in ex-2.2.19, at seeds {", ".join(str(ex.SEED_OFFSET + s) for s in SEEDS)}; the {REF_E}-epoch run at seed {ex.SEED_OFFSET + ex.SCOUT_SEED} is the ex-2.2.18 `no-four` run. Each run stored, per held-out context, the true op, the posterior over the seven ops, the query pair, and the model answer distribution over the grid (in float16). The Bayes answer distribution is rebuilt from the posterior and the answer table, and the KL from it agrees with the stored KL to within {KL_DRIFT:.4f} on average per run, the difference coming from the float16 storage.
"""
