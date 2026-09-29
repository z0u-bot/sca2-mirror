# title: Ex 2.2.18: dropping ops with similar answers, a scout

# The design constants and the refs come from `experiment.py` beside this script; the answer table and the posterior
# come from ex-2.2.16 through it, and the seed yardstick from ex-2.2.17's published evaluation.
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import AxesRow, figure_html, light_dark, themed

X = ex.ex2216
ALL_OPS: tuple[str, ...] = tuple(X.OP_NAMES)
SETS: tuple[str, ...] = tuple(s.name for s in ex.OP_SETS)
SINGLES: tuple[str, ...] = tuple(f"no-{op}" for op in ex.PARTNER)
YARDSTICK = tuple(f"sweep-{ex.PEAK_LR:g}-s{s}" for s in range(3))
# A context is confident when the posterior on its true op is above this, as in ex-2.2.17.
CONFIDENT = 0.99


# --- Helpers -------------------------------------------------------------------------------------------------


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str, *, text_cols: int = 1) -> str:
    """An authored result table in the shared report style; the first *text_cols* columns are text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>"
        + "".join(f"<td{' class=num' if i >= text_cols else ''}>{cell_html(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def rule_color() -> str:
    return light_dark("#333", "#ccc")


# --- Loading ------------------------------------------------------------------------------------------------


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists: one `get_refs` and one `get_many`."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict:
    assert path is not None, "not published yet"
    return json.loads(path.read_text())


with tempfile.TemporaryDirectory() as _tmp:
    _refs = [ex.EVAL_REF, ex.TRAJ_REF, ex.ex2217.EVAL_REF, *(ex.EVAL_ARRAYS_REF.format(label=s) for s in SETS)]
    _files = fetch(_refs, Path(_tmp))
    EVAL = read_json(_files[ex.EVAL_REF])
    TRAJ = read_json(_files[ex.TRAJ_REF])
    RUNS = {r["label"]: r for r in EVAL["runs"]}
    STATS = EVAL["op_sets"]
    PRIOR = {r["label"]: r for r in read_json(_files[ex.ex2217.EVAL_REF])["runs"] if r["label"] in YARDSTICK}
    ARRAYS: dict[str, dict[str, np.ndarray]] = {}
    for _s in SETS:
        with np.load(cast(Path, _files[ex.EVAL_ARRAYS_REF.format(label=_s)])) as _z:
            ARRAYS[_s] = {k: _z[k] for k in _z.files}


def ops_of(s: str) -> tuple[str, ...]:
    return tuple(RUNS[s]["ops"])


def score(s: str, key: str = "eem", op: str | None = None, runs: dict | None = None) -> float:
    r = (runs or RUNS)[s]["task"][key]
    return r["all"] if op is None else r["per_op"][list((runs or RUNS)[s].get("ops", ALL_OPS)).index(op)]


def skill(s: str, op: str | None = None) -> float:
    """The share of the way from the floor to the ceiling of its own op set that a run got."""
    e, c, f = (score(s, k, op) for k in ("eem", "ceiling", "floor"))
    return (e - f) / (c - f)


def gap(s: str, op: str | None = None, runs: dict | None = None) -> float:
    return score(s, "ceiling", op, runs) - score(s, "eem", op, runs)


YARD_EEM = [score(s, runs=PRIOR) for s in YARDSTICK]
YARD_GAP = [gap(s, runs=PRIOR) for s in YARDSTICK]
YARD_SPREAD = max(YARD_EEM) - min(YARD_EEM)
# The seed range of each op's gap in ex-2.2.17's three runs of this recipe: the per-op yardstick.
YARD_OP_SPREAD = {
    op: max(gap(s, op, PRIOR) for s in YARDSTICK) - min(gap(s, op, PRIOR) for s in YARDSTICK) for op in ALL_OPS
}


# --- The leak onto the dropped op ------------------------------------------------------------------------------


@memo
def answer_table():
    """The answers of all eleven ops on every pair, so the answers of a dropped op stay defined."""
    return X._get_posterior().build_table(X.TABLE)


@memo
def leak(p16: np.ndarray, op_ids: np.ndarray, post: np.ndarray, pair: np.ndarray, ops: tuple[str, ...]) -> dict:
    """On the confident contexts of each partner op in *ops*, the mass the run puts on colors its dropped op can
    give and the partner cannot; and the same for the Bayes predictive of the full op set, which is near zero on
    these contexts. Answers come from the full table, so they are defined whether or not the op was trained on.
    """
    table = answer_table()
    p = p16.astype(float)
    rows = np.arange(len(p))
    out = {}
    for dropped, partner in ex.PARTNER.items():
        if partner not in ops:
            continue
        sel = (op_ids == ops.index(partner)) & (post[rows, op_ids] > CONFIDENT)
        a, b = ALL_OPS.index(partner), ALL_OPS.index(dropped)
        gives = np.zeros((2, sel.sum(), p.shape[1]), dtype=bool)
        for i, o in enumerate((a, b)):
            idx, ok = table.idx[o, pair[sel]], table.prob[o, pair[sel]] > 0
            for j in range(idx.shape[1]):
                gives[i, np.flatnonzero(ok[:, j]), idx[ok[:, j], j]] = True
        beyond = gives[1] & ~gives[0]
        out[dropped] = {
            "leak": float((p[sel] * beyond).sum(1).mean()),
            "partner_mass": float((p[sel] * gives[0]).sum(1).mean()),
            "n": int(sel.sum()),
        }
    return out


LEAK = {
    s: leak(ARRAYS[s]["p"], ARRAYS[s]["op_ids"], ARRAYS[s]["posterior"], ARRAYS[s]["query_pair"], ops_of(s))
    for s in SETS
}

# --- Figures ------------------------------------------------------------------------------------------------


@memo
def scores_draw(alt_text: str, caption: str) -> str:
    @themed(name="scores", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained")
        axes = cast(AxesRow, axes)
        x = np.arange(len(SETS))
        ax = axes[0]
        ax.plot(x, [score(s, "ceiling") for s in SETS], "_", ms=22, mew=2, color=rule_color(), label="Bayes ceiling")
        ax.plot(x, [score(s, "floor") for s in SETS], "_", ms=22, mew=1, color="0.6", label="floor")
        ax.plot(x, [score(s) for s in SETS], "o", color="C0", label="model")
        ax.axhspan(min(YARD_EEM), max(YARD_EEM), color="C0", alpha=0.15, lw=0, label="ex-2.2.17, three seeds")
        ax.set_ylabel("held-out EEM")
        ax.set_ylim(0, 0.7)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax = axes[1]
        ax.bar(x, [gap(s) for s in SETS], color="C0", width=0.6)
        ax.axhspan(min(YARD_GAP), max(YARD_GAP), color="C0", alpha=0.15, lw=0)
        ax.set_ylabel("gap to own ceiling")
        for a in axes:
            a.set_xticks(x, SETS, rotation=30, ha="right", fontsize=8)
        return fig

    return _plot()


@memo
def gaps_draw(alt_text: str, caption: str) -> str:
    @themed(name="gaps", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        m = np.full((len(ALL_OPS), len(SETS)), np.nan)
        for j, s in enumerate(SETS):
            for op in ops_of(s):
                m[ALL_OPS.index(op), j] = gap(s, op)
        fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
        lim = np.nanmax(np.abs(m))
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=lim, aspect="auto")
        for (i, j), v in np.ndenumerate(m):
            if np.isnan(v):
                ax.text(j, i, "dropped", ha="center", va="center", fontsize=6, color="0.5")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w" if v < lim * 0.6 else "k")
        ax.set_xticks(range(len(SETS)), SETS, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(ALL_OPS)), ALL_OPS, fontsize=8)
        fig.colorbar(im, ax=ax, label="gap to own ceiling (EEM)", shrink=0.8)
        return fig

    return _plot()


@memo
def traj_draw(alt_text: str, caption: str) -> str:
    @themed(name="trajectories", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
        for i, s in enumerate(SETS):
            t = TRAJ[s]
            st = STATS[s]
            y = (np.array(t["eem"]) - st["floor"]) / (st["ceiling"] - st["floor"])
            ax.plot(np.array(t["step"]) / 1e3, y, color=f"C{i}", lw=1.3 if s != "full" else 2, label=s)
        ax.axhline(1, ls="--", color=rule_color(), lw=1)
        ax.set_xlabel("step (thousands)")
        ax.set_ylabel("skill on the probe set")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, frameon=False, ncols=2)
        return fig

    return _plot()
