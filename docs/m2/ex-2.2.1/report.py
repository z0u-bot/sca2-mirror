# title: Ex 2.2.1: suppressing red in the anchored transformer

r"""
# Ex 2.2.1: suppressing *red* in the anchored transformer

/// tip |
<!-- tl;dr -->
The first intervention on an anchored transformer. We take the D2.1 checkpoints and remove the anchor axis from the residual stream, then score completion accuracy on lines with a red operand and on lines without. The removal works, and it scales with how red the line is. It stays inside the bound the placed geometry sets. Zeroing the axis weights does the same job. Selectivity is only partial: the loss on non-red lines comes from the syntax positions, and it goes away when we edit the operand positions only.
///
"""

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D

# The document's directory is on sys.path while it runs, so the design constants and refs
# come from the experiment module rather than a copy of them in the prose.
import experiment as ex
from mini.lit import memo, stop
from mini.store import project_store
from mini.vis import AxesRow, figure_html, light_dark, smooth_step, smooth_step_area, smooth_step_marks, themed
from sca.intervention import Subspace, angle_between, projection, shaped_suppression
from sca.vis_grading import GRID_RGB, I_RED, GradingCloud, quantile_stack

SLICE_NAMES = ["emb", "1", "2", "3", "4"]
POS_NAMES = ["op1", "+", "op2", "=", "ans", "⏎"]
BIN_NAMES = ["≤0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "≥0.8"]
ARM_INK = {
    "primary": ("#d40000", "#f44"),
    "operands": ("#e07000", "#fa4"),
    "shaped": ("#8030c0", "#c8f"),
    "ablate": ("#2060c0", "#7af"),
    "control": ("#555", "#aaa"),
}
# One ink per arm the figures compare, as (light, dark) pairs for `light_dark`.


PROBES_REF = f"reports/m2/{ex.PRIMARY.exp}/probes"


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists: one `get_refs` and one `get_many` for the lot.

    The loads themselves are not memoized: a ref can move to new data under the same name, which no code fingerprint sees, so the store is asked each render and the cache holds what is computed from the data.
    """
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def load_results(files: dict[str, Path | None]) -> tuple[dict, dict[str, np.ndarray]] | None:
    """Metrics and the stacked per-run arrays, or None if unpublished."""
    m_path, a_path = files[ex.METRICS_REF], files[ex.ARRAYS_REF]
    if m_path is None or a_path is None:
        return None
    with np.load(a_path) as z:
        return json.loads(m_path.read_text()), {k: z[k] for k in z.files}


@memo
def probe_lines(tokens: np.ndarray, r1: np.ndarray, r2: np.ndarray, redness: np.ndarray) -> tuple[ex.Lines, np.ndarray]:
    """The probe set as the scorer grouped it, and the redness per line.

    The op1 column walks the palette in order, so the color token ids can be read off the set itself
    and the scorer's own grouping code runs without a checkpoint's tokenizer.
    """
    from sca.data.named_colors import GRIDS, grid_palette

    names = grid_palette(GRIDS["v216"])
    ids = tokens[:: len(tokens) // len(names), 0]
    tokenizer = SimpleNamespace(stoi=dict(zip(names, ids, strict=True)), vocab_size=int(tokens.max()) + 1)
    return ex.read_lines(tokens, r1, r2, redness, tokenizer, 512), np.maximum(r1, r2)


def span(v: np.ndarray, fmt: str = ".3f") -> str:
    """Seed mean with the seed range beside it."""
    return f"{v.mean():{fmt}} ({v.min():{fmt}}–{v.max():{fmt}})"


def span2(v: np.ndarray, fmt: str = ".3f") -> str:
    """Seed mean with the seed range beside it."""
    greater_half_range = max(v.mean() - v.min(), v.max() - v.mean())
    return f"{v.mean():{fmt}} ±{greater_half_range:{fmt}}"


# Loaded here, ahead of "## Findings" below, since the Findings prose reads the derived
# statistics; the Marimo notebook could place this cell anywhere and run it by dependency.
with tempfile.TemporaryDirectory() as _tmp:
    files = fetch([ex.METRICS_REF, ex.ARRAYS_REF, PROBES_REF], Path(_tmp))
    loaded_results = load_results(files)
    probes_path = files[PROBES_REF]
    assert probes_path is not None, "the probe set is missing from the store"
    with np.load(probes_path) as _z:
        probes = {k: _z[k] for k in ("probe_tokens", "r1", "r2", "redness")}
if loaded_results is None:
    stop("_Results are not published yet; the analysis renders once they are._")
metrics, arrays = loaded_results
lines, dose = probe_lines(probes["probe_tokens"], probes["r1"], probes["r2"], probes["redness"])
runs: dict[str, list[dict]] = {
    c.name: [r for r in metrics["runs"] if r["condition"] == c.name] for c in (ex.PRIMARY, ex.CONTROL)
}
assert [len(runs[c.name]) for c in (ex.PRIMARY, ex.CONTROL)] == [ex.PRIMARY.seeds, ex.CONTROL.seeds]


def stat(iv: str, key: str, group: str | None = None, cond: str = ex.PRIMARY.name) -> np.ndarray:
    """One value per seed of a condition, for one statistic of one intervention."""
    cells = [r["interventions"][iv][key] for r in runs[cond]]
    return np.array(cells if group is None else [c[group] for c in cells], float)


def clean(key: str, group: str, cond: str = ex.PRIMARY.name) -> np.ndarray:
    return np.array([r["clean"][key][group] for r in runs[cond]], float)


def bins(iv: str, cond: str = ex.PRIMARY.name) -> np.ndarray:
    """(seeds, bins) damage per dose bin."""
    return stat(iv, "damage_bins", cond=cond)


def per_line(iv: str, key: str, cond: str = ex.PRIMARY.name) -> np.ndarray:
    """(seeds, ...) one intervention's per-line arrays, stacked over the seeds of a condition."""
    return np.stack([arrays[f"{r['label']}/{iv}/{key}"] for r in runs[cond]]).astype(np.float32)


def bound_map(cond: str = ex.PRIMARY.name) -> np.ndarray:
    """(slices, positions) seed-mean clean bound: arcsin of the 99th-percentile non-red alignment."""
    return np.array([r["clean"]["bound"] for r in runs[cond]], float).mean(0)


def ratio_map(iv: str, cond: str = ex.PRIMARY.name) -> np.ndarray:
    """(slices, positions) seed-mean 99th-percentile non-red write over the seed-mean bound."""
    return stat(iv, "q99_write_nonred", cond=cond).mean(0) / bound_map(cond)


def floor(key: str, group: str) -> float:
    """The resolution for one statistic: the pooled between-seed sd over the anchored arms, times the factor."""
    var = np.array([stat(iv, key, group).var(ddof=1) for iv in ("primary", *ex.ARMS)])
    return ex.RESOLUTION_SD * float(np.sqrt(var.mean()))


def ex_alpha_diff() -> dict[str, float]:
    return metrics["alpha_max_diff"]


r"""
## Findings
"""

find_red = stat("primary", "acc", "red")
find_deficit = clean("acc", "nonred") - stat("primary", "acc", "nonred")
find_ops = clean("acc", "nonred") - stat("operands", "acc", "nonred")
find_bin_damage = bins("primary").mean(0)
find_bin_dip = float(max(0.0, (find_bin_damage[:-1] - find_bin_damage[1:]).max()))
find_ratio = ratio_map("primary")[1:]
find_worst_site = np.unravel_index(int(np.argmax(find_ratio)), find_ratio.shape)
# REVIEW: H4's Findings line said the ratio was "below the bound" at every site before the answer.
# It is 1.02 at (slice 1, `+`), so the line now quotes the ratio, matching the H4 section's own
# sentence. Verify: `ratio_map("primary")[1:, :ANSWER_POS]` should stay at or under the H4 tolerance.
find_abl_red = stat("ablate", "acc", "red")
find_abl_def = clean("acc", "nonred") - stat("ablate", "acc", "nonred")

rf"""
**H1 (suppression bites) — holds.** Seed-mean accuracy on red lines under the primary intervention is {find_red.mean():.2f}, against a gate of {ex.RED_ACC_GATE:g}; the highest single seed is {find_red.max():.2f}.

**H2 (selective) — partial.** The seed-mean drop in accuracy on non-red lines is {find_deficit.mean():.3f}. That is outside the {ex.TASK_GATE:g} gate but inside the {ex.TASK_PARTIAL:g} partial gate. One seed accounts for most of it ({find_deficit.max():.3f}); the median seed is at {np.median(find_deficit):.3f}. The `operands` arm, which skips the syntax positions, drops only {find_ops.mean():.3f}. So the side-effect is due to the constant component in the syntax positions, which is the case the contrary clause of H2 named.

**H3 (grades with redness) — holds.** Seed-mean damage across the five redness bins[^redness] is {", ".join(f"{v:.2f}" for v in find_bin_damage)}. The largest dip between adjacent bins is {find_bin_dip:.3f} (allowed {ex.GRADE_DIP:g}), and the top bin exceeds the bottom by {find_bin_damage[-1] - find_bin_damage[0]:.2f} (gate {ex.GRADE_SPAN:g}).

**H4 (the write bound) — holds.** At the four deeper slices the seed-mean 99th-percentile write on non-red lines is at most {find_ratio.max():.2f} times the clean bound (gate {ex.WRITE_TOLERANCE:g}); the largest is at slice {find_worst_site[0] + 1}, position `{POS_NAMES[find_worst_site[1]]}`. At every site before the answer it is at most {ratio_map("primary")[1:, : ex.ANSWER_POS].max():.2f} times the bound. Read seed by seed rather than on the mean, two seeds exceed the tolerance at a site before the answer.

**H5 (permanent removal) — holds.** Under `ablate`, accuracy on red lines is {find_abl_red.mean():.2f} (gate {ex.RED_ACC_GATE:g}) and the drop on non-red lines is {find_abl_def.mean():.3f} (gate {ex.TASK_GATE:g}).

The recomputed clean alignment maps match the ones ex-2.1.10 published to within {max(ex_alpha_diff().values()):.4f} on every run.

[^redness]: The *redness* of a line is the redness of its redder operand. *Damage* is how much probability the correct answer loses under intervention, clean minus intervened.
"""

r"""
/// admonition | How to read this report
This report was preregistered: the method, hypotheses, and decision thresholds were frozen before the experiment ran (commit `1a42bc0`). Results replaced the `TODO` placeholders in place; everything conceived after seeing the data is under [Exploratory analyses](#exploratory-analyses), marked as post hoc.

While the draft was written we ran a one-seed smoke test of the operator library (seed 0 of each condition), to check the code path from checkpoint to score. The gates come from the clean statistics instead: the published alignment maps, the task-gate width the D2.1 experiments used, and the reference values from M1. None of the smoke-test numbers are quoted here.

One smoke-test observation was stated in advance. At one syntax position, the alignment arriving at the second slice under intervention sat above the clean envelope on that seed. We kept the deeper-slice clause of H4 as the design states it; the nine-seed read is in the H4 section.

One design change was made after the smoke test and before the run. The post-hoc tier's directions were first fitted on the op1 states and applied at every position, and on the control seed the oblique fit edited the syntax states heavily, an extrapolation rather than an erasure. The tier now fits and applies at the same sites, as the Method describes. No gate changed.
///

## Why this experiment

D2.1 anchored *red* on the first axis of the residual stream, and the geometry landed where we asked: red states aligned, the non-red bulk contained, grading roughly sensible, task accuracy unchanged. It never ran an intervention. The point of placing a concept is that removing it afterwards has a predictable effect and a side-effect we can bound, and we have not tested that claim in a transformer.

The test uses checkpoints already in the store. Projecting e₁ out of the stream at inference time costs nothing to train and about a second per run to score, which makes it the most information per dollar in the [D2.2 plan](../d2.2/design.md).

It is also where the first three risks in that plan get their first data:
(a) that suppression does not bite even on *red*,
(b) that the bound is loose or wrong once later blocks sit between the write and the output, and
(c) that the response to suppression is undesigned. Only the first is retired here. The plan shares the second with the layer sweep and the third with the fallback experiment.

Three findings from the M1 autoencoder result become hypotheses here. Suppression there removed *red* almost completely, left orthogonal colors untouched, and degraded warm colors in proportion to their alignment with the axis. Those become H1, H2, and H3.

The bound does not transfer as it stands. In the autoencoder, the decoder was a single linear readout from the point of intervention, so the geometry bounded the output directly. Here every later block sits between the write and the logits.

So we state the bound as layer-local, following the [kickoff lessons](/todo/science/d21-kickoff-carry-over-lessons.md): the geometry bounds the immediate write at each intervened slice (H4). Behavior is then a separate prediction, scored by the task gate (H2).

These checkpoints have no fallback term, so whatever a red line decodes to after suppression is the untrained response of the model itself. M1 called that spoofing, and it is where the seed spread in ex-2.9.1 came from. We measure it here without a gate; it is the reference the fallback experiment after this one has to improve on.

/// details | Glossary
- **line** — one equation, `c1 + c2 = answer`, six word-level tokens. The probe set enumerates all 5,832 closed lines of the 216-color grid: every color as op1 against each of its 27 closed partners.
- **residual stream** — the running vector for each token, which every block reads from and adds back into.
- **slice** ($\ell$) — a depth at which the residual stream is read: the embedding, plus the stream after each of the four blocks. Five slices.
- **alignment** ($\alpha$) — $\cos(h, e_1)$, a state's cosine to the anchor axis. States are unit-norm, so it is the e₁ component itself.
- **redness** — of a color, the label affinity from M1, $r(1 - g/2 - b/2)$ on the unit cube. Of a line, the redness of its redder operand, $\max(r_1, r_2)$; the D2.2 plan calls this the line's dose. Either operand can be the red one, since either could draw the label in ex-2.1.10.
- **red lines** — redness ≥ 0.8: an operand among the five reddest colors of the grid. 365 lines.
- **non-red lines** — redness ≤ 0.2. 1,689 lines.
- **visible operand** — on a red line, the less red operand: the one the model still sees as itself after the red operand's axis component is removed.
- **response** — the probability the model puts on the correct answer, read from the log-softmax at the `=` position. Accuracy is whether that answer is the argmax.
- **damage** — the clean response minus the intervened response, per line, in probability units.
- **write** — the rotation the operator applies to one state, in radians: the angle between the state as it arrived at the operator and the state the next block consumed.
- **clean** — the un-intervened forward pass of the same checkpoint.
///

## Method

No training. Every run is a published checkpoint of ex-2.1.10, scored on that experiment's published probe set through the eval contract in [`sca.intervention`](/src/sca/intervention.py): the same lines, the same code, and the same statistics for every condition, arm, and baseline.

### The intervention

Axis projection at full strength: at each slice, every state loses its e₁ component and is put back onto the sphere, since the next block expects a unit vector. That re-projection is part of the edit.

For a state with alignment $\alpha$, the surviving components are rescaled by $1/\sqrt{1 - \alpha^2}$, and the whole edit amounts to a rotation by $\arcsin|\alpha|$. That formula depends only on the alignment the state arrived with, so we can compute the bound from the published alignment maps before the intervention runs.

The primary intervention acts at all five slices and all six positions. The operator at slice $\ell$ sees a state the blocks built from already-edited inputs, so only at the embedding does it see the clean stream. At every later slice, the arriving alignment measures whether the blocks put the axis back.

### Measurements

One teacher-forced pass per run per intervention, over all 5,832 lines. From each pass:

- **The response** per line: the log-softmax over the vocabulary at the `=` position. From it we read the probability on the correct answer, the argmax, and the mass outside the color vocabulary.
- **The write** per (slice, line, position): the angle between the arriving state and the edited state. The pipeline asserts that this equals $\arcsin|\alpha_{\text{arriving}}|$ at every site, so where a map reads more easily we report the write as that alignment.
- **The clean alignment map** per run, from the un-intervened pass. This is the same array ex-2.1.10 published, recomputed here so every quantity comes from one code path.
- **The final-state displacement** per line: the angle between the intervened and clean states at the last slice, `=` position. That is the state the logits are read from. We report it in radians, the same scale as a write.
- **The decoded write** per (slice, position): a ridge probe for the RGB of the operand, fitted per run on the clean states at that site, then read on the arriving state and on the edited state. The difference is what the write removed, in color units. The angle says how far a state moved; this says whether what moved was redness (E6).

**The undesigned response**, on red lines under the primary intervention, without a gate. We record the mass outside the color vocabulary; how far the decoded answer sits from the true mix, in unit-cube units; which answer it is (the visible operand, the red operand itself, the true answer, a one-step neighbor of it, or something else); and seed agreement, the fraction of red lines on which at least five of the nine seeds decode the same answer. These are the reference numbers for the fallback experiment.

**Noise floor.** Clean accuracy on every group is at or near 1.0 in every run, so the seed scatter that matters is in the intervened statistics. For each group statistic we compute the pooled between-seed standard deviation per condition before reading any comparison. A difference between arms or tiers smaller than twice that value is reported as not resolved. Gates score seed means against fixed thresholds and do not use the floor.

### Conditions

Two conditions from ex-2.1.10, both trained on the same corpus with the same seeds and code path:

- **anchored** (`either-t100`, 9 seeds) — the D2.1 recipe: pooled either-operand labels, the anchor at λ = 0.1, the anti-subspace term at the ex-2.1.8 operating point. This is the primary condition, and the one every hypothesis scores.
- **un-anchored** (`lam0`, 3 seeds) — the anchor weight at zero. Nothing was placed on e₁, so projecting it out applies the operator to a model with nothing on the axis. That calibrates what a write of this kind costs when it removes no concept.

**Arms**, on the anchored condition. Each changes one thing from the primary intervention, and we report the same statistics for each without gates:

| arm | what changes | question |
| --- | --- | --- |
| `operands` | positions restricted to op1 and op2 | does the edit need the syntax positions, whose embeddings have a constant component on the axis? |
| `embedding` | slice 0 only | is removing the token's component enough, or do the blocks re-derive red? |
| `last` | slice 4 only | the one-readout case, closest to M1's geometry |
| `shaped` | M1's shaped suppression, threshold α = 0.5, in place of the plain projection | does leaving the low-alignment bulk alone keep the removal and cut the collateral? |
| `ablate` | the weights that read and write e₁ zeroed, then re-normalized | the permanent removal from M1, scored under H5 |

The shaped suppression is M1's alternative to the plain projection.<!-- REVIEW: this arm was named `lobe` at the freeze, after the polar plots in M1's appendix; in M1 a lobe is the plot, so the arm, the operator, and the result key were renamed and the scoring re-run. Nothing else changed. --> Where the projection removes the whole e₁ component of every state, the shaped suppression removes only a fraction $h(\alpha)$ of it, set by the alignment the state arrived with. Below a threshold $a$ it removes nothing; above it, the fraction rises linearly to all of it at $\alpha = 1$ (in the notation of the M1 appendix, $b = 1$ and $p = 1$).

Alignment here is a cosine: 1 for a state pointing along e₁, 0 for one at right angles to it. A threshold of $a = 0.5$ sits 60° from the axis. States with a negative alignment are left alone too.

Removing part of the component shortens the state, taking it off the sphere, which is where the M1 suppression left it. Here the stream is unit-norm, so both operators re-normalize afterward, and removal plus re-normalization amounts to a rotation away from e₁, in the plane spanned by e₁ and the state. The figure draws that rotation, and where each direction in the plane lands.
"""


def unit_state(alpha: np.ndarray) -> np.ndarray:
    """A unit state at alignment *alpha*, in the plane of e₁."""
    return np.stack([alpha, np.sqrt(1 - alpha**2)], -1).astype(np.float32)


def op_rotation(op, alpha: np.ndarray) -> np.ndarray:
    """The write in degrees that *op* applies to a unit state at alignment *alpha*."""
    h = unit_state(alpha)
    return np.degrees(angle_between(h, np.asarray(op(h))))


def op_landing(op, alpha: np.ndarray) -> np.ndarray:
    """The alignment a unit state at alignment *alpha* leaves *op* with."""
    return np.asarray(op(unit_state(alpha)))[..., 0]


def polar_chord(ax: PolarAxes, t: float, to: float, color) -> None:
    """A chord from angle *t* to angle *to* on the rim, bowed toward the center in proportion to the turn, so nearby rotations stay apart (M1's diagram helper)."""
    s = np.linspace(0, 1, 60)
    bow = 0.97 * abs(to - t) / np.pi + 0.03
    ax.plot(t + (to - t) * s, 1 - bow * (1 - np.abs(2 * s - 1) ** 2.2), color=color, lw=0.5)


def polar_panel(ax: PolarAxes, name: str, op, ink, rim, threshold: float | None = None) -> None:
    """One operator on the unit circle, e₁ at the top: its rotation lobe, scaled so 90° reaches the rim, and a chord per direction to the dot where it lands. *threshold* draws the operator's alignment threshold as two dashed radials."""
    theta = np.linspace(-np.pi, np.pi, 721)
    h = np.stack([np.cos(theta), np.sin(theta)], -1).astype(np.float32)
    ax.set_theta_zero_location("N")
    write = angle_between(h, np.asarray(op(h))) / (np.pi / 2)
    ax.plot(theta, np.ones_like(theta), color=rim, lw=0.5)
    ax.fill(theta, write, color=ink, alpha=0.15, lw=0)
    ax.plot(theta, write, color=ink, lw=1)
    for t in np.linspace(-np.pi, np.pi, 25)[:-1] + np.pi / 24:  # offset, so no state sits on the axis itself
        o = np.asarray(op(np.array([[np.cos(t), np.sin(t)]], np.float32)))[0]
        to = float(np.arctan2(o[1], o[0]))
        if abs(to - t) > 1e-3:
            polar_chord(ax, t, to, rim)
        ax.plot(to, [1], "o", ms=1, color=rim)  # where the state lands
    if threshold is not None:
        a = float(np.arccos(threshold))
        for sign in (1, -1):
            ax.plot([sign * a, sign * a], [0, 1], ls="--", lw=0.6, color=rim)
        ax.text(-a, 0.5, "a", ha="left", va="bottom", fontsize=7)
    ax.set_rmax(1.15)
    ax.set_rticks([])
    ax.set_thetagrids([0, 180], ["e₁", "−e₁"])
    ax.tick_params(labelsize=7)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_title(name, fontsize=8, pad=8)


def alignment_marks(
    ax: plt.Axes, top: float, rim, syntax_alpha: tuple[float, float], red_alpha: float, threshold: float
) -> None:
    """On an alignment axis: the operator's threshold dashed, and the two alignments it was set between, a band for the syntax embeddings and a line for pure red, labeled at height *top*."""
    ax.axvspan(*syntax_alpha, color=light_dark("#0001", "#fff1"), lw=0)
    ax.axvline(red_alpha, color=rim, lw=0.5)
    ax.axvline(threshold, color=rim, lw=0.6, ls="--")
    for x, label in ((np.mean(syntax_alpha), "+, ="), (red_alpha, "pure red")):
        ax.text(x, top, label, ha="center", va="bottom", fontsize=6, color=rim)
    ax.tick_params(labelsize=7)


shaped_name = f"shaped suppression, a = {ex.SHAPED['a']:g}"
suppression_ops = {
    "projection": projection(Subspace.direction(np.array([1.0, 0.0]))),
    shaped_name: shaped_suppression(
        Subspace.direction(np.array([1.0, 0.0])), a=ex.SHAPED["a"], b=ex.SHAPED["b"], p=ex.SHAPED["p"]
    ),
}
suppression_ink = {"projection": ARM_INK["primary"], shaped_name: ARM_INK["shaped"]}
# Where the clean map ex-2.1.10 published puts the states the threshold was set between: the constant
# component of the `+` and `=` embeddings, and pure red at op1 (see `experiment.SHAPED`).
SYNTAX_ALPHA, RED_ALPHA = (0.3, 0.4), 0.9

op_red_write = {name: float(op_rotation(op, np.array([RED_ALPHA]))[0]) for name, op in suppression_ops.items()}
op_red_land = {name: float(op_landing(op, np.array([RED_ALPHA]))[0]) for name, op in suppression_ops.items()}


@memo
@themed(
    name="operators",
    alt_text="""
        Four panels. Two polar views of the unit circle with e₁ at the top: the projection rotates every direction to the equator, and its rotation lobe peaks at both poles; the shaped suppression with threshold 0.5 leaves every direction more than 60 degrees from e₁ where it is and rotates the rest part way. A line chart of rotation against alignment: the projection rises from zero at α = 0 to 90 degrees at α = 1, and the shaped suppression stays at zero until 0.5 and then rises to meet it at 1. A line chart of the landing alignment against the arriving one: the projection sits at zero throughout; the shaped suppression follows the identity line up to 0.5, then bends down through the threshold to reach zero at α = 1.
    """,
    caption=f"""
        **The two operators, in the plane of e₁ and the state.** Left and middle, each chord runs from a unit state at some angle from e₁ (top) to the dot where the operator moves it; the filled lobe is the size of that rotation against the direction, scaled so 90° reaches the rim. Third, the same rotation against the alignment α. Fourth, the alignment the state leaves with against the one it arrived with; the dotted line is no change. The threshold $a = {ex.SHAPED["a"]:g}$ is dashed, with the alignments it was set between marked from the published clean map: the `+` and `=` embeddings (band) and pure red at op1 (line).
    """,
)
def operators_plot() -> plt.Figure:
    fig = plt.figure(figsize=(8.6, 2.3), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.4, 1.1])
    rim = light_dark("#0008", "#fff8")
    for i, (name, op) in enumerate(suppression_ops.items()):
        ax = cast(PolarAxes, fig.add_subplot(gs[0, i], projection="polar"))
        polar_panel(ax, name, op, light_dark(*suppression_ink[name]), rim, ex.SHAPED["a"] if name == shaped_name else None)  # fmt: skip

    ax = fig.add_subplot(gs[0, 2])
    alpha = np.linspace(-1, 1, 401)
    for name, op in suppression_ops.items():
        ax.plot(alpha, op_rotation(op, alpha), color=light_dark(*suppression_ink[name]), lw=1.3, label=name)
    alignment_marks(ax, 92, rim, SYNTAX_ALPHA, RED_ALPHA, ex.SHAPED["a"])
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 30, 60, 90])
    ax.set_xlabel("alignment α of the arriving state", fontsize=8)
    ax.set_ylabel("rotation applied, degrees", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left")

    ax = fig.add_subplot(gs[0, 3])
    alpha = np.linspace(0, 1, 401)
    ax.plot(alpha, alpha, color=rim, lw=0.5, ls=":")
    for name, op in suppression_ops.items():
        ax.plot(alpha, op_landing(op, alpha), color=light_dark(*suppression_ink[name]), lw=1.3)
    alignment_marks(ax, 1.02, rim, SYNTAX_ALPHA, RED_ALPHA, ex.SHAPED["a"])
    ax.axhline(ex.SHAPED["a"], color=rim, lw=0.6, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("alignment arriving", fontsize=8)
    ax.set_ylabel("alignment leaving", fontsize=8)
    return fig


rf"""
{operators_plot()}

At $a = {ex.SHAPED["a"]:g}$ the shaped suppression leaves the syntax embeddings where they are. It rotates a pure-red operand at the embedding by about {op_red_write[shaped_name]:.0f}°, where the projection rotates it by {op_red_write["projection"]:.0f}°. So the red operand sits partway up the ramp rather than at the top.

A pure-red operand arrives at α = {RED_ALPHA:g} and leaves at {op_red_land[shaped_name]:.2f}, past the threshold and about where the syntax embeddings sit: the falloff sets how much of the component is removed, and the re-normalization sets where the state lands. Above the threshold the order reverses, so the more aligned a state arrives, the less aligned it leaves (fourth panel). The repulsion form from M1 sets the landing alignment instead, and would hold the red operand at the threshold. We did not test it here; the [repulsion item](/todo/science/repulsion-sets-the-landing-alignment.md) has the case for it.

**The post-hoc tier**, on the un-anchored condition, no gates. For each `lam0` run we fit one direction per slice and site on that run's own clean states, then apply it at that slice, at the positions it was fitted on, through the same scorer. Two sites: the operand positions, and the `=` position the answer is decoded from. The label is the redness a state can have seen under causal attention: the first operand's at op1, the higher of the two after that, which at `=` is the line's redness. Three fits beside e₁ itself: diff-in-means[^dim] between the states labelled at or above the red threshold and the rest, a ridge probe[^probe] for that redness, and LEACE[^leace] for it. The fit and the edit share a distribution because an oblique eraser needs them to: LEACE reads its coefficient along a covariance-whitened direction, so a state far from the fitted cloud can draw an edit of any size. Beside each fitted direction sits e₁ at the same footprint, so the calibration is like for like. The operand site pairs with the `operands` arm; the decode site is the sharpest lever a post-hoc method can be given, an edit where the answer is read off. This calibrates what a searched-for direction removes from a model that was never asked to place one.

[^dim]: The unit vector pointing from the mean non-red state to the mean red state. It is the simplest fitted direction, and the one most steering work uses.

[^probe]: A linear readout of redness, fitted by least squares with an ℓ₂ penalty. The direction is its weight vector, normalized. It finds what can be read off the state linearly, which need not be what the blocks use.

[^leace]: Least-squares concept erasure ([Belrose et al., 2023](https://arxiv.org/abs/2306.03819)). It zeroes the covariance between the states and the label, and is the smallest linear edit, in the covariance-whitened norm, after which no linear probe can recover the concept. The projection is oblique: it removes the covariance-weighted direction along a different one. It usually edits less than the other two, since it removes only the linear trace.

**The strength sweep**, on the anchored condition: the primary intervention at γ ∈ {{0.25, 0.5, 0.75, 1}}, where γ is the fraction of the e₁ component that is removed. Exploratory (E1).

## Hypotheses

All on the anchored condition under the primary intervention unless stated; seed means over nine runs.

- **H1.** Suppression bites. Seed-mean exact-match accuracy on red lines falls to 0.5 or below, from a clean accuracy of 1.0. Partial: between 0.5 and 0.8. Contrary: accuracy stays above 0.8, which would mean the model reads the color of the operand from somewhere the axis does not reach.
- **H2.** Suppression is selective. Seed-mean accuracy on non-red lines stays within 0.02 of the clean accuracy. Partial: within 0.05. The gate is the width every D2.1 task gate used.

    Projecting out an axis costs something even where nothing was placed on it. So we report the un-anchored row under the same operator beside this one, to say how much of any deficit comes from the operator rather than from the removal.
- **H3.** Damage grades with redness. We bin every line by its redness at edges 0.2, 0.4, 0.6, and 0.8.<!-- REVIEW: the frozen text said "dose" here and throughout; the quantity is the same (the redness of the redder operand) and is now called the line's redness, since "dose" did not say what it was. No gate changed. --> Seed-mean damage is non-decreasing across the five bins, allowing a dip between adjacent bins of at most 0.02, and damage in the top bin exceeds the bottom bin by at least 0.5. Partial: one dip larger than 0.02, with the ends still ordered. We prescribe no shape between the ends. The D2.1 alignment response was sigmoid in redness rather than linear, so the grading may be a step rather than the smooth $\cos^2$ curve M1 saw.
- **H4.** The write on non-red lines stays within the bound the clean geometry sets, at every slice. At a given (slice, position) site, the bound is $\arcsin$ of the 99th-percentile clean alignment over the 1,689 non-red lines there. We take that per run from the un-intervened map, then average over the nine seeds. The measurement is the 99th-percentile write over the same lines under the primary intervention, again per run and then seed-averaged (each quantile is the 17th-largest of 1,689 values within a run; averaging is all the seeds do). At the embedding the two are equal by construction, and the pipeline asserts it. At each of the four deeper slices, and at every position, the measured write is at most 1.25 times the bound. Partial: one site over by no more than a factor of two. Contrary: a block re-writes the axis on non-red lines after it was removed upstream. That would be a bypass, and it would bring the question of the layer sweep forward.
- **H5.** Permanent removal is selective, given the anti-subspace term. Under the `ablate` arm, the H1 and H2 gates both hold: accuracy on red lines at or below 0.5, and non-red accuracy within 0.02 of clean. Partial: one of the two holds. M1 found ablation selective only when repulsion had cleared the axis, and the D2.1 recipe has that term, so this is the transformer form of M1's headline result.

    Ablation and the projection at every slice give nearly the same forward pass: the reads see no e₁ either way, so the two differ only in the gain applied to the output of a block before it joins the stream. The seed scatter under `ablate` should therefore match the scatter under the primary intervention. On red lines both scatter because these checkpoints have no fallback term, so the response there is untrained. That is why every gate reads a seed mean and every table shows the seed range.
"""

results_rows = {
    "primary": ("anchored", "primary"),
    "operands": ("anchored", "operands"),
    "embedding": ("anchored", "embedding"),
    "last": ("anchored", "last"),
    "shaped": ("anchored", "shaped"),
    "ablate": ("anchored", "ablate"),
    "control": ("un-anchored", "primary"),
}


def results_row(title: str, cond: str, iv: str) -> str:
    c = ex.PRIMARY.name if cond == "anchored" else ex.CONTROL.name
    red, nonred = stat(iv, "acc", "red", c), stat(iv, "acc", "nonred", c)
    deficit = clean("acc", "nonred", c) - nonred
    b = lambda v, ok: f"<b>{v}</b>" if ok else v  # noqa: E731
    return (
        f"<tr><td>{title}, <code>{iv}</code></td>"
        f"<td class='num'>{clean('acc', 'red', c).mean():.3f}</td>"
        f"<td class='num'>{b(span2(red, '.2f'), red.mean() <= ex.RED_ACC_GATE)}</td>"
        f"<td class='num'>{span2(stat(iv, 'damage', 'red', c), '.2f')}</td>"
        f"<td class='num'>{span2(nonred)}</td>"
        f"<td class='num'>{b(f'{deficit.mean():.3f}', deficit.mean() <= ex.TASK_GATE)}</td>"
        f"<td class='num'>{span2(stat(iv, 'damage', 'nonred', c))}</td>"
        f"<td class='num'>{stat(iv, 'offvocab', 'red', c).mean():.3f}</td></tr>"
    )


results_table = f"""
<div class="report-table-scroll"><table class="report-table">
  <thead><tr>
    <th>condition, intervention</th>
    <th class="num">clean red acc</th>
    <th class="num">red acc ↓</th>
    <th class="num">red damage ↑</th>
    <th class="num">non-red acc ↑</th>
    <th class="num">non-red deficit ↓</th>
    <th class="num">non-red damage ↓</th>
    <th class="num">off-vocab, red</th>
  </tr></thead>
  <tbody>{"".join(results_row(t, c, iv) for t, (c, iv) in results_rows.items())}</tbody>
</table></div>
"""
results_caption = f"""
Removal and selectivity by arm. Each value is the seed mean, with the seed range in parentheses. Accuracy is exact match on the answer token; damage is the clean answer probability minus the intervened one, in probability units; the deficit is clean accuracy minus intervened accuracy on non-red lines. Off-vocab is the mass outside the color vocabulary on red lines. Bold marks a value inside its gate: red accuracy at or below {ex.RED_ACC_GATE:g} (H1, and H5 for <code>ablate</code>), non-red deficit at or below {ex.TASK_GATE:g} (H2, H5). The arms and the un-anchored row have no gate and are shown for the comparisons the sections below read.
"""

figure_html(results_table, caption=results_caption, class_="report-figure")

r"""
## Suppression bites (H1)
"""

h1_p = per_line("primary", "p_ans")[:, lines.red]  # (seeds, red lines)
h1_red = stat("primary", "acc", "red")


@memo
@themed(
    name="red-response",
    alt_text="""
        A histogram of the probability on the correct answer over the red lines of all nine seeds, stacked by seed. Nearly all the mass is in the leftmost bin, near zero, with a thin tail reaching toward one.
    """,
    caption="""
        **The response on red lines.** The intervened probability on the correct answer over the 365 red lines, as a share of all (seed, line) pairs, with the nine seeds stacked in alternating shades. A bimodal response would show as mass at both ends.
    """,
)
def red_response_plot() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 2.2), layout="constrained")
    shades = [light_dark("#d40000", "#f44"), light_dark("#f08080", "#a03030")]
    ax.hist(
        list(h1_p),
        bins=20,
        range=(0, 1),
        stacked=True,
        weights=[np.full(h1_p.shape[1], 1 / h1_p.size)] * len(h1_p),
        color=[shades[s % 2] for s in range(len(h1_p))],
        lw=0,
    )
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("p(correct answer) under intervention", fontsize=8)
    ax.set_ylabel("share of (seed, line) pairs", fontsize=8)
    ax.tick_params(labelsize=7)
    return fig


rf"""
Under the primary intervention the seed-mean accuracy on red lines falls to {h1_red.mean():.2f}, from a clean accuracy of {clean("acc", "red").mean():.3f}. The gate is {ex.RED_ACC_GATE:g}, and the seed that keeps the most red lines keeps {h1_red.max():.2f} of them. The un-anchored condition under the same operator loses nothing ({stat("primary", "acc", "red", ex.CONTROL.name).mean():.3f}): the operator removes the concept where one was placed, and finds nothing to remove where none was.

Read line by line, the removal is close to complete on nearly every red line rather than an average over lines lost and lines kept. The median intervened probability on the correct answer is {np.median(h1_p):.3f} across seeds and lines, and {np.mean(h1_p < 0.1):.0%} of (seed, line) pairs fall below 0.1.

{red_response_plot()}

**H1 holds.** The contrary case was that the model would read the color of the operand from what is left of the embedding off the axis. It does not: with the axis gone, the model no longer knows the operand was red.

## Selectivity (H2)
"""

h2_deficit = clean("acc", "nonred") - stat("primary", "acc", "nonred")
h2_worst_seed = int(np.argmax(h2_deficit))
h2_ctrl = clean("acc", "nonred", ex.CONTROL.name) - stat("primary", "acc", "nonred", ex.CONTROL.name)
h2_ops = clean("acc", "nonred") - stat("operands", "acc", "nonred")
h2_shaped = clean("acc", "nonred") - stat("shaped", "acc", "nonred")
# REVIEW: the H2 diagnosis sentence said both arms "remove red as well". The `shaped` arm's red
# accuracy is 0.60, above the H1 gate, so only `operands` removes red at the primary's level; the
# sentence now gives both numbers. Verify: `stat("shaped", "acc", "red")` against RED_ACC_GATE.
h2_floor = floor("damage", "nonred")
h2_gap = stat("primary", "damage", "nonred").mean() - stat("operands", "damage", "nonred").mean()
h2_dmg = {
    c.title: (per_line("clean", "p_ans", c.name) - per_line("primary", "p_ans", c.name)).mean(0)
    for c in (ex.PRIMARY, ex.CONTROL)
}
h2_redder = lines.line_colors[np.arange(lines.n), lines.red_operand]
h2_rgb = GRID_RGB[h2_redder]
h2_stacks = {title: quantile_stack(y, h2_redder) for title, y in h2_dmg.items()}
# Line redness is a float computation, so equal levels differ in the last bits; round before grouping.
h2_dose = np.round(dose, 6)
h2_levels = np.unique(h2_dose)
assert len(h2_levels) == 29, len(h2_levels)
h2_means = {title: np.array([y[h2_dose == d].mean() for d in h2_levels]) for title, y in h2_dmg.items()}
# The dither gives a rare line its fair share of samples, which is nearly none, so the least and most
# damaged line at each redness level are drawn as opaque dots in the color of their redder operand: the
# extremes are the evidence H2 reads. Per condition, (min, max) line indices interleaved per level.
h2_at = [np.flatnonzero(h2_dose == d) for d in h2_levels]
h2_extremes = {
    title: np.array([i[f(y[i])] for i in h2_at for f in (np.argmin, np.argmax)]) for title, y in h2_dmg.items()
}


@memo
@themed(
    name="damage-by-redness",
    alt_text="""
        Two grading clouds of damage against line redness, with a mean line, and at each redness level a thin rule from the least to the most damaged line, capped by small dots. Anchored: the cloud sits on the floor up to a redness of about 0.6, then rises steeply through the 0.6 to 0.8 band to the top of the panel. At redness 0.7 the rule spans the whole panel. Un-anchored: the cloud sits on the floor throughout, with a few rules reaching up to isolated lines at about 0.3.
    """,
    caption="""
        **Damage against line redness.** Seed-mean damage under the primary intervention, per line, on the anchored condition (left) and the un-anchored one (right). The cloud is the distribution of damage over the lines at each redness level, dithered and interpolated between levels; the line is the mean, and each thin vertical rule spans the least to the most damaged line at that level, with a dot in the color of that line's redder operand. The vertical rules are the H3 bin edges.
    """,
)
def damage_by_redness_plot() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharex=True, sharey=True, layout="constrained")
    axes = cast(AxesRow, axes)
    halo = [withStroke(linewidth=3, foreground=light_dark("#fff2", "#0002"))]
    for ax, (title, y) in zip(axes, h2_dmg.items(), strict=True):
        for e in ex.DOSE_EDGES:
            ax.axvline(e, color=light_dark("#0002", "#fff2"), lw=0.6, zorder=0)
        GradingCloud(ax, h2_stacks[title], (-0.05, 1), k=45, dpr=2, lerp=True, clip_on=False)
        ax.plot(h2_levels, h2_means[title], lw=0.5, c=light_dark("#000", "#fff"), zorder=3, path_effects=halo)
        i = h2_extremes[title]
        ax.vlines(h2_levels, y[i[0::2]], y[i[1::2]], lw=0.2, color=light_dark("#0006", "#fff6"), zorder=3)
        ax.scatter(h2_dose[i], y[i], s=2, c=h2_rgb[i], edgecolors=light_dark("#000", "#fff"), lw=0.15, zorder=3)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("line redness", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("damage, seed mean", fontsize=8)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(-0.05, 1)
    return fig


rf"""
The seed-mean drop in accuracy on non-red lines is {h2_deficit.mean():.3f}: outside the {ex.TASK_GATE:g} gate, inside the {ex.TASK_PARTIAL:g} partial gate. The spread across seeds is wide. Seed {h2_worst_seed} loses {h2_deficit[h2_worst_seed]:.3f} of its non-red accuracy and the median seed loses {np.median(h2_deficit):.3f}, with {int((h2_deficit <= ex.TASK_GATE).sum())} of the nine seeds inside the gate on their own. The un-anchored condition under the same operator loses {h2_ctrl.mean():.3f}, so the operator itself is close to free. The loss comes from what the anchored blocks do with an edited syntax state.

The contrary clause of H2 predicted where the damage would come from, and the arms agree. The `operands` arm, which leaves the four syntax positions alone, loses {h2_ops.mean():.3f}. The `shaped` arm, whose threshold sits above the constant component of the syntax embeddings, loses {h2_shaped.mean():.3f}. The `operands` arm still removes red (red accuracy {stat("operands", "acc", "red").mean():.2f}); the `shaped` arm removes only part of it ({stat("shaped", "acc", "red").mean():.2f}), trading some of the removal for the extra selectivity (E4).

So editing the syntax positions is what costs us the non-red lines. Those embeddings have a constant component on the axis, and the blocks read that component as part of the syntax rather than as redness. In damage terms the primary intervention and the `operands` arm differ by {h2_gap:.3f}, against a resolution floor of {h2_floor:.3f}, so at the seed level they are not separated either way. The accuracy read above is the one the gate scores.

{damage_by_redness_plot()}

**H2 is partial.** The removal is selective at the operand positions, and the loss on non-red lines is a cost of editing the whole stream, concentrated in a minority of seeds. What that means for the bound is read in the H4 section.

## Grading (H3)
"""

h3_series = {
    "primary": ("primary", bins("primary")),
    "operands": ("operands", bins("operands")),
    "shaped": ("shaped", bins("shaped")),
    "ablate": ("ablate", bins("ablate")),
    "control": ("un-anchored", bins("primary", ex.CONTROL.name)),
}
h3_b = h3_series["primary"][1].mean(0)
h3_dip = float(max(0.0, (h3_b[:-1] - h3_b[1:]).max()))
h3_lb = h3_series["shaped"][1].mean(0)


@memo
@themed(
    name="damage-by-bin",
    alt_text="""
        A step chart of seed-mean damage over five redness bins, with seed-range bands. The primary, operands, and ablate arms sit near zero for the three lowest bins, rise to about 0.2 in the 0.6 to 0.8 bin, and reach about 0.9 in the top bin. The shaped arm stays at zero until the top bin, where it reaches about 0.4 with a wide band. The un-anchored condition is flat at zero.
    """,
    caption=f"""
        **Damage per redness bin.** Seed-mean damage in each of the five redness bins, with the band spanning the seed range, for the primary intervention, three arms, and the un-anchored condition under the primary operator. H3 asks the primary line to be non-decreasing within a dip of {ex.GRADE_DIP:g} and to span at least {ex.GRADE_SPAN:g} from the first bin to the last.
    """,
)
def damage_by_bin_plot() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 2.8), layout="constrained")
    x = np.arange(len(BIN_NAMES))
    for key, (_title, v) in h3_series.items():
        ink = light_dark(*ARM_INK[key])
        ax.fill_between(x, v.min(0), v.max(0), color=ink, alpha=0.15, lw=0, zorder=0)
    for key, (title, v) in h3_series.items():
        ink = light_dark(*ARM_INK[key])
        ax.plot(x, v.mean(0), color=ink, marker="o", ms=3, lw=1.3, label=title)
    ax.set_xticks(x, BIN_NAMES)
    ax.set_xlabel("line redness", fontsize=8)
    ax.set_ylabel("damage, seed mean", fontsize=8)
    ax.set_ylim(-0.05, 1)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    return fig


rf"""
Seed-mean damage across the five bins under the primary intervention is {", ".join(f"{v:.3f}" for v in h3_b)}. There is one dip, {h3_dip:.3f} between the first two bins, inside the {ex.GRADE_DIP:g} allowance. It is the share of the H2 deficit that falls in the lowest bin rather than anything about the grading. The top bin exceeds the bottom by {h3_b[-1] - h3_b[0]:.2f}, against a gate of {ex.GRADE_SPAN:g}.

The shape is a step rather than a ramp: the three lowest bins sit on the floor, the 0.6–0.8 bin takes {h3_b[3]:.2f}, and the red bin {h3_b[4]:.2f}. That follows the S-shaped alignment response D2.1 measured, where the alignment of a color rises steeply only near the red corner. The `shaped` arm sharpens the step further, to {h3_lb[3]:.3f} in the 0.6–0.8 bin and {h3_lb[4]:.2f} in the red bin, since its threshold leaves every state below an alignment of {ex.SHAPED["a"]:g} alone. The un-anchored condition is flat at {h3_series["control"][1].mean():.3f}.

{damage_by_bin_plot()}

**H3 holds.** The contrary case was a middle bin damaged more than the bin above it, and no such inversion appears. The per-color view is in E3.
"""


def draw_grading_grid(ax: plt.Axes, a: np.ndarray, *, row_names: bool, scale: bool) -> None:
    """One row per slice, one grading-cloud slot per position, keyed by the op1 color; the overlay is pure red."""
    sw = 0.7
    red_c, grey_c = light_dark("#d40000", "#f44"), light_dark("#eee", "#111")
    xs = np.arange(len(POS_NAMES))
    for si in range(len(SLICE_NAMES)):
        shift = Affine2D().translate(0, si) + ax.transData
        reds = a[si, I_RED, :]
        smooth_step_area(ax, xs, reds, ramp=1 - sw, color=grey_c, transform=shift, zorder=0, fillet=3)
        for pi in range(len(POS_NAMES)):
            GradingCloud(ax, a[si, :, pi], span=(pi - sw / 2, pi + sw / 2), transform=shift, clip_on=False)
        smooth_step_marks(ax, xs, reds, ramp=1 - sw, color=red_c, lw=1.3, transform=shift, clip_on=False, fillet=3)
        ax.axhline(si, color=light_dark("#aaaa", "#333a"), lw=0.5)
    ax.set_xlim(-0.5, len(POS_NAMES) - 0.5)
    ax.set_ylim(0, len(SLICE_NAMES))
    ax.set_xticks(xs, POS_NAMES)
    ax.set_yticks(np.arange(len(SLICE_NAMES)) + 0.5, SLICE_NAMES if row_names else [""] * len(SLICE_NAMES))
    ax.tick_params(axis="x", labelsize=6, bottom=False)
    ax.tick_params(axis="y", labelsize=7, left=False)
    if scale:
        sec = ax.secondary_yaxis("right")
        sec.set_yticks(np.arange(len(SLICE_NAMES) + 1), ["0", "1"] + [""] * (len(SLICE_NAMES) - 1))
        sec.tick_params(labelsize=6, direction="out")
        sec.set_ylabel("α, 0–1 per row", fontsize=7)


r"""
## The write bound (H4)
"""

from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

h4_clean_q = np.array([r["clean"]["alpha_q99_nonred"] for r in runs[ex.PRIMARY.name]], float).mean(0)
h4_arr_q = stat("primary", "q99_alpha_nonred").mean(0)
h4_ratio = ratio_map("primary")
h4_seed_ratio = (
    stat("primary", "q99_write_nonred")[:, 1:] / np.array([r["clean"]["bound"] for r in runs[ex.PRIMARY.name]])[:, 1:]
)  # (seeds, deeper slices, positions)
h4_per_seed = h4_seed_ratio.max(axis=(1, 2))
h4_seed_sites = sorted({POS_NAMES[int(np.argmax(r.max(0)))] for r in h4_seed_ratio})
h4_seed_before = h4_seed_ratio[:, :, : ex.ANSWER_POS].max(axis=(1, 2))
h4_over = [(s, p) for s in range(1, len(SLICE_NAMES)) for p in range(len(POS_NAMES)) if h4_ratio[s, p] > 1]
h4_before = h4_ratio[1:, : ex.ANSWER_POS].max()


def h4_by_color(a: np.ndarray) -> np.ndarray:
    """(slices, colors, positions) seed-mean alignment keyed by the op1 color, from (seeds, slices, lines, positions)."""
    n_colors = len(GRID_RGB)
    return a.reshape(len(a), len(SLICE_NAMES), n_colors, lines.n // n_colors, lines.n_pos).mean(axis=(0, 3))


h4_clouds = {
    "clean": h4_by_color(per_line("clean", "alpha")),
    "arriving, primary": h4_by_color(per_line("primary", "alpha_pre")),
    "arriving, `embedding` arm": h4_by_color(per_line("embedding", "alpha_pre")),
}
h4_red_return = h4_clouds["arriving, `embedding` arm"][1:, I_RED, 0]


@memo
@themed(
    name="write-bound-maps",
    alt_text="""
        Two line charts over the six token positions, one line per slice in shades from light to dark. Left, the clean 99th-percentile non-red alignment: between 0.15 and 0.42 at the prompt positions, smaller at the answer and newline. Right, the same under the primary intervention: the embedding line is identical, and the four deeper lines sit at or below their clean values at the prompt positions.
    """,
    caption=f"""
        **The bound and the write, per site.** The {ex.WRITE_QUANTILE}th-percentile alignment over the non-red lines at each (slice, position), seed mean, on the shared 0–1 scale. Left, the clean map, whose arcsine is the bound; right, the alignment arriving at the operator under the primary intervention, whose arcsine is the write. The embedding line is the same in both by construction. Slices run from the embedding (lightest) to the last block (darkest).
    """,
)
def write_bound_maps_plot() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.4), sharex=True, sharey=True, layout="constrained")
    axes = cast(AxesRow, axes)
    cmap = LinearSegmentedColormap.from_list(
        "depth", [light_dark("#f6b0b0", "#5a1a1a"), light_dark("#8a0000", "#ff7070")]
    )
    x = np.arange(len(POS_NAMES))
    for ax, (title, m) in zip(
        axes, [("clean (the bound)", h4_clean_q), ("arriving under intervention", h4_arr_q)], strict=True
    ):
        for s, name in enumerate(SLICE_NAMES):
            ink = cmap(s / (len(SLICE_NAMES) - 1))
            smooth_step(ax, x, m[s], breaks=False, ramp=0.3, color=ink, lw=0.7, fillet=3)
            smooth_step(ax, x, m[s], breaks=True, ramp=0.3, color=ink, lw=1.4)
            ax.plot([], [], color=ink, lw=1.2, label=name)  # one legend entry per slice
        ax.set_title(title, fontsize=8)
        ax.set_xticks(x, POS_NAMES)
        ax.tick_params(labelsize=7)
    axes[0].set_xlim(-0.5, len(POS_NAMES) - 0.5)
    axes[0].set_ylim(0, 0.5)
    axes[0].set_ylabel(f"|α|, {ex.WRITE_QUANTILE}th pct of non-red", fontsize=8)
    axes[1].legend(fontsize=6, frameon=False, title="slice", title_fontsize=6, ncol=5)
    return fig


@memo
@themed(
    name="arriving-clouds",
    alt_text="""
        Three grading grids, each five rows of six cloud slots. Clean: the op1 slot rises steeply to pure red at every slice, and the syntax slots show a flat band. Arriving under the primary intervention: the embedding row is unchanged and every deeper row is flat near zero at every slot, the red overlay included. Arriving under the embedding arm: the deeper rows show pure red climbing back part way at op1 and op2, and dipping below zero at the equals sign.
    """,
    caption="""
        **The arriving alignment per color, at every site.** Each panel is one row per slice (embedding at the bottom) and one grading-cloud slot per position, each cloud the alignment of the states whose op1 color is the cloud's color, against that color's redness; the red line traces pure red across the positions. Left, the clean map ex-2.1.10 published; middle, the alignment arriving at each operator under the primary intervention; right, the same under the `embedding` arm, where only the token embedding was edited and the blocks run un-edited after it.
    """,
)
def arriving_clouds_plot() -> plt.Figure:
    fig, axes = plt.subplots(1, len(h4_clouds), figsize=(7.4, 2.4), squeeze=False)
    row = cast(AxesRow, axes[0])
    for i, (ax, (title, a)) in enumerate(zip(row, h4_clouds.items(), strict=True)):
        draw_grading_grid(ax, a, row_names=i == 0, scale=i == len(row) - 1)
        ax.set_title(title, fontsize=8, pad=6)
    return fig


def h4_bold(v: float) -> str:
    return f"<b>{v:.2f}</b>" if v <= ex.WRITE_TOLERANCE else f"{v:.2f}"


h4_rows = "".join(
    f"<tr><td>{SLICE_NAMES[s]}</td>"
    + "".join(f"<td class='num'>{h4_bold(h4_ratio[s, p])}</td>" for p in range(len(POS_NAMES)))
    + "</tr>"
    for s in range(len(SLICE_NAMES))
)
h4_table = f"""
<div class="report-table-scroll"><table class="report-table">
  <thead><tr><th>slice</th>{"".join(f"<th class='num'>{p}</th>" for p in POS_NAMES)}</tr></thead>
  <tbody>{h4_rows}</tbody>
</table></div>
"""
h4_caption = f"""
The write over the bound, per site: the seed-mean {ex.WRITE_QUANTILE}th-percentile non-red write under the primary intervention divided by the seed-mean bound. The embedding row is the identity check. Bold marks a ratio at or below the H4 tolerance of {ex.WRITE_TOLERANCE:g}.
"""

rf"""
At the four deeper slices, the seed-mean write on non-red lines is at most {h4_ratio[1:].max():.2f} times the bound. The sites over one are {", ".join(f"(slice {s}, `{POS_NAMES[p]}`)" for s, p in h4_over)}. The largest are at the newline, where the bound is smallest ({bound_map()[1:, -1].min():.3f} to {bound_map()[1:, -1].max():.3f} radians) and where nothing downstream is read for the answer. At every site before the answer the ratio is at most {h4_before:.2f}, so the blocks do not put the axis back on non-red lines.

Per seed, the largest deeper-site ratio runs from {h4_per_seed.min():.2f} to {h4_per_seed.max():.2f}, at position {" or ".join(f"`{p}`" for p in h4_seed_sites)}. At the sites before the answer it runs from {h4_seed_before.min():.2f} to {h4_seed_before.max():.2f}, and {int((h4_seed_before > ex.WRITE_TOLERANCE).sum())} of the nine seeds are above the tolerance on their own. The gate reads the seed mean, as the hypothesis states it; the per-seed spread is the number the layer sweep starts from.

{write_bound_maps_plot()}

{figure_html(h4_table, caption=h4_caption, class_="report-figure")}

The per-color view shows the same thing color by color. Under the primary intervention every row after the embedding is flat: the red colors arrive at the deeper operators with an alignment of {h4_clouds["arriving, primary"][1:, I_RED, 0].max():.2f} or less at op1, down from {h4_clouds["clean"][1:, I_RED, 0].min():.2f} or more in the clean pass, and the non-red bulk is where it was. Under the `embedding` arm, where the blocks run un-edited after the token edit, pure red climbs back to {h4_red_return.max():.2f} at op1 by the last slice and is written below zero at `=`. So when nothing stops them, the blocks re-derive part of the axis from an off-axis input; the later slices of the primary intervention are what keep it off.

{arriving_clouds_plot()}

**H4 holds.** The bound is intact at every site before the answer. A bypass would show as the axis coming back on non-red lines at a deeper slice, as it does under the `embedding` arm; under the primary intervention no slice puts it back, so the blocks do not re-derive the concept.

The H2 deficit is a different thing. A rotation of the syntax states can stay inside the bound and still be an input the blocks never met in training, and what they make of that input can reach the answer (E5).<!-- REVIEW: this paragraph called the H2 deficit "the amplification case the hypothesis names". H4 names a bypass, not amplification, and E5 finds no arm whose readout displacement exceeds its summed write, so the deficit is now described as the downstream effect of a bounded write. Verify: E5's ratio is at most 0.21 on every seed under the primary intervention, and reaches 1.0 on one seed under the `embedding` arm and no higher. -->

## Permanent removal (H5)
"""

h5_red, h5_pred = stat("ablate", "acc", "red"), stat("primary", "acc", "red")
h5_def = clean("acc", "nonred") - stat("ablate", "acc", "nonred")
h5_pdef = clean("acc", "nonred") - stat("primary", "acc", "nonred")
h5_gap = stat("primary", "damage", "nonred").mean() - stat("ablate", "damage", "nonred").mean()

rf"""
Under the `ablate` arm, accuracy on red lines is {span(h5_red, ".2f")}, against {span(h5_pred, ".2f")} under the projection. The drop on non-red lines is {h5_def.mean():.3f}, against {h5_pdef.mean():.3f}. Both H5 gates hold: red accuracy at or below {ex.RED_ACC_GATE:g}, non-red drop at or below {ex.TASK_GATE:g}. Ablation and projection behave alike on red lines, as the hypothesis said they should.

On non-red lines ablation costs less than the projection at every slice, {stat("ablate", "damage", "nonred").mean():.3f} in damage against {stat("primary", "damage", "nonred").mean():.3f}. The gap of {h5_gap:.3f} is inside the resolution floor of {floor("damage", "nonred"):.3f}, so the two are not separated at the seed level. The seed that loses most under the projection loses {h5_def[int(np.argmax(h5_pdef))]:.3f} under ablation. Two things may explain the smaller cost. A permanent removal has no re-normalization gain to apply per slice. And once the token embeddings have lost their constant component on the axis, the syntax positions have none either, which is where H2 placed the side-effect.

**H5 holds.** This is the transformer form of the headline result from M1: with the anti-subspace term in the recipe, zeroing the weights that read and write the axis removes the concept and little else.

## The undesigned response
"""

undes_comp = np.array([r["interventions"]["primary"]["composition_red"] for r in runs[ex.PRIMARY.name]], float)
undes_frac = undes_comp / undes_comp.sum(1, keepdims=True)
undes_mean = undes_frac.mean(0)
undes_agree = metrics["agreement_red"]
# Is a miss the mix of the visible operand with some grid color less red than the red operand? Mixing is
# a channel-wise round-half-up mean on the 16-level cube (`sca.data.colors.mix`); the grid is its 6-level
# sub-grid, so a guess is "explained" when some grid color mixes with the visible operand to give it.
undes_rows = np.arange(lines.n)[lines.red]
undes_lv = np.rint(GRID_RGB * 15).astype(int)
undes_redness = GRID_RGB[:, 0] * (1 - GRID_RGB[:, 1] / 2 - GRID_RGB[:, 2] / 2)
undes_vis = lines.line_colors[undes_rows, 2 - lines.red_operand[undes_rows]]
undes_redop = lines.line_colors[undes_rows, lines.red_operand[undes_rows]]
undes_ans = lines.line_colors[undes_rows, ex.ANSWER_POS]
undes_mixes = (undes_lv[undes_vis][:, None, :] + undes_lv[None, :, :] + 1) // 2  # (red lines, candidate X, channel)
undes_wrong = undes_less = undes_explained = 0
for r in runs[ex.PRIMARY.name]:
    g = lines.tok2color[arrays[f"{r['label']}/primary/guess"]][undes_rows]
    miss = (g >= 0) & (g != undes_ans)
    hits = (undes_mixes == undes_lv[np.maximum(g, 0)][:, None, :]).all(2) & (
        undes_redness[None, :] < undes_redness[undes_redop][:, None] - 1e-9
    )
    undes_wrong += int(miss.sum())
    undes_less += int((miss & (undes_redness[np.maximum(g, 0)] < undes_redness[undes_ans] - 1e-9)).sum())
    undes_explained += int((miss & hits.any(1)).sum())
undes_order = ["true", "neighbor", "visible_operand", "red_operand", "other"]
undes_names = {
    "true": "true answer",
    "neighbor": "one-step neighbor",
    "visible_operand": "visible operand",
    "red_operand": "red operand",
    "other": "other",
}


@memo
@themed(
    name="undesigned-composition",
    alt_text="""
        Nine stacked horizontal bars, one per seed, each splitting the red lines into what they decode to. In every seed the largest share is a one-step neighbor of the true answer, about half; the true answer and the visible operand take about a tenth each; the rest is other colors. The red operand never appears.
    """,
    caption="""
        **What red lines decode to, per seed.** Each bar splits the 365 red lines under the primary intervention by the decoded answer. Greys are the true mix (dark) and its one-step neighbors on the grid (light); blue is the visible operand, red the red operand itself, and the pale segment every other color.
    """,
)
def undesigned_composition_plot() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 2.4), layout="constrained")
    shades = [
        light_dark(c, d)
        for c, d in [
            ("#222", "#eee"),
            ("#999", "#777"),
            ARM_INK["ablate"],
            ARM_INK["primary"],
            ("#eee", "#2a2a2a"),
        ]
    ]
    left = np.zeros(len(undes_frac))
    for j, k in enumerate(undes_order):
        v = undes_frac[:, ex.COMPOSITION.index(k)]
        ax.barh(np.arange(len(undes_frac)), v, left=left, color=shades[j], label=undes_names[k], height=0.7, lw=0)
        left += v
    ax.set_yticks(np.arange(len(undes_frac)), [f"seed {s}" for s in range(len(undes_frac))])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.tick_params(labelsize=7)
    ax.set_xlabel("share of red lines", fontsize=8)
    ax.legend(fontsize=6, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    return fig


rf"""
No gate. With the axis gone, a red line still decodes to a color: the mass outside the color vocabulary is {stat("primary", "offvocab", "red").mean():.3f} (seed range {stat("primary", "offvocab", "red").min():.3f}–{stat("primary", "offvocab", "red").max():.3f}). On average the decoded answer sits {stat("primary", "guess_dist_red").mean():.2f} unit-cube units from the true mix, where one grid step is {1 / 5:.1f}.

We expected the answer to sit nearer the visible operand than the true mix. It does not. Across the nine seeds, {undes_mean[ex.COMPOSITION.index("neighbor")]:.0%} of red lines decode to a one-step neighbor of the true answer, {undes_mean[ex.COMPOSITION.index("true")]:.0%} to the true answer itself, {undes_mean[ex.COMPOSITION.index("visible_operand")]:.0%} to the visible operand, and {undes_mean[ex.COMPOSITION.index("other")]:.0%} to some other color. The red operand is never returned.

A neighbor is still a miss. Accuracy is exact match, and on those lines the probability on the true answer falls to near zero (H1), so in probability terms the damage on red lines is close to complete. In color terms it is small.

The misses are not random colors. Of the {undes_wrong:,} (seed, line) pairs that decode to a wrong grid color, {undes_less / undes_wrong:.1%} are less red than the true answer,[^lessred] and {undes_explained / undes_wrong:.1%} are what the line would mix to if the red operand were replaced by some grid color less red than it. That holds whether the red operand is pure red or one of the four other colors of the red group.

So the model keeps mixing, and reads the red operand as a paler or duller color than it is. That response is better behaved than the spoofing we saw in M1, and it is the reference the fallback experiment has to improve on.

[^lessred]: By the redness label, $r(1 - g/2 - b/2)$: less red in the red channel, or more green or blue.

Which neighbor a line lands on varies from seed to seed. Under the primary intervention, at least {ex.AGREE_SEEDS} of the nine seeds decode the same answer on {undes_agree["primary"]:.0%} of red lines; the figure is {undes_agree["operands"]:.0%} under the `operands` arm and {undes_agree["ablate"]:.0%} under `ablate`. The `shaped` arm, which removes only part of the red state, agrees on {undes_agree["shaped"]:.0%}, because most of its lines still decode to the true answer.

{undesigned_composition_plot()}

## The post-hoc tier
"""

posthoc_fits = [f for f in ex.POSTHOC if f != "axis"]
POSTHOC_CTRL = ex.CONTROL.name


def posthoc_cell(iv: str) -> tuple[float, float, float]:
    return (
        float(stat(iv, "damage", "red", POSTHOC_CTRL).mean()),
        float(stat(iv, "damage", "nonred", POSTHOC_CTRL).mean()),
        float(stat(iv, "acc", "red", POSTHOC_CTRL).mean()),
    )


posthoc_tier = {
    site: {(f, s): posthoc_cell(f"{f}@{s}:{site}") for f in ex.POSTHOC for s in ex.SLICES} for site in ex.POSTHOC_SITES
}
posthoc_anch = {
    arm: (float(stat(arm, "damage", "red").mean()), float(stat(arm, "damage", "nonred").mean()))
    for arm in ("primary", "operands", "shaped", "ablate")
}
posthoc_fit_ink = {
    "diff-in-means": ("#2060c0", "#7af"),
    "probe": ("#209060", "#6d9"),
    "leace": ("#8030c0", "#c8f"),
    "axis": ("#555", "#aaa"),
}


@memo
@themed(
    name="posthoc-tradeoff",
    alt_text="""
        A scatter of red damage against non-red damage. The anchored arms sit at the top left: red damage near 0.9 with non-red damage near zero. The post-hoc marks on the un-anchored model lie along a diagonal band: those that remove most red also remove half or more of the non-red response, and those with little non-red cost remove little red. The axis on the un-anchored model sits at the origin.
    """,
    caption="""
        **Red damage bought per unit of non-red damage.** One mark per (fit, slice, site) on the un-anchored condition, seed mean: circles at the operand site, squares at the decode site, labelled by slice, colored by fit. The stars are the anchored arms under the placed axis. A selective removal sits high and to the left. The dashed line is equal damage on both groups; a mark below it would remove more from non-red lines than from red ones, and none does.
    """,
)
def posthoc_tradeoff_plot() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=0.7, color=light_dark("#0004", "#fff4"), zorder=0)
    markers = {"operands": "o", "decode": ">"}
    for site, cells in posthoc_tier.items():
        for (f, s), (r, n, _) in cells.items():
            ink = light_dark(*posthoc_fit_ink[f])
            ax.scatter(
                n,
                r,
                marker=markers[site],
                s=28,
                color=ink,
                edgecolors="none",
                alpha=0.85,
                label=f"{f}, {site}" if s == 0 else None,
            )
            if r > 0.05 or n > 0.05:
                ax.annotate(SLICE_NAMES[s], (n, r), xytext=(3, 2), textcoords="offset points", fontsize=6, color=ink)
    offsets = {"primary": (8, 6), "operands": (8, -4), "ablate": (8, -14), "shaped": (8, -3)}
    for i, (arm, (r, n)) in enumerate(posthoc_anch.items()):
        star = light_dark("#d40000", "#f44")
        ax.scatter(
            n, r, marker="*", s=70, color=star, edgecolors="none", zorder=3, label="anchored arms, e₁" if i == 0 else None
        )  # fmt: skip
        ax.annotate(arm, (n, r), xytext=offsets[arm], textcoords="offset points", fontsize=6)
    ax.set_xlim(-0.02, 1)
    ax.set_ylim(-0.02, 1)
    ax.set_xlabel("non-red damage (want low)", fontsize=8)
    ax.set_ylabel("red damage (want high)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    return fig


def posthoc_table(site: str) -> str:
    rows = "".join(
        f"<tr><td>{f}</td>"
        + "".join(
            f"<td class='num'>{posthoc_tier[site][f, s][0]:.2f} / {posthoc_tier[site][f, s][1]:.2f}</td>"
            for s in ex.SLICES
        )
        + "</tr>"
        for f in ex.POSTHOC
    )
    return f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr><th>fit, at {site}</th>{"".join(f"<th class='num'>{SLICE_NAMES[s]}</th>" for s in ex.SLICES)}</tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    """


posthoc_best_ops = max(((f, s) for f in posthoc_fits for s in ex.SLICES), key=lambda k: posthoc_tier["operands"][k][0])  # fmt: skip
posthoc_best_dec = max(((f, s) for f in posthoc_fits for s in ex.SLICES), key=lambda k: posthoc_tier["decode"][k][0])  # fmt: skip
posthoc_r_o, posthoc_n_o, _ = posthoc_tier["operands"][posthoc_best_ops]
posthoc_r_d, posthoc_n_d, _ = posthoc_tier["decode"][posthoc_best_dec]
posthoc_p_r, posthoc_p_n = posthoc_anch["primary"]
posthoc_probe0 = posthoc_tier["operands"]["probe", 0]
# REVIEW: the closing sentence of this paragraph said no fitted direction came "within an order of
# magnitude of the placed axis on red damage per unit of non-red damage". As a ratio that is not what
# the table shows: `probe@1:operands` reaches 121 against the primary's 26, on a red damage of 0.09.
# The claim now reads on the two ends separately. Verify: `posthoc_strong` below, against `posthoc_anch["primary"]`.
posthoc_strong = min(n for cells in posthoc_tier.values() for (r, n, _) in cells.values() if r >= 0.5)

rf"""
No gate. On the un-anchored condition, e₁ itself does nothing at either site (red damage at most {max(posthoc_tier[s]["axis", k][0] for s in posthoc_tier for k in ex.SLICES):.3f}). That is the calibration: the operator only removes what sits on the direction it is given.

Each fitted direction removes something, and the pattern is the same at both sites. At the operand site only the embedding and the first block matter. A direction fitted at a deeper slice removes nothing, so the answer is formed from the operand states before the second block reads them. At the decode site the first two blocks matter. At the embedding there the state is the same on every line, so there is nothing to fit: LEACE returns the identity, and the other two fits return a direction with no label in it, whose small effect is an edit of the shared `=` state.

{posthoc_tradeoff_plot()}

What separates a placed axis from a found direction is the trade-off between the two kinds of damage. The strongest post-hoc removal at the operand site, {posthoc_best_ops[0]} at slice {SLICE_NAMES[posthoc_best_ops[1]]}, buys {posthoc_r_o:.2f} of red damage for {posthoc_n_o:.2f} of non-red damage. At the decode site, {posthoc_best_dec[0]} at slice {SLICE_NAMES[posthoc_best_dec[1]]} buys {posthoc_r_d:.2f} for {posthoc_n_d:.2f}. The anchored primary intervention buys {posthoc_p_r:.2f} for {posthoc_p_n:.3f}, and the `operands` arm {posthoc_anch["operands"][0]:.2f} for {posthoc_anch["operands"][1]:.3f}. Only one post-hoc direction has a small non-red cost, the probe at the embedding, and it is a mild edit in both respects: {posthoc_probe0[0]:.2f} of red damage for {posthoc_probe0[1]:.3f}.

So no fitted direction reaches the combination the placed axis gets. Every post-hoc row that removes at least half the red response costs {posthoc_strong:.2f} or more in non-red response, an order of magnitude above the cost of the anchored primary intervention, and the rows that cost little non-red response remove little red.

{figure_html(posthoc_table("operands"), caption="")}

{figure_html(posthoc_table("decode"), caption="")}

/// details | Reading the tables
Each cell is seed-mean red damage / non-red damage for the direction fitted at that slice and site, applied at the same site, on the three un-anchored runs. We expected the decode site to remove more red than the operand site, and it does for diff-in-means and LEACE at the first two blocks; the non-red cost there is about the same size as the red removal. We also expected LEACE to edit less than diff-in-means, since it removes only the linear trace. Instead it costs more non-red response at the operand site and about the same at the decode site, so on these states the linear trace of redness is not a small part of the state.
///

The un-anchored model does not keep redness on one linear direction that the mixing computation leaves alone. What the probe reads at the embedding is mostly separable from the task. The mean-difference and LEACE directions are not: the states that differ between red and non-red lines differ along directions the blocks also use for the mix. The per-slice pattern says where the two part company, and it is the first block.

## Exploratory analyses

Preregistered as exploratory, no gates.
"""

e1_gammas = list(ex.GAMMAS)
e1_ivs = [f"gamma-{g:g}" if g != ex.PRIMARY_GAMMA else "primary" for g in e1_gammas]
e1 = {
    k: np.stack([stat(iv, key, g) for iv in e1_ivs], 1)
    for k, (key, g) in {"red acc": ("acc", "red"), "non-red damage": ("damage", "nonred")}.items()
}


@memo
@themed(
    name="strength-sweep",
    alt_text="""
        Two panels against the fraction of the axis removed, from 0.25 to 1. Left, red accuracy falls from about 0.9 to about 0.1, steepest between 0.5 and 0.75, with a seed band widening in the middle. Right, non-red damage rises from near zero to a few hundredths, with one seed's band reaching about 0.17.
    """,
    caption="""
        **The strength sweep (E1).** Seed mean and seed range of red accuracy and non-red damage under the primary intervention at γ ∈ {0.25, 0.5, 0.75, 1}, the fraction of the e₁ component removed. The write at strength γ is closed-form in the arriving alignment, so the x axis could equally be drawn as the write on the red operand.
    """,
)
def strength_sweep_plot() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.4), layout="constrained")
    axes = cast(AxesRow, axes)
    for ax, (title, v) in zip(axes, e1.items(), strict=True):
        ink = light_dark(*ARM_INK["primary"])
        ax.fill_between(e1_gammas, v.min(0), v.max(0), color=ink, alpha=0.15, lw=0, zorder=0)
        ax.plot(e1_gammas, v.mean(0), color=ink, marker="o", ms=3, lw=1.3)
        ax.set_xticks(e1_gammas)
        ax.set_xlabel("γ, fraction of e₁ removed", fontsize=8)
        ax.set_ylabel(title, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylim(0, 1)
    return fig


# %%

# E2 reads the H1 table. E3: damage against clean alignment at the red operand, per slice.
e3_alpha = per_line("clean", "alpha")  # (seeds, slices, lines, positions)
e3_a_red = e3_alpha[:, :, np.arange(lines.n), lines.red_operand].mean(0)  # (slices, lines)
e3_dmg = (per_line("clean", "p_ans") - per_line("primary", "p_ans")).mean(0)
e3_rgb = GRID_RGB[lines.line_colors[np.arange(lines.n), lines.red_operand]]


@memo
@themed(
    name="damage-by-alignment",
    alt_text="""
        Five scatter panels, one per slice, of seed-mean damage against the clean alignment of the redder operand at that slice, marks colored by that operand. At every slice the marks sit on the floor up to an alignment of about 0.6, then rise steeply to near one; the transition sits at a higher alignment at the deeper slices, where every red color is already near one.
    """,
    caption="""
        **Damage against the clean alignment of the redder operand (E3).** One mark per line, seed-mean damage under the primary intervention against the seed-mean clean alignment at the position of the line's redder operand, one panel per slice. Marks are colored by that operand.
    """,
)
def damage_by_alignment_plot() -> plt.Figure:
    fig, axes = plt.subplots(1, len(SLICE_NAMES), figsize=(7.4, 1.9), sharex=True, sharey=True, layout="constrained")
    axes = cast(AxesRow, axes)
    for s, ax in enumerate(axes):
        ax.scatter(e3_a_red[s], e3_dmg, c=e3_rgb, s=4, alpha=0.6, linewidths=0, clip_on=False)
        ax.set_title(f"slice {SLICE_NAMES[s]}", fontsize=8)
        ax.tick_params(labelsize=6)
    axes[0].set_xlim(-0.3, 1)
    axes[0].set_ylim(-0.05, 1)
    axes[0].set_ylabel("damage", fontsize=8)
    fig.supxlabel("clean α of the redder operand", fontsize=8)
    return fig


# %%

# E5: the gain from the prompt writes to the final-state displacement.
E5_ARMS = {
    "embedding": "slice 0",
    "primary": "every slice",
    "operands": "every slice, operands only",
    "last": "slice 4",
}
e5_gain = {iv: stat(iv, "disp", "nonred") / stat(iv, "write_prompt_sum", "nonred") for iv in E5_ARMS}
e5_gain_red = {iv: stat(iv, "disp", "red") / stat(iv, "write_prompt_sum", "red") for iv in E5_ARMS}


def e5_row(iv: str) -> str:
    cells = []
    for g, gain in (("nonred", e5_gain), ("red", e5_gain_red)):
        cells += [
            f"<td class='num'>{v:.2f}</td>"
            for v in (stat(iv, "write_prompt_sum", g).mean(), stat(iv, "disp", g).mean(), gain[iv].mean())
        ]
    return f"<tr><td><code>{iv}</code></td><td>{E5_ARMS[iv]}</td>{''.join(cells)}</tr>"


e5_table = f"""
<div class="report-table-scroll"><table class="report-table">
  <thead>
    <tr><th rowspan="2">arm</th><th rowspan="2">writes at</th><th colspan="3">non-red lines</th><th colspan="3">red lines</th></tr>
    <tr>{"".join("<th class='num'>summed write</th><th class='num'>displacement</th><th class='num'>ratio</th>" for _ in range(2))}</tr>
  </thead>
  <tbody>{"".join(e5_row(iv) for iv in E5_ARMS)}</tbody>
</table></div>
"""
e5_caption = """
Readout displacement against the summed prompt write (E5). Seed means, in radians. Summed write: over the four prompt positions and every edited slice. Displacement: how far the `=` state at the last slice moved from its clean position. Ratio: the mean across seeds of the per-seed ratio of the two.
"""

# E4: the shaped arm's arriving states at depth, on red lines at op1, against the clean pass.
E4_MID = slice(1, 4)
e4_shaped_arrive = per_line("shaped", "alpha_pre")[:, E4_MID][:, :, lines.red, 0].mean()
e4_clean_arrive = per_line("clean", "alpha")[:, E4_MID][:, :, lines.red, 0].mean()
e4_shaped_again = (per_line("shaped", "theta")[:, E4_MID][:, :, lines.red, 0] > 1e-3).mean()
e4_shaped_comp = np.array([r["interventions"]["shaped"]["composition_red"] for r in runs[ex.PRIMARY.name]]).sum(0)
e4_shaped_comp = dict(zip(ex.COMPOSITION, e4_shaped_comp / e4_shaped_comp.sum(), strict=True))

# E6: the decoded write per site, under the primary intervention.
e6_dec = np.stack([per_line("primary", "decode", ex.PRIMARY.name)]).squeeze(0).mean(0)
# (slices, positions, target, group, channel)
e6_g = {g: ex.GROUPS.index(g) for g in ("red", "nonred")}


def e6_row(s: int) -> str:
    cells = []
    for pos, t in ((0, 0), (2, 1)):
        for g in ("red", "nonred"):
            cells += [f"<td class='num'>{v:+.3f}</td>" for v in e6_dec[s, pos, t, e6_g[g]]]
    return f"<tr><td>{SLICE_NAMES[s]}</td>{''.join(cells)}</tr>"


e6_table = f"""
<div class="report-table-scroll"><table class="report-table">
  <thead>
    <tr><th rowspan="2">slice</th><th colspan="6">at op1, decoding op1</th><th colspan="6">at op2, decoding op2</th></tr>
    <tr>{"".join(f"<th class='num'>{g} Δ{c}</th>" for _ in range(2) for g in ("red", "non-red") for c in "RGB")}</tr>
  </thead>
  <tbody>{"".join(e6_row(s) for s in range(len(SLICE_NAMES)))}</tbody>
</table></div>
"""
e6_caption = """
The decoded write (E6): the seed-mean change in the ridge-decoded RGB of the operand at its own position, edited state minus arriving state, under the primary intervention, on red lines and on non-red lines. A probe read on an edited state is an extrapolation, so the numbers are a direction of change in color units rather than a calibrated color.
"""
e4_emb_q = stat("embedding", "q99_alpha_nonred").mean(0)
e4_clean_q = np.array([r["clean"]["alpha_q99_nonred"] for r in runs[ex.PRIMARY.name]], float).mean(0)

rf"""
**E1 — strength.** The response is graded in γ rather than all-or-nothing: seed-mean red accuracy is {", ".join(f"{v:.2f}" for v in e1["red acc"].mean(0))} at γ = {", ".join(f"{g:g}" for g in e1_gammas)}, with the steepest drop between 0.5 and 0.75, and the seed spread widest in the middle of the sweep. Non-red damage is {", ".join(f"{v:.3f}" for v in e1["non-red damage"].mean(0))} over the same points, and the seed with the H2 deficit shows it from γ = 0.5 on.

{strength_sweep_plot()}

**E2 — where the edit acts.** The `embedding` arm bites (red accuracy {stat("embedding", "acc", "red").mean():.2f}) at the largest non-red cost of any arm, a drop of {(clean("acc", "nonred") - stat("embedding", "acc", "nonred")).mean():.3f} on non-red lines. The `last` arm barely bites ({stat("last", "acc", "red").mean():.2f}) and costs nothing. So the concept is read off the token, in the first two blocks, and the last block does not read it. That is the same depth profile the post-hoc tier found on the un-anchored model.

When only the token is edited, the blocks partly re-derive the axis (the H4 clouds). At the prompt positions, the non-red 99th-percentile alignment arriving at the last slice under that arm is {e4_emb_q[4, : ex.ANSWER_POS].max():.2f}, against {e4_clean_q[4, : ex.ANSWER_POS].max():.2f} clean. That partial return, together with the large non-red cost, is one transformer data point on the bypass question the D1.3 post left open: removing the axis at the input alone is neither a clean removal nor a cheap one, and editing at every slice is what makes the primary intervention selective.

**E3 — damage against alignment.** Read line by line, damage against the clean alignment of the redder operand rises more steeply than the $\cos^2$ curve M1 saw. At the embedding, lines whose operand sits near zero alignment are untouched, the middle of the scale takes partial damage with a wide spread, and the reddest colors, at 0.7 and above, lose nearly everything. At the deeper slices the clean pass has already pulled every red color to an alignment near one, so the rise sits at the top of the scale and the middle colors spread along it. The per-color view gives the same picture: the colors that lose their lines are the five reddest, and the middle bins of H3 are made of lines whose operand sits on the steep part of the D2.1 alignment response.

{damage_by_alignment_plot()}

**E4 — the shaped suppression.** With a threshold of {ex.SHAPED["a"]:g}, the `shaped` arm leaves every syntax state and every non-red state alone: its non-red damage has magnitude {abs(stat("shaped", "damage", "nonred").mean()):.5f}, and at every site its 99th-percentile write on non-red lines falls to at most {np.array([r["interventions"]["shaped"]["q99_write_nonred"] for r in runs[ex.PRIMARY.name]], float).mean(0).max():.3f} radians. That is the H4 map with the syntax columns emptied.

It removes about half of the red response (accuracy {span(stat("shaped", "acc", "red"), ".2f")}), with the widest seed spread of any arm. The reason is that the alignment of the red operand at the embedding, 0.90 for pure red, sits partway up the ramp of the falloff rather than at its top (see the operator figure in the Method).

The edit at the embedding sends pure red past the threshold, and the blocks then bring the red operand partway back. At the three middle slices it arrives with a mean alignment of {e4_shaped_arrive:.2f} under the shaped suppression, against {e4_clean_arrive:.2f} clean, and {e4_shaped_again:.0%} of those states sit above the threshold and are written again.<!-- REVIEW: an earlier draft said "the deeper slices see a state already pulled toward the threshold" without measuring it. The arriving alignment under the `shaped` arm is stored, and the sentence now quotes it. Verify: `shaped/alpha_pre` at op1 on red lines, slices 1-3, against `clean/alpha`. -->

What it decodes to is still a color: on red lines the mass outside the color vocabulary is {stat("shaped", "offvocab", "red").mean():.3f}, {e4_shaped_comp["true"]:.0%} of the decodes are the true answer, and {e4_shaped_comp["neighbor"]:.0%} are a one-step neighbor of it. The shaped suppression is the selective operator. A lower threshold or a steeper ramp would remove more of red, and the repulsion form from M1 would bound where the red operand lands. Both are open ([shaped-suppression item](/todo/science/shaped-suppression-rather-than-projecting-whole-axis.md), [repulsion item](/todo/science/repulsion-sets-the-landing-alignment.md)).

**E5 — how much of the write reaches the answer.** The logits are read from one state, the `=` state at the last slice. Its displacement is how far it moved, intervened against clean. We compare that with the writes summed over every edited slice and all four prompt positions, since attention moves information across positions. Both quantities are angles.

A ratio near one would say the rotations reach the readout state whole; above one, that the blocks grow them; below one, that they shrink or cancel them. The `last` arm is the reference with no block in between: its writes are all at the last slice, so the readout state moves by its own write and no more, and its ratio is the share of the four positions that this write accounts for.

{figure_html(e5_table, caption=e5_caption, class_="report-figure")}

On non-red lines the ratio under the primary intervention is {e5_gain["primary"].mean():.2f} (seed range {e5_gain["primary"].min():.2f}–{e5_gain["primary"].max():.2f}), about the same as the reference. The `embedding` arm says why. A write at the embedding alone reaches the readout state at nearly its summed size, a ratio of {e5_gain["embedding"].mean():.2f}, so the blocks carry an edit forward rather than shrink it.

Under the primary intervention the edits at the deeper slices remove what the blocks rebuilt, and the readout state ends {stat("primary", "disp", "nonred").mean():.2f} radians from clean. That is less than half the displacement under the `embedding` arm, from twice the summed write. The `operands` arm moves the readout state by {stat("operands", "disp", "nonred").mean():.2f} radians, so most of that displacement comes from editing the syntax positions.<!-- REVIEW: was "So the blocks pass on a small fraction of the rotation", read off the primary ratio alone. The `embedding` arm's ratio is near one, so an uncorrected write is carried forward nearly whole, and the low primary ratio comes from the deeper edits removing what the blocks rebuilt. Verify: `e5_gain["embedding"]` per seed. -->

The seed with the H2 deficit has the largest readout displacement of the nine, {stat("primary", "disp", "nonred").max():.2f} radians against a seed mean of {stat("primary", "disp", "nonred").mean():.2f}, and its summed write is no larger than what the other seeds see.

On red lines the readout state moves {stat("primary", "disp", "red").mean():.2f} radians under the primary intervention, a ratio of {e5_gain_red["primary"].mean():.2f}. That is the designed case: on those lines the edit is meant to change the answer.

**E6 — what the write decodes to.** At the operand positions on red lines, the write reads as a fall in decoded redness with a rise in green and blue of about the same size, at every slice. The change is largest at the embedding, where the alignment is removed in one step. On non-red lines the same direction of change appears at the deeper slices, at about a third of the size; at the embedding the sign reverses, since what is removed there is a small negative alignment. The three channels stay in a fixed ratio of roughly (−1, +0.85, +0.85) throughout. The probe reads the axis as redness against the other two channels, so this says the rotation is along what the probe calls red and nothing else the probe can see.

{figure_html(e6_table, caption=e6_caption, class_="report-figure")}

## Discussion

The D2.2 plan can now say that the placed axis is where red is read from, in the transformer as in the autoencoder. Removing it removes the concept, in proportion to how red the line is, and the write stays inside the bound the clean geometry set. There is more to say about the operator and about depth.

The whole-stream projection costs a little on non-red lines, and that cost comes from the syntax positions, whose embeddings have a constant component on the axis. Two arms avoid it: the edit restricted to the operand positions, and the shaped suppression, which by construction leaves the low-alignment bulk alone. Of the two, only the shaped suppression needs no knowledge of the syntax of the line, so it is the natural operator for the anchored-op experiments and for the [shaped-suppression item](/todo/science/shaped-suppression-rather-than-projecting-whole-axis.md).

At the M1 threshold the removal by the shaped suppression is only partial: the embedding alignment of the red operand sits inside the ramp, and the blocks bring the operand partway back at the deeper slices. So the threshold is something to tune, and the strength sweep says the response to a partial removal is graded, which gives the tuning something to grade against.

Repulsion is the shape to reach for when the dose is a target alignment rather than a fraction removed, since it sets the landing alignment where the suppression leaves it to the re-normalization ([repulsion item](/todo/science/repulsion-sets-the-landing-alignment.md)).

On depth: the concept is read from the operand states before the second block, the last block does not read it at all, and a token-only removal is neither clean nor cheap, because the blocks partly re-derive the axis from an off-axis input. That is the case for the layer sweep, and it says what the sweep should find. A removal acting at the embedding and the first block should do the work of the primary intervention, and one that starts later should not.

The fallback experiment has its reference numbers. Without a fallback term, a red line decodes in the color vocabulary to an answer near the true mix, computed as if the red operand were a less red color, and which near miss it lands on varies by seed. A fallback that is trained rather than left to the model should make the response reproducible across seeds; the seed-agreement fraction is the number to move.

The bound held layer by layer, and one seed still paid a cost on non-red lines from a bounded write. The layer-local statement is the one we can defend from the geometry we placed. What the blocks then do with a bounded rotation is a property of the trained model, and the seed spread says it varies. To bound the behavior rather than the write, the anti-subspace term would have to reach the syntax positions, or the operator would have to skip them.
"""
