# title: Experiment 2.9.4: closed-loop regularizer weights

import matplotlib.pyplot as plt
import numpy as np

from mini.lit import memo, stop
from mini.vis import light_dark, themed
from sca.colorcube import TRAJ_STRIDE, classify, load_results

# Store refs published by experiment.py (kept in sync by hand).
METRICS_REF = "reports/ex-2.9.4/metrics"
TRAJS_REF = "reports/ex-2.9.4/trajectories"

LAM_CAP = 0.15  # the ceiling of the anchor dual

r"""
# Experiment 2.9.4: closed-loop regularizer weights

/// tip |
<!-- tl;dr -->
We replaced the timed regularizer anneal with a feedback controller, so that each weight climbs while its constraint is being violated and settles back once it is met. It kind of worked but not very well, and it adds complexity.
///

[Ex-2.9.3](../ex-2.9.3/report.py) traced anchoring failures to an instability late in training. The anchored solution is metastable during the high learning-rate plateau: the regularizers hold it in place, and the timed anneal takes that support away while the optimizer is still hot. We fixed it statically, by halving the peak learning rate. That works, but it sets the timing by hand.

This experiment lets the regularizer weights respond to the training signals, so a weight climbs while its constraint is being violated and settles back down once the constraint is met.

The mechanism is a small feedback controller (dual ascent with hysteresis[^dual]) acting on the anchor and anti-anchor weights. It reads only signals available during training, with no privileged ground-truth probe:

- Each controlled term keeps a running average of its own raw value (an EMA[^ema]). The anchor term is measured on labeled samples only, so its average updates on the ~6% of batches that carry a label.
- The weight λ rises while its average sits above an engage threshold, decays (5× faster) once it falls below a release threshold, and holds steady in the band between the two. That keeps the ordinary early transient from winding the weight up, and lets λ return to zero in a healthy run.
- λ is capped at 0.15, close to the dopesheet constant 0.1 from earlier experiments.

The anti-subspace weight is left uncontrolled. Its raw value has a floor set by the red samples themselves, and a sensor that cannot see labels has no way to tell that floor apart from leakage. So we hold it at its small late value instead of annealing it. Learning rate and `separate` follow the dopesheet as before.

We ran ten conditions, 32 seeds each, scored by the ex-2.9.2 `redirect` intervention: {static timed-anneal, controller} × peak LR {0.10 risky, 0.05 safe} with the fallback term; the same pair with the fallback term removed, at 0.10, since ex-2.9.3 showed that is where catastrophic failures actually appear; and a sensitivity grid over the controller parameters (targets ×0.75, ×1.5; gains ×0.5, ×2). The experiment is [`experiment.py`](./experiment.py):

```bash
bin/mini run docs/m1/ex-2.9.4/experiment.py --app modal --max-containers 16
```

[^dual]: Dual ascent enforces a constraint by putting a price on it. While
the constraint is being violated the price (here the weight λ) rises, which pushes the optimizer to satisfy it; once it is met, the price relaxes. Hysteresis means the price uses two thresholds with a gap between them rather than one, the way a thermostat lets the room drift a little before switching, so the weight doesn't chatter on and off.

[^ema]: Exponential moving average: a running average that weights recent
samples more than old ones, so it follows a changing signal without keeping its whole history.
"""

loaded = load_results(METRICS_REF, TRAJS_REF)
if loaded is None:
    stop(
        "No results yet — run the experiment (it publishes metrics to the store on completion):\n\n"
        "```bash\nbin/mini run docs/m1/ex-2.9.4/experiment.py --app modal --max-containers 16\n```"
    )
metrics, trajs = loaded


def cond(name: str, peak_lr: float | None = None) -> list[dict]:
    return [r for r in metrics if r["cond"] == name and (peak_lr is None or r["peak_lr"] == peak_lr)]


def traj(r: dict, key: str) -> np.ndarray:
    return trajs[f"{r['run']:03d}_{key}"]


def rd(rs: list[dict]) -> np.ndarray:
    return np.array([r["interventions"]["redirect"]["score"] for r in rs])


def n_cat(rs: list[dict]) -> int:
    return sum(classify(r) == "catastrophic" for r in rs)


steps = np.arange(len(traj(metrics[0], "z0_red"))) * TRAJ_STRIDE

sn, cn = cond("static-nofb"), cond("ctrl-nofb")
s_bad = sorted(r["seed"] for r in sn if classify(r) == "catastrophic")
c_bad = sorted(r["seed"] for r in cn if classify(r) == "catastrophic")
rescued = [s for s in s_bad if classify(next(r for r in cn if r["seed"] == s)) == "clean"]

rf"""
**{sum(1 for _ in (r for r in cond("static") + cond("ctrl")))} + {len(sn) + len(cn)} + 128 runs completed** across ten conditions. This is a clear negative result. Without the fallback term, the feedback loop does what it was designed to do on the seeds the static schedule loses: its {n_cat(sn)} catastrophic seeds ({", ".join(map(str, s_bad))}) train cleanly under control (redirect scores {", ".join(f"{rd([next(r for r in cn if r['seed'] == s)])[0]:.2f}" for s in rescued)}). But the controller introduces {n_cat(cn)} new failures on other seeds ({", ".join(map(str, c_bad))}), so the overall failure rate does not improve. With the fallback term on (the recipe we adopted), no condition has a single catastrophic failure, so there is nothing left for the controller to rescue. So we will probably keep the static schedule.

## The fallback-free test: rescues and new failures

Ex-2.9.3 showed that catastrophic failures only appear when there's no fallback term and a high LR. That makes it the fair place to test a rescue mechanism. The charts below show anchor progress and leakage for all 32 seeds under each variant.

The static schedule failures are in the anchor panel: the anchor forms, then falls away late in training. With the controller, the same panel shows the feedback at work, with runs dipping hard in the middle of the plateau and getting pulled back up to 1.

The response has a cost, though, and it doesn't always succeed. The leak panel shows runs where a sustained penalty drags pinkish labeled colors onto the axis until the geometry is spoiled, and one run loses the anchor outright with its weight pinned at the cap.
"""

nofb_cells = [("static-nofb", "static (timed anneal)"), ("ctrl-nofb", "controller")]
nofb_ncat = {c: sum(classify(r) == "catastrophic" for r in cond(c)) for c, _ in nofb_cells}


@memo
@themed(
    name="nofb-trajectories",
    alt_text=f"""
    Four charts in a two-by-two grid. Rows are the static schedule and the controller, both without the fallback term; columns are anchor progress and leakage, 32 seeds each, with catastrophic runs drawn in color over gray healthy ones. Top left, static anchor progress: all runs reach 1 by step 750; of the {nofb_ncat["static-nofb"]} colored runs, one falls away late to about 0.6 while the other keeps its anchor, failing instead by reconstruction collapse. Top right, static leak: the colored runs climb to {max(r["leak"] for r in cond("static-nofb")):.2f}. Bottom left, controller anchor progress: colored runs dip sharply in the middle of the plateau, one to about minus 0.25, and the feedback pulls most of them back to 1; one ends near 0.15, not recovered. Bottom right, controller leak: {nofb_ncat["ctrl-nofb"]} colored runs blow up, the worst reaching {max(r["leak"] for r in cond("ctrl-nofb")):.2f}, from over-anchoring under a sustained maximum penalty.
    """,
)
def nofb_trajectories_plot() -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), sharex=True, sharey="col")
    gray = light_dark("#9aa5b155", "#6b768655")
    hues = ["#d55e00", "#0072b2", "#009e73", "#cc79a7", "#e69f00"]
    for row, (c, label) in enumerate(nofb_cells):
        rs = cond(c)
        bad = sorted((r for r in rs if classify(r) == "catastrophic"), key=lambda r: r["seed"])
        for col, key in enumerate(("z0_red", "leak")):
            ax = axes[row, col]
            for r in rs:
                if classify(r) != "catastrophic":
                    ax.plot(steps, traj(r, key), color=gray, lw=1)
            for hue, r in zip(hues, bad, strict=False):
                ax.plot(steps, traj(r, key), color=hue, lw=1.6, label=f"seed {r['seed']}")
            ax.axvspan(750, 1500, color=light_dark("#1a5f8a", "#6ab0d4"), alpha=0.07, lw=0)
            ax.grid(alpha=0.3)
            ax.legend(loc="center left", fontsize=8)
        axes[row, 0].set_ylabel(f"{label}\nz₁(pure red)")
    axes[0, 1].set_ylabel("leak (mean |z₁|, non-red)")
    axes[1, 1].set_ylabel("leak (mean |z₁|, non-red)")
    axes[0, 0].set_ylim(-1.05, 1.05)
    axes[0, 1].set_ylim(0, 0.9)
    for ax in axes[1]:
        ax.set_xlabel("step")
    axes[0, 0].set_title("Anchor progress")
    axes[0, 1].set_title("Leakage")
    fig.suptitle("Without the fallback term: the controller keeps every anchor, but spoils other runs", y=0.99)
    return fig


nofb_trajectories_plot()

# %%

cn_ = cond("ctrl-nofb")
bad_ = [r for r in cn_ if classify(r) == "catastrophic"]
sat_bad = sum(1 for r in bad_ if r["lam_anchor_mean"] > 0.13)
sat_ok = sum(1 for r in cn_ if classify(r) == "clean" and r["lam_anchor_mean"] > 0.13)
n_ok = sum(1 for r in cn_ if classify(r) == "clean")
r27 = next(r for r in cond("static-nofb") if r["seed"] == 27)
c27 = next(r for r in cn_ if r["seed"] == 27)
drop = int(np.flatnonzero(traj(r27, "z0_red") < 0.7)[-1]) * TRAJ_STRIDE

rf"""
The weight trajectory separates the two outcomes. In the rescues, λ engages while the anchor is forming, then releases once the constraint is met. On seed 27 (the worst under the static schedule, still below z₁ = 0.7 at step {drop}), the controlled run holds z₁ near 1 through the plateau, with λ decaying to {c27["lam_anchor_end"]:.2f}. In the failures the labeled anchor EMA never reaches its target, so λ climbs to the {LAM_CAP} cap and stays there: {sat_bad} of {len(bad_)} catastrophic runs had mean λ > 0.13, versus {sat_ok} of {n_ok} clean ones.

This is a problem with the sensor. The anchor term is measured on noisy labels, and a pinkish color placed off the axis on purpose looks the same, to that measurement, as a red that has drifted off it. On most seeds the average settles below target anyway; on some it can't, and overcorrects. One run lost its anchor even with its weight pinned at maximum.

## With the fallback term

The fallback term from ex-2.9.2 turns out to prevent every catastrophic failure on its own: none in 448 fallback-trained runs across this experiment and ex-2.9.3, against 7 of 224 without it.
"""

fb_cells = [
    ("static", 0.10, "static\n0.10"),
    ("ctrl", 0.10, "ctrl\n0.10"),
    ("static", 0.05, "static\n0.05"),
    ("ctrl", 0.05, "ctrl\n0.05"),
    ("ctrl-tau0.75", 0.10, "ctrl\nτ×0.75"),
    ("ctrl-tau1.5", 0.10, "ctrl\nτ×1.5"),
    ("ctrl-eta0.5", 0.10, "ctrl\nη×0.5"),
    ("ctrl-eta2", 0.10, "ctrl\nη×2"),
]
fb_scores = {lbl: rd(cond(c, lr)) for c, lr, lbl in fb_cells}
fb_leaks = {lbl: np.array([r["leak"] for r in cond(c, lr)]) for c, lr, lbl in fb_cells}
fb_ncats = {lbl: n_cat(cond(c, lr)) for c, lr, lbl in fb_cells}


@memo
@themed(
    name="fallback-cells",
    alt_text=f"""
    Two stacked strip plots over eight fallback-trained conditions: the static schedule and the controller at peak LR 0.10 and 0.05, then four controller variants at 0.10 with shifted targets (×0.75, ×1.5) and gains (×0.5, ×2). Top: redirect scores, with medians all between {min(np.median(v) for v in fb_scores.values()):.2f} and {max(np.median(v) for v in fb_scores.values()):.2f} and a few scattered outliers below 0.5. Bottom: final leakage on a log scale, with a dashed line at the 0.1 degraded threshold and a dotted line at the 0.3 catastrophic threshold; the number of catastrophic runs is printed under each condition. The τ×0.75, η×0.5, and η×2 variants show {fb_ncats["ctrl\nτ×0.75"]}, {fb_ncats["ctrl\nη×0.5"]}, and {fb_ncats["ctrl\nη×2"]} catastrophic runs with leak reaching 0.4 to 0.8; every static condition shows zero.
    """,
)
def fallback_cells_plot() -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True, height_ratios=[2, 1.6])
    rng = np.random.default_rng(0)
    accent, gray = light_dark("#1a5f8a", "#6ab0d4"), light_dark("#9aa5b1", "#6b7686")
    colors = [gray, accent, gray, accent, accent, accent, accent, accent]
    for ax, data in zip(axes, (fb_scores, fb_leaks), strict=True):
        for gi, (lbl, color) in enumerate(zip(fb_scores, colors, strict=True)):
            x0 = gi + rng.uniform(-0.13, 0.13, len(data[lbl]))
            ax.scatter(x0, data[lbl], s=14, color=color, alpha=0.75, lw=0)
            ax.plot([gi - 0.22, gi + 0.22], [np.median(data[lbl])] * 2, color=color, lw=2)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set(ylabel="score (R², redirect)", ylim=(0, 1.02))
    ref = light_dark("#000", "#fff")
    axes[1].axhline(0.1, ls="--", lw=1, color=ref, alpha=0.5)
    axes[1].axhline(0.3, ls=":", lw=1, color=ref, alpha=0.5)
    axes[1].set(ylabel="final leak", yscale="log")
    axes[1].set_xticks(range(len(fb_cells)), [lbl for _, _, lbl in fb_cells])
    for gi, lbl in enumerate(fb_scores):
        axes[1].annotate(
            f"{fb_ncats[lbl]} cat.",
            (gi, 0.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            alpha=0.8,
        )
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=v)
        for v, c in (("static (timed anneal)", gray), ("controller", accent))
    ]
    axes[0].legend(handles=handles, loc="lower left")
    axes[0].set_title("Fallback-trained conditions: feedback adds no headroom")
    return fig


fallback_cells_plot()

# %%

s5, c5 = cond("static", 0.05), cond("ctrl", 0.05)
s1, c1 = cond("static", 0.10), cond("ctrl", 0.10)


def rc(rs: list[dict]) -> float:
    return float(np.median([r["val_recon"] for r in rs]))


def lk(rs: list[dict]) -> float:
    return float(np.median([r["leak"] for r in rs]))


rf"""
At the safe peak the two approaches are interchangeable on the score (medians
{np.median(rd(c5)):.2f} for the controller vs {np.median(rd(s5)):.2f} static; floors
{rd(c5).min():.2f} vs {rd(s5).min():.2f}). The controller halves the typical leak
({lk(c5):.3f} vs {lk(s5):.3f}), since a steady low-level penalty does keep the axis
cleaner on average, but it worsens the tail past the degraded threshold and costs about
2–3× in reconstruction ({rc(c5):.6f} vs {rc(s5):.6f}, both still tiny). At the risky
peak the pattern is the same, with {n_cat(c1) + n_cat(s1)} catastrophes between the two
conditions, thanks to the fallback term.\
The sensitivity grid is fairly clear: Tightening the targets by 25%
(τ×0.75) or doubling the gain (η×2), with the fallback term still on, produces
{n_cat(cond("ctrl-tau0.75"))} and {n_cat(cond("ctrl-eta2"))} catastrophic runs
respectively, and halving the gain leaves {n_cat(cond("ctrl-eta0.5"))}. Loosening the
targets (τ×1.5) is safe ({n_cat(cond("ctrl-tau1.5"))}).

## Lessons

The idea seemed reasonable: protection on demand instead of on a timer. The controller response is fine; the measurement it relies on is the problem. The training-time signal for anchor health was the anchor loss on noisy labels, and in this experiment, that signal couldn't tell a drifting red from a pink that should be off-axis.

So the stack we keep is the plain one: the fallback term (which, across two experiments, has removed every catastrophic failure), peak LR 0.05, the original timed anneal, and cheap endpoint screening.

One engineering note. `DynamicProp.set()` in `mini.temporal` can retarget a value mid-flight, which is what a controller needs, but experiments consume schedules through `realize_timeline`, which bakes the dopesheet into a static array before training. Dopesheet keyframes and runtime `set()` calls would then compete over the same prop. So the controller in this experiment lives inside the training loop instead, carrying its dual state in the `lax.scan` carry, while the dopesheet drives the props it doesn't control. That split worked well and is worth keeping, even though the controller itself is not adopted.
"""
