"""Ex-2.2.21: the in-context grammar pilot, again — round 2 of the D2.2 quick route, on the reworked recipe.

Ex-2.2.16 stopped at its first rule: the control got a little under halfway from the floor to the Bayes ceiling, so
its anchored arms were trained and evaluated but never scored. Ex-2.2.17 to ex-2.2.20 reworked the grammar and the
recipe (the newline mask, a lower peak rate, the seven-op set, 200 epochs), and the control now reaches about nine
tenths of the way. This pilot trains the anchored arms again on that recipe and scores the rules ex-2.2.16 left
open, with the changes the report argues for: the corpus rule becomes a regression check, two label variants move
the pull toward the query `=` as references, and the hinge and operator rules merge into one rule that scores where
the anchor can be edited.

This module holds the design constants only, for the preregistration; the DAG lands once the plan is frozen.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass

DESIGN_ONLY = True


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules` (the pattern ex-2.2.16 to ex-2.2.20
    use), so this module's task bodies still cloudpickle by value for a remote worker.
    """
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


ex2219 = _load_sibling("ex-2.2.19", "ex2221_ex2219")
ex2218 = ex2219.ex2218
ex2217 = ex2219.ex2217
ex2216 = ex2219.ex2216

# --- What is inherited, unchanged --------------------------------------------------------------------------

OP_SET = ex2219.OP_SET
OP_NAMES: tuple[str, ...] = OP_SET.ops
"""The seven-op set (`no-four`) of ex-2.2.18: table A+ less `screen`, `multiply`, `hsvmix`, and `exclusion`."""
N_OPS = len(OP_NAMES)

ANCHORED_OP = ex2216.ANCHORED_OP
"""`difference`, as in ex-2.2.14 to ex-2.2.16. Its closest partner in table A+, `exclusion`, is not in the set."""
assert ANCHORED_OP in OP_NAMES

CENTRE: tuple[int, float] = ex2216.CENTRE
"""Three examples at ρ = 0.3, the only corpus condition. The report rechecks the posterior on `difference` contexts
on the seven-op table."""

CUBE_RATE = ex2216.CUBE_RATE
MIDDLE_BAND = ex2216.MIDDLE_BAND

MODEL = ex2216.MODEL
EPOCHS = 200
"""The length ex-2.2.19 adopted for the seven-op set, at the peak rate, warmup, and schedule of ex-2.2.17's recipe;
ex-2.2.20 kept the plain schedule."""
assert EPOCHS in ex2219.SCOUT_EPOCHS

PEAK_LR = ex2219.PEAK_LR
WARMUP_EPOCHS = ex2219.WARMUP_EPOCHS
NEWLINE_MASK = True
"""Every arm, the controls included: attention does not cross a line break. Ex-2.2.16's two mask arms are now the
recipe, so they leave the arm list."""

CROP_POLICY = ex2216.CROP_POLICY
LABEL_RATE = ex2216.LABEL_RATE
HOLDOUT_CONTEXTS = ex2216.HOLDOUT_CONTEXTS
VERIFY_RATE = ex2216.VERIFY_RATE
PREFIX_THRESHOLD = ex2216.PREFIX_THRESHOLD
HINGE_CAP = ex2216.HINGE_CAP
HINGE_SOFTNESS = ex2216.HINGE_SOFTNESS
"""The anchor recipe of ex-2.2.14 (λ_a, τ, and the anti-subspace schedule, each as a fraction of training) and the
label and hinge settings of ex-2.2.16, unchanged."""

# --- The arms ----------------------------------------------------------------------------------------------

LABEL_VARIANTS: tuple[tuple[str, str], ...] = (
    *ex2216.LABEL_VARIANTS,
    ("prompt", "(e) the context up to and including the query `=`: the query answer and the line break left out"),
    ("query-eq", "(f) the query `=` alone"),
    ("every-eq", "(g) every `=`, the examples' and the query's: the positions whose next token depends on the op"),
)
"""Ex-2.2.16's whole-line label and its four variants, then three new ones. Ex-2.2.16's anchored arms put most of their
alignment on the answer positions and very little at the query `=`, whose state predicts the answer (the report has
the measurement). Variant (e) leaves the query answer out of the pull, to see where the pooled term settles without
it; variant (f) pulls only the query `=`, a position oracle that shows what an anchor there would allow; variant (g) pulls every `=`, the four positions whose next token is an answer
and so depends on the op, to see whether an anchor spread over the examples' `=` settles there too. (The first
example's `=` has no evidence before it, so its pull asks for the op before the context shows it.) All three are
references for the site rule, and not labels round 3 could adopt as they stand."""

PRIMARY = "anchor-whole"
CONTROL = "control"


@dataclass(frozen=True)
class Arm:
    """One arm of the pilot: what it trains and why it is here."""

    name: str
    group: str
    """Which question the arm serves: `control`, `label`, `site`, or `verify`."""
    anchored: bool = False
    label: str = "whole"
    hinge: bool = False
    verify: bool = False
    seeds: int = 3
    note: str = ""


SEEDS_REFERENCE = 5
"""The control and the whole-line arm, which every comparison runs through, get five seeds; the others get three,
as in ex-2.2.16. A comparison with the control then pairs five seeds with three or five, which narrows the seed band
by a tenth (five against three) or a fifth (five against five) over three against three, for four more runs."""

ARMS: tuple[Arm, ...] = (
    Arm(CONTROL, "control", seeds=SEEDS_REFERENCE, note="the un-anchored control"),
    Arm(PRIMARY, "label", anchored=True, seeds=SEEDS_REFERENCE, note="the whole-line label, the primary"),
    *(
        Arm(f"anchor-{label}", "label", anchored=True, label=label, note=desc)
        for label, desc in ex2216.LABEL_VARIANTS
        if label != "whole"
    ),
    Arm("anchor-prompt", "site", anchored=True, label="prompt", note=LABEL_VARIANTS[-3][1]),
    Arm("anchor-query-eq", "site", anchored=True, label="query-eq", note=LABEL_VARIANTS[-2][1]),
    Arm("anchor-every-eq", "site", anchored=True, label="every-eq", note=LABEL_VARIANTS[-1][1]),
    Arm("anchor-hinge", "site", anchored=True, hinge=True, note="the whole-line pull capped by a hinge"),
    Arm("control-verify", "verify", verify=True, note="the control with verification lines"),
    Arm("anchor-verify", "verify", anchored=True, verify=True, note="the whole-line arm with verification lines"),
)
"""Every arm trains at the center condition on the seven-op set. The control is the reference for all of them, and
the whole-line arm for the label, site, and verification arms."""

N_RUNS = sum(a.seeds for a in ARMS)
assert N_RUNS == 40

SITE_ARMS: tuple[str, ...] = (PRIMARY, "anchor-hinge")
"""The candidates of the site rule, in its order of preference: the whole-line label, as the M3-shaped labeller,
then the same label with the hinge, a training setting M3 could also use."""

SITE_REFERENCES: tuple[str, ...] = ("anchor-prompt", "anchor-query-eq", "anchor-every-eq")
"""Scored by the site rule beside the candidates, and never chosen: they show where the anchor settles when the pull
leaves out the query answer, what an anchor at the query `=` would allow, and whether one spread over every `=`
lands at the query `=`."""

SUPPRESSION_ARMS: tuple[str, ...] = (*SITE_ARMS, *SITE_REFERENCES, CONTROL)
"""The arms the scoring-only suppression pass runs on: the candidates and references of the site rule, and the
control, whose damage under the same edit is subtracted."""


def arm(name: str) -> Arm:
    return next(a for a in ARMS if a.name == name)


SEED_OFFSET = 700
"""Fresh model seeds 700 onward; no earlier experiment has used them."""

# --- The rules -----------------------------------------------------------------------------------------------

REGRESSION_REF = "e200-lr0.00316"
"""The ex-2.2.19 runs the control is checked against: the four seeds at 200 epochs and the recipe peak rate."""

REGRESSION_TOL = ex2219.SHORTFALL_TOL
"""H1 (a): the control reproduces the recipe when its seed-mean held-out expected exact match is within this of the
seed mean of the ex-2.2.19 runs, either way: 0.015, the tolerance ex-2.2.19 gated length on."""

SEED_BAND_SD = ex2216.SEED_BAND_SD
"""The seed band between two arms at n₁ and n₂ seeds is `SEED_BAND_SD · σ · √(1/n₁ + 1/n₂)`, with σ the seed standard
deviation pooled over the two arms: ex-2.2.16's definition, generalized to unequal seed counts."""

MARGIN_KEEP = ex2216.MARGIN_KEEP
"""S1: a label variant "holds the anchor" when its seed-mean op margin is at least this share of the whole-line
margin."""

DOSE_GAMMAS = ex2216.DOSE_GAMMAS
REPULSION_LANDINGS = ex2216.REPULSION_LANDINGS
REPULSION_THRESHOLD = ex2216.REPULSION_THRESHOLD
REFLECT_GAMMA = ex2216.REFLECT_GAMMA
EDIT_SITES: tuple[str, ...] = (*ex2216.EDIT_SITES, "example answers")
SELECTIVITY_GATE = ex2216.SELECTIVITY_GATE
GRADING_MIN_DAMAGE = ex2216.GRADING_MIN_DAMAGE
"""The suppression pass and the two criteria of ex-2.2.16's operator rule (e), unchanged: the operators, their dose
axes, the three sites, a selectivity gate of 0.02 on each other op net of the control, and full-dose damage at least
half the way to the target null, net of the control."""

GRADE_DIP = 0.01
"""S2: the net drop on the anchored op may dip by at most this between adjacent doses and still count as grading, as
ex-2.2.1 and ex-2.2.2 allowed (they used 0.02, on larger drops). Half the selectivity gate, and about a quarter of a
dose step of the whole-line projection in ex-2.2.16."""

TASK_COST_TOL = 0.01
"""H1 (b): the whole-line arm may fall short of the control by at most this in seed-mean held-out expected exact
match. A little wider than the seed band of five seeds against five (about 0.009), so a miss is a cost the comparison
resolves, and narrower than the 0.015 of (a)."""

SCORED_SITES: tuple[str, ...] = ("query =", "every position")
"""S2 qualifies an operator at either of these sites. Ex-2.2.16 scored only every position; the query `=` is added
because its state predicts the answer, and `query-eq` and `every-eq` put the anchor there."""

REPORTED_SITES: tuple[str, ...] = ("query ?", "example answers")
"""Sites the suppression pass also edits, reported with no gate. The example answers are new: ex-2.2.16's anchored
arms put much of their alignment there, and the query may read the op back from them."""


def cost_per_run(epochs: int = EPOCHS) -> float:
    """Dollars of L4 time for one run, from ex-2.2.19's unanchored runs; the anchored step costs about the same."""
    return ex2219.cost_per_run(epochs)
