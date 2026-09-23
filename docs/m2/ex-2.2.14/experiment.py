"""Ex-2.2.14: anchoring an operation — the smoke test.

The first anchored-op experiment of D2.2. One operation of table A+ is anchored to e₁ on the handover setup,
with no *red* anchor, at a few seeds, and read against the alignment and task gates alone. It says whether
anchoring an op works at all before the many-seed equivalence read spends its budget, and it fixes which op
that read anchors. The other ten ops ride along at fewer seeds as a description, not a test.

Design constants only while the preregistration is in review; the DAG lands when the hypotheses freeze.
"""

from __future__ import annotations

DESIGN_ONLY = True

# --- What is inherited ---------------------------------------------------------------------------------

REFERENCE_EXPERIMENT = "m2/ex-2.2.11"
"""The grammar (table A+), the stochastic corpus at 300k lines, the whole-line span, the untied readout, the
recipe (λ_a 0.1 annealed to a 0.1 floor over the last tenth of training, τ 0.1, the anti-subspace weight from
2.5× the anchor weight to 0.3× by 90% of training), 50 epochs at d64-L4, and the task gate: all as ex-2.2.11
froze them and ex-2.2.13 confirmed at fresh seeds. Ex-2.2.13's ladder left the weight where it was, so the
recipe of record is `axis-0.1`, which is ex-2.2.11's `handover`."""

CONTROL_EXPERIMENT = "m2/ex-2.2.11"
CONTROL = "control"
CONTROL_SEEDS = 5
"""The un-anchored control: ex-2.2.11's `control` checkpoints at model seeds 100–104, served from the store.
It is the task reference and the baseline for every alignment read. Its labeller differs from this
experiment's (the control drew red labels it never used), so the comparison carries corpus-draw noise on
top of seed noise, as every cross-labeller comparison since ex-2.1.10 has."""

# --- The anchored op -----------------------------------------------------------------------------------

ANCHORED_OP = "difference"
"""The op anchored to e₁. Chosen on paper from table A+ before any run, by three criteria the design named
and one the calibration look added. Op-relevance: on 75% of its lines its answer names it alone, the most
of any commutative op in the table (`mix` 63%, `hsvmix` 59%). Rounding: it is total on the grid, so every
line has one answer and the task read is not capped by stochastic rounding (the control learns it to 0.97
held-out expected exact match against 0.43 on `mix`). Order: it is commutative, so nothing in its read
depends on which operand is which. The sweep below reads the other ten ops the same way so the choice can
be revisited on data rather than on paper."""

ANCHOR_AXIS = 0
"""e₁. The op takes the axis *red* had; there is no *red* anchor in this experiment. A model that carries
both is the follow-up once anchoring an op alone is understood."""

KEYING = "op"
"""A new labeller keying: a line draws a label when its op word is the anchored op, at `LABEL_RATE`. The
pull covers the whole line, as the handover's `line` keying does, so the label never says which position
carried it (the position-free mechanism of ex-2.1.10). The library today keys on the operands and the
answer only, so this keying is the one engineering change the experiment needs."""

LABEL_RATE = 1.0
"""Every line of the anchored op draws: about a tenth of the corpus. The red labeller labels about
`RED_LABEL_SHARE` of lines, fifty times fewer. The anchor term normalizes by the mask's own weight, so its
size per batch is unchanged and each labelled line gets a proportionally weaker pull (REVIEW: was "per-line
pull unchanged"). The matched arm below brackets that, and the label share is recorded per run."""

RED_LABEL_SHARE = 0.0018
"""The share of lines the red labeller labels under `line` keying: three draws per line at redness⁸ × 0.04
per token, averaged over the grid. Computed from the palette, not measured."""

MATCHED_RATE = 0.02
"""The arm that matches red's label share: the anchored op's lines draw at this rate, so about
`RED_LABEL_SHARE` of the corpus is labelled and each labelled line gets the pull a red line got. If the primary
and this arm agree on the margin, the label share is not what sets it; if they part, the ratio of their
margins is the price of pulling every line of a categorical concept."""

OPWORD_SPAN = 2
"""The arm that pulls the op word alone: the pull covers position 1 only (a span of two tokens minus the
first operand, in the library's role terms, or a `slot` pull at role 1 once the keying exists). Under it the
use sites are outside the mask, so a contrast at `=` or the answer at the final slice is carried there by the
blocks rather than asked for by the pull. That is the read H3 cannot make on the primary, and it is reported
beside H3 without a gate."""

WHOLE_SPAN = 6
"""The pull covers all six positions of a labelled line."""

# --- The runs ------------------------------------------------------------------------------------------

SEED_OFFSET = 300
"""Condition seed *i* trains at model seed `SEED_OFFSET + i`. Ex-2.2.13 trained at offset 200 and ex-2.2.11
and ex-2.2.12 at 100; every seed here is fresh."""

SEEDS = 5
"""Seeds for the primary condition. The smoke test asks whether the op lands and the task survives, which
five seeds resolve at the spreads ex-2.2.11 saw (the line margin's standard deviation across seeds was
0.016, the task's 0.005 on the worst op). The equivalence read that follows sets its own count."""

SWEEP_SEEDS = 3
"""Seeds for each of the other ten ops. The sweep describes; it decides nothing unless the primary misses."""

PRIMARY = f"anchor-{ANCHORED_OP}"
MATCHED_ARM = f"{PRIMARY}-matched"
OPWORD_ARM = f"{PRIMARY}-opword"
ARMS = (MATCHED_ARM, OPWORD_ARM)
"""Two arms at the primary's five seeds, outside the hypotheses and the rule."""
SWEEP = tuple(
    f"anchor-{op}"
    for op in (
        "mix",
        "screen",
        "multiply",
        "lighten",
        "darken",
        "exclusion",
        "hsvmix",
        "hue-hsv",
        "sat-hsv",
        "value-hsv",
    )
)
CONDITIONS = (PRIMARY, *ARMS, *SWEEP)
N_RUNS = SEEDS * (1 + len(ARMS)) + SWEEP_SEEDS * len(SWEEP)
assert N_RUNS == 45

# --- The measurements ----------------------------------------------------------------------------------

OP_POSITION = 1
EQUALS_POSITION = 3
ANSWER_POSITION = 4
"""Where in the six-token line the op word, `=`, and the answer sit."""

TASK_GATE = 0.02
TASK_PARTIAL = 0.05
"""H1: for each op, the primary's seed-mean held-out expected exact match is within `TASK_GATE` of the
control's seed mean; partial when every op is within `TASK_PARTIAL`. Ex-2.2.11's gate, unchanged."""

REF_M_LINE = 0.4286
"""Ex-2.2.11's `handover` line margin on the `mix` lines at twenty seeds: what *red* achieved under the same
recipe. The op margin is read against it."""

MARGIN_RATIO = 0.8
MARGIN_PARTIAL = 0.6
"""H2: the op margin — ex-2.2.3's `line_margin` with uniform line weights summing to one over the anchored op's
lines and zero elsewhere (the einsum needs a normalized weight): per slice, the labelled mean alignment minus the mean over all lines at the largest span role,
averaged over every slice — is at least `MARGIN_RATIO` of `REF_M_LINE` on the seed mean; partial from
`MARGIN_PARTIAL`. The same bar the handover set against ex-2.1.10."""

RETENTION_GATE = 0.8
"""H2 (retention): every primary run whose op margin peaks at 0.2 or more ends, after the anchor weight's
anneal, at `RETENTION_GATE` of that peak. Ex-2.2.11's rule, unchanged."""

USE_CONTRAST_RATIO = 0.5
USE_CONTRAST_PARTIAL = 0.25
"""H3: the contrast at the use sites. At `=` and at the answer position the embedding is the same token on
every line, so any difference in alignment between the anchored op's lines and the others at slices 1..L
came through the blocks. H3 holds when the seed-mean contrast (mean cosine on the anchored op's lines minus
mean cosine on the other ops' lines) at the final slice, at each of the two use sites, is at least
`USE_CONTRAST_RATIO` of the same contrast at the op position; partial from `USE_CONTRAST_PARTIAL`."""

CONTAINMENT_REF = 0.264
"""Ex-2.2.11's ᾱ at op1 on `handover` at twenty seeds: the mean alignment over every color at the first
operand, the containment statistic that rose under the untied readout and has no named mechanism. Its
categorical analogue here — the mean alignment at the op position over the other ten ops' lines — is
reported beside it and beside the control with no gate, as the open item asks."""

PROBE_RIDGE = 1e-2
"""The ridge strength of every probe scan since ex-2.1.5. The op-identity scan fits a one-hot ridge probe
per site and reports its held-out R², anchored against control. Exploratory here: it sizes the margin the
equivalence read declares."""

# --- The rule for the follow-up ------------------------------------------------------------------------

# REVIEW: the sweep fallback checked only the task gate and ranked by margin, so it could adopt an op that
# misses the H2 bar or retention; it now carries every gate the primary does, and "passes" excludes partials.
ADOPTION = f"""\
The equivalence read anchors `{ANCHORED_OP}` if the primary passes H1 and H2 in full (a partial counts as \
a miss), margin and retention both. If it does not, the read anchors the op in the sweep with the largest \
seed-mean op margin among those that pass the same gates at their three seeds: every op inside the task gate, \
the margin at or above the H2 bar, and every run clearing the retention rule, and that op is confirmed at the read's own seeds before any \
number is quoted for it. If no op qualifies, the anchored-op line stops here and the report says what gave \
way. H3 and the containment read do not enter the rule: they describe what the anchor did, and the read \
that follows measures them at its own seeds whichever way they came out."""
