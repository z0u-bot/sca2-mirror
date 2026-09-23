"""Ex-2.2.14: anchoring an operation — the smoke test.

The first anchored-op experiment of D2.2. One operation of table A+ is anchored to e₁ on the handover setup,
with no *red* anchor, at a few seeds, and scored against the alignment and task gates alone. It says whether
anchoring an op works at all before the many-seed equivalence read spends its budget, and it fixes which op
that experiment anchors. The other ten ops ride along at fewer seeds as a description, not a test.

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
It is the task reference and the baseline for every alignment measurement. Its labeller differs from this
experiment's (the control drew red labels it never used), so the comparison carries corpus-draw noise on
top of seed noise, as every cross-labeller comparison since ex-2.1.10 has."""

# --- The anchored op -----------------------------------------------------------------------------------

ANCHORED_OP = "difference"
"""The op anchored to e₁. Chosen on paper from table A+ before any run, by three criteria the design named
and one the calibration look added. Op-relevance: on 75% of its lines its answer names it alone, the most
of any commutative op in the table (`mix` 63%, `hsvmix` 59%). Rounding: it is total on the grid, so every
line has one answer and the task score is not capped by stochastic rounding (the control learns it to 0.97
held-out expected exact match against 0.43 on `mix`). Order: it is commutative, so nothing in its measurement
depends on which operand is which. The sweep below reads the other ten ops the same way so the choice can
be revisited on data rather than on paper."""

SHORT_NAMES = {
    "difference": "diff",
    "multiply": "mult",
    "exclusion": "excl",
    "hue-hsv": "hue",
    "sat-hsv": "sat",
    "value-hsv": "val",
}
"""Condition names carry the op's short name, so that `anchor-diff-opword` fits a table cell."""


def short(op: str) -> str:
    return SHORT_NAMES.get(op, op)


ANCHOR_AXIS = 0
"""e₁. The op takes the axis *red* had; there is no *red* anchor in this experiment. A model that carries
both is the follow-up once anchoring an op alone is understood."""

KEYING = "op"
"""A new labeller keying: a line draws a label when its op word is the anchored op, at `LABEL_RATE`. The
pull covers the whole line, as the handover's `line` keying does, so the label never says which position
carried it (the position-free mechanism of ex-2.1.10). The library today keys on the operands and the
answer only, so this keying is the one engineering change the experiment needs."""

LABEL_RATE = 0.02
"""The anchored op's lines draw at this rate, so about `RED_LABEL_SHARE` of the corpus is labelled, red's
share, and each labelled line gets the pull a red line got (the anchor term normalizes by the mask's own
weight). The rate is the primary rather than a pull on every line because the labels the method will have
further up the ladder — an abstract concept in natural language — will be scarce, so the scarce case is the
one the hypotheses should be scored on. The every-line arm below brackets it (REVIEW: the two swapped
places at the human's call; the rate is the more realistic labeller)."""

RED_LABEL_SHARE = 0.0018
"""The share of lines the red labeller labels under `line` keying: three draws per line at redness⁸ × 0.04
per token, averaged over the grid. Computed from the palette, not measured."""

FULL_RATE = 1.0
"""The arm in which every line of the anchored op draws: about a tenth of the corpus, fifty times red's
share. The term's size per batch is unchanged and each labelled line gets a proportionally weaker pull. If
it and the primary agree on the margin, the label share is not what sets it; if they part, the ratio of
their margins is what pulling every line of a categorical concept buys."""

LABEL_PRECISION = 0.8
NOISY_TRUE_RATE = LABEL_RATE * LABEL_PRECISION
FALSE_RATE = LABEL_RATE * (1 - LABEL_PRECISION) / 10
"""The arm with erroneous labels: the anchored op's lines draw at `NOISY_TRUE_RATE` and every other op's
lines at `FALSE_RATE`, so the labelled share of the corpus is the primary's and a fifth of the labels land
on lines that do not carry the op (REVIEW: was a fifth added on top of the primary's labels, which would
have raised the total pull as well; the wrong labels now replace true ones). A labeller for an abstract
concept in natural language will be wrong some of the time, and this arm says what a fifth of wrong labels
costs the margin and the task. The precision is a round number, not a measurement of any labeller."""

OPWORD_SPAN = 2
"""The arm that pulls the op word alone, at the primary's rate: the pull covers position 1 only (a span of
two tokens minus the first operand, in the library's role terms, or a `slot` pull at role 1 once the keying
exists). Under it the use sites are outside the mask, so a contrast at `=` or the answer at the final slice
is carried there by the blocks rather than asked for by the pull. H3 is scored on this arm for that reason;
the primary's use-site contrast is optimized by construction and is reported beside it as a manipulation
check. One position instead of six also makes each pull about six times stronger under the mask normalizer
(the ex-2.1.7 lesson), which the arm's op-position contrast beside the primary's will show."""

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

PRIMARY = f"anchor-{short(ANCHORED_OP)}"
FULL_ARM = f"{PRIMARY}-full"
OPWORD_ARM = f"{PRIMARY}-opword"
NOISY_ARM = f"{PRIMARY}-noisy"
ARMS = (FULL_ARM, OPWORD_ARM, NOISY_ARM)
"""Three arms at the primary's five seeds. H3 is scored on the op-word arm; the other two are outside the
hypotheses and the rule."""
SWEEP = tuple(
    f"anchor-{short(op)}"
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
assert N_RUNS == 50

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

RETENTION_FLOOR = 0.2
RETENTION_GATE = 0.8
"""H2 (retention): every primary run whose op margin reaches `RETENTION_FLOOR` at the start of the anchor
weight's anneal ends training at `RETENTION_GATE` of that value. Ex-2.2.11's rule and denominator (REVIEW:
the draft said "of its peak", which is the rule ex-2.2.10 replaced; a noisy plateau's high point is not a
level to hold). The learning rate is low over the anneal, so a tighter share would be defensible; one
`handover` seed in twenty ended under 0.8 in ex-2.2.11, so the bar stays where a known failure sits and
the per-seed values are reported for the equivalence experiment to set its own."""

USE_CONTRAST_FLOOR = 0.05
"""H3, scored on the op-word arm, a prediction with no decision hanging on it: some of the op reaches the
use sites. At `=` and at the answer position the embedding is the same token on every line, so any
difference in alignment between the anchored op's lines and the others at slices 1..L came through the
blocks, and under the op-word pull nothing asked for it. H3 holds when the arm's seed-mean contrast (mean
cosine on the anchored op's lines minus mean cosine on the other ops' lines) at the final slice, at each of
the two use sites, is at least `USE_CONTRAST_FLOOR` above the control's. The ratio to the contrast at the op
position is reported with no bar (REVIEW: the draft gated the ratio at 0.5, a number with no observation
behind it; the model has so far been seen to put the answer color at the use sites, not a copy of the op).
The primary's use-site contrast is reported beside it as a manipulation check."""

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
The equivalence experiment anchors `{ANCHORED_OP}` if the primary passes H1 and H2 in full (a partial \
counts as a miss), margin and retention both. If it does not, it anchors the op in the sweep with the \
largest seed-mean op margin among those that pass the same gates at their three seeds: every op inside the \
task gate, the margin at or above the H2 bar, and every run clearing the retention rule, and that op is \
confirmed at the equivalence experiment's own seeds before any number is quoted for it. If no op qualifies, \
the anchored-op line stops here and the report says what gave way. H3, the arms, and the containment \
measure do not enter the rule: they describe what the anchor did, and the equivalence experiment measures \
them at its own seeds whichever way they came out."""
