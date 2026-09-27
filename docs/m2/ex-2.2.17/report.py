# title: Ex 2.2.17: suppressing the op word

# The design constants and the refs come from `experiment.py` beside this script (the script's directory is
# on sys.path while it runs).
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import experiment as ex
from mini.lit import memo, stop
from mini.store import project_store


def conditions_html() -> str:
    """The three stored conditions, one row each."""
    head = "<tr><th>condition</th><th>from</th><th class=num>seeds</th><th>role</th></tr>"
    origin = {ex.PRIMARY: "ex-2.2.14", ex.OPWORD_ARM: "ex-2.2.14", ex.CONTROL: "ex-2.2.11"}
    rows = [
        f"<tr><td><code>{s.condition}</code></td><td>{origin[s.condition]}</td><td class=num>{s.seeds}</td><td>{s.role}</td></tr>"
        for s in ex.SOURCES
    ]
    return f'<table class="report-table dense"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def rows_html() -> str:
    """The four rows: where each acts and what it does."""
    head = "<tr><th>row</th><th>position</th><th>operator</th><th>what it does</th></tr>"
    site = {ex.OP_POSITION: "op word", ex.EQUALS_POSITION: "<code>=</code>"}

    def operator(r: ex.Row) -> str:
        if r.kind == "projection":
            return f"<code>projection</code>, γ = {r.gamma:g}"
        if r.kind == "repulsion":
            return f"<code>repulsion</code>, linear, a = {r.a:g}, b = {r.b:g}"
        return "token mask"

    rows = [
        f"<tr><td><code>{r.name}</code></td><td>{site[r.position]}</td><td>{operator(r)}</td><td>{r.role}</td></tr>"
        for r in ex.ROWS
    ]
    return f'<table class="report-table dense"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


rf"""
# Ex 2.2.17: suppressing the op word

/// tip |
<!-- tl;dr -->
Ex-2.2.14 put `difference` on e₁, and the anchor settled on the state at the op word. This pass edits that state on the stored checkpoints. It uses two operators that have a defined landing there, and compares each with masking the word. The question is whether the anchor holds more than the identity of the token.
///

## Findings

- [The edits remove the op (H1)](#the-edits-remove-the-op-h1) —
- [The damage follows op-relevance (H2)](#the-damage-follows-op-relevance-h2) —
- [The other ten ops are untouched (H3)](#the-other-ten-ops-are-untouched-h3) —
- [The axis at `=` marks the op (H4)](#the-axis-at-marks-the-op-h4) —

[The rule for the write-up](#the-rule-for-the-write-up): how much of the op-word line the D2.2 write-up reports.

## How to read this draft

This is a preregistration. The rows, the four predictions, their thresholds, and the rule for the write-up are to be fixed before the pass runs; the freeze commit will be quoted here. Results replace the `TODO` placeholders in place, and anything conceived after seeing the data goes under [exploratory analyses](#exploratory-analyses), marked as post hoc.

While drafting, a smoke check ran the scoring code on one primary seed over 600 lines of three ops, to verify that the checkpoints load and the contract checks hold. Its numbers were seen after the thresholds below were written and did not move them; they are not reported.

The [D2.2 design](../d2.2/design.md#quick-route) names this pass as round 1 of the quick route, scoring only, and asks it to keep to the measurements that decide how much of the op-word line to report. The ten sweep ops and the finer slice subsets wait.

## Why this pass

[Ex-2.2.14](../ex-2.2.14/report.py) anchored an operation for the first time. `difference` landed on e₁ at twice the margin *red* reached, held through the anneal, and cost the task nothing.

The anchor settled on the op word. On the lines of the op, the state at the op word sits at a cosine near 1 with e₁ at every slice.[^cosine] Under the whole-line pull, the use sites (`=` and the answer) have an alignment of about 0.1. With the op word alone pulled, the blocks carry a twentieth of that to `=` and none to the answer.

[^cosine]: Cosine similarity: the dot product of two unit vectors, 1 when they point the same way, 0 when perpendicular, −1 when opposite. The *alignment* of a state with e₁ is this cosine.

So the anchor may hold no more than the identity of one token. D2.2 asks whether removing the op from the axis removes the ability to perform it, and at the op word that question is confounded: masking the word would also remove the op.

The pass separates the two by running a token mask beside every edit. What the edits do beyond the mask (on the lines of the op, and on those of the other ten ops) is what the anchor adds over knowing which token names the op.

The geometry also rules out the plain projection at that site. Removing e₁ from a state that is almost all e₁ leaves a small remainder. Re-normalization then scales it up by $1/\sqrt{{1 - x_1^2}}$, so the edited state is whatever the remainder happened to be. Two operators do have a defined landing there:
(a) the reflection, which sends a state at alignment α to −α and keeps the rest; and
(b) a repulsion onto the antipode, where the rest drops out and every edited state lands on −e₁ itself.
The pair also answers a question the reflection alone cannot: whether the remainder still carries the token.

One row edits the use site `=` instead, where the whole-line pull put a little of the op. If the model reads the op from that small component, the anchor holds more than a token.

## Conditions

Nothing is trained. The pass scores fifteen stored checkpoints under the clean forward pass and four rows.

{conditions_html()}

{rows_html()}

Every row acts at all five slices, the embedding included, at one position. The answer position is not edited: the answer is read from the logits at `=`, so an edit at the answer position cannot move the completion.

**The lines.** Every probe line of every op in table A+, as built by ex-2.2.9: 5,832 lines per op, and 11,664 for each of the three order-sensitive ops, 81,648 in all. The gated measurements are taken on the primary, on the `difference` lines and on the lines of the other ten ops. The op-word arm and the control run under every row as references.

**The seeds.** Five per condition, as stored. The control seeds are those of ex-2.2.11, so every comparison with it is between seed means.

## Glossary

<dl>
<dt>Expected exact match</dt>
<dd>The model's probability mass on the colors a line's answer can be, per line, then averaged. On <code>difference</code>, whose answers are on the grid, it is the mass on the one right answer. Written <i>eem</i> in tables.</dd>
<dt>Normalized distance</dt>
<dd>The distance in grid steps from the raw answer of the line to the mean of the model's distribution over the 216 colors, divided by the distance a uniformly random color would have (chance): 0 is a perfect answer, 1 is no better than guessing. The distance of the greedy guess is quoted beside it, floored by the best grid answer. Adopted by the <a href="../answer-distance/report.py">answer-distance re-score</a>.</dd>
<dt>Op-relevance and <i>k</i></dt>
<dd>For a <code>difference</code> line, <i>k</i> is how many of the other ten ops give the same answer, and <i>r</i> = 1 − (1/10) Σ<sub>o</sub> q<sub>o</sub>(answer) is the share of the answer mass of the other ops that misses it. A <i>named</i> line has k = 0: only <code>difference</code> produces its answer. A <i>shared</i> line has k ≥ 1.</dd>
<dt>The designed null</dt>
<dd>What a model that has lost the op and nothing else would answer: the uniform mixture of the answer distributions of the other ten ops on the same operands. Its expected exact match on a line is 1 − r, so on a named line it is near zero and on a shared line it is about k/10.</dd>
<dt>Landing</dt>
<dd>The alignment with e₁ a state leaves an edit with. The reflection lands at −α; the pole at −1 exactly; the mask at the alignment of the mean state, whatever that is.</dd>
<dt>Write</dt>
<dd>The angle between the state as it arrived at an edited slice and the state the next block consumed, per slice. Closed-form for each operator, and checked against the measured angle on every line.</dd>
</dl>

Before any checkpoint is read, the op table gives the distribution of *k* over the lines of `difference`: how many of them are named, and how the shared ones split. The pass groups its per-line measurements by these bins.
"""


@memo
def relevance_table() -> str:
    from sca.data.ops import CANDIDATE_BY_NAME, OP_BY_NAME, lines, relevance

    table = OP_BY_NAME | CANDIDATE_BY_NAME
    dist = relevance(table[ex.ANCHORED_OP], tuple(table[o] for o in ex.OP_NAMES), lines())
    head = "<tr><th class=num>k</th><th class=num>share of lines</th><th class=num>null eem</th></tr>"
    body = "".join(
        f"<tr><td class=num>{k}</td><td class=num>{v:.1%}</td><td class=num>≈{k / 10:.1f}</td></tr>"
        for k, v in sorted(dist.items())
    )
    return f'<table class="report-table dense"><thead>{head}</thead><tbody>{body}</tbody></table>'


relevance_table()

r"""
The null column is the expected exact match of the designed null on a line of that bin if the answer of every other op were on the grid; the pass computes the exact value per line from the answer distributions of the ops.
"""

# --- Loading ------------------------------------------------------------------------------------


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """The published file of each ref under *into*, or None before it exists: one `get_refs` and one `get_many`."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


with tempfile.TemporaryDirectory() as _tmp:
    _files = fetch([ex.METRICS_REF], Path(_tmp))
    metrics_loaded = None if (_p := _files[ex.METRICS_REF]) is None else json.loads(_p.read_text())

if metrics_loaded is None:
    stop("_Results are not published yet: the sections below carry their frozen predictions and placeholders._")

rf"""
## The edits remove the op (H1)

**What we expect.** On the primary, each op-word edit takes `difference` out. On the named lines the seed-mean expected exact match, about 0.97 clean, falls to at most {ex.REMOVAL_GATE:g} under `reflect` and under `pole`; between {ex.REMOVAL_GATE:g} and {ex.REMOVAL_PARTIAL:g} is partial.

The mask sets the reference: it removes the word, so its own drop is what removing the op looks like, and an edit that matches it has done as much.

If `pole` clears the gate and `reflect` does not, the remainder the reflection keeps still carries the token: the model reads `difference` from the off-axis part of the state, which the pole discards.

If neither clears it, the model still applies the op with the op-word state on the far side of the sphere, so it must read the op from somewhere the axis at the op word does not reach.

The control under `reflect` is the calibration. Its op word holds almost none of e₁, so its drop on `difference` lines should stay within the task gate ({ex.TASK_GATE:g}). `pole` leaves the control untouched by construction; the pass counts the states it moves there, which should be none.

The normalized distances are quoted beside expected exact match: they say whether the removed answers move one step or across the cube.

/// admonition | TODO
One figure: per row (`reflect`, `pole`, `mask`), a column of per-seed expected exact match on the named lines, for the primary, the arm, and the control, with the gate dashed, the partial level dotted, and the failing side hatched. One table: per condition and row, the seed-mean expected exact match, its drop from clean, and the two normalized distances, on the named and the shared lines.
///

## The damage follows op-relevance (H2)

**What we expect.** With the op removed, the model answers as some mixture of the other ops, and we predict that mixture is the designed null: on the primary, under `reflect` and under `pole`, the seed-mean expected exact match in each bin of *k* is within {ex.NULL_TOL:g} of the mean 1 − r of that bin, for every bin with at least {ex.NULL_MIN_LINES} lines. The Spearman (rank) correlation between the r of a line and its drop is reported beside it.

If the drop overshoots the null on every bin, with expected exact match near zero on shared lines too, the model does not fall back to the other ops; the edited state sends it somewhere none of them go. If it undershoots on the shared bins, the model falls back to one op, or a few, that happen to agree with `difference` there. The composition below says which.

The mask is measured the same way, since the null is a claim about what a model that has lost the op does, and the mask is the plainest way to lose it.

/// admonition | TODO
One figure: expected exact match against k, one panel per op-word row, with 1 − r̄ of the null per bin as a step and the tolerance as a band around it; seed dots and seed means per bin. A table of the same with line counts.
///

## The other ten ops are untouched (H3)

**What we expect.** On the primary, `reflect` at the op word costs the other ten ops nothing: on each, the seed-mean drop in expected exact match from clean is within the task gate ({ex.SELECTIVITY_GATE:g}), or partial to {ex.SELECTIVITY_PARTIAL:g}. The control under `reflect` sits inside the same bound, and its drop is quoted beside the drop on the primary as the calibration.

`pole` is a manipulation check here: on the primary the other op words sit under 0.1 on e₁, far below the threshold of {ex.POLE_THRESHOLD:g}, so the repulsion leaves every one of their lines as it was. The pass reports the count of states it moves on those lines, which should be zero.

The mask is not expected to be selective and is not gated: it removes the word of every op, so every op should lose expected exact match under it. Its drop on each op is reported beside the drops under the edits as the size of the thing the edits leave alone.

If `reflect` costs some op more than the gate, the small alignment the other op words hold (the containment ex-2.2.14 measured, about 0.05 over the control) is load-bearing for that op, and the selectivity of the edit is a property of the threshold rather than of the axis.

/// admonition | TODO
One figure: per op, the seed-mean drop under `reflect` on the primary and on the control, with the drop under the mask as a faint reference mark, the gate dashed and the failing side hatched. A table with the seed ranges.
///

## The axis at `=` marks the op (H4)

**What we expect.** On the primary, removing e₁ at `=` (`equals`, the plain projection) leaves the completion of `difference` lines nearly where it was: the seed-mean drop in expected exact match is at most {ex.MARKER_MAX:g}. In that case the axis at `=` *marks* the op: it holds a trace that the blocks carried or the pull placed, but the model does not read the op from it.

A drop of at least {ex.CARRIER_MIN:g} would say the axis at `=` *carries* the op: the model does read that component of about 0.1. The mask row bounds how much removal that could amount to. Between the two levels the question is unresolved.

The marker outcome is the expected one, since the write at `=` is small on every condition (about 6° for an alignment of 0.1).

The op-word arm and the control run the same row as references. The arm holds about 0.05 at `=` and the control about 0.01, so both should show drops near zero. If the arm drops about as much as the primary, the model reads the trace the blocks carry. If the control drops, the projection at `=` has a cost of its own.

/// admonition | TODO
One figure: per condition, per-seed drops in expected exact match on `difference` lines under `equals`, with the marker and carrier levels as rules; beside it the same on the other ten ops as a cost check. A table with the normalized distances.
///

## The rule for the write-up

The design leaves the pass one decision: how much of the op-word line the D2.2 write-up reports.

- H1 passes on both rows and H4 comes out as marker: the anchor placed a token attribute on the axis, and suppression at the op word is masking with extra steps. The op-word line is reported briefly, as the finding that an anchor on a named op settles on its name, and the in-context grammar carries the operation question.
- H4 comes out as carrier: the use site holds the op, and suppression there is a real removal. The op-word line is reported in full, and the finer slice subsets and the ten sweep ops become worth scoring on the stored checkpoints.
- H1 misses on `reflect` alone: the remainder carries the token, which is a finding about the operators the D2.2 suppression should use, and goes to the intervention section of the design.
- H1 misses on both rows: the model reads the op from somewhere the axis at the op word does not reach, and the op-word line is reported as that finding, with the bypass test of the design brought forward.

If H4 is unresolved, the brief report stands and the `=` measurement is listed as open rather than as a marker.

H2 and H3 qualify the report rather than route it: a miss on H2 says what the fallback is, and a miss on H3 says the small alignment on the other op words is load-bearing.

## Exploratory analyses

Five descriptions are planned in advance, with no gate.

**Where the answers go.** On the named `difference` lines under each op-word row, the mass on the answer of each other op, as a composition over the ten ops: whether the fallback is the uniform mixture, one op, or a few. Beside it, the distance of the greedy guess from the answer, as a histogram in grid steps, and the mass off the color vocabulary.

**The landing, slice by slice.** For each op-word row, the alignment at the op word before and after the edit at every slice, and the alignment the next block hands on: whether a state sent to −e₁ stays there or is pulled back before the next edit, as the landing figure of ex-2.2.8 showed for a positive landing. The write angle per slice beside it, against its closed form.

The pole acts only where a state arrives at or above the threshold, so once the first edit has sent the state negative the later slices leave it alone, and the landing decays under the blocks. The reflection acts at every slice and may alternate with them.

**The target of the mask.** The norm of the mean over the states of the eleven op words before re-normalization, per slice: 1 when the op words share a state, small when they spread out. On the primary that says how far the anchored op word moved away from the other ten; on the control, how the model itself separates them. The alignment of the target with e₁ is reported with it, since the mask lands there rather than at zero.

**The op-word arm under every row.** The arm had the op word alone pulled, so its `=` holds only what the blocks carried. Its drops under each row beside those of the primary say whether the whole-line pull changed what the edits remove.

**Seed agreement.** The seed range of every gated statistic, and whether the seeds order the same way under `reflect` and `pole`.

## Discussion

This pass can settle the scope of the ex-2.2.14 result: the alignment measurements there could not tell an anchored operation from an anchored token, and suppression can.

If the edits at the op word do what the mask does and the edit at `=` does nothing, the axis holds the name of the op, and the model reads the op from that name, as a token anchor would. The pivot to the in-context grammar already assumes this outcome, so it would not be a setback. It would also put a number on how much of the op the use site holds.

The pair of operators also bears on intervention. If they agree, the remainder the reflection keeps is inert and either operator serves. If they differ, the D2.2 suppression needs to set the landing explicitly rather than leave it to re-normalization. That is the M1 argument for repulsion, applied to a categorical concept.

## Method

### The rows

Each operator acts on the unit-norm state h at one position, at every slice, and the next block consumes the result. With α = h · e₁ and u⊥ the unit vector of h with e₁ removed:

- `reflect` is the projection at γ = 2, h − 2α e₁ followed by re-normalization, which lands at −α along the same u⊥. Its write is 2 arcsin|α|.
- `pole` is the repulsion with the linear mapper: a state at α ≥ a lands at m = b, as b e₁ + √(1 − b²) u⊥, and at b = −1 the second term vanishes. States under the threshold, and states with negative α, are untouched. Its write is arccos b − arccos α on the states it moves. A state that arrives with no off-axis part has no u⊥ and is left where it is; the pass counts those.
- `mask` replaces the state at the op position with the target for the first operand of the line: the mean over the eleven op words of the clean state at that position given that operand, per slice, re-normalized. Under causal attention that state depends on the first two tokens alone, so the targets come from the 2,376 two-token prefixes, and the pass checks that the clean state on every full line equals the state of its prefix.
- `equals` is the projection at γ = 1 at `=`, with write arcsin|α|.

Every row goes through the eval contract (`sca.intervention.apply`, or its mask counterpart), and the pass asserts on every line that no other position moved and that the measured write matches the closed form.

### The measurements

Per line and pass: expected exact match against the answer distribution of the line; the distance in grid steps from the raw answer to the mean of the model's distribution over the 216 colors, over chance; the distance of the greedy guess, floored by the best grid answer and normalized the same way; the guess; and the mass off the color vocabulary. On `difference` lines, also the mass on the answer distribution of each other op, computed from the op table on the operands of the line.

The line constants come from the op table before any checkpoint is read: for each `difference` line the answer distributions of the other ten ops under stochastic rounding, their agreement with the answer of the line (r), how many of their modes equal it (k), the expected exact match of the null, 1 − r, and the normalized distance of the null.

Every gated statistic is a seed mean over the condition; the seed range is reported with it.

### Budget

Fifteen checkpoints, each under five passes over 81,648 six-token lines plus the 2,376 prefixes: a few minutes each on an L4. On CPU the smoke check took about 20 s per op on 600 lines, so a full checkpoint there is closer to an hour. `--max-containers 5 --budget 1h` on Modal, well under a dollar.

### What this pass does not do

It trains nothing, edits no operand position and no answer position, scores neither the ten sweep ops nor any slice subset, varies neither the threshold nor the landing, and does not anchor *red*. The design holds each of those for a later round or for a surprise here.
"""
