# title: Ex 2.2.12: what the stream holds on hue-hsv, and a small recipe sweep

# The design constants come from `experiment.py` beside this script (the script's directory is on
# sys.path while it runs). During preregistration that module is constants only.
import experiment as ex

# What a result section shows while the run has not published yet.
RESULTS_TO_COME = "/// admonition | TODO\n    type: warning\nResults to come.\n///"


def conditions_html() -> str:
    """The sweep's conditions, one row each, as a compact report table."""
    head = "<tr><th>condition</th><th class=num>seeds</th><th>what changes</th><th class=num>τ</th><th class=num>λ_a</th><th class=num>anti peak</th><th class=num>blocks</th><th>home of <em>red</em></th></tr>"
    rows = [
        f"<tr><td><code>{c.name}</code></td><td class=num>{c.seeds}</td><td>{c.title}</td><td class=num>{c.tau:g}</td>"
        f"<td class=num>{c.lam:g}</td><td class=num>{c.anti_peak:g}×</td><td class=num>{c.n_layer}</td><td>{c.subspace}</td></tr>"
        for c in ex.CONDITIONS
    ]
    return f'<table class="report-table dense"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


rf"""
# Ex 2.2.12: what the stream holds on `hue-hsv`, and a small recipe sweep

/// tip |
<!-- tl;dr -->
Scouting, in two parts: what ex-2.2.11's stored models kept of *red* on the one op where the projection did not remove it, and a small sweep of the recipe, with *red* anchored to a plane among the changes, to find a setup that removes cleanly on every op. Nothing here is a result; the handover re-run that follows adopts what this proposes and scores it at fresh seeds.
///

## Observations

- [Which lines survive on `hue-hsv`](#which-lines-survive-on-hue-hsv-part-1) —
- [Where the surviving hue is written](#where-the-surviving-hue-is-written-part-1) —
- [Where the op1 drift comes from](#where-the-op1-drift-comes-from-part-1) —
- [The sweep](#the-sweep-part-2) —

[The proposal](#the-proposal): —

## How to read this draft

This is a scouting round with a frozen plan. The measurements and the sweep's conditions were fixed before any run, at commit `TODO`, and the promotion rule below says in advance what counts as a proposal and what happens if nothing qualifies. Everything after that commit is either observations filled into their sections or exploratory, marked as post hoc. A scouting round scores nothing: the next preregistered experiment adopts what it needs from here and checks it on models it trains itself.

## Why this experiment

We have a training setup that nearly works, and the anchored-op experiments that follow will read their results through it. So the setup should be one we understand well: every unexplained leftover in it is a confound waiting for a later experiment, and a blind spot we can name now is one we can design around. [Ex-2.2.11](../ex-2.2.11/report.py) put *red* on one axis of a small transformer's residual stream,[^rs] taught it eleven color operations, and then projected the axis out. On ten of the eleven ops the model then lost *red*: it could no longer answer the lines whose answer needs the red operand's hue. On `hue-hsv`, the op that takes its hue from the second operand, the model kept about a quarter of those answers, a little over the gate, so the rule we froze said the setup is not adopted.

[^rs]: The *residual stream* is the running vector of activations that each layer of a transformer reads from and writes back to.

We suspect we know why, and the first part of this experiment checks it. Picture hue as a clock face with red at twelve. The anchored axis measures how red a color is, and that is the same for a color a little clockwise of red (toward orange) and a little anticlockwise (toward pink). So the one thing the axis cannot hold is which side of red a color sits on. Every other op reads the red operand through its channels, and the red channel is what the axis holds. `hue-hsv` with red at op2 is the only case that needs the side. If the model keeps the side somewhere off the axis, that is what survives the projection. Red being at zero degrees is only a coincidence; a green anchor would have the same blind spot around green.

If that holds, no amount of pulling harder will fix it, because one axis cannot hold a two-sided quantity, whereas a plane can. So the second part trains a small sweep on the handover setup: *red* anchored to a plane and to an axis, at four and at six blocks, plus the recipe's two force factors one step up each (which we expect to do nothing for `hue-hsv`, and a null there is worth having), the plane at the higher anchor weight, and two sharper settings of the pooling temperature τ, which have their own question to answer. Ex-2.2.11 also left the op1 alignment higher than its references and drifting before the anneal, and the τ conditions and one of the stored-checkpoint measurements are there to make progress on both.

The setup we sweep is ex-2.2.11's `handover`: [table A+](../ex-2.2.4/report.py#the-op-set), the stochastic corpus, the whole-line labeller, the untied readout, and ex-2.2.3's recipe. Its twenty seeds are the reference condition, and memoization makes that condition free.

## Conditions

The two parts are independent: part 2 does not wait on part 1.

### Part 1: the stored checkpoints

Part 1 trains nothing. It scores the twenty `handover` checkpoints of {ex.REFERENCE_EXPERIMENT}, with its probe set and its removal lines, under a few more edits than ex-2.2.11 scored. Where a measurement has a comparison, `handover-slot` and `handover-tied` are scored too.

### Part 2: the sweep

With the ex-2.2.11 recipe as a base, every condition changes one or two things from the reference.

{conditions_html()}

**`{ex.REF.name}`** is ex-2.2.11's candidate at its twenty seeds, and the other conditions train at its first {ex.SWEEP_SEEDS}, so every condition pairs with the reference seed for seed.

**`tau-0.03`** and **`tau-0.01`** sharpen the pool. Mellowmax at τ over a line's positions is roughly the best position minus τ times the log of the line length, and the pull it returns is a softmax at that τ, so a longer label span with the same τ spreads a little more of the pull onto positions that are not the red operand. The whole-line label made the span longer, so these two conditions ask whether a sharper pool takes ᾱ at op1 back down.

**`lam-0.2`** and **`anti-5`** are the force factors: twice the anchor weight, and twice the anti-subspace peak. They are here so that a null can be read: if the plane fixes `hue-hsv` and these do not, the fix was the shape of the home and not its strength.

**`L6`**, **`plane`**, and **`L6-plane`** with the reference are a two-by-two of depth and subspace. Under the plane conditions *red* is pulled toward the span of e₁ and e₂ rather than toward e₁. A state's alignment with the plane is the length of its projection onto the pair, which is a cosine too since every state is unit-norm, but unsigned: it says how much of the state is in the plane and nothing about the direction within it. The anchor term pulls that length toward one on the labelled lines, the anti-subspace term is its square over every live position, and the removal projects the whole plane out. Six blocks instead of four is the other capacity factor, in case the HSV ops want more depth; it changes the number of slices too, so every slice-averaged measurement is reported per slice as well.

**`plane-lam-0.2`** is the plane at the doubled anchor weight, in case the plane needs more pull than the axis did. Without it, a null on the force conditions would only say that the axis cannot be pushed harder, and a partial result on `plane` could not be told from a plane pulled too gently.

### The measurements

Two things differ from ex-2.2.11 in how the sweep is scored. The plane conditions are scored on the plane, and since an unsigned two-dimensional alignment sits higher than a signed one-dimensional one even for a state that has nothing to do with *red*, every plane measurement is compared with the control checkpoints scored the same way, which are stored and cost nothing to score. And every op is scored on the removal lines chosen by hue, as ex-2.2.11 did, with `{ex.MISSED_OP}` split by slot.

## Glossary

<dl>
<dt>Removal lines</dt>
<dd>The red lines whose answer needs the red operand's hue: some permutation of that operand's channels moves the true answer far. Ex-2.2.11's rule, unchanged. On <code>hue-hsv</code> they are the lines with red at op2.</dd>
<dt>Kept share</dt>
<dd>How much of its clean accuracy on the removal lines a model keeps after the projection. One means the projection did nothing. Zero means every removal-line answer changed, which is the gate's sense of <em>red</em> being gone; the model may still land on a near neighbour of the true answer, and how far the answers move is a separate question that ex-2.2.10's answer-cube figures ask.</dd>
<dt>Side of red</dt>
<dd>Whether the red operand sits on the red axis of the cube, leans toward orange, or leans toward pink ({", ".join(ex.SIDE_GROUPS)}). The part of the hue the anchored axis cannot hold.</dd>
<dt>ᾱ at op1</dt>
<dd>The mean alignment with the axis over every color at the first operand position. How much the colors that are not red have drifted onto the axis.</dd>
<dt>Band</dt>
<dd>The seed spread a statistic showed at ex-2.2.11's twenty seeds. A difference smaller than the band is not resolved.</dd>
</dl>

## Which lines survive on `hue-hsv` (part 1)

**What we expect.** On the `{ex.MISSED_OP}` removal lines, the kept share under the projection splits by the side of red: near zero where the red operand has G = B, and well above zero where G ≠ B. If the kept share is the same on all three sides, the axis is not the reason, and the plane conditions of part 2 lose their motivation; they run regardless, and would then be read as a capacity change with no hypothesis behind it.

[Ex-2.2.10](../ex-2.2.10/report.py#where-the-answers-go) already saw the two sides. Its answer-cube figure for `{ex.MISSED_OP}` with red at op2 shows the projected answers leaving red in two lobes, one toward orange and one toward pink, and `sat-hsv` and `value-hsv` with red at op1 fan the same way, so the side survives the projection on those ops too; it just cannot rescue an answer that needs red's saturation or value. That figure pools every line. This measurement pairs each answer with its own operand's side, which is what says whether the lobes are the side of red or something else.

/// admonition | TODO
One figure: kept share per line on the `hue-hsv` removal lines, grouped by the side of red, one panel per condition (`handover`, `handover-slot`, `handover-tied`), each dot a seed mean per line and the bar the group mean with its seed range. One table: the group means with line counts.
///

## Where the surviving hue is written (part 1)

**What we expect.** With the projection applied at the embedding only, `{ex.MISSED_OP}` keeps about what it keeps under the full projection: the side is written at the embedding, where the red operand's token is, and the blocks read it from there. If removal at the blocks alone takes as much as the full projection, the side is re-derived from the other channels inside the blocks, and the plane at the embedding would not be enough.

/// admonition | TODO
One figure: kept share on the `hue-hsv` removal lines under each of the {len(ex.BYPASS_EDITS)} edits ({", ".join(f"`{e}`" for e in ex.BYPASS_EDITS)}), seed means with ranges, `handover` only. The same edits on `mix` beside it as the op where the full projection removes cleanly.
///

## Where the op1 drift comes from (part 1)

**What we expect.** On the red lines, ᾱ at op1 is higher on the lines the whole-line labeller labelled through their answer than on the lines it labelled through an operand. That is the candidate the containment item named for the labeller's half of the rise: a line whose answer draws is pulled at every position, op1 included. If the two groups read the same, the labeller's half of the rise has another cause.

/// admonition | TODO
One figure: ᾱ at op1 on `handover`'s red lines, split by how the line earned its label ({" against ".join(ex.CONTAINMENT_SPLIT)}), per slice, with `handover-slot` beside it as the condition with no answer-labelled lines.
///

## The sweep (part 2)

**What we expect.** The plane conditions (`plane`, `L6-plane`, `plane-lam-0.2`) bring the `{ex.MISSED_OP}` kept share under the gate by more than the band, and the force conditions (`lam-0.2`, `anti-5`) do not move it. τ moves ᾱ at op1 down and nothing else. Depth on its own does little for `{ex.MISSED_OP}`, and the factorial says whether the plane needs it. Every condition keeps the task, the margin, the lead, and the contrast inside ex-2.2.11's gates; a condition that does not is reported and cannot be proposed.

/// admonition | TODO
One figure per measurement, with a paragraph between them saying what each shows: kept share on `hue-hsv`, kept share on the ten other ops pooled, ᾱ at op1, then the margin and the non-red deficit together as the cost side. In each, every condition is a column of five seed dots with its mean, the reference's twenty seeds are a band across the panel, and the gate is a line with hatching on the failing side. One table at the end of every condition and every measurement, seed means with ranges.
///

## The proposal

The promotion rule, frozen before the run:

> {ex.PROMOTION}

> {ex.TAU_PROPOSAL}

{RESULTS_TO_COME}

## Exploratory analyses

Not part of the plan. Anything we think of after seeing the data goes here, marked as post hoc.

## Discussion

/// admonition | TODO
What the measurements say about the blind spot, which condition (if any) the re-run adopts, and what the anchored-op experiments inherit. About 200 words.
///

## Method

### The side of red

A grid color's side is the sign of G − B. The seven grid colors at or above the red dose split TODO / TODO / TODO across G = B, G > B, and G < B, and the removal lines of `{ex.MISSED_OP}` carry those counts in the first figure's table.

### The plane

The home of *red* under the plane conditions is axes {ex.PLANE_AXES[0]} and {ex.PLANE_AXES[1]} of the stream, e₁ and e₂ together. Alignment is the length of a state's projection onto the pair; the anchor and anti-subspace terms, the trajectory measurements, and the projection operator all take the pair where they took the axis. Every plane measurement is compared with the control checkpoints scored on the same pair. The concept holds two coordinates of sixty-four rather than one, and its variance share counts both.

### Budget

{ex.NEW_RUNS} runs of ex-2.2.11's length at d64 (the two `L6` conditions about half again as long), about three minutes a run on an L4, plus scoring: the part 1 edits on three conditions' checkpoints, and the sweep's conditions under ex-2.2.11's operators. The reference condition is memoized from ex-2.2.11.
"""
