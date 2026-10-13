# D2.2 design: anchoring an operation the model infers

D2.2 asks whether Sparse Concept Anchoring works for a concept that is never written down: the model has to infer it from context. Each line shows a few solved equations that use the same hidden operation, and the model has to infer the op to answer the last equation. We anchor one op to the first axis of the residual stream, then remove it by editing that axis out.

Each line of the corpus is one context: three solved examples of one op, then a query under the same op, written `op1 ? op2 = answer` with a constant `?` where the op word would be. The model reads the examples, infers which op fits them, and completes the query. We can compute how strongly each context indicates each op (the posterior over ops), and we evaluate the predictive power of each model against that, with and without being edited to remove the anchored op.

On the recipe of record, the anchor holds at a small cost to the task. Turning the edit up takes `difference` out gradually, and at eight fresh seeds the other ops stayed within the selectivity criterion on every run.

Still to show: that the removal follows how strongly each context indicates the op, the write bound on the inferred op, and where the edited model lands against the target null.

How we got here is in the [index](/docs/index.md#d22-anchoring-an-operation). The [pivot](pivot.md) explains why we took the op words out of the grammar.

## The setup today

Our training recipe evolves with each experiment, as we learn more about how SCA behaves in transformers. This is our "recipe of record": by default, the next experiment should use these parameters.

Ops
: Seven: `mix`, `lighten`, `darken`, `difference`, and three HSV blend modes (`hue-hsv`, `sat-hsv`, `value-hsv`). Four ops whose answers often coincide with those of another op were dropped from the earlier table of eleven.

    From [ex-2.2.18](../ex-2.2.18/report.py) and [ex-2.2.19](../ex-2.2.19/report.py).

Rounding
: An answer that falls between grid levels rounds up or down at random, in proportion to where it falls. So the answer to a pair is a distribution over a few colors, and a model is scored on the probability it puts on them.

    From [ex-2.2.4](../ex-2.2.4/report.py) and [ex-2.2.5](../ex-2.2.5/report.py).

Contexts
: Three examples and a query. Three examples in ten show the answer another op would give (replacement op noise at 0.3), so the examples leave some contexts in doubt about their op; a few answers are random colors (cube noise at 0.02). Contexts with varying numbers of examples were tried and cost a little skill.

    From [ex-2.2.16](../ex-2.2.16/report.py); the varying counts from [ex-2.2.22](../ex-2.2.22/report.py).

Model
: d64-L4 simplified nGPT, with a readout table of its own (untied from the embedding table) and attention that stops at the line break.

    From [ex-2.2.7](../ex-2.2.7/report.py) (readout) and [ex-2.2.17](../ex-2.2.17/report.py) (mask).

Training
: 400 epochs, cosine schedule peaking near 0.003. At 200 epochs a few anchored runs missed the second rise in task skill; at 400 every run made it.

    From [ex-2.2.17](../ex-2.2.17/report.py), [ex-2.2.19](../ex-2.2.19/report.py), and [ex-2.2.20](../ex-2.2.20/report.py) (schedule); [ex-2.2.23](../ex-2.2.23/report.py) (length).

Anchored op
: `difference` on e₁, with no *red* anchor.

    From [ex-2.2.14](../ex-2.2.14/report.py).

Labels
: The whole context, on about one `difference` context in fifty. The label says only that the context uses `difference`; the model never sees it, and no other op is labelled.

    From [ex-2.2.14](../ex-2.2.14/report.py) and [ex-2.2.21](../ex-2.2.21/report.py).

Pull
: On a labelled context, the anchor term pulls blocks 2 to 4, pooled over the positions of the context at a temperature $\tau_p$ of 0.1, so most of the pull lands on the example answers. There is no cap on the alignment. A context the training window cuts short is not pulled. The embedding and block 1 are left out because, before the model has read the context, the pull finds a stand-in there (see the lean, under (b)).

    From [ex-2.2.15](../ex-2.2.15/report.py) (whole contexts), [ex-2.2.22](../ex-2.2.22/report.py) (no cap), the [τ × λ_a sweep](../tau-lambda-sweep/report.py) (τ_p), and [late-pull](../late-pull/report.py) (the slices).

Anchor schedule
: The anchor weight $\lambda_a$ (peak 0.1) warms up over the first tenth of training, holds, then eases over the last tenth to a tenth of its peak. A second term, the anti-subspace term, pushes the states of every other context off e₁. Its weight $\lambda_{\bar s}$ is set by slice, as an absolute number: 0.25 on every slice at the start, annealing by epoch 360 to a *hold* that falls off in a straight line with depth, from 0.2 at the embedding to 0.03 at block 4 (the `falloff` arm). The high hold on the first slices keeps the stand-in off e₁; the low hold on the pulled blocks leaves room for the pull.

    From [ex-2.1.10](../ex-2.1.10/report.py) (anchor weight), [slice-terms](../slice-terms/report.py) (weights by slice), and [split-anti-seeds](../split-anti-seeds/report.py) (the falloff).

Verification lines
: Allowed, since they leave completion unchanged; the `hinge` condition trains without them.

    From [ex-2.2.21](../ex-2.2.21/report.py).

Edit
: Remove a share of the component along e₁ from the state at every position and every slice. That share is the *dose*, from a quarter to all of it. Edits on a subset of slices are used to locate the op.

    From [ex-2.2.21](../ex-2.2.21/report.py).

Scoring
: Task: the probability the model puts on the right answer on held-out contexts (expected exact match), beside the Bayes ceiling, which is the best any model could do given the examples, and the control, a model trained without the anchor at the same seed. The task gate fails if the mean change against the control falls below −0.02, or the change on any one op below −0.05. Removal: the drop on `difference` as a share of the way to the *target null*, what an ideal predictor that had lost `difference` and nothing else would answer; it has to grow with the dose. Selectivity: no other op may drop by more than 0.02 at any dose (the *selectivity criterion*); the largest such drop is the *spill*. Both drops are net of what the same edit does to the control. Landing: the share of the distance to the target null that the full edit covers, context by context.

    From [ex-2.2.16](../ex-2.2.16/report.py), [ex-2.2.21](../ex-2.2.21/report.py), [ex-2.2.22](../ex-2.2.22/report.py), and [split-anti-seeds](../split-anti-seeds/report.py) (task gate).

## What we want to be able to say

Ultimately, we want to see whether SCA works in transformers as a general alignment technique. For the current, narrow domain of color-mixing, we claim:

### (a) An inferred op can be anchored without costing the task

The pull puts `difference` contexts on the axis, and the model answers as well as one trained without the anchor.

**Observations.** The anchor sits on the example answers, where the context has shown the most about its op, and hardly at all at the query `=`, where the answer is predicted. At those answers the alignment rises with the posterior, on every anchored condition, while the control stays flat near zero. It does so at a fixed answer index too, so it is not only that evidence builds up along a context. That is what we would hope to see if the anchor holds the op the model inferred, and not a shortcut such as a characteristic answer color, which would not follow the evidence.

The cost to the task is small. At eight fresh seeds and 400 epochs, the recipe of record scored 0.007 below the control on average. It missed the task gate once, on `hue-hsv` at seed 708, a little past the limit of 0.05. Every loss past that limit, in every arm of that experiment, was on one of the HSV ops. Training for 400 epochs settled the earlier question of slow seeds, which at 200 epochs could not be told apart from a cost. Still to show: that the rest of what the model represents is unchanged, which was measured for the op word and not yet for the inferred op.

From [ex-2.2.21](../ex-2.2.21/report.py) (H1, E1), [ex-2.2.22](../ex-2.2.22/report.py) (E1, E3, E5), [ex-2.2.23](../ex-2.2.23/report.py), [split-anti-seeds](../split-anti-seeds/report.py) (Decision), and [ex-2.2.14](../ex-2.2.14/report.py) (the probe scan).

### (b) Removing the op is graded and selective

Graded means that a stronger edit takes out more of `difference`. Selective means that the other ops are left as they were, at every dose. In M1 this was the dose-response curve: a stronger edit removed more of the concept, and colors unrelated to it stayed as they were.

**Observations.** On the earlier grammar, where the concept was *red*, projecting the axis out removed *red*, more so the redder the line, and some of the cost fell on lines with no red in them. That cost came through the embeddings of syntax tokens, and a readout table of their own cleaned them. With that change, removal was clean on ten of eleven ops.

On the in-context grammar, an edit at the query `=` has little to act on, since the anchor sits on the example answers, and none met both criteria. The edit at every position does work: turning it up takes `difference` out step by step. Editing the example answers alone takes out most of the op, so it seems the query takes the op from the examples.

Trained to 400 epochs on the earlier recipe, which pulled every slice, the edit spilled onto other ops on nearly every run that learned the HSV ops. The spill came from a stand-in, the *lean*. At the embedding and block 1 the model has not yet read the context, so it meets the pull with a token-level substitute: it loads lightness onto e₁ in the color embedding table (darker colors are the nearest thing to a `difference` answer), or latches a syntax embedding onto e₁. The lean fades with depth, but the edit at those slices takes the lightness away from every op. Pulling only blocks 2 to 4 cut the lean to about a quarter, and holding the anti-subspace weight high on the first two slices kept it off.

On that recipe, with the anti weight falling off with depth, the full edit removed 0.85 of the way to the target null on average at eight fresh seeds, and stayed within the selectivity criterion on every seed. Holding the anti weight at one level on every slice lost the anchor altogether on one seed in each arm.

Still to show: that the damage on each context follows how much its answer depended on `difference`, which the target null predicts context by context.

From [ex-2.2.1](../ex-2.2.1/report.py), [ex-2.2.7](../ex-2.2.7/report.py), and [ex-2.2.11](../ex-2.2.11/report.py) (*red*); [ex-2.2.21](../ex-2.2.21/report.py) (E2), [ex-2.2.22](../ex-2.2.22/report.py) (E1, E2), [ex-2.2.23](../ex-2.2.23/report.py), [spill-by-position](../spill-by-position/report.py), [embedding-lean](../embedding-lean/report.py), [late-pull](../late-pull/report.py), [slice-terms](../slice-terms/report.py), and [split-anti-seeds](../split-anti-seeds/report.py) (H1, H2).

### (c) The edited model answers in a predictable way

Once `difference` is removed, the model should answer as an ideal predictor that never knew `difference` but still weighs the other ops by how well they fit the examples: the target null. If it does, the edit has a destination we can state in advance, and we may not need to train one.

**Observations.** Looked at after the fact, the edited `hinge` runs answered `difference` contexts much as that predictor does: the mass leaves the `difference` answer and goes to the other ops in about the proportions the posterior gives. On three fresh runs, counted by op over all contexts, the mass goes about where the target null puts it. Context by context the match is looser: the edit moved each run a little under half of the way to the target null, short of the half we flat-out guessed at. Most of the distance that remains is on answers another op could give, shared among them differently from the target null; the edit moves little mass onto colors no op gives.

On the recipe of record the full edit lands about half of the way (0.52 on average at eight fresh seeds).

So the anchor alone gives the edit a direction and not the whole distance. Whether a trained fallback, a designed response to the edit, would close the rest, or whether the recipe can be improved so that it does, is open.

From [ex-2.2.21](../ex-2.2.21/report.py) (E2, post hoc), [ex-2.2.22](../ex-2.2.22/report.py) (H2, E4), and [split-anti-seeds](../split-anti-seeds/report.py).

### (d) The side-effects are bounded by construction, and we know at which layers the op is held

Because we placed the axis, we can say in advance how far an edit moves each state: the write bound.

**Observations.** The write bound held on *red*: at every slice past the embedding, the move on lines with no red in them stayed within the bound, or at most a quarter over it, as we allowed in advance. *Red* was read in the first two blocks. For the inferred op, the alignment on `difference` contexts rises through the stack.

With the pull on blocks 2 to 4, editing block 1 alone removes most of the op on every seed, though nothing pulls block 1; we don't yet know why. Editing blocks 2 to 4 alone removes nearly all of it on most runs and falls short on the same seed in every arm, which fits the seed rather than the terms deciding where it misses. Pooling the pull over slices as well as positions settled it on block 4 on most seeds, and on the query answer on one, where the op could not be removed; it was not adopted.

Still to show: the write bound on the inferred op.

From [ex-2.2.1](../ex-2.2.1/report.py) (H4), [ex-2.2.22](../ex-2.2.22/report.py) (E1), [late-pull](../late-pull/report.py) (E2), [slice-terms](../slice-terms/report.py) (E1, E3), and [split-anti-seeds](../split-anti-seeds/report.py) (E1).

### (e) Scarce, noisy labels are enough

About one `difference` context in fifty is labelled, and the label says only that the context uses `difference`. That is the shape of label we expect to have for a concept in natural language, where labels will be scarce and sometimes wrong.

**Observations.** At that rate the op is anchored, on every condition tried. On the op word, labelling one `difference` line in fifty placed the op as well as labelling every `difference` line, and so did the same scarce labeller with a fifth of its labels on the wrong op. The labels here are already wrong in one sense: a label marks the op that generated the context, even when its examples favor another op.

Still to show: wrong labels on the inferred op.

From [ex-2.2.14](../ex-2.2.14/report.py) (the label arms) and [ex-2.2.21](../ex-2.2.21/report.py) (S1).

## What comes next

Each step below is written as the opening line we hope to write once it has run.

*An anchor that ramps up with depth.* An anchor weight that rises from the early slices to block 4, mirroring the falloff of the anti weight, did as well as the step at block 2, so neither term needs a slice boundary chosen by hand. (Running now.)

*The recipe sweep around the falloff.* Around the recipe of record, removal and spill moved smoothly with the anchor weight, the pool temperature, and the anti weights, so the recipe sits on a broad region and not a narrow edge.

*A pooled anti term.* Pooling the anti-subspace term over slices, so that its weight goes where the states sit closest to e₁, kept the stand-in off e₁ without a weight chosen for each slice.

*Where the edited model ends up.* A fallback trained toward the target null brought the edited model the rest of the way there, with less scatter across seeds than the anchor alone gave; or the anchor alone turned out to be enough once the recipe settled, and no fallback was needed.

*The layer sweep.* Editing the first few slices removed as much of the op as editing all of them, and anchoring those slices alone kept the task score and the selectivity, so the bound could be stated for the slices that matter. Beside it, a baseline trained on the same labels (SGTM) showed what anchoring adds over a method that does not place the concept.

## Out of scope

The verification measurements (D2.3), though verification lines are allowed in the corpus. Topic markers, and anchoring at fine-tune time, both open questions in the [pivot](pivot.md#open-questions). Several ops on separate axes. RMU and SAE baselines, which wait for D2.3.

<details markdown="1"><summary>About this page</summary>

Rewritten 2026-10-06, to say where D2.2 stands in one sitting. The plan as it stood through ex-2.2.22, with its route, risk table, and decisions, is [archived](design-2026-10.md) and in git. Updated 2026-10-07 with the [ex-2.2.22](../ex-2.2.22/report.py) decision: no cap, every slice, three examples per context. Updated 2026-10-13 for the work since: 400 epochs (ex-2.2.23), the lean and the pull on blocks 2 to 4 (embedding-lean, late-pull), and the anti weight by slice with the falloff as the recipe of record (slice-terms, split-anti-seeds).

</details>
