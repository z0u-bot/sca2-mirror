# Where the lean comes from, and what to try next

/// tip |
<!-- lede -->
A review of the D2.2 runs to date, asking why the edit that removes `difference` spills onto other ops, and what would stop it. On the stored checkpoints the spill comes from the first two slices. The pull there cannot be met by anything contextual, so it is met by a stand-in: lightness loaded onto e₁ in the color embedding table, or a syntax token latched to e₁. The edit at those slices then takes the lightness away from every op. Editing only the later blocks removes `difference` with almost no spill. The anti-subspace term as weighted cannot hold the stand-in off, which is why its weight has barely moved the result. The most promising next run keeps the pull off the first two slices, holds them off e₁ with the anti term or a hard constraint, and edits only where it pulls.
///

This is a reading of stored data, with no new training. It covers the 24 anchored and 24 control checkpoints of [ex-2.2.23](/docs/m2/ex-2.2.23/report.py) (12 seeds, at 200 and 400 epochs), and the states of 14 trials of the τ × λ_a sweep on the no-emb arm. The analysis scripts and the numbers behind every figure and table sit beside this file under `analysis/`.

## Observations

- [E1](#what-e₁-holds-on-the-other-ops-e1) — At the example answers of other ops, e₁ holds a per-example judgement of whether that one example fits `difference`. The spill does not follow it, so the spill is not the concept firing on the wrong contexts.
- [E2](#lightness-on-e₁-in-the-embedding-table-e2) — Every anchored run loads lightness onto e₁ in the color embedding table, darker colors further along. The lean is about three times larger on runs with no latch, and the latch decides how the run spills.
- [E3](#the-lean-fades-with-depth-e3) — The lean is strongest at the embedding and fades through the blocks. It is there at 200 and 400 epochs and on the no-emb arm alike, and the readout table has none of it.
- [E4](#the-edit-by-slice-e4) — Editing only blocks 2 to 4 removes most of `difference` with almost no spill. Editing only the embedding gives half the removal and all the spill.

## Where things stand

The recipe of record pulls every slice toward e₁ on `difference` contexts, with the pooled mellowmax term at τ = 0.1 and an anti-subspace term on every live position, and edits by projecting e₁ out at every position and slice. After ex-2.2.23 the picture was: the anchor holds and the edit removes `difference` on nearly every run trained to 400 epochs, but on most of those runs the edit also takes another op down well past the selectivity criterion, usually `darken`. The spill-by-position report then found that removal and spill share the example answers, and that on most runs one syntax token embedding lies along e₁: a latch. Runs latched on `⏎` barely spill, and runs latched on `,` spill most. The τ × λ_a sweep on the no-emb arm found that the spill persists with no latch at all, and that λ_a barely matters.

Two things were open: what carries the spill when there is no latch, and why the anti term does nothing about it. This review takes up both.

The table below lists what has been tried on the spill so far, so that the proposals at the end can be checked against it.

| What was changed | Where | What happened |
|---|---|---|
| A cap on the pull (0.8, 0.9, 0.95, none) | ex-2.2.21, ex-2.2.22 E2 | The cap does not put the spills in order; dropped |
| The pull kept off the embedding, off the last slice, or on the middle slices only (the anti term left the same slices; the edit stayed at every slice) | ex-2.2.22 E1 | Every restriction spilled more; other ops sat further along e₁ at the first two slices |
| Training for 400 epochs instead of 200 | ex-2.2.23 | The slow seeds finish; the edit removes `difference` on nearly every run and spills on most |
| The edit applied to one set of positions at a time | spill-by-position | Removal and spill share the example answers; the latch was found |
| τ and λ_a swept on the no-emb arm | τ × λ_a sweep | An edge at τ ≈ 0.07; λ_a barely matters; spill with no latch |
| The anti anneal started earlier; pooling scored over roles | schedules at the rise (closed) | The anti anneal nearly coincides with the learning-rate cosine; a peak over roles tracks the spill |

## Scope

The measurements are the ones ex-2.2.23 and spill-by-position use. The task score of an op is the EEM of the model's answer on held-out contexts of that op. Removal is the drop in the `difference` score under the edit, net of the same edit on the paired control run, as a share of the way from the clean score to the target null (the score of an ideal predictor that had lost the op). Spill is the largest net drop on any other op, with the criterion at 0.02. In E4 the edit is restricted to a set of slices and is otherwise the full edit: every position, the whole e₁ component removed.

Three cautions. The checkpoints are the ones the recipe was tuned on, so the seed-to-seed spread here is the spread the recipe has, with nothing held out. The slice-restricted edits of E4 are on models trained with every slice pulled, so they say where the removable part of the concept sits in those models, and only a new run can say whether a pull kept off the early slices would put it in the later blocks. And the comparison of the two regularizer weights at the end is a rough estimate, good to a factor of a few, meant to show which way the balance tips.

## What e₁ holds on the other ops (E1)

The first question was whether the spill is the concept itself firing on contexts that are not `difference`. The example-evidence report showed that at an example answer the anchor follows the posterior given that one example, so a `mix` context whose first example happens to be consistent with `difference` would be expected to carry some alignment there.

It does, strongly. On the 400-epoch runs the alignment at an example answer of another op is near 0.9 when that example fits `difference` on its own and near 0.1 when it does not, on every seed, latched or not. So e₁ at the example answers holds a clean per-example judgement, and other ops' contexts get it wherever an example fits.

But that is not what spills. Taken context by context on the other ops, the drop in score under the edit does not rise with the number of fitting examples in the context, nor with the alignment at them; the correlation is slightly negative on every run, around −0.1. The contexts that lose most under the edit are ones with no fitting example at all. So the spill is carried by something the edit removes at every context of the affected op, whatever its examples say, and E2 looks for it in the one place the pull reaches that no context can: the embedding table.

## Lightness on e₁ in the embedding table (E2)

At the embedding slice the state at a position is the token embedding alone; nothing has been read from the context yet. The pull asks each labeled `difference` context to have some position with α near 1 at that slice too. The only ways to meet it are at the token level. One is a latch: a syntax token every context contains, put on e₁. The other is a statistical stand-in: tokens that are more common in `difference` contexts sit further along e₁ than tokens that are not. `difference` answers are the channel-wise absolute difference of the operands, so they are darker than the colors around them, and lightness is such a stand-in.

![Three scatter panels of the e₁ component of each color embedding against the lightness of the color, marks colored by the color itself. The control is a cloud with no trend. The anchored run latched on comma is a narrow band that falls from about 0.2 at the dark end to 0 at the light end. The anchored run with no latch is a wide wedge that falls from about 0.7 at the dark end to 0 at the light end.](emb-lightness.png)

**Lightness on e₁ in the color embedding table, three 400-epoch runs.** Each mark is one of the 216 color embeddings, colored by the color it names; x is the lightness of the color (its mean channel), y its component along e₁. Left: the control at seed 700. Middle: the anchored run at seed 701, latched on `,`. Right: the anchored run at seed 700, with no latch. The title of each panel gives the correlation and the standard deviation of the component over the table.

Every anchored run has this lean, with the correlation between lightness and the e₁ component near −0.7 on all 24, against about zero on the controls. Its size depends on the latch. Where a syntax token sits on e₁ the pull is met at that token and the color table barely moves, so the spread of the e₁ component over the colors is small. Where nothing latches, the colors take the whole pull and the spread is about three times larger. The figure below sets the spill against the lean, for all 24 anchored runs.

![Two scatter panels of spill against a measure of the lean, marks shaped by the latch and colored by training length. Left, against the spread of the e₁ component over the color table: the comma-latched runs spill most at a small spread, the no-latch runs spill moderately at a large spread, and the newline-latched runs sit at the criterion with a small spread. Right, against the lightness correlation in the states of other ops at slice 2: spill rises as that correlation moves away from zero, with the newline-latched runs closest to zero and lowest.](spill-vs-lean.png)

**Spill against the lean, all 24 anchored runs of ex-2.2.23.** Left: against the standard deviation of the e₁ component over the color embeddings. Right: against the correlation between lightness and α at the color positions of other ops, at slice 2. Red is 400 epochs, blue 200; circles have no latch, up-triangles are latched on `,`, down-triangles on `⏎`, and the diamond on `?`. The dashed rule is the selectivity criterion.

The three groups behave differently. The runs latched on `⏎` barely spill, since that token comes after the answer and nothing downstream reads it, and their color table hardly leans. The runs latched on `,` have a small lean and the largest spill, because the `,` embedding sits between the operands of every example and the edit removes it from every op; spill-by-position saw the same. The runs with no latch have the largest lean and spill moderately, onto `darken` on all but one of them. The right panel puts the three on one line: the further the lightness correlation in the states has moved from zero, the more the run spills.

## The lean fades with depth (E3)

If the lean is a property of the embedding table, it should be strongest at the embedding and fade as the blocks add contextual signal. The figure below follows the lightness correlation in the states of other ops through the slices, for the ex-2.2.23 groups and for the no-emb sweep trials.

![A line chart of the lightness correlation at color positions of other ops against slice, from the embedding to slice 4. All anchored groups start near −0.7 at the embedding. The no-latch group fades to −0.6, −0.4, −0.25, and −0.2; the two latched groups jump to about −0.2 at slice 1 and fade toward −0.07; the no-emb sweep trials follow the no-latch group. The control stays at zero. Faint lines show the individual runs.](lightness-by-slice.png)

**The lightness correlation by slice.** Each line is the seed mean of the correlation between lightness and α at the color positions of other ops, with the individual runs as faint lines behind. Red circles: 400-epoch runs with no latch. Red up-triangles: latched on `,`. Blue down-triangles: 200-epoch runs latched on `⏎`. Gray squares: the controls. Teal diamonds: the eleven no-emb trials of the τ × λ_a sweep, τ from 0.05 to 0.3.

Three things to see. The lean is there at 200 epochs as at 400, so it comes with the pull, and training longer only grows it on the runs that lose their latch. The no-emb trials lean just as far at the embedding although that slice is not pulled: the pull at block 1 moves the embedding beneath it, and the anti term is off at slice 0 on that arm, so there is nothing to hold the table in place. And on the latched runs the lean in the states falls away at slice 1, which fits the latch taking the pull from the color table. The readout table has no lightness on e₁ on any run (the correlation is near zero), so the lean is on the input side only.

## The edit by slice (E4)

The last question is whether the early slices are where the spill is done. The figure below applies the full edit to one set of slices at a time, on every anchored run, and shows the net drop on each op.

![Two dot panels, one per training length, of the net drop in score for each of seven ops under five slice-restricted edits, seeds jittered behind the seed mean. At 400 epochs every edit that includes block 1 takes difference down by about 0.6; the embedding-only edit takes it down by 0.4. On the other ops the embedding-only and every-slice edits sit at 0.1 to 0.25, the block-1-only and blocks-1-to-4 edits at about 0.05, and the blocks-2-to-4 edit at zero. At 200 epochs the drops on other ops are small for every edit.](edit-by-slice.png)

**The edit restricted to a set of slices, by op.** Each column is an op and each color a slice set; the large mark is the seed mean and the small marks behind it are the runs. The y axis is the drop in task score under the edit, net of the paired control. The dashed rule is the selectivity criterion.

| Slices edited | Removal, 400 | Spill, 400 | Removal, 200 | Spill, 200 |
|---|---:|---:|---:|---:|
| Embedding only | 0.52 (0.03 to 0.79) | 0.26 (0.00 to 0.52) | 0.11 (−0.13 to 0.56) | 0.06 (0.00 to 0.23) |
| Block 1 only | 0.79 (0.68 to 0.95) | 0.12 (0.03 to 0.27) | 0.74 (0.00 to 0.95) | 0.05 (−0.01 to 0.23) |
| Embedding and block 1 | 0.83 (0.76 to 0.92) | 0.26 (0.03 to 0.51) | 0.70 (−0.05 to 0.92) | 0.08 (0.00 to 0.24) |
| Blocks 1 to 4 | 0.87 (0.80 to 0.93) | 0.12 (0.02 to 0.26) | 0.79 (0.08 to 0.93) | 0.05 (0.00 to 0.24) |
| Blocks 2 to 4 | 0.74 (0.09 to 0.92) | 0.02 (−0.01 to 0.07) | 0.53 (−0.05 to 0.95) | 0.01 (−0.01 to 0.03) |
| Block 4 only | 0.00 (−0.02 to 0.01) | 0.00 | −0.01 (−0.02 to 0.01) | 0.00 |
| Every slice | 0.85 (0.77 to 0.92) | 0.25 (0.02 to 0.47) | 0.76 (0.02 to 0.90) | 0.07 (−0.01 to 0.22) |

**Removal and spill under each slice-restricted edit**, seed mean and range over the 12 runs at each training length. Removal is the net drop on `difference` as a share of the way to the target null; spill is the largest net drop on another op.

At 400 epochs the embedding slice alone gives half the removal and all the spill: the spill under the embedding-only edit is the same size as under the full edit, run by run. Blocks 2 to 4 alone give most of the removal and a spill within the criterion on 11 of the 12 runs. Block 4 alone does nothing, which matches the readout having no lightness on e₁ and says the concept is read out of the stream before the last block. The 200-epoch runs spill little under any edit because most of them are latched on `⏎`, and their removal under the late edit is lower because the three unlatched ones lean at the embedding like the 400-epoch runs.

The one caution is in the range of the late edit. On three of the seven unlatched 400-epoch runs (seeds 700, 708, and 710) the edit on blocks 2 to 4 removes little, at 0.14, 0.60, and 0.09, while block 1 alone removes most of it on the same runs. On those runs the model has come to compute `difference` partly out of the lightness the lean put on e₁, so taking the lean away is part of how the full edit works. That is what makes a late-only edit on the current recipe a weak fix, and it is why the proposal below changes the pull rather than the edit.

## Why the anti term does not hold the lean off

The anti-subspace term is the mean of cos² against e₁ over every live position and every slice, at a weight that starts at 2.5× the anchor weight, holds at 0.3× for most of training, and anneals toward a floor. It should be what keeps the color table off e₁. Two things keep it from doing so.

The first is normalization. The anti term divides by the number of live positions times the number of slices, so each color embedding is one of a few thousand terms in a mean, and moving one of them along e₁ costs almost nothing. The pull divides by the number of labeled contexts, and mellowmax at τ = 0.1 concentrates the gradient of each context on its best position, so the embedding of the darkest token in a `difference` context gets most of that context's pull. A rough count puts the pull on a color embedding at an order of magnitude or more above the cost the anti term charges for it at the hold ratio, and further above it once the anti term anneals.[^weights] A lean of this size is then the cheapest way the model has to satisfy the slice-0 pull, and the anti term is only a mild drag on it.

[^weights]: A rough count: at the hold the anti weight is about 0.03, spread over roughly 300 live positions per window and five slices, so one embedding at cos² = 0.3 costs on the order of 0.03 × 0.3 / 1500 per window. The pull on the same embedding, where it is the best position in a labeled context, is on the order of 0.1 × (1 − α) / (labeled contexts per window, about 3) per window at each of the slices it is pulled on. The ratio is in the tens, and it depends on how many contexts the token appears in, so only the order of magnitude is meaningful.

The second is that λ_a never changed the balance. The anti weight is given relative to the anchor weight, so the sweep that varied λ_a scaled both terms together and left the ratio between them as it was, which is why it barely mattered. The one change that would have moved the ratio, the flat high anti weight tried in D2.1, cost the margin, and the trailing schedule that replaced it was tuned for the cube-wide drift of the word-token grammar, where the pull could be met by the states and did not need a stand-in.

The no-emb arm fits the same account. Leaving slice 0 out of `anchor_slices` leaves both terms off it, so the pull at block 1 moves the embedding beneath it and nothing holds the table in place (E3); and the edit still touches slice 0, so the lean is still removed from every op. That is why ex-2.2.22 found every slice restriction spilled more, with the states of other ops further along e₁ at the first two slices: the restriction took the anti term off the slices where the stand-in lives while the edit went on removing it.

## What to try next

The account above says the stand-in is forced by pulling where no contextual concept can exist, and is removed by editing there. The fix is to do neither, and to keep the early slices held off e₁ while the pull acts on the later ones. In order of how much I would expect each to tell us for its cost:

1. **Pull on blocks 2 to 4, anti term on every slice, edit on the pulled slices.** This is the run the review points to. It separates whether the concept can be pulled to the later blocks without a stand-in at the early ones, and whether the removal then holds, which the three low-removal runs in E4 leave open. On the stored runs the late edit gave a removal around three quarters with a spill within the criterion, so that is the point of comparison. It needs a small change to the training step, since `anchor_slices` currently restricts both terms together (`src/sca/anchoring.py`): the anti term wants its own slice set. Report the every-slice edit beside the pulled-slices edit, since the two differ on the stored runs and a reader will want to know whether the early slices stay clean. Four seeds at 400 epochs would do; the recipe-of-record runs at those seeds are the control.

2. **The same, with a hard constraint in place of the anti term at the embedding.** `clean_embedding_rows` already projects e₁ out of named embeddings after each step (ex-2.2.7 used it on the syntax tokens). Applied to every token it removes the latch and the table lean at no cost, and leaves block 1 to the anti term. It is a stronger statement than the anti term and simpler to reason about, so it is worth one arm beside the first proposal rather than a separate experiment. If the first proposal still leans the table through block 1, this arm says whether holding the table is enough.

3. **The anti-subspace sweep from the backlog**, in a form that changes the ratio at the early slices. A sweep of the global weight or of the anneal start is unlikely to move the spill, for the reasons above: λ_a scales both terms, and the anneal timing mattered in D2.1 for a drift the states could carry. What could move it is a per-slice anti weight, or the term normalized per labeled context like the pull, so the two are on the same footing. I would run that after the first proposal, and only if the early slices still lean when they are not pulled.

4. **The lean-timing scout from the backlog.** Part of its question is answered here: the lean is in the table by 200 epochs and grows on the runs that lose their latch by 400. What it would add is when the `⏎` latch gives way to the table lean, since most 200-epoch runs are latched on `⏎` and clean, and most 400-epoch runs are not. That would matter if the first proposal fails, as early stopping before the hand-off would then be the remaining lever; otherwise a per-slice α trajectory is a cheap addition to the first proposal's runs rather than a scout of its own.

5. **Keep the recipe and edit only blocks 2 to 4.** No training is needed, and E4 has the result: within the criterion on most runs, with little removal on three unlatched seeds. It is a fallback if a late-only pull cannot hold the concept, and it would change what the edit claims, since the early slices would keep the lean the edit no longer removes.

What I would not spend runs on: the cap (ex-2.2.22 settled it), τ below 0.07 (the sweep found the edge), and leaving slices out of the pull without also keeping the anti term on them (ex-2.2.22 did that and it spilled more).

## Glossary

Latch
:   A syntax token embedding on e₁: a token every context contains, so putting it on the anchor direction satisfies the pull at the embedding for every context at once. Found by spill-by-position.

Lean
:   Lightness on e₁ in the color embedding table: darker colors sit further along the anchor direction, a stand-in for `difference` at the slices where nothing contextual exists.

Stand-in
:   Something token-level that satisfies the pull where no contextual concept can: a latch or a lean.

Removal
:   How much of `difference` the edit takes out: the net drop in its score under the edit, as a share of the way from the clean score to the target null.

Spill
:   The largest net drop in score on another op under the edit. The selectivity criterion is 0.02.

Target null
:   The score an ideal predictor would get on `difference` contexts if it had lost that op and answered from the remaining ones.
