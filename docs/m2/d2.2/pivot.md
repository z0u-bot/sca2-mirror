# D2.2 pivot: an operation the model has to infer

*Draft for discussion, 2026-09-23.* A proposal to change which concept D2.2 anchors. The machinery stays the same, and so do the claims in the [design](design.md#what-we-want-to-be-able-to-say). Nothing here is adopted until it has been through review.

## Why

[Ex-2.2.14](../ex-2.2.14/report.py) anchored `difference` and every gate passed, but the anchor went to the easiest place it could reach. The pull put the op word's embedding on e₁, at a cosine near 1, and the op position stays there at every slice. When only the op word was pulled, the blocks carried a twentieth of that alignment to `=` and none of it to the answer.

So the anchored concept is an attribute of one token: which op this word names. That is the D2.1 result again, with a categorical attribute in place of a graded one. Suppression at the op word will very likely remove the op, but deleting the word would do the same, so it cannot show what anchoring adds. The layer-local bound also loses its meaning there, because a state that is all concept has nothing left to rescale.

Every op in the current grammar is named by a word, so choosing a different op will not change this.

## What M3 needs from D2.2

M3's lead target is sycophancy. No single token names it. It is inferred from the context and used far from where its evidence appears, and its labels will be scarce and noisy. Before M3, D2.2 should tell us:

1. whether SCA can anchor a concept the model has to *compute* from context, rather than look up from a token;
2. whether suppressing that concept removes the ability, graded and selective, within a bound set before the intervention;
3. where in the depth such a concept is best anchored (the layer sweep);
4. whether scarce, noisy labels are enough.

The current design answers (4), with the caveat that a scarce labeller only had to find one embedding row. It answers (2) only in the token-mask sense, and (1) not at all.

## The proposal

**Main line: take the op word out of the grammar.** A context shows a few solved lines of one op, written with a neutral symbol in place of the op word, then asks for a new line under the same op:

```
red ~ blue = magenta
white ~ cyan = red
yellow ~ red =
```

Here the op is `difference`, and the model has to work that out from the first two lines to complete the third.

The op becomes a latent task that the model infers from examples. The anchored concept is "the op in this context is `difference`", and no token carries it.

This setting has a property the color grammar lacked. For any context, the posterior over ops given the solved lines can be computed from the op table. That gives three things for free:

- a **graded stimulus**, how strongly the context points to `difference`, which plays the part redness played for *red*;
- a **designed null** for suppression: the answer distribution under the posterior with `difference` removed;
- a **label-noise model**: a labeller that labels contexts by their posterior rather than by their true op is realistically noisy.

This is also where there is prior post-hoc work to compare against. Function vectors (Todd et al., 2023, arXiv:2310.15213) and task vectors (Hendel et al., 2023, arXiv:2310.15916) find that an in-context task gets compressed into a direction in the residual stream. SCA's version places that direction during training, where those methods search for it afterwards.

**Bridge: keep the pull off the op word.** On the current grammar, pull only at the use sites (`=` and the answer), or only past the embedding slice. The op word is then no longer an easy target, and the anchor has to sit on a state the blocks compute. This is cheap, since ex-2.2.14 already has span masks and [ex-2.2.7](../ex-2.2.7/report.py) had a blocks-only arm; the [span-location item](/todo/science/locating-concept-inside-labeled-span-without-being.md) queued the same exclusion for *red*. The risk is that the model copies the op word's identity to `=` through attention, which would be a lookup one hop later. It stays useful as an arm under the main line, because it separates "anchor where the concept is named" from "anchor where it is used".

## Sequence

1. **Suppress `difference` on the stored ex-2.2.14 checkpoints.** Scoring only, as planned in the [design](design.md#suppress-the-operation-and-the-operands): the op-word edit against a token mask, and the use-site edits on the whole-line primary. It turns "trivial" into a measurement, and it tells the bridge where to start.
2. **The bridge**, a small pilot on the current grammar.
3. **The in-context grammar**: an op table without op words, the posterior over ops, and a regression check that the model learns the task at all. Like ex-2.2.3, this is a grammar change and needs its own control. The model may need more depth or width than d64-L4 to infer a task in context.
4. **Anchor the latent op**, then **suppress it**, then the **layer sweep** and the **SGTM baseline**, as in the current design.

## What carries over

The eval contract and the operator library; the fallback term, whose null now comes from the posterior; the labellers, which gain a context-level keying; the untied readout; the op table A+ and its op-relevance, which now decide how many shots it takes to pin down an op; and the scoring conventions from ex-2.2.9 onward. The *red* line of work is finished as a D2.2 prerequisite and needs nothing more for this.

## How this changes D2.3

D2.3 asks whether suppression can degrade *completion* while *verification* survives: the analogue of a model that can recognize a behavior without producing it.

- **The concept for D2.3 becomes the latent op.** Completion means answering the query under the inferred op; verification means judging whether a shown line follows it. That maps onto M3 more closely than *red* does: recognizing sycophancy while not producing it.
- **The asymmetry question gets harder, and more informative.** Both tasks must infer the same op from the same context. If they read one shared state, suppression hits both, and the asymmetry has to come from confining the anchor to the part of the stream only completion reads. That is the open [confinement item](/todo/science/can-anchor-confined-part-stream.md), which moves from optional to central. With the op word in the grammar, the question would have been easier to answer and would have told us little.
- **The [concept swap](/todo/science/redirect-between-two-anchored-ops.md) gets a natural form.** Redirecting one inferred op to another is "make the model act as if the examples showed op Y". It is the steering claim M3 would want (steer from sycophantic toward candid), and the posterior still gives per-line ground truth.
- **Contexts get longer.** The [packed-context leakage item](/todo/science/packed-contexts-open-cross-line-leakage.md), filed for M3, reaches D2.3 sooner.

## Open questions

- How many shots per context, and whether to vary the count. Fewer shots give a flatter posterior and a wider graded range, but harder training.
- Whether the neutral symbol stays constant or varies, and whether contexts mix ops (which would be closer to M3, and harder).
- Whether d64-L4 is enough, and what we would conclude if the control cannot learn the task.
- Where the label sits: on the whole context, on the query line alone, or on its `=`.
- Whether the bridge is worth running if the scoring pass already shows the use-site edits are inert.
