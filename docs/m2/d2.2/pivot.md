# D2.2 pivot: an operation the model has to infer

*Draft for discussion, 2026-09-23.* A proposal to change which concept D2.2 anchors. The machinery stays the same, and so do the claims in the [design](design.md#what-we-want-to-be-able-to-say). Nothing here is adopted until it has been through review.

## Why

[Ex-2.2.14](../ex-2.2.14/report.py) anchored `difference` and every gate passed. The anchor landed on the op word's embedding: the pull put that row on e₁ at a cosine near 1, and the op position stays there at every slice. When only the op word was pulled, the blocks carried a twentieth of that alignment to `=` and none of it to the answer.

So the anchored concept is an attribute of one token: which op this word names. That is the D2.1 result again, with a categorical attribute in place of a graded one. Suppression at the op word will very likely remove the op, but deleting the word token would do the same, so it cannot show what anchoring adds. The projection's dose axis also collapses there: its re-normalizing gain, 1/√(1−x₁²), is unbounded at full alignment, so a state that is all concept has no partial dose.

Every op in the current grammar is named by a word, so choosing a different op will not change this.

## What M3 needs from D2.2

M3's lead target is sycophancy. No single token names it. It is inferred from the context and used far from where its evidence appears, and its labels will be scarce and noisy. Before M3, D2.2 should tell us:

1. whether SCA can anchor a concept the model has to *compute* from context, rather than look up from a token;
2. whether suppressing that concept removes the ability, graded and selective, within a bound set before the intervention;
3. where in the depth such a concept is best anchored (the layer sweep);
4. whether scarce, noisy labels are enough.

The current design answers (4), with the caveat that a scarce labeller only had to find one embedding row. It answers (2) only in the token-mask sense, and leaves (1) unmeasured.

## The proposal

**Take the op word out of the grammar.** Each line is one context: a few solved examples of one op, written with a neutral symbol in place of the op word, then a query under the same op:

```
red~blue=magenta, white~cyan=red, yellow~red=
```

Here the op is `difference`, and the model has to work that out from the first two examples to complete the third. Op words never appear in the corpus, so the op is always inferred; a corpus that sometimes named it would give the anchor a token to land on again. The symbol `~` stays constant. It could be dropped, since the frame is fixed, but it keeps the example's shape close to the current grammar and costs one token.

The op becomes a latent task that the model infers from examples. The anchored concept is "the op in this context is `difference`", and no token carries it.

**One context per line** keeps the line as the unit that the existing code already counts in. The labeller's `line` keying already draws once per context, and the holdouts and the per-line scoring already count contexts. It also keeps the eval inside one line per forward pass, so the [packed-context leakage item](/todo/science/packed-contexts-open-cross-line-leakage.md) stays an M3 question. Training windows still read the previous lines, which hold other contexts with independent ops; the model should learn to ignore them, and an attention mask that resets at `\n` would make that certain. The costs: a line grows from 6 tokens to about 20 with three shots, and a line width that varies with the shot count breaks the fixed `LINE_TOKENS` indexing, so a varying count needs padding to the longest line.

For any context, the posterior over ops given the examples can be computed from the op table. That gives us three things to design with:

- a **graded stimulus**, how strongly the context points to `difference`, which plays the part redness played for *red*;
- a **designed null** for suppression: the answer distribution under the posterior with `difference` removed;
- a **label-noise model**: a labeller that labels contexts by their posterior rather than by their true op is realistically noisy.

The graded stimulus needs care. Clean examples pin the op down fast: on table A+, one clean shot already gives a posterior of 1.0 for three contexts in four, and three shots do so for 99%. So the shot count alone grades very little. What does grade it is replacing some shown answers with the answer *another* op would give. With a replacement rate near 0.3 and three or four shots, the posterior on the true op spreads across the whole range (with three shots at 0.35: about a quarter of contexts above 0.95, 45% between 0.5 and 0.95, 30% below 0.5). The query's target stays the true op's answer, so the noisy examples are evidence to weigh, and the model's reliance on them is what the stimulus measures. Varying the shot count on top of this is probably worthwhile, subject to the padding cost above.

There is prior post-hoc work to compare against. Function vectors (Todd et al., 2023, arXiv:2310.15213) and task vectors (Hendel et al., 2023, arXiv:2310.15916) find that an in-context task is compressed into a direction in the residual stream, carried by a few attention heads to the final position, from about the middle layers on. SCA's version places that direction during training, where those methods search for it afterwards. We keep the color domain: their tasks lean on a pretrained model's knowledge (antonyms, capitals), and ours keeps a computable posterior, the op table, and the checkpoints and machinery we already have. What carries over from their work is the prior on *where*: the query's `=`, and mid-depth, which is where the label should sit first and where the layer sweep should start.

## Sequence

1. **Suppress `difference` on the stored ex-2.2.14 checkpoints.** Scoring only, as planned in the [design](design.md#suppress-the-operation-and-the-operands): the op-word edit against a token mask, and the use-site edits on the whole-line primary. It turns "the anchor is a token" into a measurement. Its outcome decides how much of the old line to report, and it does not decide whether to pivot: if the use-site edits move the answer, that is a result worth writing up beside the pivot, and the new grammar still answers (1).
2. **The in-context grammar**: the one-context-per-line format, the replacement noise, the posterior over ops, and a regression check that the model learns the task. Like ex-2.2.3, this is a grammar change and needs its own control. We expect d64-L4 to be enough: the model has to compare each example against the op table and pool the evidence, with no variables to track. If the control cannot learn it, the next step is a wider or deeper control before anything is anchored.
3. **Anchor the latent op**, then **suppress it**, then the **layer sweep** and the **SGTM baseline**, as in the current design.

## Alternatives considered

- **Keep the op word and pull only at the use sites** (`=` and the answer), or only past the embedding slice. This is cheap in compute, but the model can copy the op word's identity to `=` through attention, which makes it a lookup one hop later, and it costs a round of our time to learn that. Dropped.
- **Keep the op word and report the token-anchor result as it is.** This answers (4) and a narrow form of (2), and leaves M3 to find out whether SCA reaches a computed concept.
- **Adopt the function-vector tasks.** Covered above: they need a pretrained model and give up the computable posterior.

## Failure modes

- **The control does not learn the task.** Covered in step 2.
- **The anchor lands on a shortcut.** The model might key on a surface feature that correlates with the op, such as a characteristic answer color. The posterior makes this checkable: a context whose examples are ambiguous between two ops should give an intermediate alignment, and a shortcut would not track it.
- **The pull does not take.** The op is spread across the examples, so the concept may form only at the query, late in depth. The layer sweep covers this, and the function-vector prior says where to start.

## What carries over

The eval contract and the intervention library (`sca.intervention`: projection, reflection, repulsion, weight ablation); the fallback term, which worked on *red* in [ex-2.2.2](../ex-2.2.2/report.py) and whose null now comes from the posterior; the labellers, with `line` keying now meaning one context; the untied readout; the op table A+ and its op-relevance, which now decide how many examples it takes to pin down an op; and the scoring conventions from ex-2.2.9 onward. The *red* line of work is finished as a D2.2 prerequisite and needs nothing more for this.

## How this changes D2.3

D2.3 asks whether suppression can degrade *completion* while *verification* survives: the analogue of a model that can recognize a behavior without producing it.

- **The concept for D2.3 becomes the latent op.** Completion means answering the query under the inferred op; verification means judging whether a shown example follows it. That maps onto M3 more closely than *red* does: recognizing sycophancy while not producing it.
- **The asymmetry question gets harder, and more informative.** Both tasks must infer the same op from the same context. If they read one shared state, suppression hits both, and the asymmetry has to come from confining the anchor to the part of the stream only completion reads. That is the open [confinement item](/todo/science/can-anchor-confined-part-stream.md), which moves from optional to central. With the op word in the grammar, the question would have been easier to answer and would have told us little.
- **The [concept swap](/todo/science/redirect-between-two-anchored-ops.md) gets a natural form.** Redirecting one inferred op to another is "make the model act as if the examples showed op Y". It is the steering claim M3 would want (steer from sycophantic toward candid), and the posterior still gives per-context ground truth.

## Open questions

- How many examples per context: fixed at three or four, or varied (which needs padding).
- Whether contexts should ever mix ops, for example two ops interleaved with a cue that says which applies to the query. That is closer to M3, where the relevant behavior depends on cues within a conversation, and much harder. It is out of scope for D2.2.
- Where the label sits: on the whole context, on the query alone, or on its `=`. The function-vector prior favors the query's `=`.
