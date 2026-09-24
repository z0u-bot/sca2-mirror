# D2.2 pivot: an operation the model has to infer

*Draft for discussion, 2026-09-23.* A proposal to change which concept D2.2 anchors. The machinery stays the same, and so do the claims in the [design](design.md#what-we-want-to-be-able-to-say). Nothing here is adopted until it has been through review.

## Why

[Ex-2.2.14](../ex-2.2.14/report.py) anchored `difference` and every gate passed. The anchor landed on the op word embedding: the pull put that row on e₁ at a cosine near 1, and the op position stays there at every slice. When only the op word was pulled, the blocks carried a twentieth of that alignment to `=` and none of it to the answer.

So the anchored concept is an attribute of one token: which op this word names. That is the D2.1 result again, with a categorical attribute in place of a graded one. Suppression at the op word will very likely remove the op, but deleting the word token would do the same, so it cannot show what anchoring adds. The dose axis of the projection also collapses there: its re-normalizing gain, 1/√(1−x₁²), is unbounded at full alignment, so a state that is all concept has no partial dose.

Every op in the current grammar is named by a word, so choosing a different op will not change this.

## What M3 needs from D2.2

The lead target for M3 is sycophancy. No single token names it. It is inferred from the context and used far from where its evidence appears, and its labels will be scarce and noisy. Before M3, D2.2 should tell us:

1. whether SCA can anchor a concept the model has to *compute* from context, rather than look up from a token;
2. whether suppressing that concept removes the ability, graded and selective, within a bound set before the intervention;
3. where in the depth such a concept is best anchored (the layer sweep);
4. whether scarce, noisy labels are enough.

The current design answers (4), with the caveat that a scarce labeller only had to find one embedding row. It answers (2) only in the token-mask sense, and leaves (1) unmeasured.

## The proposal

**Take the op word out of the grammar.** Each line is one context: a few solved examples of one op, written with a neutral symbol in place of the op word, then a query under the same op:

```
red?blue=magenta, white?cyan=red, yellow?red=
```

Here the op is `difference`, and the model has to work that out from the first two examples to complete the third. Op words never appear in the corpus, so the op is always inferred; a corpus that sometimes named it would give the anchor a token to land on again.

The symbol `?` stays constant, and reads as "some op". It carries no information, so the frame would work without it, but it keeps the shape of each equation close to the current grammar. It also gives the model a position between the operands that it may use for its own computation, perhaps to gather the op; whether it does is an [open question](#open-questions).

The op becomes a latent task that the model infers from examples. The anchored concept is "the op in this context is `difference`", and no token carries it.

**One context per line** keeps the line as the unit the existing code counts in: `line` keying in the labeller draws once per context, and the holdouts and per-line scoring count contexts. It also keeps the eval to one line per forward pass, so the [packed-context leakage item](/todo/science/packed-contexts-open-cross-line-leakage.md) stays an M3 question. Training windows still read the previous lines, which hold other contexts with independent ops; the model should learn to ignore them, and an attention mask that resets at `\n` would make that certain.

Lines get longer, from 6 tokens to about 20 with three examples, and their length can vary. The code finds the line and role of a token by arithmetic on a fixed length (`LINE_TOKENS`, 6 today: line = position ÷ 6, role = position mod 6). A variable length needs two integer arrays the size of the corpus instead, the line and role of every token, computed once from the positions of `\n` and read by lookup. That costs a little memory and no padding. Training windows are random crops of the packed corpus, so the first context in a window is usually cut short; the block size should fit at least two whole contexts.

For any context, the posterior over ops given the examples can be computed from the op table. That gives us three things to design with:

- a **graded stimulus**, how strongly the context points to `difference`, which plays the part redness played for *red*;
- a **designed null** for suppression: the answer distribution under the posterior with `difference` removed;
- a **label-noise model**: a labeller that labels contexts by their posterior rather than by their true op is realistically noisy.

The graded stimulus needs care. Clean examples pin the op down fast: on table A+, one clean example already gives a posterior of 1.0 for three contexts in four, and three examples do so for 99%. So the example count alone grades very little. What does grade it is **replacement noise**: showing, in some examples, the answer *another* op would give. With a replacement rate near 0.3 and three or four examples, the posterior on the true op spreads across the whole range (with three examples at 0.35: about a quarter of contexts above 0.95, 45% between 0.5 and 0.95, 30% below 0.5). A wrong answer drawn at random from the color cube grades much less, because no op produces it: it removes the evidence of that example without pointing anywhere else. It can still ride along at a low rate, so that the model learns to discount examples that fit nothing. In every case the query (the final equation) is clean and its target is the answer under the true op, so the noisy examples are evidence to weigh, and how much the model relies on them is what the stimulus measures.

There is prior post-hoc work to compare against. Function vectors (Todd et al., 2023, arXiv:2310.15213) and task vectors (Hendel et al., 2023, arXiv:2310.15916) study pretrained language models given in-context examples of word-to-word functions over natural concepts: antonyms, country to capital, English to French. They find the task compressed into a direction in the residual stream, carried by a few attention heads to the final position, from about the middle layers on. The SCA version places that direction during training, where those methods search for it afterwards. We keep the color domain: their tasks need the knowledge of a pretrained model, and ours keeps a computable posterior, the op table, and the checkpoints and machinery we already have.

Their results suggest mid-depth, at the position where the answer forms, which for us is the query `=`. That is a place to look first, and a weak prior on where the op lives: it may be assembled earlier, at the query `?` or across the examples, or stay spread out. So the label is a binary one on the whole context, which is also the only form an M3 labeller can give, and where the anchor ends up is something we measure.

## Sequence

1. **Suppress `difference` on the stored ex-2.2.14 checkpoints.** Scoring only, as planned in the [design](design.md#suppress-the-operation-and-the-operands): the op-word edit against a token mask, and the use-site edits on the whole-line primary. It turns "the anchor is a token" into a measurement. Its outcome decides how much of the old line to report, and it does not decide whether to pivot: if the use-site edits move the answer, that is a result worth writing up beside the pivot, and only the new grammar can show whether SCA anchors a concept the model computes.
2. **The in-context grammar**: the one-context-per-line format, replacement noise, the posterior over ops, and a regression check that the model learns the task. Like ex-2.2.3, this is a grammar change and needs its own control. We expect d64-L4 to be enough: the model only has to compare each example against the op table and pool the evidence, with no variables to track. If the control cannot learn it, the next step is a wider or deeper control before anything is anchored.
3. **Anchor the latent op**, then **suppress it**, then the **layer sweep** and the **SGTM baseline**, as in the current design.

## Alternatives considered

- **Keep the op word and pull only at the use sites** (`=` and the answer), or only past the embedding slice. This is cheap in compute, but the model can copy the identity of the op word to `=` through attention, which makes it a lookup one hop later, and it costs a round of our time to learn that. Dropped.
- **Keep the op word and report the token-anchor result as it is.** This shows that scarce, noisy labels are enough, and that suppression works in the sense a token mask does. It leaves M3 to find out whether SCA reaches a concept the model computes.
- **Adopt the function-vector tasks.** Covered above: they need a pretrained model and give up the computable posterior.

## Failure modes

- **The control does not learn the task.** Covered in step 2.
- **The anchor lands on a shortcut.** The model might key on a surface feature that correlates with the op, such as a characteristic answer color. The posterior makes this checkable: a context whose examples are ambiguous between two ops should give an intermediate alignment, and a shortcut would not track it.
- **The label asks for the concept before it can exist.** Attention is causal, so the tokens of the first example in a context have seen nothing that identifies the op, and no position holds it at the embedding slice. A binary label over the whole context, pulling at every slice, asks for something those states cannot have. Training could then miss the margin there or satisfy it with a shortcut, and a strong pull could cost the task. The binary whole-line label stays the default all the same: it is what an M3 labeller can give, and ex-2.2.14 used the same form. Two cheap arms measure the cost: one leaves out the embedding slice, and one weights the pull at each position by the posterior given the tokens before it, so the weight rises through the context as the evidence arrives. That weighting is a soft label, and it uses ground truth an M3 labeller would not have, which is one more reason to keep it an arm.

## What carries over

The eval contract and the intervention library (`sca.intervention`: projection, reflection, repulsion, weight ablation); the fallback term, which worked on *red* in [ex-2.2.2](../ex-2.2.2/report.py) and whose null now comes from the posterior; the labellers, whose `line` keying now labels one context, all of its examples and the query; the untied readout; the op table A+ and its op-relevance, which now decide how many examples it takes to pin down an op; and the scoring conventions from ex-2.2.9 onward. The *red* line of work is finished as a D2.2 prerequisite and needs nothing more for this.

## How this changes D2.3

D2.3 asks whether suppression can degrade *completion* while *verification* survives: the analogue of a model that can recognize a behavior without producing it.

- **The concept for D2.3 becomes the latent op.** Completion means answering the query under the inferred op; verification means judging whether a shown example follows it. That maps onto M3 more closely than *red* does: recognizing sycophancy while not producing it.
- **The asymmetry question gets harder, and more informative.** Both tasks must infer the same op from the same context. If they read one shared state, suppression hits both, and the asymmetry has to come from confining the anchor to the part of the stream only completion reads. That is the open [confinement item](/todo/science/can-anchor-confined-part-stream.md), which moves from optional to central. With the op word in the grammar, the question would have been easier to answer and would have told us little.
- **The [concept swap](/todo/science/redirect-between-two-anchored-ops.md) gets a natural form.** Redirecting one inferred op to another is "make the model act as if the examples showed op Y". It is the steering claim M3 would want (steer from sycophantic toward candid), and the posterior still gives per-context ground truth.

## Open questions

- How many examples per context: fixed at three or four, or varied.
- Whether the model uses `?` for computation. A control trained without it would say whether the frame needs it, and the states at the query `?` are a candidate site for the op.
- Whether contexts should ever hold more than one true op (replacement noise shows another op's answer, but the context still has one true op). A form is sketched below; it is out of scope for D2.2.
- Whether to use soft labels in M3: a labeller that reports its confidence in the op of a context, as a natural-language classifier could.

**A sketch of mixed ops.** A tag sets the op for the examples that follow it, until another tag replaces it, and each tag stands for an op that is inferred as before:

```
a: red?blue=magenta, b: red?blue=purple, a: white?cyan=red, yellow?red=
```

Here `a` is `difference` and `b` is another op. The query has no tag of its own, so it takes the op of the most recent one. That is closer to M3, where the relevant behavior depends on cues earlier in a conversation that stay in force until something changes them. It is also a binding task, the kind of state tracking the current grammar does not ask for, so it may need a larger model.
