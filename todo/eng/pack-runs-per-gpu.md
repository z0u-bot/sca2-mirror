---
status: open
tags: [performance, modal, experiments]
opened: 2026-09-27
---
# Pack several training runs onto one GPU

Training a d64-L4 model leaves the GPU mostly idle, so the cheapest way to cut the training bill is to give each GPU more to do per step without changing any run's recipe.

The evidence is from a timing probe of ex-2.2.14's `train_one` (2,970 steps, two containers per cell; the table is in [`eng/determinism.md`](/eng/determinism.md)). The steady-state step took about 10.5 ms on an L4, 9.3 ms on an A10, and 7.7 ms on an L40S, which has roughly four times the L4's compute. A step that barely speeds up on a much larger card is latency-bound: kernel launches, Python dispatch, and the `float(loss)` sync each step, rather than arithmetic. Host-side batch sampling is not the cause (about 2 s of an 80 s task).

Per step, the L4 is already the cheapest card on Modal's price list (about 2.3 µ$ against 3.1 for T4, 2.9 for A10, 4.2 for L40S), so a GPU swap has nothing to offer. Larger batches would fill the card, but batch size is part of the recipe and would change what the experiments measure.

What keeps the recipe is running several cells in one step: `jax.vmap` the train step over a leading axis of models, optimizer states and batches, so a container trains, say, all seeds of one condition at once. If the step time stays near flat as the stack grows, five seeds cost about as much GPU time as one.

Things to settle before adopting it:

- **Numerics.** Batched kernels are a different computation, so a packed run will not reproduce an unpacked one bit for bit. Fine for a new experiment; it can't be slipped under an existing one's memo.
- **Memo granularity.** One task per pack rather than per cell, so a failure retrains the whole pack, and a label or result has to be split back out per cell for `eval_one`.
- **Per-cell schedules.** Anchor and anti weights are scalars per step today; they would become per-model vectors, which is easy for cells sharing a condition and fiddlier across conditions.
- **Measure first.** A short probe of step time against stack size (1, 2, 4, 8) on an L4 says where it stops being flat. The 2026-09-27 probe was a throwaway mini experiment: it wrapped `train_one` and swapped `sca.compute.training.sample_anchored_batches` for a generator that timestamps each batch, one role per GPU/flag variant with `single_use_containers=True`, run under `MINI_PROFILE=dev`.
