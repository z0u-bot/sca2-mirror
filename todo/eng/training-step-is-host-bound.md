---
status: open
tags: [performance, modal, experiments]
opened: 2026-09-28
---
# The anchored training step is host-bound on an L4

Three timing probes in September 2026 (dev profile, ex-2.2.14's corpus and primary arm, 12 epochs, one L4 per task) looked for the cheapest way to cut the L4 bill, which is most of Modal's charge (about $77 of $89 so far this month). They all point the same way: the GPU waits on the host. A fourth, after a fix to a recompile, sized the main lever. This item records the numbers so that [`pack-runs-per-gpu`](./pack-runs-per-gpu.md) and any later lever start from them.

**Deferring the per-step host sync does nothing.** `train_anchored` calls `float()` on three losses every step, which forces a device sync. A variant that kept the losses on the device and drained them once per trajectory record gave bit-identical results and no speedup: 10.3, 10.8 ms/step for the current loop against 10.9, 11.5 for the deferred one, ignoring one slow container on each side. Since it would also move the memo fingerprint of every experiment that trains through `train_anchored`, the change was reverted.

**The step is almost all dispatch.** Timing the pieces in isolation: the host-side batch sampler costs 0.7 ms per batch; a step without a sync costs 8–9 ms, and nearly all of that is spent in the Python call itself (dispatch time equals total time); moving the batches onto the device first still costs about 8 ms. So the time goes to Python dispatch and CUDA calls under gVisor, before the GPU has much to do.

**A CPU reservation doesn't help.** Train roles with `cpu=` unset, 2 and 4 (four replicas each) gave 10–12 ms/step in the healthy containers of every variant. The process used 1.2–1.5 CPU-seconds per wall-second throughout, so it wasn't starved; the cgroup throttling counters aren't readable under gVisor. `nvidia-smi` put steady GPU utilization at 12–30% and power at 30–37 W (the L4 is rated at 72 W), and the dashboard agreed at 5–15%.

**About one container in three or four runs 1.5–2.5× slower** (16–32 ms/step), on every piece of the step at once, with no pattern by region, cloud, or CPU reservation. Since CPU time per wall-second was the same in slow containers, they look like slower hosts rather than starved ones. Recording the CPU model from `/proc/cpuinfo` alongside the step rate would confirm it.

**The "unexplained floor" was recompilation.** The first two probes of the step alone reported 50–75 ms/step, which turned out to be compile time inside the timed loops. `Scale` built its `(1,)` gains from a bare Python float, so they started weakly typed; a step handed them back strong, and the jitted step compiled again for the new signature, then once more when Adam's moments followed a step later. Every training task paid three compiles where one would do: about 15 s of an 85 s `train_one` on an L4. Fixed in `Scale` with an explicit dtype (weights bit-identical on CPU; a model test guards it), which moves the memo fingerprint of every task that builds a model. That only matters when an experiment is re-ticked, since published reports read pinned artifacts.

## Several steps per dispatch: about 3× faster, same numbers

With the recompiles removed, a fourth probe timed the step `train_anchored` uses, every variant in the same container, on four containers (us-east and us-central), 384 steps per variant:

| variant | ms per seed-step |
| --- | --- |
| one dispatch per step, three syncs (the current loop) | 9.0, 10.0, 11.7, 16.4 |
| the same with no syncs | 7.8, 8.7, 9.1, 13.1 |
| `lax.scan` over 8 steps, one sync per dispatch | 3.6, 3.9, 4.1, 4.4 |
| `lax.scan` over 32 steps | 3.1, 3.4, 3.4, 3.4 |
| 4 seeds vmapped, one dispatch per step | 5.9, 6.3, 6.8, 7.5 |
| 4 seeds vmapped × scan over 32 | 4.6–4.8 |
| 8 seeds vmapped × scan over 32 | 6.2–6.3 |

The scanned runs ended with weights bit-identical to the per-step loop on the GPU (max abs difference 0.0 after 384 steps), so scan changes cost and nothing else. It also levels the hosts: the slow container went from 16.4 to 3.4 ms, since there is much less host work left to be slow at. Packing seeds does not add to it: with scan the L4 is doing real work, and 8 seeds cost about 8× one. So scan, which keeps one cell per task, is the lever; [`pack-runs-per-gpu`](./pack-runs-per-gpu.md) has little left to offer at d64.

## What scanning `train_anchored` needs

The loop does per-step host work that a scan has to batch up:

- The anchor and anti weights are schedule values computed on the host per step. They become a length-K array per dispatch.
- `emit_metrics` and the `_Window` means read every step's losses. The scan returns them stacked, and the host emits them after the dispatch.
- Trajectory records (every `traj_stride` steps) and epoch ends (validation, checkpoints) need the model at a given step, so a dispatch must not cross one. Shortening the dispatch there gives a new shape and another compile. Padding it to K with a per-step "live" flag, where a dead step passes the state through, keeps one compile at the price of a few wasted steps per boundary.
- Batches keep their draw order: an epoch's training batches are all drawn before its validation crops, whatever the dispatch size.
- `emit_progress` and the watchdog see progress once per dispatch, which is fine at K = 8–32 (well under a second).

K = 8 gets most of the gain; K = 32 a little more. `train_model` (the unanchored loop) has the same shape and could share the helper.
