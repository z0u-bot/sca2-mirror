---
status: open
tags: [performance, modal, experiments]
opened: 2026-09-28
---
# The anchored training step is host-bound on an L4

Three timing probes in September 2026 (dev profile, ex-2.2.14's corpus and primary arm, 12 epochs, one L4 per task) looked for the cheapest way to cut the L4 bill, which is most of Modal's charge (about $77 of $89 so far this month). They all point the same way: the GPU waits on the host. This item records the numbers so that [`pack-runs-per-gpu`](./pack-runs-per-gpu.md) and any later lever start from them.

**Deferring the per-step host sync does nothing.** `train_anchored` calls `float()` on three losses every step, which forces a device sync. A variant that kept the losses on the device and drained them once per trajectory record gave bit-identical results and no speedup: 10.3, 10.8 ms/step for the current loop against 10.9, 11.5 for the deferred one, ignoring one slow container on each side. Since it would also move the memo fingerprint of every experiment that trains through `train_anchored`, the change was reverted.

**The step is almost all dispatch.** Timing the pieces in isolation: the host-side batch sampler costs 0.7 ms per batch; a step without a sync costs 8–9 ms, and nearly all of that is spent in the Python call itself (dispatch time equals total time); moving the batches onto the device first still costs about 8 ms. So the time goes to Python dispatch and CUDA calls under gVisor, before the GPU has much to do.

**A CPU reservation doesn't help.** Train roles with `cpu=` unset, 2 and 4 (four replicas each) gave 10–12 ms/step in the healthy containers of every variant. The process used 1.2–1.5 CPU-seconds per wall-second throughout, so it wasn't starved; the cgroup throttling counters aren't readable under gVisor. `nvidia-smi` put steady GPU utilization at 12–30% and power at 30–37 W (the L4 is rated at 72 W), and the dashboard agreed at 5–15%.

**About one container in three or four runs 1.5–2.5× slower** (16–32 ms/step), on every piece of the step at once, with no pattern by region, cloud, or CPU reservation. Since CPU time per wall-second was the same in slow containers, they look like slower hosts rather than starved ones. Recording the CPU model from `/proc/cpuinfo` alongside the step rate would confirm it.

**One number is unexplained.** In the breakdown probe, a loop that synced every step ran at 73–76 ms/step, and a `filter_vmap` over K seeds cost 74, 80, 111, 154, 238 ms/step for K = 1, 2, 4, 8, 16. The real training loop, which also syncs every step, runs at about 10 ms, so something in the probe's harness (a recompile, a host transfer of the vmapped state) adds a floor. The vmap numbers can't be read as packing costs until that is found; a cleaner probe would reuse `train_anchored`'s own jitted step.

## Levers, in the order to try them

1. **Several steps per dispatch.** Wrap K steps in one `lax.scan` over a device-resident block of K batches, so the host pays one dispatch for K steps. The numerics are unchanged per step, but the memo fingerprint of `train_anchored` moves, so it should land with other changes to that function (and after PR #221, which edits the same loop). A probe of K = 1, 8, 32 on the primary arm would size the gain.
2. **Packing seeds** per [`pack-runs-per-gpu`](./pack-runs-per-gpu.md), once a clean vmap probe replaces the numbers above.
3. **Early retry of slow outliers.** mini already flags a task under a third of its siblings' median rate (`_slow_outlier`); a task running at 2× could instead be restarted on another container early, if the slow-host explanation holds.

A small mini addition would make all of this cheaper to watch: sample GPU utilization through NVML in the worker and emit it with the task's metrics, so a run's own record shows whether the GPU was busy.
