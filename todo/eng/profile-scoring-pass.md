---
status: done
tags: [performance, modal, experiments]
opened: 2026-09-27
closed: 2026-09-27
---
# Find where the scoring pass spends its GPU time

In ex-2.2.12 and ex-2.2.13 the scoring role used more L4 time than training did: 160 against 91 GPU-minutes in ex-2.2.12, and 477 against 324 in ex-2.2.13 (median 161 s per `score_one`, against 107 s per training run).

Settled 2026-09-27: it was recompilation. `intervention.apply` built a new jitted lambda per (operator, subspace), so one `score_one` compiled 112 programs and spent about three quarters of its wall time compiling. `apply` now calls one module-level `_forward_jit` with the operator as an `eqx.Module` argument: on L4 a `score_one` went from a median 164 s to about 35 s. JAX's persistent compilation cache would not help across Modal containers (the key includes a per-container GPU topology fingerprint). Measurements and the conventions that came out of it are in [eng/gpu-efficiency.md](/eng/gpu-efficiency.md).

The change moves the code fingerprint of every task that reaches `sca.intervention`, so re-waking an ex-2.2.x experiment re-runs its scoring once. Outputs match the old code to float32 rounding on L4 (bit-identical on CPU), the same spread the old code shows between two containers.

Left over: the helpers in `sca/compute/evaluation.py` and `sca/compute/geometry.py` wrap `eqx.filter_jit(model.…)` per call. Each is called about once per task, so it costs a compile or two per task; worth moving to module level the next time one of them is edited.
