# Spending GPU time on arithmetic

*Part of the [engineering notes](./README.md).*

Our models are small (d64, four layers), so on a GPU most tasks are bound by latency and fixed costs: kernel launches, compilation, container start, uploads. The arithmetic is the cheap part. That shapes which choices matter, and this note records them as conventions for new experiments, with the measurements behind each.

## Conventions

1. **Default to an L4.** A training step takes about 10.5 ms on an L4, 9.3 ms on an A10 and 7.7 ms on an L40S, while the hourly price goes $0.80, $1.10, $1.95. Per step, the L4 is cheapest; a T4 ($0.59) is half the speed. Bigger batches would use the larger cards better, but batch size is a science parameter here, so it isn't ours to tune for cost. Measurements in [determinism](./determinism.md#re-measured-at-full-loop-length-2026-09-27).
2. **Keep the deterministic XLA flags.** At full loop length they cost a few seconds of compilation per task and nothing per step, and without them replicas of one run drift apart (see [determinism](./determinism.md)).
3. **Jit once, at module level, and pass what varies as arguments.** JAX caches a compiled function per *function object*, keyed on the shapes and static values of its arguments. A closure or lambda built inside a loop (`eqx.filter_jit(lambda m, x: m(x, op))`) is a new function object each time, and whatever it closes over is baked into the program, so every call compiles from scratch. Define the jitted function once, and pass the model, the batch and any operator in as arguments. An operator with array parameters is an `eqx.Module`: arrays are leaves (traced, so a new subspace of the same shape reuses the program) and Python scalars that select code paths are `eqx.field(static=True)`. `_stream_axis` in `sca/anchoring.py` and `_forward_jit` in `sca/intervention.py` are the models to copy.
4. **Keep shapes fixed across calls.** Each new shape is a new compile. Batch with a fixed `batch_size` and let the last chunk be the only odd one. A static argument (`slices`, `axes`) compiles once per value, which is fine for a handful of values and costly for hundreds.
5. **When a task is slower than its arithmetic, count compiles first.** `jax.config.update("jax_log_compiles", True)` logs one line per compile with the function name, and `jax.monitoring.register_event_duration_secs_listener` receives `/jax/core/compile/backend_compile_duration` events to total the time. A name that repeats dozens of times is the lead.
6. **Leave the volume commit alone.** A task commits the experiment Volume after writing its result and before recording `DONE`, so a reader that sees `DONE` can always read the result. It costs about 3 s per task. Modal's background sync would get the bytes there eventually, but with no ordering against the record.

## The persistent compilation cache

On Modal, mini points `JAX_COMPILATION_CACHE_DIR` at `/hf-cache/jax`, on the same workspace-wide Volume as the Hugging Face cache (`_attach_hf_cache` in `mini/modal_apparatus.py`). Its key covers the lowered program, the XLA flags, the JAX and jaxlib versions and the device kind, so a hit returns the executable a fresh compile would build: numerics don't depend on whether the cache was warm, and the memo key doesn't see it. Like the HF cache, it's disposable. Deleting it costs recompiles, and a torn entry makes JAX warn and compile again. JAX's default threshold applies, so only compiles longer than a second are written. A role that sets `JAX_COMPILATION_CACHE_DIR` in its own `env=` keeps its own value. Locally nothing changes; set the variable yourself if you want the cache there.

## What we measured: the scoring pass (2026-09-27)

In ex-2.2.12 and ex-2.2.13 the scoring role used more L4 time than training: 160 against 91 GPU-minutes, and 477 against 324. One ex-2.2.13 `score_one` compiled 112 programs, 74 of them the forward pass in `intervention.apply`, which built a new jitted lambda for every (operator, subspace) pair. On CPU:

| variant | wall | compiling | compiles |
| --- | --- | --- | --- |
| before | 225 s | 166 s | 112 |
| before, cold persistent cache | 68 s | 25 s | |
| before, warm persistent cache | 44 s | 2.5 s | |
| jit once | 53 s | 22 s | 51 (11 of them `_forward`) |
| jit once, warm persistent cache | 30 s | 0.5 s | |

The cold-cache row is faster than "before" because the cache also dedups identical programs within one process: the 74 lambdas lowered to a few distinct programs. Every variant's output was bit-identical to the original.

GPU_RESULTS_PLACEHOLDER
