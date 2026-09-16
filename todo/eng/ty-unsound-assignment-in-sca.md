---
status: open
tags: [tooling, typing, memoization]
opened: 2026-09-15
---
# Turn on `unsound-assignment` for `src/sca/`, when there's a re-run budget

[`ty-unsound-assignment`](./ty-unsound-assignment.md) enabled ty's `unsound-assignment` rule everywhere it was free, and switched it off for `src/sca/**` in `[[tool.ty.overrides]]`. Eleven findings live under that override. This item is the other half: the fix is small, written, and understood — what it costs is a re-run.

## Why it was deferred

mini keys a memoized task on the *source text* of every project helper it can reach: `_collect_sources` stores `_without_docstrings(inspect.getsource(fn))` per dependency, `_code_fingerprint` hashes the lot, and `Ctx._classify` re-runs a stale DONE record ("bias to over-invalidate"). An annotation-only edit is therefore a full invalidation of everything downstream of it.

Measured against the eleven-finding fix: 76 experiment functions across 21 experiments change fingerprint, including every `train_one` in M2 and most `eval_one`s. Nothing breaks and no published report changes — refs are content-addressed and already written — but the next `bin/mini run` on any of those experiments retrains on GPU. That is a real bill, and whose call it is to spend is the point of this item rather than something a tech-debt session should decide.

The cheapest way to take it is alongside work that was going to re-run anyway, or with `--keep-stale` bounding a sweep to the cells that actually need new code.

## The fix

Three `equinox` entry points return `Any` or a bare `PyTree`, which drops the model's type; everything else follows from that. A small `src/utils/typed_eqx.py` pins it back on once, at the boundary:

```python
def apply_updates[M](model: M, updates: PyTree) -> M:
    return cast(M, eqx.apply_updates(model, updates))

def inference_mode[M](model: M) -> M:
    return cast(M, eqx.nn.inference_mode(model))

def filter_checkpoint[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    return cast(Callable[P, R], eqx.filter_checkpoint(fn))
```

Then: swap `eqx.apply_updates` for the wrapper in `sca/training/loop.py`, `sca/anchoring.py` and `sca/fallback.py` (which also clears the two cascaded `model.normalize_weights()` findings — they were only `Any` because the line above them was); swap `eqx.nn.inference_mode` in `sca/compute/evaluation.py`; and give `Block.__call__`, `Block._step`, `MLP.__call__`, `CausalSelfAttention.__call__`, `NGPT.__call__`, `RotaryEncoding.__call__` and `RotaryEncoding._rotate_half` the return annotations they are missing. The three `run_block = eqx.filter_checkpoint(lambda block, h: block(h, enc))` sites become an annotated `def` under the wrapper as a decorator, because a lambda has nowhere to put the parameter types.

## The bug it finds

With `apply_updates` typed, ty reports at `sca/anchoring.py:310` that `clean_embedding_rows` expects `NGPT` and is being handed a `LanguageModel`. That is accurate: `make_anchored_train_step`'s `train_step` declares `model: LanguageModel`, and the `clean_rows` path reads `model.transformer.wte`, which only `NGPT` has. It works today because `NGPT` is the only model we train, and the `Any` from `eqx.apply_updates` is what kept it quiet.

Worth resolving rather than casting past: either narrow with an `isinstance` guard at the call site (cheap, and says out loud that the tied-table fix is nGPT-specific), or widen `clean_embedding_rows` to whatever protocol actually covers "has an embedding table". The guard runs at trace time under `filter_jit`, so it costs nothing per step.
