---
status: done
tags: [mini, memoization]
opened: 2026-09-27
closed: 2026-11-08
---

# Memo evidence misses sibling modules loaded by path

Experiments that reuse another experiment's code, or a helper beside their own `experiment.py`, load it by file path with `importlib.util.spec_from_file_location` and leave it out of `sys.modules`, so the task bodies still cloudpickle by value (ex-2.2.15 loads ex-2.2.14 this way; ex-2.2.16 loads ex-2.2.14 and `posterior.py`). The memo evidence walk reads imports from source and never sees these loads, so an edit to `docs/m2/ex-2.2.16/posterior.py` does not invalidate `prepare_corpus_condition`, `eval_one`, or `suppress_one`, which all compute with it. Until this is fixed, bump a task's code by hand after editing such a module, or clear its memo.

Two directions: teach the walk to follow a module-level `spec_from_file_location` whose path is a literal relative to `__file__`, or give experiments a small helper (`mini.load_sibling(path)`) that loads the module and registers its source as evidence for any task that reaches the returned handle.

## Notes

**2026-11-03, maintenance** — Not blocked on scheduling. A fix moves the evidence of the ex-2.2.15 and ex-2.2.16 tasks that reach the path-loaded modules, but both experiments are finished, and stored results and published reports stay as they are. If either is run again, `mini run --keep-stale-done` serves the DONE results as they are.

**2026-11-08, tech debt** — Fixed in `mini.memo`, though the gap was narrower than the item says. A module loaded by path and bound to a global is reached through the live object, so `sib.helper(x)` already landed on `helper` by the attribute-chain walk; ex-2.2.15 and every `ex2219 = _load_sibling(...)` experiment were covered. The hole was a module handed on as a value: ex-2.2.16 returns its `posterior.py` module from `_get_posterior()` and its tasks call `P.build_table(...)` on a local, which no walk followed. The walk now folds in a path-loaded module whole (its source and what its functions reach) wherever a function loads it by name and uses it as something other than an attribute root. Two smaller holes went with it: an absolute `__file__` path joined the evidence of any task that loads a sibling at call time, and `importlib.util.module_from_spec` counted as project source because it lives in a frozen stdlib module.

The cost is wider than the 11-03 note expected. Fingerprinting every module-level function of every `docs/**/experiment.py` before and after the change, the evidence of 146 functions in 40 experiments moved (not counting the `_load_*` helpers themselves). Most of those are drivers, but the training tasks of ex-2.2.17 through ex-2.2.22, anneal-confirm, filtered-corpus, fixed-labels, late-spill-scout, loss-rate, and soft-pool-scout reach ex-2.2.16's corpus preparation and so its posterior. A later tick of any of these re-trains unless it passes `mini run --keep-stale-done`.

A task that loads a sibling inside its own body (`_load('sib').helper(x)`) is still a hole: there is no live module to follow and no import statement to read. `test_modules_loaded_by_path_are_evidence` pins it.
