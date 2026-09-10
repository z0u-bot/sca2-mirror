---
status: open
tags: [ci, mini, tests]
opened: 2026-09-09
---
# `blocking_phase` can lose a race with the watchdog on a slow runner

`tests/mini/test_watchdog.py::test_blocking_phase_lets_a_post_loop_upload_finish` failed once on CI ([run 34319699866](https://github.com/z0u/sca2/actions/runs/34319699866), on `5a3233b`) and passes locally three times in a row. The record's error was:

```
WatchdogStall: no step progress in 0s (watchdog 0.25s) at step 3/3
```

Two things in that line: the stall is under a second (`:.0f` rounds it to `0s`), so the watchdog polled within one or two poll intervals of the last step; and the named limit is the 0.25 s step watchdog, not the 30 s phase the task had declared. So at the moment the watchdog checked, the phase was not yet visible to it. The task enters `blocking_phase` right after its last `emit_progress`, so on a loaded runner the gap between the emission and the phase's registration can outlast a 0.25 s watchdog whose poll is `timeout_s / 4`.

Worth a look at how a phase reaches the watchdog (`src/mini/_watchdog.py`, `_widest_phase`, and `mini.progress.blocking_phase`): if registration goes through the same channel as progress and is ordered after the emission, the fix might be as small as having the phase's `__enter__` register before it returns, or having the test's watchdog poll less tightly. A watchdog this tight is a test setting; production watchdogs are seconds to minutes, so a real run would need a very long scheduling gap to hit this.
