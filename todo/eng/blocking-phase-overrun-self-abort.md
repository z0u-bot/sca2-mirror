---
status: open
tags: [mini, reliability, storage]
opened: 2026-09-22
---
# A blocking phase that outlives its budget should abort itself

In the ex-2.2.12 scoring stage (2026-09-22), two `score_one` tasks entered their final `put` of a ~10 MB score `.npz` and never left it. The record's blocking phase had a budget of about two minutes (`phase_at` to `phase_until`); `watch` flagged them (`heartbeat stale — worker may be dead`, exit 3) three minutes past that budget, and they were still in the same phase ten minutes past it, with fifteen sibling tasks of the same shape completing normally. `cancel --key` and `retry --key` cleared both, and the retries finished in the usual time, so the upload itself had hung rather than the store being down.

Two things would make this cheaper. First, a blocking phase past its budget could abort the worker the way the step watchdog does (`WatchdogStall`, with a stack dump), so a hung transfer becomes a fast, retryable failure instead of a stale record that a human has to cancel by hand and that holds a GPU container meanwhile. The phase already carries its deadline, so the worker knows when to fire. Second, `watch --json` exits 3 again on every call while a flagged task stays flagged, so a wake loop that has already acted on (or decided to wait out) an attention entry cannot wait for the rest of the stage without hand-rolling a loop; an `--ack <key>` or "attention only for events since the last call" would let the loop continue.

Separately, the `experiment-monitor` agent that launched this run waited on the driver with a `pgrep -f` loop whose pattern matched its own shell, so it never returned; the agent definition now says so in one line. The wake loop in `running.md` is the right tool and the fix is behavioural rather than in `mini`.
