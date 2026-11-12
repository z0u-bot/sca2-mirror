---
status: open
tags: [experiments, storage]
opened: 2026-11-12
---
# Kept-checkpoint uploads that stall at the end of training

In the flat-anti experiment (48 runs on L40S cards), 10 training tasks sat in the `put` of their kept checkpoints (`put ex-2.2.21-flat-anti-…-kept`) for about eight minutes after their last step. The heartbeat was frozen, the blocking-phase budget the `put` declares had expired, and the GPU still showed about a quarter of its use. The watchdog did not abort them. Each one was cancelled and retried by hand, and every retry trained from scratch, which about doubled the cost of the run ($17.87 against an estimate of $8).

Two things would help. First, find out why the watchdog let a `put` that had passed its declared budget run on, since the budget is meant to bound the span. Second, a retry after training should start from the trained state where it can, rather than re-running 400 epochs to repeat an upload. The second one probably means storing the final checkpoint before the kept ones, or letting a task resume past its last completed phase.
