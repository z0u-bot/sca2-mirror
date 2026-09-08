---
status: done
tags: [cli]
opened: 2026-09-01
closed: 2026-09-08
bundle: cli-devx
---
# Status redraw causes flicker in terminal

When running `mini watch ...`, it draws one row per task, each with a progress bar. The first bar updates OK, but the others all flicker when they are redrawn.

```
❯ mini watch ex-2.2.1
score_run-0d869395648e                     ━━━━━━━━ 100% 0:00:00
score_run-1b4eea3e6d55  !! IndexError: ... ━━━━━━━━   0% 0:00:00
score_run-3fd612c1f6d3                     ━━━━━━━━ 100% 0:00:00
```

Perhaps it needs to be batched into a single update or [un]buffered.

Observed in VS Code. Unsure if it happens in other terminal emulators.

## Notes

**2026-09-08, tech debt** — Fixed by repainting on change rather than on a timer (`mini.monitor._Bars`). Buffering was already right, so that guess was off: Rich writes each frame as one buffered write, and the flicker was in how many frames there were and what each one does. `LiveRender.position_cursor` erases every row from the bottom upward before the frame is rewritten top down, so each row is blank for part of a frame — and the further down a row sits, the longer that gap lasts, which is why the top bar looked settled while the rest did not.

The frames were the surplus. Rich drives a `Progress` from a thread that repaints ten times a second whether or not anything moved, while `watch` polls the durable records twice a second, so most frames redrew a picture identical to the one on screen. `_Bars` now holds `auto_refresh=False` and calls `refresh()` itself: when a bar's own label, step or span moves; once a second while a clock is still counting; at Rich's own cadence while a bar with no known span is pulsing, so the one case that needs frames keeps them. A settled watch repaints not at all. Measured over three seconds of three tasks stepping once a second, on a forced-terminal console: 103 erase-line codes before, 16 after.

Two things fell out of it. A settled task's clock is now stopped, so its elapsed reading freezes where it settled instead of counting on — and stopping it is what lets a finished watch go quiet, since nothing is left counting. `stop_task` had to be made once-only: Rich re-stamps the stop time on every call, so polling a finished run would have crept the reading forward.

Not touched: `RichProgressDisplay` (`mini/progress_display.py`), the local-apparatus display, which builds its own `Progress` and has the same ten-frames-a-second idle repaint. Its update model is a blocking `queue.get(timeout=1.0)` rather than a poll loop, so the same treatment would fit it neatly — the `Empty` branch is already a one-second tick — but nobody has reported flicker there and it was left alone rather than changed unobserved.
