---
status: done
tags: [local, security]
opened: 2026-09-28
closed: 2026-10-11
---
# Warn when a role's env overlay looks like it carries a credential

A queued local task's env overlay (a role's `env=`, `[tool.mini] env`) waits in a launch spec until a worker slot frees (`mini.local_apparatus._stage_spec`). The spec is owner-only and sits in memory where there's a tmpfs, but on macOS it falls back to `~/.local/state` on disk. The overlay is meant for config (thread caps, XLA flags); a secret belongs in the shell's environment, which never reaches the spec. A warning at staging time for keys that look like credentials (`*_TOKEN`, `*_SECRET`, `*_KEY`, `*PASSWORD*`) would catch one passed through by mistake. A warning rather than a refusal, since a name match can misfire on an ordinary config key. Low priority: nothing passes a secret this way today.

## Notes

**2026-10-11** — Done. `LocalApparatus.w` warns once per key when an `env=` overlay (a role's, or `[tool.mini] env`, which reaches it the same way) has a name ending in `TOKEN`, `SECRET`, or `KEY`, or containing `PASSWORD`. The check sits where the overlay enters rather than in `_stage_spec`, which runs per task and would repeat the warning. The current project overlay (`XLA_FLAGS`) does not match.
