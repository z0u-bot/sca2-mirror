---
status: open
tags: [modal, storage, devops]
opened: 2026-09-06
bundle: env-hardening
---

# Modal memo state ignores the active profile

The storage pair is profile-scoped, but the Modal control plane is not. `control_dict_name()` is `f"mini-cp-{name}"` and the volume is `app.name` (`src/mini/modal_apparatus.py:70`, `:590`), both keyed by experiment name alone. So a `MINI_PROFILE=dev` run of an experiment that already ran in production finds production's memo records and skips the work it thinks is done, while whatever it does write lands in the dev bucket — a run that reads as memoized and complete with its outputs split across two stores.

Nothing has hit this: it needs a Modal token in a dev-profile session, and the environments that default to `dev` (the Claude Code web env) have Modal off, while the one with Modal (the devcontainer) keeps `dev` opt-in for this reason. So it is latent rather than live — but it is the thing standing between the web environment and a Modal token, which is what would let a web session run experiments rather than only edit code. See `todo/eng/non-prod.md` for how the two environments are configured.

The natural fix mirrors what `PROFILE_KEYS` already does for storage: fold the active profile into the Modal namespace, so `mini-cp-<name>` becomes `mini-cp-<profile>-<name>` under a profile and stays unchanged on the base configuration. That keeps production names and their existing state untouched, needs no discipline from the caller, and gives dev runs their own volume as well as their own memo records. The cheaper alternative — refusing or prefixing experiment names under a profile in `mini run` — leaves the interactive path (a notebook constructing a `ModalApparatus` itself) uncovered, so the naming belongs in `control_dict_name` rather than in the CLI.

Worth checking against `gc` and `mini rm` while doing it: both address the control plane by the same name (`src/mini/gc.py:348`, `src/mini/__main__.py:97`), so they need to resolve the profile the same way or they will look at the wrong dict. `todo/eng/mini-rm-experiment-memo-state.md` is the related gap in deleting memo state at all.
