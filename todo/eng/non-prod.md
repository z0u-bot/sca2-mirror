---
status: partial
tags: [devops, security, publishing, storage]
opened: 2026-09-05
bundle: env-hardening
---

# Create a non-prod environment

We push to a single bucket and dataset repo. A second pair, for development, would keep work *on* the storage and publishing machinery off production: the `hf`-marked integration tests (which write probe commits into `z0u/sca2-pub` on every run), a `sync_export` or `gc --store` change mid-development, an agent building a store feature.

## Notes

**2026-09-05, port** — The mechanism is in, ported from mi-ni. A `[tool.mini.profiles.<name>]` table names a second pair and `MINI_PROFILE` selects it; the two storage keys come from the profile alone, so a half-written one falls to the local store rather than reaching production, and `app`/`env`/`region` are inherited so the compute is unchanged. Under a profile `./go publish` writes its pins to a gitignored `.mini/publish.<profile>.lock`, leaving `publish.lock` as the record CI and the site read. `mini run --app modal` forwards the profile to its workers, and `tests/mini/test_hf_store.py` picks a `dev` profile itself whenever one is defined. The storage reference in the `mi-ni` skill describes it; [`eng/environments.md`](/eng/environments.md) has the reasoning.

What is left is the human half: create `z0u/sca2-store-dev` and `z0u/sca2-pub-dev`, add the `[tool.mini.profiles.dev]` table, mint a dev-only token, and set `MINI_PROFILE=dev` (or the two env vars, for an environment configured by variable) in the engineering environments. The `mi-ni` skill's storage reference is the runbook. Until that happens the profile machinery is inert and every session uses production, as before.

**2026-09-06, verified** — The pair exists (`z0u/sca2-store-dev`, `z0u/sca2-pub-dev`, both created 09-05) and the `[tool.mini.profiles.dev]` table is in `pyproject.toml`. `MINI_PROFILE=dev ./go auth --check` reports `profile dev` with the dev pair; `publish_lock()` resolves to `.mini/publish.dev.lock`; `uv run pytest -m hf` passes 7/1-skipped against the dev pair, and the probe commits land in `sca2-pub-dev`, so the tests are off production now.

Two parts of the runbook are unfinished, both deliberate-looking. The devcontainer token is fine-grained but scoped to all four repos ("sca2 MBA M5 devc prod+dev rw"), so it is not the dev-only credential that makes forgetting the profile safe; that fits a checkout used for both science and machinery, but the boundary is then the profile rather than the token. And `MINI_PROFILE` is unset here, so anything other than the `hf` tests still uses production unless the variable is set per command. The web environment's `MINI_STORE_BUCKET` / `MINI_PUBLISH_REPO` weren't checked from here.

**2026-09-06, wired up** — The remaining human half is done bar one deliberate choice. `MINI_PROFILE=dev` is opt-in rather than the default here, because Modal memo state is keyed by experiment name with no profile component, so a dev run of an experiment that already ran in production finds its memo records while the artifact bytes stay in the production bucket. Defaulting to dev would put every session one step from that. AGENTS.md now carries the rule and that caveat, both session-start hooks print which pair the session writes to, and `./go auth --check` reports which pairs the token can write (`token: write on production and dev`).

The devcontainer token stays scoped to all four repos, since this checkout does science as well as machinery work; the profile is the boundary here rather than the token, which `./go auth --check` now makes visible. Closing this leaves only the Claude Code web environment's `MINI_STORE_BUCKET` / `MINI_PUBLISH_REPO`, which can't be checked from the devcontainer.
