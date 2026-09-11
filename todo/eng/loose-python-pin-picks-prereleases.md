---
status: open
tags: [ci, tooling]
opened: 2026-09-11
---
# A cached prerelease interpreter can satisfy `.python-version`

`.python-version` holds `3.14`, and `pyproject.toml` asks for `>=3.14`. Both are happy with `3.14.0rc2`, so a container that has rc2 in the uv cache builds its venv on the release candidate and never fetches 3.14.7. That happened in this environment on 2026-09-11: the whole test suite collapsed at import time with `TypeError: _eval_type() got an unexpected keyword argument 'prefer_fwd_module'`, because pydantic 2.13.4 calls a `typing._eval_type` signature that only exists in the final release. `uv python install 3.14.7 && uv sync -p 3.14.7` cleared it, and the lock file was unchanged — so the resolution was always correct and only the interpreter was behind.

The failure mode is unpleasant because it looks like a code problem. Forty collection errors with a pydantic traceback reads as "someone broke the models", and the interpreter version appears in one line of pytest's header that is easy to skim past.

Options, roughly in order of how much they cost: pin `.python-version` to a full version and accept editing it a few times a year; raise the floor to `>=3.14.1` in `requires-python`, which excludes every 3.14.0 prerelease and needs no maintenance (a bare `>=3.14` matches prereleases of 3.14.0 under PEP 440, `>=3.14.1` cannot); or have `scripts/install.sh` compare `sys.version_info.releaselevel` against `'final'` and say something when it isn't. The middle one looks cheapest, and CI is unaffected either way since `setup-uv` starts from a clean cache.
