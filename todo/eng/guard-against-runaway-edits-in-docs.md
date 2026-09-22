---
status: open
tags: [ci, tooling, docs]
opened: 2026-09-22
---
# A tripwire for runaway edits to files under `docs/`

On 2026-09-22 a commit on the ex-2.2.12 branch ([#206](https://github.com/z0u/sca2/pull/206)) grew `docs/index.md` from 283 lines to 101,883 — the new index entry repeated 25,400 times. It reached GitHub and was only noticed on review of the PR diff, which took a history rewrite to undo.

The mechanism is settled, reconstructed byte-for-byte from the blob: a substitution with an **empty pattern** over the whole file. `re.sub('', entry, text)` (and `str.replace('', entry)`) matches at every position rather than raising, so the 974-character replacement was inserted between all 25,400 characters of the file — the original text survives one character at a time, each copy prefixed by the character it displaced. Whatever produced it was ad hoc and is gone (the session was cleared), but the shape says the old text it meant to match came back empty — a failed search returning `''` rather than an error — and the substitution then went ahead against nothing.

Two possible responses, and they are not exclusive:

- **A tripwire, which catches the class rather than the cause.** A check comparing each changed file under `docs/` against its base-branch size, failing on an implausible growth factor (say 10× and more than a few hundred lines), with an escape for a legitimately large addition. It could live beside `scripts/unpublished_reports.py`, which already diffs the branch against `origin/main` with no store access and no write token, and could run in the same CI step and pre-push hook.
- **A convention for scripted edits.** Any substitution run over a file from a script asserts its match count before writing — one match expected, one match found. This is the more general lesson, and it belongs in the Python style skill rather than in CI.

The tripwire is the one worth building: it holds regardless of which tool made the edit, and the class of failure it catches (a quiet, enormous, plausible-looking write) is exactly the kind that survives to a push.
