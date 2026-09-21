---
status: done
tags: [storage, mini, reliability]
opened: 2026-09-16
closed: 2026-09-20
---
# Retry transient 5xx responses on bucket batch writes

`HfStore._write_blob` makes one `batch_bucket_files` call per put with no retry, so a single transient server error fails the whole task. In the ex-2.2.9 run on 2026-09-16, the Hugging Face bucket's `/batch` endpoint returned `500 Internal Server Error` twice in twenty minutes: once inside a `score_one` (four minutes of compute lost) and once inside `publish_results`, the last step of the DAG. Both cleared on `bin/mini retry`, at the cost of a driver relaunch and a human noticing.

A bounded retry with exponential backoff (three or four attempts over a minute, on 5xx and connection errors only, never on 4xx) around the batch add in `_write_blob`, and probably around `_pull_blobs` and `_paths_info`, would absorb this. Keep it inside `hf_store.py` so the `LocalStore` path and the memo semantics stay untouched: the retry wraps one idempotent HTTP call (a CAS write keyed by content hash), so a duplicate attempt cannot corrupt anything.

Worth checking first whether `huggingface_hub` already exposes a retry knob for bucket calls; if it does, turning that on is the smaller change.

## Notes

**2026-09-20, Claude** — Done, with two corrections to the premise above.

*`huggingface_hub` already retries, but not far enough.* Version 1.26.0 — pinned since 2026-09-10, so the one running when ex-2.2.9 failed — wraps both metadata endpoints we call (`/batch` and `/paths-info`) in `http_backoff`: five attempts on 408/429/5xx and on transport errors, waiting 1s and doubling to a cap of 8s. So "one call with no retry" was not the shape of it; a blip of a few seconds never reached us, and what took the run down was an incident outlasting that ~23s ceiling. Two gaps were left to close: the ceiling itself, and the Xet byte transfer under `add`/`download`, which carries no retry of its own. `HFStore` now wraps its whole bucket call surface in a second, wider ring — four attempts waiting 10s, 20s and 40s — taking what the store rides out from ~23s to a bit over two minutes.

*"never on 4xx" needed one exception, and it is measured.* Two processes that commit the same Xet hash at the same moment race in the bucket's dedup, and one comes back `422 Unprocessable Entity`. Against the dev bucket: ten concurrent writes of distinct content all clean, concurrent writes of identical content failing about half the time, every failure clearing on the next attempt, and threads inside one process — which share a Xet session — never racing. A content-addressed CAS makes that collision ordinary, since two workers whose `has` probes both miss on the same bytes will both write them. It was also the cause of a live flake: `pytest -m hf` was scoring 1–5 passed with 2–6 errors under xdist, because the integration suite's write probe used a constant `b"{}"` payload in every worker. The probe now uses unique bytes, and the gated suite is 7/7 across three consecutive runs.

The rationale is in `eng/storage-backend.md`; the ring and its five offline tests are in `src/mini/hf_store.py` and `tests/mini/test_hf_store_retry.py`.
