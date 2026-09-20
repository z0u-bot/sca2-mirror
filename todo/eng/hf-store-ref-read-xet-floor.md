---
status: open
tags: [storage, mini, reports, performance]
opened: 2026-09-20
---
# Read refs over plain HTTP rather than a Xet download group

`HfStore._read_refs` pulls the ref JSONs with `download_bucket_files`, which opens a Xet download group. Profiling `./go render docs/m2/ex-2.2.10/report.py` on 2026-09-20 put the group's `__exit__` at 1.9 s per call and `new_file_download_group` at another 0.35 s, on top of the ~0.3 s paths-info request: about 2.5 s to fetch a few hundred bytes, however many refs are in the batch. The blobs themselves came from the warm cache in 0.1 s.

The batching already in place (one `_read_refs` per `get_refs`) means a report pays this once if it resolves its refs in one call, and ex-2.2.10 now does. It still sets the floor for every render, `bin/mini` status read, and experiment wake that touches a ref. A ref is a tiny file at a known path, so a plain `GET` on the bucket's resolve URL (with the token) would skip the Xet session; `get_bucket_file_metadata` already goes that way and costs ~0.27 s. Worth checking whether `huggingface_hub` exposes a non-Xet download for bucket files, and if not, whether the resolve URL is stable enough to hit with `httpx` from `_read_refs`. Keep `_paths_info` for present-versus-absent.
