---
status: done
tags: [storage, mini, reports, performance]
opened: 2026-09-20
closed: 2026-09-20
---
# Read refs without paying the Xet download floor

`HfStore._read_refs` pulled the ref JSONs with `download_bucket_files`, which opens a Xet download group. Profiling `./go render docs/m2/ex-2.2.10/report.py` on 2026-09-20 put the group's `__exit__` at 1.9 s per call and `new_file_download_group` at another 0.35 s, on top of the ~0.3 s paths-info request: about 2.5 s to fetch a few hundred bytes, however many refs are in the batch. The blobs themselves came from the warm cache in 0.1 s.

## Notes

**2026-09-20, closing** — A plain `GET` on the bucket's resolve URL was the first idea, and it is not the answer: the hub redirect is 0.3 s, but the CDN it points at takes ~1.2 s per file, every time, presumably reconstructing it from Xet chunks. Five refs in parallel came to 1.2–1.5 s, a floor of over a second still.

What landed instead: `_paths_info` already returns each ref's `xet_hash`, a content hash, so `_read_refs` now keeps a local cache of payloads under `<warm cache>/ref-cache/<xet_hash>` and downloads only the refs whose hash it has not seen. A re-pointed ref gets a new hash and downloads once; an unchanged one costs nothing beyond the shared paths-info call. A warm `_read_refs` of five refs went from ~3 s to ~0.4 s, and the warm render of ex-2.2.10 (with its figures memoized, see [`lit-warn-unmemoized-figures`](./lit-warn-unmemoized-figures.md)) from 5 s to 1.9 s. Nothing walks the cache root beyond `cas/`, so the new directory is outside gc's view, and its entries are immutable by construction.
