---
status: done
tags: [reports, publishing, tooling]
opened: 2026-09-21
closed: 2026-09-22
---
# pymdown-extensions 12 rewrites three extensions we render every report with

`pymdown-extensions` 12.0 is available and the lock sits at 11.0.2. The floor in `pyproject.toml` is `>=10.21`, so a plain `uv lock --upgrade` takes it — which is why this is written down rather than left to be discovered by whoever runs that next.

The 12.0 release rewrote `BetterEm` for CommonMark compliance and rebuilt `tilde`, `caret` and `mark` on top of the new logic. We enable all three (`mini.lit.page.EXTENSIONS`), so this is not a transitive bump: it changes how `~~strike~~`, `^^insert^^` and `==mark==` parse in every report body and figure caption. The specific behaviour change upstream calls out is that those three now allow mid-word operations by default, where earlier versions suppressed them; `smart_delete`, `smart_insert` and `smart_mark` restore the old behaviour. The release also warns of "subtle differences between legacy behavior and new CommonMark behavior" in emphasis nesting generally, which reaches `*` and `_` as well.

The decision is which side of that to sit on. Pinning the legacy options keeps every published report rendering as it does today and costs three lines of config. Taking the new behaviour is the better long-run answer — it is what CommonMark does, and what a reader pasting our Markdown elsewhere will get — but it wants a render diff across the report corpus first, because a stray `_` inside an identifier in prose is the sort of thing that only shows up rendered. The corpus is large enough that eyeballing is not the method: the useful shape is to render every report under both versions and diff the HTML, which `./go render` can do per report but nothing currently does in bulk.

Worth noting the same release removed Arithmatex's deprecated format options. We configure `pymdownx.arithmatex` with `generic` and `smart_dollar`, both current, so that removal looks clear of us — but it is worth confirming against the changelog rather than from this note.

Until then the lock stays at 11.x. Nothing is broken; this is a change we should make deliberately and check, not absorb in a dependency refresh.

## Notes

**2026-09-22, resolved** — Took 12.0.1 plain, no legacy flags, no config change. The corpus render diff came back empty.

Method: the conversion is a pure function of the Markdown text, so it doesn't need the reports run twice. I rebuilt the woven Markdown (`.mini/lit/<key>/index.md`) for every report and converted all of it under 11.0.2 and under 12.0.1 through a standalone copy of `mini.lit.page`'s dialect, in two isolated `uv` environments. Three sweeps, all byte-identical: 33 woven report documents (1.8 MB of HTML), every string literal harvested with `ast` from all 63 `docs/**/*.py` (2.3 MB — this is what covers figure captions, which reach `index.md` already rendered and so are invisible to the first sweep), and all 246 Markdown files in the repo. A positive control on mid-word `~~`/`^^`/`==` and on emphasis-nesting cases reproduced the upstream change, so the harness was live rather than comparing nothing. The corpus contains no instance of `~~`, `^^` or `==` at all, which is why the sweeps are empty.

Two corrections to the body above. First, the BetterEm worry does not reach us: we do not enable `pymdownx.betterem`, so `*` and `_` are base python-markdown and unchanged — the probe confirmed identical nesting output. Second, `smart_delete`/`smart_insert` would *not* have restored 11.x rendering anyway: under 12 they suppress the `del`/`ins` reading mid-word, but the residue then parses as nested sub/sup (`foo<sub><sub>bar</sub></sub>baz`) where 11.x gave `foo<sub>~bar</sub>~baz`. So the legacy pin was never the identity option it looked like, which is a further argument for taking 12 plain.

Arithmatex is clear, and empirically rather than from the changelog: `smart_dollar` is still in `ArithmatexExtension.config` under 12.0.1, and python-markdown raises `KeyError` on an unknown config key, so a removed option would have failed the sweep loudly instead of being ignored.

Then, once the container had credentials, the end-to-end check the sweeps were standing in for: render all 30 reports against the store under 11.0.2, restore 12.0.1, render all 30 again, and diff the weave. Same sources, same data, only the library moved. Two reports differed — ex-2.1.1 and ex-2.1.2 — and the whole difference was the random `clipPath` ids that sublines emit, which also differ between two renders on one version. Normalised, all 62 files compare equal. Written up separately as [subline-svg-ids-random-per-render](./subline-svg-ids-random-per-render.md).

`./go check` passes (typecheck, format, links, lint, 982 tests).
