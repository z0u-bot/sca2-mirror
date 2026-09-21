---
status: open
tags: [reports, publishing, tooling]
opened: 2026-09-21
---
# pymdown-extensions 12 rewrites three extensions we render every report with

`pymdown-extensions` 12.0 is available and the lock sits at 11.0.2. The floor in `pyproject.toml` is `>=10.21`, so a plain `uv lock --upgrade` takes it — which is why this is written down rather than left to be discovered by whoever runs that next.

The 12.0 release rewrote `BetterEm` for CommonMark compliance and rebuilt `tilde`, `caret` and `mark` on top of the new logic. We enable all three (`mini.lit.page.EXTENSIONS`), so this is not a transitive bump: it changes how `~~strike~~`, `^^insert^^` and `==mark==` parse in every report body and figure caption. The specific behaviour change upstream calls out is that those three now allow mid-word operations by default, where earlier versions suppressed them; `smart_delete`, `smart_insert` and `smart_mark` restore the old behaviour. The release also warns of "subtle differences between legacy behavior and new CommonMark behavior" in emphasis nesting generally, which reaches `*` and `_` as well.

The decision is which side of that to sit on. Pinning the legacy options keeps every published report rendering as it does today and costs three lines of config. Taking the new behaviour is the better long-run answer — it is what CommonMark does, and what a reader pasting our Markdown elsewhere will get — but it wants a render diff across the report corpus first, because a stray `_` inside an identifier in prose is the sort of thing that only shows up rendered. The corpus is large enough that eyeballing is not the method: the useful shape is to render every report under both versions and diff the HTML, which `./go render` can do per report but nothing currently does in bulk.

Worth noting the same release removed Arithmatex's deprecated format options. We configure `pymdownx.arithmatex` with `generic` and `smart_dollar`, both current, so that removal looks clear of us — but it is worth confirming against the changelog rather than from this note.

Until then the lock stays at 11.x. Nothing is broken; this is a change we should make deliberately and check, not absorb in a dependency refresh.
