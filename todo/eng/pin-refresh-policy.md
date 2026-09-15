---
status: done
tags: [ci]
opened: 2026-08-21
closed: 2026-09-15
---
# Nothing tells us when a pinned Action goes stale

`astral-sh/setup-uv` is now pinned to a full version (`@v10.0.1`) in both workflows, because upstream stopped publishing floating major tags at v8 — a mutable `@v8` is a supply chain the pin can't see, so naming the release is their recommendation. That trade lands the freshness problem on us: the pin will sit at v10.0.1 until someone edits two files, and nothing in the repo will mention it. The Node 20 pins took about eleven months to get written down (deprecation announced 2025-09-19, item opened 2026-08-19), and that was with a *warning* printed on every single run; an exact pin is quieter than that.

The obvious answer is a `.github/dependabot.yml` with the `github-actions` ecosystem on a monthly interval, which opens a PR per stale pin and understands full-version and SHA pins alike. That is a process change rather than a code one — it means a bot opening PRs on this repo and every one of them spending a CI round — so it wants a human decision rather than a quiet commit. Worth weighing against the alternative of grouping the updates (`groups:` in the config keeps it to one PR per interval), or of doing nothing and re-checking by hand whenever CI is being touched anyway.

Scope is small either way: two pinned Actions across two workflows. `actions/checkout` still publishes floating majors, so today only setup-uv actually needs the reminder — but that is a property of one upstream, not a guarantee, and the same immutable-release argument is spreading.

## Notes

**2026-09-15, housekeeping** — Closing: the decision this item asked for has been made, and it went to the third option rather than to dependabot. [PR #167](https://github.com/z0u/sca2/pull/167) added the weekly `deps-routine` agent and the `./go deps` check behind it, and `./go deps --actions` compares every `uses:` pin against the newest tag upstream — which is the reminder the item said nothing in the repo would give us. The routine's own instructions name this item as the reason that check exists.

It is not a paper answer: two runs have landed since, [#173](https://github.com/z0u/sca2/pull/173) clearing 19 transitive security advisories and [#174](https://github.com/z0u/sca2/pull/174) taking ty 0.0.73 → 0.0.80 and ruff 0.16.1 → 0.16.7. The pins themselves are unchanged (`astral-sh/setup-uv@v10.0.1`, `actions/checkout@v7` in both workflows), which under the routine's "plain staleness doesn't qualify" rule is the expected outcome rather than a missed one.

What the routine buys over dependabot is the thing the item was weighing: no PR per stale pin, and a judgement call each week about whether a pin falling behind is worth a CI round. If that judgement turns out to be made by nobody — a few months of runs with no Action ever mentioned — the answer is to revisit `scripts/deps.sh`'s filter rather than to reopen the dependabot question.
