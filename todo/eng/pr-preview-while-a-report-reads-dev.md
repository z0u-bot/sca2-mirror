---
status: open
tags: [publishing, storage, reports, ci]
opened: 2026-09-11
---
# A PR preview for a report whose data is still on dev

The gap, raised by Sandy while the storage rule was being rewritten (the line between the pairs is publishing; see the `mi-ni` skill's storage reference, "Which pair a run uses"). Early in a report's life the figures change on every pass, and if the experiment is being prototyped under `MINI_PROFILE=dev`, so does the data behind them. The PR preview serves only pinned bundles from the production publish repo, so getting a draft page in front of a reviewer means a production run and a production publish per iteration, for a report whose numbers are not yet the ones that will be published. The question was whether a PR could serve a report from the dev pair while that is the case.

What already covers most of it, without a change:

- `MINI_PROFILE=dev ./go preview` assembles the site locally from the dev runs and serves it. It costs the export and nothing else, and it is the loop for self-review.
- `MINI_PROFILE=dev ./go publish` writes bundles to the dev publish repo, with pins in the gitignored `.mini/publish.dev.lock`, so a dev publish never moves the production record.
- The `skip-publish-check` label lets a drafting PR push without a publish per push; the pin moves at the freeze, once, from the production run.
- The cost a production publish adds over a preview is an upload of a few megabytes and one commit on the publish repo. The export, which is the slow part, is paid either way.

What serving a PR preview from dev would need, which is why it is filed rather than done:

- A committed dev pin for the PR, which is the record the profile design keeps out of git on purpose ([`eng/environments.md`](/eng/environments.md)).
- The site build's token gaining read on the private dev publish repo, and per-report logic in `build_site.py` to choose a tier.
- A guard that a dev pin never reaches `main`, at the seam most likely to leak it: a report previewed from dev is one forgotten republish away from merging as a dev read.
- The preview becoming a page whose numbers are not the published numbers, on the surface reviewers use to check them.

Recommendation for now: no. Use the local preview for iteration, the label while drafting, and publish from production at the freeze. Reopen if the human-review loop on dev data turns out to happen often enough that a local server is the bottleneck; the cheapest version then is a preview served from the dev publish repo under a distinct URL prefix with the pin held in the PR description rather than in a committed lock.
