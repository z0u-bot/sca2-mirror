---
status: open
tags: [ex-2.2.3, metrics, methodology, task-grammar, D2.2]
opened: 2026-09-10
---
# `mix` has its own probe partners, which confounds every per-op comparison against it

`sca.data.ops.probe_lines` builds two partner sets, not six: `probe_partners(n_probe, seed)[int(op is not MIX)]`. `mix` gets D2.1's on-grid partners so that its gated statistics are read on the lines D2.1 read them on, and the other five ops share a single random draw of 27 partners per color.

That is the right call for the gates — it keeps ex-2.1.10's reference values comparable — but it means `mix` is the one op that cannot be compared to the others on the op alone. Ex-2.2.3's E1 ran into this: contrast sits below the five shared-partner ops on every candidate, by 0.006 to 0.029, which is four times the equivalence band on the recipe arms. Split the six ops into `mix` and the five, and the five agree with each other to within the band on every candidate, so the whole apparent op effect is the `mix` column. The report now draws E1 that way and says so, but it cannot say whether the step is the partner set or the rule.

Cheapest way to separate them: add a seventh probe set, `mix` lines drawn from the shared partner draw, alongside the on-grid one. The gates keep reading the on-grid set; E1 reads the shared one. That is one more entry in the probes archive and no change to any gate. Worth doing before D2.2 leans on a per-op placement claim.

The same split would sharpen E1's m_line read, which currently exceeds the band on `t48` and `t12` with per-seed deviations wider than the band — five seeds do not resolve it, and the candidate explanation (the five sets share op1 colors and partners but not answers, so the answer distribution moves the margin) is itself a probe-set effect.

## Notes

**2026-09-10, Claude** — the split is deliberate and has been documented in `probe_partners` since the module's first commit (8eacacc6): the gates have to read the lines ex-2.1.10 read them on, so `mix` keeps D2.1's on-grid partners, and the five want the same operand pairs under different op words, so they share a draw. For `mix` those two constraints conflict and the design took the gates, which is the right way round. `N_PROBE` is pinned by mix's structure too — the assert in `probe_partners` says the on-grid partner count sets the probe count for every op. What was missing was E1's *reading* of it, not the design; ex-2.2.3's report now splits the six ops before comparing them.

A mechanism worth checking when the seventh probe set exists: the two partner sets differ in how often the operands are near-ties on redness. Over all (color, partner) pairs, `|Δredness| < 0.05` on 21.1% of mix's on-grid pairs against 14.8% of the shared draw, while partner redness and dose match to three decimals. Contrast is built from `g1 = p1(1 − p2)` and `g2 = (1 − p1)p2`, which converge as `p1 → p2`, so a probe set with half again as many near-ties would read a lower contrast with the placement unchanged. That is a plausible mechanism, not a measured cause: the shared-partner `mix` set would settle it.
