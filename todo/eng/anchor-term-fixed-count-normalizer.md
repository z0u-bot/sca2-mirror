---
status: open
tags: [anchoring, methodology]
opened: 2026-09-10
---
# Offer a fixed-count normalizer in the anchor term

`sca.anchoring.anchor_term` divides by the realized mask weight, so the per-position pull scales inversely with how many positions a labeller marks. That entangles the pull's *placement* with its *strength*: ex-2.1.7's op1-only factor pulled ~3.9× fewer positions than the span pull and so pulled each ~3.9× harder, and nothing in the design said so (see `check-for-tautological-hypotheses.md`, second paragraph). The reading survived only because the ceiling arm happened to separate strength from placement.

A cleaner design normalizes by a fixed count (the expected labeled-position count under the span labeller, say, or a constant passed in) so that changing the mask changes only which positions are pulled. `anchor_term` and `pooled_anchor_term` should offer this; existing conditions keep the realized-mask form so memo fingerprints are unchanged, and a future experiment that varies the mask adopts the fixed form and reports both invariants (positions pulled, per-position weight) for every arm.
