---
status: done
tags: [reports, publishing]
opened: 2026-09-22
closed: 2026-09-22
---
# Pad the page clip when a page is only a little taller than the reader's native size

On the reMarkable, a page whose height is only a little over the native page height renders zoomed out: the reader scales the whole page to fit, so the text comes out small. Sandy saw it on ex-2.2.11's Glossary page (about 700 pt, against pages of 1000 pt and more that render at full size).

Proposal: when the PDF export clips a section to a page and the height lands in the awkward band (roughly 1x to 2x the native height), pad the clip so the page is clearly taller and the reader falls back to a scroll rather than a zoom-to-fit. The band and the native height are for the export script to hold, since they depend on the device.

## Notes

**2026-09-22, Fable** — Done in `mini.report_print`, as Sandy's three-band rule: a clipped page shorter than one screen (the reMarkable's 4:3) is padded to one, a page between one and two screens is padded to two, and a taller page is cut to its content plus the margin as before (`padded_height`). The middle band is the zoom-to-fit case; at two screens the reader scrolls.
