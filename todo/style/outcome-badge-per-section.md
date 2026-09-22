---
status: done
tags: [reports, structure]
opened: 2026-09-17
closed: 2026-09-22
---
# An outcome badge in each hypothesis section's heading row

A preregistered report has one verdict per hypothesis, and today it lives in two places: the Findings list at the top, and a callout at the foot of the section, after the figures and tables. Reading a section on the reMarkable, Sandy sketched a small box beside each `##` heading reading "Pass" / "Outcome", so the verdict is visible where the section starts and the reader knows what the evidence below is going to add up to.

Mechanically this is a `<mark>` or a `<span class="badge">` written into the heading cell from the same `h*_verdict` call the callout uses, styled in `report.css` next to `.markdown mark.draft` (which already floats a badge beside the title). The wording should match the Findings list, where ex-2.2.9 now writes "gate cleared" / "gate missed" for the gated hypotheses and "held" / "did not hold" for the predictions, so a badge never reads as an answer to the section's question. Alternatively, use an icon/symbol like ○/×/~/⋯.

Open question: whether the footer callout stays once the badge exists. It carries the sentence of detail the badge can't, so probably yes, shortened.

Mockup CSS:

```css
.paragraph:has(+ :is(h1, h2)) > mark:only-child {
  &.pass {
    color: var(--grass-11);
  }
  &.miss {
    color: var(--red-11);
  }
  &.partial {
    color: var(--yellow-11);
  }
}
.paragraph:has(+ h2) > mark:only-child {
  font-size: 1.5em;
}
```

## Notes

**2026-09-22, Fable** — Done, together with [verdict-badge-at-the-heading](verdict-badge-at-the-heading.md), which is the same ask from the ex-2.2.11 review; the note there says what landed. On the open question: the footer callout stays, and the badge is hoisted from it at page-build time, so nothing is written twice. The wording is the admonition's title (Pass / Miss / Partial), which is what ex-2.2.9 and ex-2.2.11 both emit; a report that prefers "gate cleared" can change its title once the words in `mini.reports.VERDICTS` include it.
