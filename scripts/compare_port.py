#!/usr/bin/env python
"""Check a report ported to a literate script against the Marimo render it replaces.

The port is mechanical, so the test is mechanical too: the same prose, the same figures. The two Markdown faces spell their markup differently (Marimo's render writes an admonition as a blockquote and a figure as an image link; the woven script keeps the ``/// tip`` block and the ``<figure>`` HTML ``themed`` emits), so both are reduced to plain text before comparing, paragraph by paragraph: emphasis, code spans, tags, and Marimo's ``<!-- @output -->`` markers all go, and a figure becomes one line naming its asset stem and its alt text. What is left is what a reader would read, and it should match line for line; the diff shows where it does not.

    uv run scripts/compare_port.py .mini/renders/m2/ex-2.1.12.md .mini/exports/m2/ex-2.1.12/index.md

Render the Marimo notebook first (``./go render docs/<key>/report.py``) and keep the file, since the notebook goes when the port lands; then export the script (``./go preview --no-serve docs/<key>/report.py``) and compare. Exit status is the number of differing hunks, so a clean port is a zero.
"""

from __future__ import annotations

import argparse
import difflib
import html
import re
import sys
from pathlib import Path, PurePosixPath

_FIGURE_HTML = re.compile(r"<figure\b.*?</figure>", re.DOTALL | re.IGNORECASE)
_IMG_TAG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_ATTR = {k: re.compile(rf'\b{k}\s*=\s*"([^"]*)"', re.IGNORECASE) for k in ("src", "alt")}
_MD_IMAGE = re.compile(r"!\[(?P<alt>(?:[^\[\]]|\[[^\]]*\])*)\]\((?P<src>[^)\s]+)[^)]*\)")
_OUTPUT_MARKER = re.compile(r"<!-- @output:\w+ -->\n?")
_STYLE = re.compile(r"<style\b.*?</style>", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3}|`+)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_SPACE = re.compile(r"[ \t]+")
_FOOTNOTE_PREFIX = re.compile(r"\[\^\d+-")  # Marimo numbers a footnote label by its cell (``[^8-recompute]``)


def _stem(src: str) -> str:
    """The figure's asset stem: the file name without its theme suffix or extension."""
    name = PurePosixPath(src.split("?")[0]).stem
    return re.sub(r"-(light|dark)$", "", name)


def _figure_line(src: str, alt: str) -> str:
    return f"[figure {_stem(src)}] {html.unescape(alt).strip()}"


def _reduce_figure_html(m: re.Match) -> str:
    """A ``<figure>`` block to a figure line (the light variant of a themed pair) plus its caption's text."""
    block = m.group(0)
    imgs = [(a["src"].search(t), a["alt"].search(t)) for t in _IMG_TAG.findall(block) for a in (_ATTR,)]
    lines = []
    seen: set[str] = set()
    for src, alt in imgs:
        if src is None or (stem := _stem(src.group(1))) in seen:
            continue
        seen.add(stem)
        lines.append(_figure_line(src.group(1), alt.group(1) if alt else ""))
    for m in _MD_IMAGE.finditer(block):  # Marimo's render puts an image link inside the figure tags
        if (stem := _stem(m["src"])) not in seen:
            seen.add(stem)
            lines.append(_figure_line(m["src"], m["alt"]))
    caption = re.search(r"<figcaption\b[^>]*>(.*?)</figcaption>", block, re.DOTALL | re.IGNORECASE)
    if not lines:  # a table or other HTML body: its text, cell by cell
        body = block[: caption.start()] if caption else block
        lines.append(_plain(re.sub(r"</(?:t[dhr]|br)>", " ", body, flags=re.IGNORECASE)))
    if caption:
        lines.append(_plain(caption.group(1)))
    return "\n\n" + "\n\n".join(lines) + "\n\n"


def _plain(text: str) -> str:
    """One paragraph as a reader would read it: no tags, links, emphasis, or code spans."""
    text = _TAG.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _EMPHASIS.sub("", text)
    text = text.replace("\\", "")  # the escapes Marimo's render adds (``\*\*``)
    text = _FOOTNOTE_PREFIX.sub("[^", text)
    return _SPACE.sub(" ", html.unescape(text)).strip()


def reduce(text: str) -> list[str]:
    """The paragraphs of a report's Markdown face as plain text, whichever dialect wrote it."""
    text = _OUTPUT_MARKER.sub("", text)
    text = _STYLE.sub("", text)
    text = _FIGURE_HTML.sub(_reduce_figure_html, text)
    text = _MD_IMAGE.sub(lambda m: "\n\n" + _figure_line(m["src"], m["alt"]) + "\n\n", text)
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        lines = []
        for line in para.splitlines():
            line = line.strip()
            if line in ("///", "<!-- tl;dr -->"):
                continue  # an admonition's fences; the Marimo render has none
            if line.startswith("/// "):  # an admonition opens: its title is text the Marimo render shows too
                line = line.partition("|")[2].strip()
                if not line:
                    continue
            line = line.removeprefix("> ").removeprefix(">")  # the blockquote Marimo renders an admonition as
            if re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", line):
                continue  # a pipe table's separator row
            if line.startswith("|") and line.endswith("|"):
                line = line.strip("|").replace("|", " ")  # a pipe table's cells, as the HTML table's are
            if line.startswith("<!-- ") and line.endswith(" -->"):
                lines.append(line)  # a review note: kept as is, it is part of the report
                continue
            lines.append(line)
        if reduced := _plain(" ".join(lines)):
            out.append(reduced)
    return out


def compare(before: Path, after: Path) -> int:
    a, b = reduce(before.read_text("utf-8")), reduce(after.read_text("utf-8"))
    figs_a = [p for p in a if p.startswith("[figure ")]
    figs_b = [p for p in b if p.startswith("[figure ")]
    print(f"{before}: {len(a)} paragraph(s), {len(figs_a)} figure(s)")
    print(f"{after}: {len(b)} paragraph(s), {len(figs_b)} figure(s)")
    if missing := {f.split("] ")[0] for f in figs_a} - {f.split("] ")[0] for f in figs_b}:
        print(f"figures missing from the port: {', '.join(sorted(missing))}")
    diff = list(difflib.unified_diff(a, b, str(before), str(after), n=1, lineterm=""))
    hunks = sum(1 for line in diff if line.startswith("@@"))
    if diff:
        print()
        for line in diff:
            print(line[:400] + (" …" if len(line) > 400 else ""))
    print(f"\n{hunks} differing hunk(s)" if hunks else "\nSame prose, same figures.")
    return hunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=Path, help="the Marimo render (.mini/renders/<key>.md)")
    ap.add_argument("after", type=Path, help="the woven script (.mini/exports/<key>/index.md)")
    args = ap.parse_args()
    sys.exit(min(compare(args.before, args.after), 125))


if __name__ == "__main__":
    main()
