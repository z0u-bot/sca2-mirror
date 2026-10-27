#!/usr/bin/env python
"""Files under ``docs/`` that this branch grew implausibly — the runaway-edit check.

A scripted substitution whose search text came back empty matches at every position rather than raising (``re.sub('', new, text)``, ``text.replace('', new)``), so it writes the replacement between every pair of characters. The file still holds its original text, one character per copy, and the diff looks like a large but ordinary addition. That happened once to ``docs/index.md``, which grew from 283 lines to over 100,000 and was noticed only on review of the PR diff (``todo/eng/guard-against-runaway-edits-in-docs.md``).

This check catches the class rather than the cause: it compares each file under ``docs/`` that the branch changed with its size at the merge base, and names any that grew both by a large factor and by a large amount. Both conditions matter. A stub that becomes a full report grows many times over but stays within the size of the largest reports, and a long report that gains a section grows a lot but by a small factor. The absolute floors sit above the largest report in the repo, so a legitimate rewrite doesn't trip them. Lines and bytes are both measured, since a runaway edit to a one-line file (JSON, SVG) adds bytes without adding lines.

A file new on the branch has no base to compare with, and a deleted one can't have grown, so both are skipped. Like ``unpublished_reports.py``, this is pure git: no store access and no write token, so it runs in CI and in the pre-push hook alike.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

FACTOR = 10  # how many times larger than at the base
MIN_ADDED_LINES = 3_000  # above the longest report (about 2,400 lines)
MIN_ADDED_BYTES = 250_000  # above the largest report (about 150 KB)


@dataclass(frozen=True)
class Growth:
    path: str
    lines: tuple[int, int]
    size: tuple[int, int]

    @property
    def runaway(self) -> bool:
        (l0, l1), (b0, b1) = self.lines, self.size
        return (l1 >= FACTOR * max(l0, 1) and l1 - l0 >= MIN_ADDED_LINES) or (
            b1 >= FACTOR * max(b0, 1) and b1 - b0 >= MIN_ADDED_BYTES
        )

    def __str__(self) -> str:
        (l0, l1), (b0, b1) = self.lines, self.size
        return f"{self.path}: {l0:,} → {l1:,} lines, {b0:,} → {b1:,} bytes"


def git(*args: str, root: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True)


def blob(rev: str, path: str, root: Path) -> bytes:
    show = git("show", f"{rev}:{path}", root=root)
    if show.returncode != 0:
        sys.exit(f"git show {rev}:{path} failed\n{show.stderr.decode().strip()}")
    return show.stdout


def growth(base: str, root: Path = ROOT) -> list[Growth]:
    """Each file under ``docs/`` modified since the merge base of *base* and ``HEAD``, with its size then and now.

    Committed content only, at both ends: the question is what a push would send. Renames are read as a delete and an add (``--no-renames``), so a moved file counts as new and is skipped.
    """
    mb = git("merge-base", base, "HEAD", root=root)
    if mb.returncode != 0:
        sys.exit(f"git merge-base against '{base}' failed — is that ref fetched?\n{mb.stderr.decode().strip()}")
    merge_base = mb.stdout.decode().strip()
    diff = git(
        "diff", "--name-only", "--no-renames", "--diff-filter=M", "-z", merge_base, "HEAD", "--", "docs", root=root
    )
    if diff.returncode != 0:
        sys.exit(f"git diff failed\n{diff.stderr.decode().strip()}")

    def measure(path: str) -> Growth:
        old, new = blob(merge_base, path, root), blob("HEAD", path, root)
        return Growth(path, (old.count(b"\n"), new.count(b"\n")), (len(old), len(new)))

    return [measure(p) for p in diff.stdout.decode().split("\0") if p]


def runaway(base: str, root: Path = ROOT) -> list[Growth]:
    return [g for g in growth(base, root) if g.runaway]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", help="the ref to compare against, e.g. origin/main")
    args = ap.parse_args()

    for g in runaway(args.base):
        print(g)


if __name__ == "__main__":
    main()
