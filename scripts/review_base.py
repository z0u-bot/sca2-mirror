#!/usr/bin/env python
"""Export reports as they were at a git ref, as the baseline a re-review print is marked against (``./go preview --since <ref>``).

A reader reviews a report on paper, rounds apart, and from the second round on wants to see what changed since the version they last annotated. The site build marks that in the print's margin (:mod:`mini.review_marks`), comparing the report's page with the page as it was at *ref*. This script makes the second one: it checks *ref* out in a throwaway worktree and runs that checkout's own exporter over each named report, so the baseline is woven by the code the reader saw it woven by. The current environment runs it, with the checkout's ``src/`` first on the path, which is fine for the recent refs this is for; a ref old enough to need other dependencies fails, and says so.

A baseline lands at ``.mini/review/<sha>/<key>/`` and is kept, so the next print against the same ref reuses it. A report that did not exist at *ref* has none, and its print carries no marks.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mini.reports import export_dir, export_key
from mini.review_marks import baseline_dir

ROOT = Path(__file__).parent.parent.resolve()
REVIEW = ROOT / ".mini" / "review"


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def resolve(ref: str) -> str:
    """The full commit id of *ref*."""
    return git("rev-parse", "--verify", f"{ref}^{{commit}}")


def export_at(sha: str, reports: list[Path]) -> None:
    """Export each of *reports* as it was at *sha*, into :func:`baseline_dir`; those already there are skipped."""
    todo = []
    for report in reports:
        rel = report.resolve().relative_to(ROOT).as_posix()
        if (baseline_dir(sha, export_key(report)) / "index.html").is_file():
            print(f"  baseline {rel} @ {sha[:7]}: already exported")
        elif subprocess.run(["git", "cat-file", "-e", f"{sha}:{rel}"], cwd=ROOT, capture_output=True).returncode:
            print(f"  baseline {rel} @ {sha[:7]}: not in that commit, so its print is unmarked")
        else:
            todo.append(rel)
    if not todo:
        return
    checkout = REVIEW / "checkouts" / sha
    shutil.rmtree(checkout, ignore_errors=True)
    git("worktree", "prune")
    git("worktree", "add", "--detach", str(checkout), sha)
    try:
        env = os.environ | {
            "PYTHONPATH": os.pathsep.join(filter(None, [str(checkout / "src"), os.environ.get("PYTHONPATH")]))
        }
        cmd = [sys.executable, "scripts/export_reports.py", *todo]
        print(f"  baseline @ {sha[:7]}: exporting {len(todo)} report(s) from a checkout of that commit")
        if subprocess.run(cmd, cwd=checkout, env=env).returncode:
            sys.exit(
                f"review_base: exporting at {sha[:7]} failed (see above); a ref that old may need its own environment"
            )
        for rel in todo:
            # The checkout's exporter writes under the checkout's own .mini/.
            src = checkout / export_dir(ROOT / rel).relative_to(ROOT)
            dest = baseline_dir(sha, export_key(ROOT / rel))
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(src, dest)
            print(f"  baseline {rel} @ {sha[:7]} -> {dest.relative_to(ROOT)}")
    finally:
        git("worktree", "remove", "--force", str(checkout))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", help="the commit the reader last reviewed (any git ref)")
    ap.add_argument("reports", nargs="+", type=Path, help="report scripts, e.g. docs/m2/ex-2.2.13/report.py")
    args = ap.parse_args()
    export_at(resolve(args.ref), args.reports)


if __name__ == "__main__":
    main()
