"""Tests for the runaway-edit check — files under `docs/` that grew implausibly."""

import re
import subprocess
from pathlib import Path

import pytest

from tests.conftest import load_script

runaway_edits = load_script("runaway_edits")

_INDEX = "".join(f"- [Experiment {i}](ex-{i}/)\n" for i in range(300))


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def commit(repo: Path, message: str) -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


def flagged(repo: Path) -> set[str]:
    return {g.path for g in runaway_edits.runaway("main", root=repo)}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """An index page and a stub report on `main`, with a `work` branch checked out."""
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "commit.gpgsign", "false")
    (docs := tmp_path / "docs").mkdir()
    (docs / "index.md").write_text(_INDEX)
    (docs / "stub.py").write_text("# title: A stub\n")
    (docs / "data.json").write_text('{"a": 1}')
    commit(tmp_path, "docs")
    git(tmp_path, "checkout", "-q", "-b", "work")
    return tmp_path


def test_an_empty_pattern_substitution_is_flagged(repo):
    """The incident: an entry inserted between every pair of characters."""
    index = repo / "docs" / "index.md"
    index.write_text(re.sub("", "- [A new experiment](ex-new/)\n", index.read_text()))
    commit(repo, "add an index entry")
    assert flagged(repo) == {"docs/index.md"}


def test_a_one_line_file_is_measured_in_bytes(repo):
    data = repo / "docs" / "data.json"
    data.write_text("x" * 300_000)
    commit(repo, "a runaway one-liner")
    assert flagged(repo) == {"docs/data.json"}


def test_an_ordinary_addition_is_not_flagged(repo):
    index = repo / "docs" / "index.md"
    index.write_text(index.read_text() + "- [A new experiment](ex-new/)\n")
    commit(repo, "add an index entry")
    assert flagged(repo) == set()


def test_a_stub_grown_into_a_full_report_is_not_flagged(repo):
    """Many times larger, but within the size of a real report."""
    (repo / "docs" / "stub.py").write_text("# title: A stub\n" + '"""A paragraph of prose."""\n' * 2_000)
    commit(repo, "write the report")
    assert flagged(repo) == set()


def test_new_files_are_skipped(repo):
    (repo / "docs" / "big.md").write_text("line\n" * 50_000)
    commit(repo, "a large new page")
    assert flagged(repo) == set()


def test_only_docs_are_checked(repo):
    (repo / "notes.md").write_text("a\n")
    commit(repo, "outside docs")
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "work")
    git(repo, "checkout", "-q", "work")
    (repo / "notes.md").write_text("a\n" * 50_000)
    commit(repo, "a runaway edit outside docs")
    assert flagged(repo) == set()


def test_commits_on_the_base_are_not_ours(repo):
    """Three-dot semantics: growth that landed on `main` meanwhile is not the branch's."""
    git(repo, "checkout", "-q", "main")
    (repo / "docs" / "data.json").write_text("x" * 300_000)
    commit(repo, "growth on main")
    git(repo, "checkout", "-q", "work")
    assert flagged(repo) == set()
