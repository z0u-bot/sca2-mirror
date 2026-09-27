"""The memoized local backend caps its concurrent workers and queues the rest (``mini.local_apparatus.launch_queued``)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console

from mini.experiment import Experiment
from mini.local_apparatus import LocalApparatus, _launch_env, _stage_spec, launch_queued, spec_path
from mini.monitor import drive_and_watch
from mini.orchestration import tick
from mini.runs import RunState


def _timed(x):
    import time

    start = time.time()
    time.sleep(0.4)
    return x, start, time.time()


def _sweep(name: str, n: int) -> Experiment:
    return Experiment(name=name, main=lambda ctx: ctx.map(_timed, list(range(n))))


def _max_overlap(spans: list[tuple[float, float]]) -> int:
    events = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans])
    live = peak = 0
    for _, step in events:
        live += step
        peak = max(peak, live)
    return peak


def test_workers_cap_concurrent_tasks(tmp_path: Path):
    app = LocalApparatus("capped", max_workers=2, data_dir=tmp_path / "capped")
    results = drive_and_watch(_sweep("capped", 5), app, poll=0.01, console=Console(quiet=True))
    assert [x for x, _, _ in results] == list(range(5))
    assert _max_overlap([(s, e) for _, s, e in results]) <= 2


def test_queue_drains_without_a_driver(tmp_path: Path):
    """One tick and no watch: each exiting worker hands its slot to the next queued task."""
    app = LocalApparatus("detached", max_workers=1, data_dir=tmp_path / "detached")
    done, _ = tick(_sweep("detached", 3), app)
    assert not done
    store = app.memo_store()
    queued = [r for r in store.records() if not r.get("pid")]
    assert len(queued) == 2  # one started, two waiting
    deadline = time.time() + 30
    while any(r.get("state") == RunState.RUNNING for r in store.records()):
        assert time.time() < deadline, "queue did not drain"
        time.sleep(0.05)
    assert {r["state"] for r in store.records()} == {RunState.DONE}
    assert not any(spec_path(store, r["key"]).exists() for r in store.records())  # removed at launch


def test_call_staged_by_an_earlier_attempt_is_not_launched(tmp_path: Path):
    """A key claimed under a new gen but not yet re-staged must wait, not start the old call."""
    app = LocalApparatus("stale", max_workers=1, data_dir=tmp_path / "stale")
    store = app.memo_store()
    store.records_backend.write("k", {"key": "k", "state": RunState.RUNNING, "gen": "new", "created_at": 0})
    store.write_call("k", _timed, (1,), gen="old")
    _stage_spec(store, "k", "old", {})
    assert launch_queued(store) == []
    assert "pid" not in store.record("k")


def test_launch_spec_lives_outside_the_project(tmp_path: Path):
    """The spec may hold an env overlay, so it sits under the state home, readable by its owner only."""
    store = LocalApparatus("spec", data_dir=tmp_path / "project" / ".mini" / "spec").memo_store()
    _stage_spec(store, "k", "g1", {"XLA_FLAGS": "--xla_cpu_enable_fast_math=false"})
    path = spec_path(store, "k")
    assert path.is_relative_to(tmp_path / "xdg-state")
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert json.loads(path.read_text()) == {"gen": "g1", "env": {"XLA_FLAGS": "--xla_cpu_enable_fast_math=false"}}


def test_launch_env_undoes_the_launchers_own_overlay(monkeypatch):
    """A worker launching a sibling passes on the sibling's overlay, not its own."""
    monkeypatch.setenv("SHARED", "base")
    monkeypatch.delenv("ONLY_MINE", raising=False)
    monkeypatch.delenv("MINI_TASK_BASE_ENV", raising=False)
    mine = _launch_env({"SHARED": "mine", "ONLY_MINE": "1"})  # as the tick launches "mine"
    # Now act as that worker, launching a sibling with an overlay of its own.
    for k, v in mine.items():
        monkeypatch.setenv(k, v)
    sibling = _launch_env({"OTHER": "2"})
    assert sibling["SHARED"] == "base"
    assert "ONLY_MINE" not in sibling
    assert sibling["OTHER"] == "2"
    assert json.loads(sibling["MINI_TASK_BASE_ENV"]) == {"OTHER": None}
