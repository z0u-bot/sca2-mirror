"""The memoized local backend caps its concurrent workers and queues the rest (``mini.local_apparatus.launch_queued``)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console

from mini.experiment import Experiment
from mini.local_apparatus import LocalApparatus, _launch_env, _stage_env, launch_queued
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


def test_call_staged_by_an_earlier_attempt_is_not_launched(tmp_path: Path):
    """A key claimed under a new gen but not yet re-staged must wait, not start the old call."""
    app = LocalApparatus("stale", max_workers=1, data_dir=tmp_path / "stale")
    store = app.memo_store()
    store.records_backend.write("k", {"key": "k", "state": RunState.RUNNING, "gen": "new", "created_at": 0})
    store.write_call("k", _timed, (1,), gen="old")
    _stage_env(store, "k", "old", {})
    assert launch_queued(store) == []
    assert "pid" not in store.record("k")


def test_launch_env_undoes_the_launchers_own_overlay(tmp_path: Path, monkeypatch):
    """A worker launching a sibling passes on the sibling's overlay, not its own."""
    store = LocalApparatus("env", data_dir=tmp_path / "env").memo_store()
    store.root.mkdir(parents=True)
    monkeypatch.setenv("SHARED", "base")
    monkeypatch.delenv("ONLY_MINE", raising=False)
    _stage_env(store, "mine", "g1", {"SHARED": "mine", "ONLY_MINE": "1"})
    _stage_env(store, "sibling", "g2", {"OTHER": "2"})
    # Now act as the worker for "mine": its overlay is applied and it points at its staged env.
    monkeypatch.setenv("SHARED", "mine")
    monkeypatch.setenv("ONLY_MINE", "1")
    monkeypatch.setenv("MINI_TASK_ENV_FILE", str(store.root / "mine.env"))
    env = _launch_env(store, "sibling")
    assert env["SHARED"] == "base"
    assert "ONLY_MINE" not in env
    assert env["OTHER"] == "2"
    assert env["MINI_TASK_ENV_FILE"] == str(store.root / "sibling.env")
    assert json.loads((store.root / "sibling.env").read_text())["gen"] == "g2"
