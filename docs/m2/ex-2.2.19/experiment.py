"""Ex-2.2.19: training length and seeds for the seven-op set.

Ex-2.2.18 trained ex-2.2.17's recipe once on each of several smaller op sets. Dropping `screen`, `multiply`, `hsvmix`,
and `exclusion` (`no-four`) raised the Bayes ceiling and narrowed the gap to it, at one seed, and the skill curves
suggested that the last fifth of the 400-epoch schedule added little. This experiment asks how short a `no-four` run
can be, in two stages:

1. A scout at model seed 600 (the initialization of the ex-2.2.18 `no-four` run) at 50, 100, and 200 epochs. A
   frozen rule picks the length to confirm from these runs and the ex-2.2.18 run at 400.
2. Three fresh model seeds (601-603) at the chosen length and at 400 epochs, so that each comparison of lengths is
   paired on the initialization. Seeds 601 and 602 are also the initializations of ex-2.2.17's `sweep-0.00316-s1`
   and `-s2` on the full op set, so the 400-epoch runs pair with those too.

Design only for now: the constants the preregistration quotes. The DAG lands with the implementation.
"""

from __future__ import annotations

import importlib.util
import sys

DESIGN_ONLY = True


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules` (the pattern ex-2.2.16 to ex-2.2.18
    use), so this module's task bodies still cloudpickle by value for a remote worker.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / name / "experiment.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex2218 = _load_sibling("ex-2.2.18", "ex2219_ex2218")
ex2217 = ex2218.ex2217
ex2216 = ex2218.ex2216

# --- What is inherited, unchanged --------------------------------------------------------------------------

OP_SET = ex2218.op_set("no-four")
"""The seven-op set, and its corpus condition as ex-2.2.18 built it (the same corpus, held-out set, and probe set)."""

PEAK_LR = ex2218.PEAK_LR
WARMUP_EPOCHS = ex2218.WARMUP_EPOCHS
"""A warmup of fixed length (5 epochs) at every length, as ex-2.2.17 fixed it from round 2 on. At 50 epochs this is
a tenth of the run, ex-2.2.16's own rule; at 400 it is an eightieth."""

EPOCH_UNIT = ex2216.EPOCHS
"""Ex-2.2.16's length, 50 epochs: the unit ex-2.2.17 measured its lengths in."""

REFERENCE_EPOCHS = ex2218.EPOCHS
"""400 epochs, the length of ex-2.2.17's recipe and of every ex-2.2.18 run."""

SEED_OFFSET = ex2218.SEED_OFFSET

# --- Stage 1: the scout ---------------------------------------------------------------------------------------

SCOUT_EPOCHS: tuple[int, ...] = (50, 100, 200)
"""The lengths the scout trains, in epochs: one, two, and four times ex-2.2.16's length. The 400-epoch point is the
ex-2.2.18 `no-four` run, which starts from the same weights."""

SCOUT_SEED = 0
"""Seed index 0, model seed 600: the initialization of the ex-2.2.18 `no-four` run."""

SHORTFALL_TOL = 0.015
"""How far below the 400-epoch run a shorter run may fall in held-out expected exact match and still count as keeping
most of its skill. It is the largest gain over the last fifth of training that ex-2.2.18 (E4) logged, and about half
the seed range of ex-2.2.17's three runs of this recipe on the full set."""

# --- Stage 2: fresh seeds -------------------------------------------------------------------------------------

CONFIRM_SEEDS: tuple[int, ...] = (1, 2, 3)
"""Seed indices of the confirmation runs, model seeds 601-603. None of them helped choose the length."""

MAX_CONFIRM_LENGTHS = 2
"""The chosen length, and twice it as a fallback when that is still shorter than the reference."""

PARTIAL_TOL = 0.03
"""The partial band of H1: about the seed range of ex-2.2.17's three runs on the full set."""


def chosen_lengths(pick: int) -> tuple[int, ...]:
    """The lengths stage 2 trains beside the reference, given the length the selection rule picked."""
    return tuple(e for e in (pick, 2 * pick) if e < REFERENCE_EPOCHS)[:MAX_CONFIRM_LENGTHS]


def cost_per_run(epochs: int) -> float:
    """Dollars of L4 time for one run, scaled from ex-2.2.18's $0.26 at 400 epochs."""
    return 0.26 * epochs / REFERENCE_EPOCHS
