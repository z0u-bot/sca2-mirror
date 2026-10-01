"""Ex-2.2.20: a high-rate head start before the recipe schedule.

In ex-2.2.19, the 50-epoch scout run at peak rate 0.00562 ended with a higher skill than any of the longer runs had
reached by epoch 50. This experiment trains the seven-op set (`no-four`) for 200 epochs on a schedule of two cycles:
50 epochs of warmup and cosine at that higher peak, then 150 epochs of warmup and cosine at the recipe peak, 0.00316.
It is one run per model seed, with the optimizer state carried across the boundary, at the four model seeds of the
200-epoch runs of ex-2.2.19 (600-603), so that each run pairs with a plain 200-epoch run and a 400-epoch run from the
same initialization.

The corpus condition is rebuilt from ex-2.2.18's seed, as ex-2.2.19 did, which gives the same corpus. Training and
evaluation are ex-2.2.18's own tasks.

This module holds the design constants only, for the preregistration; the DAG lands after the freeze.
"""

from __future__ import annotations

import importlib.util
import sys

DESIGN_ONLY = True


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules` (the pattern ex-2.2.16 to ex-2.2.19
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


ex2219 = _load_sibling("ex-2.2.19", "ex2220_ex2219")
ex2218 = ex2219.ex2218
ex2217 = ex2219.ex2217
ex2216 = ex2219.ex2216

# --- What is inherited, unchanged --------------------------------------------------------------------------

OP_SET = ex2219.OP_SET
"""The seven-op set, with the corpus, held-out set, and probe set that ex-2.2.18 built."""

LO_LR = ex2219.PEAK_LR
"""The recipe peak rate, 0.00316: the peak of the second cycle, and of the plain runs this compares with."""

HI_LR = ex2219.SCOUT_LRS[1]
"""The higher peak rate of the ex-2.2.19 scout, 0.00562: the peak of the first cycle."""

WARMUP_EPOCHS = ex2219.WARMUP_EPOCHS
"""Each cycle warms up over 5 epochs, the warmup of the recipe at every length."""

MIN_LR_FACTOR = 0.01
"""The recipe warms up from, and anneals to, 1% of its peak. The first cycle warms up from 1% of its own peak, as the
ex-2.2.19 scout run did, and anneals to 1% of the second peak, where the second warmup starts, so the schedule has no
jump."""

EPOCHS = 200
"""The length of every run: the length ex-2.2.19 adopted."""

REFERENCE_EPOCHS = ex2219.REFERENCE_EPOCHS
"""400 epochs, the length of the recipe before ex-2.2.19."""

HEAD_START_EPOCHS = 50
"""The length of the first cycle: the ex-2.2.19 scout run that prompted this experiment."""

SEED_OFFSET = ex2219.SEED_OFFSET
SEEDS: tuple[int, ...] = (ex2219.SCOUT_SEED, *ex2219.CONFIRM_SEEDS)
"""Seed indices 0-3, model seeds 600-603: the four initializations of the 200-epoch runs of ex-2.2.19."""

GATE_SEEDS: tuple[int, ...] = ex2219.CONFIRM_SEEDS
"""The seeds H1 is scored on, 601-603, as in ex-2.2.19. Seed 600 is reported beside them: the observation behind
this experiment came from its runs."""

SHORTFALL_TOL = ex2219.SHORTFALL_TOL
PARTIAL_TOL = ex2219.PARTIAL_TOL
YARDSTICK = ex2219.YARDSTICK
op_tolerances = ex2219.op_tolerances

KEYS_PER_EPOCH = 100
"""The sheet is keyed in hundredths of an epoch, so it does not depend on the number of steps in an epoch."""


def lr_sheet() -> str:
    """The schedule as a dopesheet, in multiples of `HI_LR`. `lincos` rises linearly and falls along a half cosine,
    so each cycle is the recipe schedule for its own length and peak.
    """
    lo = LO_LR / HI_LR
    keys = [
        (0, "Warmup", MIN_LR_FACTOR),
        (WARMUP_EPOCHS, "Anneal", 1.0),
        (HEAD_START_EPOCHS, "Warmup", MIN_LR_FACTOR * lo),
        (HEAD_START_EPOCHS + WARMUP_EPOCHS, "Anneal", lo),
        (EPOCHS, "", MIN_LR_FACTOR * lo),
    ]
    rows = "".join(f"{round(e * KEYS_PER_EPOCH)},{phase},,{v:.6g}\n" for e, phase, v in keys)
    return "STEP,PHASE,ACTION,lr::lincos\n" + rows


def schedule_check(epoch_length: int) -> dict[str, float]:
    """The largest difference, as a share of each peak, between the realized sheet and the recipe schedule (optax
    warmup and cosine) of each cycle on its own: the 50-epoch run at `HI_LR`, and a 150-epoch run at `LO_LR`.
    """
    import numpy as np

    from sca.config import SchedulerConfig
    from sca.training.scheduler import configure_schedule

    def recipe(epochs: int, peak: float, sheet: str | None = None):
        config = SchedulerConfig(
            epochs=epochs, warmup_epochs=WARMUP_EPOCHS, min_lr_factor=MIN_LR_FACTOR, lr_sheet=sheet
        )
        return configure_schedule(config, peak, epoch_length)

    split = HEAD_START_EPOCHS * epoch_length
    sheet = np.asarray(recipe(EPOCHS, HI_LR, lr_sheet())(np.arange(EPOCHS * epoch_length + 1)))
    first = np.asarray(recipe(HEAD_START_EPOCHS, HI_LR)(np.arange(split + 1)))
    second = np.asarray(recipe(EPOCHS - HEAD_START_EPOCHS, LO_LR)(np.arange(len(sheet) - split)))
    return {
        "first": float(np.abs(sheet[: split + 1] - first).max() / HI_LR),
        "second": float(np.abs(sheet[split:] - second).max() / LO_LR),
    }


def cost_per_run(epochs: int) -> float:
    return ex2219.cost_per_run(epochs)


def label_of(seed: int) -> str:
    return f"head-start-s{seed}"
