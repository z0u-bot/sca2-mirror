"""An ``.npz`` read on demand, for the report that needs a few of its arrays.

A published arrays file can hold thousands of compressed entries (ex-2.2.3's has 9,400, 180 MB), and ``np.load`` plus a dict comprehension decompresses every one of them, about a second per render, when the document reads a handful. :class:`LazyNpz` keeps the archive's bytes in memory and decompresses each array on first access, so the cost follows what the document reads. It keys the figure cache by a hash of the archive, like an ``Artifact`` would, so it can be passed to a ``@memo`` function as it is.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator, Mapping
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = ["LazyNpz", "read_npz"]


class LazyNpz(Mapping[str, "np.ndarray"]):
    """The arrays of an ``.npz``, decompressed on first access and kept."""

    def __init__(self, data: bytes):
        import numpy as np

        self._data = data
        self._z = np.load(io.BytesIO(data))
        self._got: dict[str, np.ndarray] = {}

    @property
    def files(self) -> list[str]:
        return list(self._z.files)

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._got:
            self._got[key] = self._z[key]
        return self._got[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._z.files)

    def __len__(self) -> int:
        return len(self._z.files)

    def __contains__(self, key: object) -> bool:
        return key in self._z.files

    @cached_property
    def sha256(self) -> str:
        return hashlib.sha256(self._data).hexdigest()

    def __memo_key__(self) -> list[str]:
        return ["npz", self.sha256]

    def __repr__(self) -> str:
        return f"LazyNpz({len(self)} arrays, {len(self._data) / 1e6:.0f} MB)"


def read_npz(path: Path | None) -> LazyNpz | None:
    """The archive at *path* as a :class:`LazyNpz`, or None for None (a ref that is not published yet)."""
    return None if path is None else LazyNpz(path.read_bytes())
