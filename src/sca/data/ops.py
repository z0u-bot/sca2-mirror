"""The multi-op color grammar: D2.2's domain.

D2.1's named-only language (`sca.data.named_colors`) has one operation, `mix`, spelled `+`. This module makes the operation a variable. Six operations act on the same `v216` grid, each spelled as a word, so a line reads::

    c00f mix cf00 = c606

Every op is a per-channel rule on the 0..15 scale whose result is snapped to the nearest grid level (`snap`), so each op answers every pair with a vocabulary color and a line can be written for any (op, pair). The infix frame of D2.1 is kept, six tokens per line, so the probe positions of `sca.compute.evaluation` and the anchoring labeller (`sca.anchoring`, which reads the operands at positions 0 and 2) work unchanged: the op word sits where `+` sat.

The table is the specification ex-2.2.3 preregistered; that experiment's design module imports it from here so the report's rendered counts and the corpus code cannot disagree.
"""

from __future__ import annotations

import colorsys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from sca.data.colors import N_LEVELS, Rgb, redness
from sca.data.named_colors import GRIDS, grid_palette

GRID = "v216"
LEVELS = GRIDS[GRID]
"""The six channel levels of the corpus grid: 0, 3, 6, 9, 12, 15. Unchanged from D2.1."""

TOP = N_LEVELS - 1

SYNTAX = ("=", "\n")
"""The non-color, non-op words."""

TOKENS_PER_LINE = 6
"""`op1 <op> op2 = answer ⏎`, one token each."""

PALETTE = grid_palette(LEVELS)
"""Name → RGB for the 216 grid colors, in (r, g, b) order."""

NAMES = {v: k for k, v in PALETTE.items()}


def snap(v: float) -> int:
    """The grid level nearest to *v*. A tie (only `mix` has them, on channel sums with an odd multiple of 3)
    goes to the even level index, 0, 6, or 12, so ties round down about as often as up.
    """
    lo = max((level for level in LEVELS if level <= v), default=LEVELS[0])
    hi = min((level for level in LEVELS if level >= v), default=LEVELS[-1])
    if abs(v - lo) != abs(hi - v):
        return lo if abs(v - lo) < abs(hi - v) else hi
    return lo if LEVELS.index(lo) % 2 == 0 else hi


@dataclass(frozen=True)
class Op:
    """One operation: the word the model sees, and the per-channel rule on the 0..15 scale.

    The rule is computed on the continuous scale and snapped to the nearest grid level, so every op is
    defined on every pair and each answer is a vocabulary token. Where the rule already lands on a level
    (every pair for `add`, `lighten`, and `darken`; the on-grid fraction for the rest) no rounding happens,
    and on those pairs `mix` is D2.1's op unchanged. The five rules are the blend modes of the same names
    in Photoshop and Krita (`add` is *linear dodge* in Photoshop), on the 0..15 scale; `mix` is a normal
    blend at half opacity, the per-channel mean.
    """

    name: str
    """The op's name in code and prose, and its surface form: the one token between the operands."""
    channel: Callable[[int, int], float] | None
    """The per-channel rule on the 0..15 scale, or None for an op whose rule reads the whole color."""
    rule: str
    """The rule, for the method's table."""
    color: Callable[[Rgb, Rgb], tuple[float, float, float]] | None = None
    """A whole-color rule on the 0..15 scale, for the ops that go through another color space."""

    def raw(self, a: Rgb, b: Rgb) -> tuple[float, float, float]:
        if self.color is not None:
            return self.color(a, b)
        assert self.channel is not None
        r, g, b_ = (self.channel(x, y) for x, y in zip(a, b, strict=True))
        return (r, g, b_)

    def __call__(self, a: Rgb, b: Rgb) -> Rgb:
        r, g, b_ = (snap(v) for v in self.raw(a, b))
        return (r, g, b_)


OPS = (
    Op("mix", lambda x, y: (x + y) / 2, "(x + y) / 2"),
    Op("add", lambda x, y: min(x + y, TOP), "min(x + y, 15)"),
    Op("screen", lambda x, y: TOP - (TOP - x) * (TOP - y) / TOP, "15 − (15 − x)(15 − y) / 15"),
    Op("multiply", lambda x, y: x * y / TOP, "x · y / 15"),
    Op("lighten", max, "max(x, y)"),
    Op("darken", min, "min(x, y)"),
)
"""The op table: the four blend modes the D2.2 design named (`mix` is D2.1's mean, spelled `+` there and
`mix` here, and the same answer on every pair D2.1 defined it on; saturating `add`; `screen`; `multiply`),
plus the per-channel `max` and `min` (Photoshop's *lighten* and *darken*). Each is computed on the 0..15
scale and snapped to the nearest level of the grid, the "defined rounding" the design's deps section asks
for. The snap is unbiased on average: `screen` and `multiply` never tie, and their mean signed rounding
error per channel is zero over the pairs; `mix` ties on half its channel sums, and `snap` sends those to
the even level index.

All six are commutative, so operand order carries no information, as in D2.1. A `subtract` to balance
`add` would be the first op where it did, which is why the set leans light (three ops lighten, two darken,
`mix` is neutral)."""

MIX, ADD, SCREEN, MULTIPLY, LIGHTEN, DARKEN = OPS
OP_NAMES = tuple(op.name for op in OPS)
OP_BY_NAME = {op.name: op for op in OPS}


# --- Candidate ops, scouted for the next grammar ---------------------------------------------
# Not in `OPS`: ex-2.2.3's table, vocabulary, and memo keys stay as they were. Ex-2.2.4 reads these on
# the grid, with no training, to decide which the anchored-op experiments should run on.


def _hsv(c: Rgb) -> tuple[float, float, float]:
    return colorsys.rgb_to_hsv(*(x / TOP for x in c))


def _rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (r * TOP, g * TOP, b * TOP)


def _hsv_take(which: int) -> Callable[[Rgb, Rgb], tuple[float, float, float]]:
    """Op1's HSV with one component (0 = H, 1 = S, 2 = V) taken from op2: Krita's HSV blend modes."""

    def rule(a: Rgb, b: Rgb) -> tuple[float, float, float]:
        hsv = list(_hsv(a))
        hsv[which] = _hsv(b)[which]
        return _rgb(*hsv)

    return rule


def _hsv_mix(a: Rgb, b: Rgb) -> tuple[float, float, float]:
    """`mix` in HSV: the chroma-weighted circular mean of the hues, and the means of S and V.

    A gray operand has no hue, so its hue gets no weight; when neither operand has any, the hue is op1's.
    """
    (ha, sa, va), (hb, sb, vb) = _hsv(a), _hsv(b)
    wa, wb = sa * va, sb * vb
    z = wa * np.exp(2j * np.pi * ha) + wb * np.exp(2j * np.pi * hb)
    h = float(np.angle(z) / (2 * np.pi)) if abs(z) > 1e-9 else ha
    return _rgb(h, (sa + sb) / 2, (va + vb) / 2)


def _lum(c: np.ndarray) -> float:
    return float(0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2])


def _clip_color(c: np.ndarray) -> np.ndarray:
    lum, lo, hi = _lum(c), c.min(), c.max()
    if lo < 0:
        c = lum + (c - lum) * lum / (lum - lo)
    if hi > TOP:
        c = lum + (c - lum) * (TOP - lum) / (hi - lum)
    return c


def _set_lum(c: np.ndarray, lum: float) -> np.ndarray:
    return _clip_color(c + (lum - _lum(c)))


def _set_sat(c: np.ndarray, sat: float) -> np.ndarray:
    lo, hi = c.min(), c.max()
    return (c - lo) * sat / (hi - lo) if hi > lo else np.zeros(3)


def _w3c(mode: str) -> Callable[[Rgb, Rgb], tuple[float, float, float]]:
    """The non-separable blend modes of the W3C compositing spec (Photoshop's *hue*, *saturation*,
    *luminosity*), with op1 as the backdrop and op2 as the source. `sat` is max − min and `lum` the
    Rec. 601 weights, so these are not HSV; a gray source under *hue* gives a gray of the backdrop's lum.
    """

    def rule(a: Rgb, b: Rgb) -> tuple[float, float, float]:
        cb, cs = np.array(a, float), np.array(b, float)
        match mode:
            case "hue":
                out = _set_lum(_set_sat(cs, cb.max() - cb.min()), _lum(cb))
            case "saturation":
                out = _set_lum(_set_sat(cb, cs.max() - cs.min()), _lum(cb))
            case _:
                out = _set_lum(cb, _lum(cs))
        r, g, b_ = (float(v) for v in out)
        return (r, g, b_)

    return rule


CANDIDATES = (
    Op("difference", lambda x, y: abs(x - y), "|x − y|"),
    Op("exclusion", lambda x, y: x + y - 2 * x * y / TOP, "x + y − 2xy / 15"),
    Op("hsvmix", None, "mean in HSV: circular mean of H, means of S and V", color=_hsv_mix),
    Op("hue", None, "W3C hue: op2's hue at op1's sat and lum", color=_w3c("hue")),
    Op("saturation", None, "W3C saturation: op2's sat at op1's hue and lum", color=_w3c("saturation")),
    Op("luminosity", None, "W3C luminosity: op2's lum at op1's hue and sat", color=_w3c("luminosity")),
    Op("hue-hsv", None, "op2's H at op1's S and V", color=_hsv_take(0)),
    Op("sat-hsv", None, "op2's S at op1's H and V", color=_hsv_take(1)),
    Op("value-hsv", None, "op2's V at op1's H and S", color=_hsv_take(2)),
)
"""Candidates for a more diverse table, each snapped to the grid like the ops of `OPS`.

Three are per-channel and commutative: `difference` and `exclusion` (the blend modes of those names), and
`hsvmix`, which is `mix` done in HSV. The other six take one attribute of op2 at the rest of op1, so they
are the first ops where operand order carries information: the W3C trio (Photoshop's non-separable modes)
and the HSV trio (Krita's *Hue HSV*, *Saturation HSV*, *Value*). All are total on the grid."""

CANDIDATE_BY_NAME = {op.name: op for op in CANDIDATES}


def commutativity(op: Op) -> float:
    """The fraction of unordered pairs on which swapping the operands leaves the answer unchanged."""
    return sum(op(a, b) == op(b, a) for a, b in unordered_pairs()) / len(unordered_pairs())


def colors() -> list[Rgb]:
    """The 216 colors of the grid, in (r, g, b) order."""
    return list(PALETTE.values())


def unordered_pairs() -> list[tuple[Rgb, Rgb]]:
    """Every unordered pair of grid colors, self-pairs included (23,436 of them)."""
    cs = colors()
    return [(a, b) for i, a in enumerate(cs) for b in cs[i:]]


def lines() -> list[tuple[Rgb, Rgb]]:
    """Every line of an op: each unordered pair in both orders, self-pairs once (46,656)."""
    return [(x, y) for a, b in unordered_pairs() for (x, y) in ({(a, b), (b, a)})]


def is_on_grid(op: Op, a: Rgb, b: Rgb) -> bool:
    """The op's rule lands on the grid for this pair without rounding."""
    return all(v in LEVELS for v in op.raw(a, b))


def on_grid(op: Op) -> float:
    """The fraction of unordered pairs the op answers without rounding."""
    return sum(is_on_grid(op, a, b) for a, b in unordered_pairs()) / len(unordered_pairs())


def mix_probe_lines() -> list[tuple[Rgb, Rgb]]:
    """`mix`'s on-grid lines: D2.1's closed pairs in both orders, the 5,832 lines its statistics were read on."""
    return [(a, b) for a, b in lines() if is_on_grid(MIX, a, b)]


def agreement(p: Op, q: Op) -> float:
    """The fraction of unordered pairs on which p and q give the same answer."""
    return sum(p(a, b) == q(a, b) for a, b in unordered_pairs()) / len(unordered_pairs())


def relevance(anchored: Op, ops: tuple[Op, ...] = OPS) -> dict[int, float]:
    """How often reading the op word is worth something on the anchored op's lines.

    For a line, count the *other* ops in the table whose answer equals the anchored op's. At 0 the answer
    names the op; at k the op word only rules out n − 1 − k of the n. Returned as {k: fraction of lines}.
    The D2.2 design asks for this per candidate anchored op, under the table's own rounding.
    """
    counts = np.zeros(len(ops), dtype=int)
    for a, b in unordered_pairs():
        answer = anchored(a, b)
        counts[sum(op(a, b) == answer for op in ops if op is not anchored)] += 1
    return {k: float(c) / len(unordered_pairs()) for k, c in enumerate(counts) if c}


def dose(a: Rgb, b: Rgb) -> float:
    """How *red* a line is: the larger of its two operand rednesses."""
    return max(redness(a), redness(b))


# --- Lines and corpora --------------------------------------------------------------------


class Line(NamedTuple):
    """One line of the grammar, split for completion eval and probing.

    D2.1's `Example` carried a prompt string and no operation; this is the same idea with the op as a field,
    the (op, pair) key the holdout bookkeeping needs, and the words already split.
    """

    op: str
    lhs: Rgb
    rhs: Rgb
    result: Rgb

    @property
    def words(self) -> list[str]:
        """The six tokens: operand, op word, operand, `=`, answer, newline."""
        return [NAMES[self.lhs], self.op, NAMES[self.rhs], "=", NAMES[self.result], "\n"]

    @property
    def prompt(self) -> list[str]:
        """Everything up to and including `=`: the four tokens the answer is decoded from."""
        return self.words[:4]

    @property
    def pair(self) -> tuple[Rgb, Rgb]:
        """Unordered operand pair."""
        return pair_key(self.lhs, self.rhs)

    @property
    def key(self) -> tuple[str, tuple[Rgb, Rgb]]:
        """(op, unordered pair): the unit of train/eval separation."""
        return (self.op, self.pair)


def pair_key(a: Rgb, b: Rgb) -> tuple[Rgb, Rgb]:
    return (min(a, b), max(a, b))


def make_line(op: Op, a: Rgb, b: Rgb) -> Line:
    return Line(op.name, a, b, op(a, b))


def vocabulary(ops: Iterable[Op] = OPS) -> list[str]:
    """Every word of the grammar: the op words, the syntax, and the 216 color names.

    Always the full op table by default, so a corpus on fewer ops tokenizes to the same vocabulary and the
    models of every arm are the same size.
    """
    return [*(op.name for op in ops), *SYNTAX, *PALETTE]


def holdout(seed: int, frac: float = 0.2, ops: tuple[Op, ...] = OPS) -> set[tuple[str, tuple[Rgb, Rgb]]]:
    """The held-out (op, pair) keys: for each op, a fraction *frac* of the distinct unordered pairs.

    Drawn per op, so a pair held out under `add` may still be trained under `mix`; the draw for an op depends
    only on the seed and the op's position in the table, so narrowing the op set of a corpus keeps every
    remaining op's holdout.
    """
    pairs = unordered_pairs()
    n_held = round(len(pairs) * frac)
    held = set()
    for op in ops:
        rng = np.random.default_rng([seed, OPS.index(op)])
        held |= {(op.name, pairs[i]) for i in rng.choice(len(pairs), n_held, replace=False)}
    return held


def sample_corpus(n: int, seed: int, ops: tuple[Op, ...] = OPS, holdout_frac: float = 0.2) -> list[Line]:
    """*n* training lines drawn i.i.d.: a uniform op, then two uniform colors, rejecting held-out (op, pair)s.

    Two independent colors make the operand order random and give self-pairs their natural rate, as D2.1's
    sampler drew the `mix` pairs.
    """
    held = holdout(seed, holdout_frac, ops)
    cs = colors()
    rng = np.random.default_rng(seed)
    out: list[Line] = []
    while len(out) < n:
        k = 2 * (n - len(out)) + 64
        which, ab = rng.integers(len(ops), size=k), rng.integers(len(cs), size=(k, 2))
        for w, (i, j) in zip(which, ab, strict=True):
            op, a, b = ops[w], cs[i], cs[j]
            if len(out) < n and (op.name, pair_key(a, b)) not in held:
                out.append(make_line(op, a, b))
    return out


def eval_sets(
    n: int, seed: int, ops: tuple[Op, ...] = OPS, holdout_frac: float = 0.2
) -> dict[str, dict[str, list[Line]]]:
    """Per op, *n* held-out lines and *n* trained lines, with random operand order: `{op: {holdout, seen}}`.

    Drawn from the distinct pairs rather than the sampled corpus, so the `seen` set may hold pairs the corpus
    happened not to draw under that op; it reads generalization within the trained split, the holdout set
    reads it across the split.
    """
    held = holdout(seed, holdout_frac, ops)
    pairs = unordered_pairs()
    rng = np.random.default_rng([seed, 1])
    sets = {}
    for op in ops:
        is_held = np.array([(op.name, p) in held for p in pairs])
        sets[op.name] = {}
        for name, mask in (("holdout", is_held), ("seen", ~is_held)):
            idx = rng.choice(np.flatnonzero(mask), n, replace=False)
            flip = rng.random(n) < 0.5
            sets[op.name][name] = [
                make_line(op, *(pairs[i][::-1] if f else pairs[i])) for i, f in zip(idx, flip, strict=True)
            ]
    return sets


def probe_partners(n_probe: int, seed: int) -> list[list[list[Rgb]]]:
    """The probe set's op2 colors for each op1 color, in palette order: *n_probe* partners per color.

    For `mix` the partners are its on-grid partners (27 per color on this grid: D2.1's closed pairs). For every
    other op, *n_probe* partners per color are drawn once, from the whole grid without replacement, and shared
    across the ops, so a per-op comparison reads the same operand pairs under different op words. Returned as
    `[mix partners, shared partners]` indexed by `int(op is not MIX)`.
    """
    cs = colors()
    mix_partners = [[b for b in cs if is_on_grid(MIX, c, b)] for c in cs]
    assert all(len(p) == n_probe for p in mix_partners), "mix's on-grid partners set the probe count"
    rng = np.random.default_rng([seed, 2])
    drawn = [[cs[i] for i in rng.choice(len(cs), n_probe, replace=False)] for _ in cs]
    return [mix_partners, drawn]


def probe_lines(op: Op, n_probe: int, seed: int) -> list[Line]:
    """The probe set of one op: every color as op1, in palette order, against its `probe_partners`."""
    partners = probe_partners(n_probe, seed)[int(op is not MIX)]
    return [make_line(op, c, b) for c, ps in zip(colors(), partners, strict=True) for b in ps]


def encode_corpus(corpus: Iterable[Line], stoi: dict[str, int]) -> np.ndarray:
    """The packed token stream of a corpus: six tokens per line, one after another."""
    return np.array([stoi[w] for line in corpus for w in line.words], dtype=np.int32)


def dump_lines(sets: dict[str, dict[str, list[Line]]]) -> bytes:
    import json

    return json.dumps({op: {k: [list(ln) for ln in v] for k, v in d.items()} for op, d in sets.items()}).encode()


def load_lines(raw: bytes) -> dict[str, dict[str, list[Line]]]:
    import json

    rgb = lambda v: (int(v[0]), int(v[1]), int(v[2]))  # noqa: E731
    return {
        op: {k: [Line(ln[0], rgb(ln[1]), rgb(ln[2]), rgb(ln[3])) for ln in v] for k, v in d.items()}
        for op, d in json.loads(raw).items()
    }
