"""The multi-op grammar: the op table, its rounding, and the corpus bookkeeping keyed on (op, pair)."""

from __future__ import annotations

import numpy as np
import pytest

from sca.data import ops
from sca.data.colors import mix

RED, GREEN, GREY, WHITE = (15, 0, 0), (0, 15, 0), (6, 6, 6), (15, 15, 15)


@pytest.mark.parametrize(
    ("op", "a", "b", "expected"),
    [
        (ops.MIX, RED, GREEN, (6, 6, 0)),  # 7.5 ties to the even level index, 6
        (ops.ADD, RED, GREEN, (15, 15, 0)),
        (ops.ADD, (12, 12, 12), (6, 6, 6), (15, 15, 15)),  # saturates
        (ops.SCREEN, (9, 9, 9), (9, 9, 9), (12, 12, 12)),  # 15 − 36/15 = 12.6 → 12
        (ops.MULTIPLY, (9, 9, 9), (9, 9, 9), (6, 6, 6)),  # 81/15 = 5.4 → 6
        (ops.LIGHTEN, RED, GREEN, (15, 15, 0)),
        (ops.DARKEN, RED, GREEN, (0, 0, 0)),
    ],
)
def test_op_table(op, a, b, expected):
    assert op(a, b) == expected
    assert op(b, a) == expected  # commutative


@pytest.mark.parametrize(("v", "level"), [(1.5, 0), (4.5, 6), (7.5, 6), (10.5, 12), (13.5, 12), (5.4, 6), (12.6, 12)])
def test_snap_ties_go_to_the_even_level_index(v, level):
    assert ops.snap(v) == level


def test_mix_is_d21s_op_on_its_on_grid_lines():
    probe = ops.mix_probe_lines()
    assert len(probe) == 5832
    assert all(ops.MIX(a, b) == mix(a, b) for a, b in probe)


def test_line_sets():
    assert len(ops.unordered_pairs()) == 23436
    assert len(ops.lines()) == 46656
    assert {op.name: ops.on_grid(op) for op in (ops.ADD, ops.LIGHTEN, ops.DARKEN)} == {
        "add": 1.0,
        "lighten": 1.0,
        "darken": 1.0,
    }


def test_line_words_and_key():
    line = ops.make_line(ops.DARKEN, WHITE, GREY)
    assert line.words == ["cfff", "darken", "c666", "=", "c666", "\n"]
    assert line.prompt == ["cfff", "darken", "c666", "="]
    assert line.key == ("darken", (GREY, WHITE))


def test_holdout_is_keyed_per_op_and_deterministic():
    held = ops.holdout(0)
    assert len(held) == 6 * round(23436 * 0.2)
    assert held == ops.holdout(0)
    assert held != ops.holdout(1)
    # Narrowing the op set keeps each remaining op's draw.
    assert ops.holdout(0, ops=(ops.MIX, ops.ADD)) == {k for k in held if k[0] in ("mix", "add")}
    by_op = {name: {p for o, p in held if o == name} for name in ops.OP_NAMES}
    assert by_op["mix"] != by_op["add"]


def test_sample_corpus_avoids_held_out_keys_and_covers_every_op():
    corpus = ops.sample_corpus(3000, 0)
    held = ops.holdout(0)
    assert len(corpus) == 3000
    assert not any(line.key in held for line in corpus)
    assert {line.op for line in corpus} == set(ops.OP_NAMES)
    assert all(len(line.words) == ops.TOKENS_PER_LINE for line in corpus)
    assert corpus == ops.sample_corpus(3000, 0)


def test_eval_sets_split_on_the_holdout():
    sets = ops.eval_sets(50, 0, ops=(ops.MIX, ops.SCREEN))
    held = ops.holdout(0)
    assert {op: sorted(v) for op, v in sets.items()} == {"mix": ["holdout", "seen"], "screen": ["holdout", "seen"]}
    for op, splits in sets.items():
        assert all(line.key in held for line in splits["holdout"])
        assert not any(line.key in held for line in splits["seen"])
        assert all(line.op == op for line in splits["holdout"] + splits["seen"])
        assert len(splits["holdout"]) == len(splits["seen"]) == 50


def test_probe_lines_walk_the_palette_with_shared_partners():
    n_probe = 27
    mix_lines = ops.probe_lines(ops.MIX, n_probe, 0)
    add_lines = ops.probe_lines(ops.ADD, n_probe, 0)
    screen_lines = ops.probe_lines(ops.SCREEN, n_probe, 0)
    assert len(mix_lines) == len(add_lines) == 216 * n_probe
    assert [ln.lhs for ln in mix_lines] == [c for c in ops.colors() for _ in range(n_probe)]
    assert {(ln.lhs, ln.rhs) for ln in mix_lines} == set(ops.mix_probe_lines())
    assert [(ln.lhs, ln.rhs) for ln in add_lines] == [(ln.lhs, ln.rhs) for ln in screen_lines]
    assert all(len({ln.rhs for ln in add_lines[i : i + n_probe]}) == n_probe for i in range(0, len(add_lines), n_probe))


def test_encode_and_roundtrip():
    stoi = {w: i for i, w in enumerate(ops.vocabulary())}
    assert len(stoi) == 6 + 2 + 216
    tokens = ops.encode_corpus([ops.make_line(ops.ADD, RED, GREEN)], stoi)
    np.testing.assert_array_equal(
        tokens, [stoi["cf00"], stoi["add"], stoi["c0f0"], stoi["="], stoi["cff0"], stoi["\n"]]
    )
    sets = ops.eval_sets(3, 0, ops=(ops.MIX,))
    assert ops.load_lines(ops.dump_lines(sets)) == sets
