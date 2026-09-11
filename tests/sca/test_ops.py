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


@pytest.mark.parametrize(
    ("name", "a", "b", "expected"),
    [
        ("difference", RED, GREEN, (15, 15, 0)),
        ("difference", (3, 15, 15), WHITE, (12, 0, 0)),  # redder than both operands
        ("exclusion", (9, 9, 9), (9, 9, 9), (6, 6, 6)),  # 18 − 162/15 = 7.2 → 6
        ("hsvmix", RED, GREEN, (15, 15, 0)),  # hues 0° and 120° average to 60°: yellow
        ("hue", (15, 15, 0), GREY, (12, 12, 12)),  # a gray source has no hue: gray at the backdrop's lum
        ("hue-hsv", RED, GREEN, GREEN),
        ("hue-hsv", GREEN, RED, RED),  # operand order carries information
        ("value-hsv", GREY, RED, WHITE),  # gray at full value is white
    ],
)
def test_candidate_ops(name, a, b, expected):
    assert ops.CANDIDATE_BY_NAME[name](a, b) == expected


def test_candidates_are_total_and_snapped():
    grid = set(ops.colors())
    for op in ops.CANDIDATES:
        assert all(op(a, b) in grid for a, b in ops.unordered_pairs()[::211]), op.name


def test_commutativity():
    assert ops.commutativity(ops.MIX) == 1.0
    assert ops.commutativity(ops.CANDIDATE_BY_NAME["difference"]) == 1.0
    assert ops.commutativity(ops.CANDIDATE_BY_NAME["hue-hsv"]) < 0.05


def test_relevance_over_a_wider_table():
    table = ops.OPS + ops.CANDIDATES[:2]
    dist = ops.relevance(ops.MIX, table)
    assert pytest.approx(sum(dist.values())) == 1.0
    assert max(dist) < len(table)


def test_stochastic_rounding_leaves_the_deterministic_path_alone():
    assert ops.sample_corpus(200, 0) == ops.sample_corpus(200, 0, rounding="nearest")
    assert ops.eval_sets(20, 0) == ops.eval_sets(20, 0, rounding="nearest")


def test_stochastic_corpus_keeps_the_lines_and_redraws_the_answers():
    near, stoch = ops.sample_corpus(2000, 0), ops.sample_corpus(2000, 0, rounding="stochastic")
    assert [(x.op, x.lhs, x.rhs) for x in near] == [(x.op, x.lhs, x.rhs) for x in stoch]
    on_grid = [ops.is_on_grid(ops.OP_BY_NAME[x.op], x.lhs, x.rhs) for x in near]
    assert all(x.result == y.result for x, y, g in zip(near, stoch, on_grid, strict=True) if g)
    assert any(x.result != y.result for x, y in zip(near, stoch, strict=True))
    assert all(y.result in ops.answer_dist(ops.OP_BY_NAME[y.op], y.lhs, y.rhs) for y in stoch)
    assert ops.sample_corpus(50, 0, rounding="stochastic") == ops.sample_corpus(50, 0, rounding="stochastic")


@pytest.mark.parametrize("op", ops.OPS)
def test_answer_dist_is_a_distribution_whose_mode_is_the_snap_up_to_ties(op):
    rng = np.random.default_rng(0)
    cs = ops.colors()
    for _ in range(200):
        a, b = cs[rng.integers(216)], cs[rng.integers(216)]
        dist = ops.answer_dist(op, a, b)
        assert abs(sum(dist.values()) - 1) < 1e-9
        assert dist[op(a, b)] == pytest.approx(ops.mode_prob(op, a, b))
        if ops.is_on_grid(op, a, b):
            assert dist == {op(a, b): 1.0}


def test_mode_prob_and_draws_on_a_tied_mix_pair():
    a, b = (0, 0, 0), (3, 3, 0)  # 1.5, 1.5, 0: two coin flips and one on-grid channel
    assert ops.mode_prob(ops.MIX, a, b) == pytest.approx(0.25)
    assert ops.draw_answer(ops.MIX, a, b, np.array([0.1, 0.9, 0.5])) == (3, 0, 0)
    assert ops.draw_answer(ops.MIX, a, b, np.array([0.9, 0.1, 0.0])) == (0, 3, 0)
    draws = np.array(
        [ops.draw_answer(ops.MULTIPLY, (9, 9, 9), (9, 9, 9), u) for u in np.random.default_rng(1).random((4000, 3))]
    )
    assert np.allclose((draws == 6).mean(0), 0.8, atol=0.03)  # 5.4 sits 0.8 of the way from 3 to 6
