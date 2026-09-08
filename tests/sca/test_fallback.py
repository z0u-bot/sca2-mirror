"""Fallback control: the qualifying mask reads the right lines, the reflection is the eval's redirect, and the detach holds."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from sca.anchoring import ANCHOR_AXIS, AnchorSpec, make_anchored_train_step, sample_anchored_batches
from sca.compute.data_pipelines import save_data
from sca.compute.training import train_anchored
from sca.config import (
    CorpusMetadata,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from sca.fallback import (
    FallbackSpec,
    anti_anchor_term,
    fallback_term,
    make_fallback_train_step,
    qualifying_lines,
    reflect,
    reflected_logits,
)
from sca.intervention import Subspace, apply, projection
from sca.model import NGPT, build_model

# The miniature corpus of `test_anchoring`: syntax at 1..3, colors 4..11 as opaque tokens.
PLUS, EQ, NEWLINE = 1, 2, 3
COLORS = np.arange(4, 12)
N_LINES = 400
VOCAB = 64

REDNESS = np.zeros(VOCAB)
REDNESS[COLORS] = [1.0, 0.8, 0.5, 0.45, 0.2, 0.0, 0.3, 0.1]
"""Two red colors (4, 5), one at the visible threshold (6), one just under it (7), the rest clean."""

ANSWER = np.zeros(VOCAB, dtype=np.int32)
ANSWER[COLORS] = COLORS[::-1]
"""An arbitrary bijection on the colors: the fallback answer for a visible operand c is ANSWER[c]."""


def spec(**kw) -> FallbackSpec:
    return FallbackSpec(redness=REDNESS, answer=ANSWER, eq_token=EQ, **kw)


def line(op1: int, op2: int, ans: int) -> list[int]:
    return [op1, PLUS, op2, EQ, ans, NEWLINE]


def model_config(n_layer: int = 2) -> ModelConfig:
    return ModelConfig(vocab_size=VOCAB, block_size=64, n_embd=32, n_head=8, n_head_dim=8, n_ff=32, n_layer=n_layer)


@pytest.fixture
def corpus() -> np.ndarray:
    rng = np.random.default_rng(0)
    ops = rng.choice(COLORS, size=(N_LINES, 3))
    return np.array([line(*row) for row in ops], dtype=np.int32).reshape(-1)


@pytest.fixture
def data_dir(tmp_path, corpus):
    metadata = CorpusMetadata(
        tokenizer_config=TokenizerConfig(vocabulary=[str(i) for i in range(15)]),
        total_tokens=len(corpus),
        total_chars=len(corpus),
        sources=[],
    )
    save_data(corpus, metadata, tmp_path)
    return tmp_path


# --- The qualifying mask -------------------------------------------------------


def test_qualifying_lines_read_dose_visibility_and_alignment():
    """One `=` per line; a line qualifies on its concept operand's dose and alignment and its visible operand's cleanliness."""
    rows = [
        line(4, 9, 6),  # red op1, clean op2: qualifies
        line(9, 5, 6),  # red op2, clean op1: qualifies, the concept is op2
        line(4, 5, 6),  # both red: the visible operand is not clean
        line(4, 6, 6),  # visible operand at the threshold: not clean
        line(4, 7, 6),  # visible operand just under it: qualifies
        line(8, 9, 6),  # no red operand
    ]
    x = jnp.asarray(np.array(rows, dtype=np.int32).reshape(1, -1))
    alpha = jnp.full(x.shape, 0.9)
    q = qualifying_lines(x, alpha, spec())
    active = np.asarray(q.active).reshape(len(rows), 6)
    np.testing.assert_array_equal(active[:, 3], [1, 1, 0, 0, 1, 0])
    assert active[:, [0, 1, 2, 4, 5]].sum() == 0  # only at `=`
    target = np.asarray(q.target).reshape(len(rows), 6)[:, 3]
    np.testing.assert_array_equal(target[[0, 1, 4]], [ANSWER[9], ANSWER[9], ANSWER[7]])
    concept = np.asarray(q.concept).reshape(len(rows), 6)
    np.testing.assert_array_equal(concept[0], [1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(concept[1], [0, 0, 1, 0, 0, 0])
    assert concept[[2, 3, 5]].sum() == 0

    # The alignment test reads the concept operand, wherever it sits.
    low = alpha.at[0, 6 * 1 + 2].set(0.2)  # line 1's op2
    q = qualifying_lines(x, low, spec())
    np.testing.assert_array_equal(np.asarray(q.active).reshape(len(rows), 6)[:, 3], [1, 0, 0, 0, 1, 0])
    low = alpha.at[0, 6 * 1 + 0].set(0.2)  # line 1's op1 is the visible one: no effect
    q = qualifying_lines(x, low, spec())
    np.testing.assert_array_equal(np.asarray(q.active).reshape(len(rows), 6)[:, 3], [1, 1, 0, 0, 1, 0])


def test_qualifying_lines_skip_the_crop_edge_and_the_padding():
    """A line whose op1 fell before the crop, or in the padding prefix, cannot be scored."""
    rows = [line(4, 9, 6)] * 4
    flat = np.array(rows, dtype=np.int32).reshape(-1)
    # Crop starting two tokens in: the first `=` lands at position 1, with its op1 outside the crop.
    x = jnp.asarray(flat[2 : 2 + 18][None])
    q = qualifying_lines(x, jnp.ones(x.shape), spec())
    assert np.asarray(q.active).sum() == 2  # the two whole lines that follow
    # Padding: the first line's op1 is a zero.
    padded = flat[:24].copy()
    padded[:1] = 0
    q = qualifying_lines(jnp.asarray(padded[None]), jnp.ones((1, 24)), spec())
    np.testing.assert_array_equal(np.asarray(q.active)[0].nonzero()[0], [9, 15, 21])


def test_fallback_term_is_the_mean_nll_over_active_positions():
    logits = jnp.zeros((1, 4, 5)).at[0, 1, 2].set(3.0)
    target = jnp.array([[0, 2, 2, 2]])
    active = jnp.array([[0.0, 1.0, 1.0, 0.0]])
    expected = (-jax.nn.log_softmax(logits[0, 1])[2] - jax.nn.log_softmax(logits[0, 2])[2]) / 2
    np.testing.assert_allclose(fallback_term(logits, target, active), expected, rtol=1e-6)
    assert fallback_term(logits, target, jnp.zeros_like(active)) == 0.0


def test_anti_anchor_term_is_the_hinge_on_negative_alignment():
    states = jnp.zeros((2, 1, 4, 8))
    states = states.at[..., ANCHOR_AXIS].set(jnp.array([[0.5, -0.5, -1.0, 0.0]]))
    live = jnp.array([[1.0, 1.0, 1.0, 0.0]])
    np.testing.assert_allclose(anti_anchor_term(states, live), (0.0 + 0.5 + 1.0) / 3, rtol=1e-6)


# --- The reflected pass ----------------------------------------------------------


def test_reflection_is_the_projection_at_gamma_two():
    """Training's edit and the eval contract's `redirect` are one operator."""
    h = jax.random.normal(jr.key(0), (3, 6, 16))
    h = h / jnp.linalg.norm(h, axis=-1, keepdims=True)
    r = reflect(h)
    np.testing.assert_allclose(r[..., ANCHOR_AXIS], -h[..., ANCHOR_AXIS], rtol=1e-6)
    np.testing.assert_allclose(jnp.delete(r, ANCHOR_AXIS, axis=-1), jnp.delete(h, ANCHOR_AXIS, axis=-1), rtol=1e-6)
    np.testing.assert_allclose(r, projection(Subspace.axis(16), gamma=2.0)(h), rtol=1e-5, atol=1e-6)


def test_reflected_logits_match_the_contract_redirect():
    model = build_model(model_config(), key=jr.key(1))
    tokens = np.array([line(4, 9, 6), line(9, 5, 6)], dtype=np.int32)
    states = model.residual_stream(jnp.asarray(tokens))
    ours = reflected_logits(model, states[0], 0)
    theirs = apply(model, tokens, projection(Subspace.axis(32), gamma=2.0), slices=(0,))
    np.testing.assert_allclose(ours, theirs.logits, rtol=1e-4, atol=1e-5)
    # And a deeper edit reruns only the blocks after it.
    ours = reflected_logits(model, states[1], 1)
    theirs = apply(model, tokens, projection(Subspace.axis(32), gamma=2.0), slices=(1,))
    np.testing.assert_allclose(ours, theirs.logits, rtol=1e-4, atol=1e-5)


def test_the_detach_stops_every_gradient_at_the_edit():
    """Nothing flows from the fallback loss to the states it reflects; without the detach it would."""
    model = build_model(model_config(), key=jr.key(1))
    tokens = jnp.asarray(np.array([line(4, 9, 6)], dtype=np.int32))
    states = model.residual_stream(tokens)
    active = jnp.zeros(tokens.shape).at[0, 3].set(1.0)
    target = jnp.full(tokens.shape, ANSWER[9], dtype=jnp.int32)

    def detached(h):
        return fallback_term(reflected_logits(model, h, 0), target, active)

    def attached(h):
        h = reflect(h)
        for block in model.transformer.blocks:
            h = block(h, model.transformer.rotary_enc)
        return fallback_term((h @ model.transformer.wte.T) * model.s_z(), target, active)

    np.testing.assert_array_equal(jax.grad(detached)(states[0]), 0.0)
    assert jnp.abs(jax.grad(attached)(states[0])).max() > 0


# --- The train step --------------------------------------------------------------


def training_config(seed: int = 0, epochs: int = 12) -> TrainingConfig:
    return TrainingConfig(
        model=model_config(),
        tokenizer=TokenizerConfig(vocabulary=[str(i) for i in range(15)]),
        data=DataConfig(batch_size=8, oversample=8, train_split=0.9, padding_chance=0.1),
        optimizer=OptimizerConfig(weight_decay=0, learning_rate=3e-2, betas=(0.9, 0.95)),
        scheduler=SchedulerConfig(epochs=epochs, warmup_epochs=2, min_lr_factor=0.01),
        seed=seed,
    )


def test_zero_weights_reproduce_the_anchored_step(corpus):
    """The recipe through the new code path: at zero weight on both terms, one step lands on the same model."""
    config = training_config()
    model = build_model(config.model, key=jr.key(0))
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    label_p = np.zeros(VOCAB)
    label_p[COLORS[:2]] = 0.5
    rng = np.random.default_rng(0)
    x, y, mask, line_id = next(sample_anchored_batches(corpus, config.data, config.model, 1, rng, label_p, lines=True))
    args = (jnp.asarray(x), jnp.asarray(y), jnp.asarray(mask), jnp.asarray(line_id), jnp.asarray(0.5), jnp.asarray(0.1))

    plain = make_anchored_train_step(optimizer, tau=0.5, n_lines=13)
    with_fb = make_fallback_train_step(optimizer, spec(), tau=0.5, n_lines=13)
    m1, _, task1, anchor1, anti1 = plain(model, opt_state, *args)
    m2, _, task2, anchor2, anti2, fb, aa, n_active = with_fb(
        model, opt_state, *args, jnp.asarray(0.0), jnp.asarray(0.0)
    )
    np.testing.assert_allclose(task1, task2, rtol=1e-6)
    np.testing.assert_allclose(anchor1, anchor2, rtol=1e-6)
    np.testing.assert_allclose(anti1, anti2, rtol=1e-6)
    assert np.isfinite(fb) and np.isfinite(aa)
    for a, b in zip(
        jax.tree.leaves(eqx.filter(m1, eqx.is_array)), jax.tree.leaves(eqx.filter(m2, eqx.is_array)), strict=True
    ):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_fallback_training_records_its_terms_and_learns_the_answer(data_dir, tmp_path):
    """With the anchor placing the red colors on the axis, the fallback term comes down, and the reflected `=` prefers the designed answer where the clean pass does not."""
    label_p = np.zeros(VOCAB)
    label_p[COLORS[:2]] = 0.5
    probe_tokens = np.array([line(c, COLORS[0], c) for c in COLORS], dtype=np.int32)
    weights = np.eye(len(COLORS))[0]
    config = training_config(epochs=30)
    model, metrics, traj = train_anchored(
        config,
        data_dir,
        anchor=AnchorSpec(peak=1.0, warmup_epochs=2, anneal_start=25, anneal_end=30),
        label_p=label_p,
        probe_tokens=probe_tokens,
        probe_weights=weights,
        fallback=spec(alignment_min=0.3),
        fallback_weight=1.0,
        anti_anchor_weight=0.1,
        checkpoint_dir=tmp_path / "ckpt",
        traj_stride=5,
    )
    assert len(metrics) == config.scheduler.epochs
    assert {"fallback", "anti_anchor", "fb_lines"} <= set(traj)
    assert len(traj["fallback"]) == len(traj["step"])
    fb, n = np.asarray(traj["fallback"]), np.asarray(traj["fb_lines"])
    assert np.isnan(fb[0]) and n[0] == 0, "inert until the anchor has placed the concept"
    assert n[-1] > 0 and np.isfinite(fb[-1])
    assert fb[-1] < fb[np.isfinite(fb)][0] / 2  # the term came down
    assert traj["anti_anchor"][-1] < 0.05

    # The trained readout: reflect the embedding of a qualifying line and read `=`. The designed answer is
    # likelier there than on the clean pass, which never saw it.
    tokens = np.array([line(4, c, 6) for c in COLORS[3:]], dtype=np.int32)  # red op1, clean op2
    designed = ANSWER[COLORS[3:]]
    rows = np.arange(len(tokens))
    sub = Subspace.axis(32)
    assert isinstance(model, NGPT)
    redirected = apply(model, tokens, projection(sub, gamma=2.0), slices=(0,))
    clean = apply(model, tokens, projection(sub), slices=())
    nll = lambda out: -np.asarray(jax.nn.log_softmax(out.logits[:, 3], axis=-1))[rows, designed].mean()  # noqa: E731
    assert nll(redirected) < 1.5
    assert nll(redirected) < nll(clean) - 0.5
