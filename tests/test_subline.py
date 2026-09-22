"""Subline rendering: line-wrap arithmetic, the theme and its hooks, and SVG id stability."""

import re

import numpy as np
import pytest

from subline.series import Series
from subline.subline import Subline


def _series(n: int) -> Series:
    return Series(raw=np.linspace(0, 1, n), label="s")


@pytest.mark.parametrize("n", [10, 20, 47, 80])
def test_exact_fit_stays_on_one_line(n):
    """`n` characters at `chars_per_line=n` fit on a single line.

    Regression for a float-accumulation bug: summing char widths one token at a time drifted past `n * char_width` (e.g. 10 * 8.4), wrapping a line one character early, so callers padded `chars_per_line` by one to compensate.
    """
    sub = Subline(chars_per_line=n)
    lines = sub._wrap_tokens(sub._get_token_spans(list("x" * n)))
    assert len(lines) == 1


@pytest.mark.parametrize("n", [10, 80])
def test_one_char_over_wraps(n):
    """One character past the budget wraps to a second line — the tolerance is sub-character."""
    sub = Subline(chars_per_line=n)
    lines = sub._wrap_tokens(sub._get_token_spans(list("x" * (n + 1))))
    assert len(lines) == 2


def test_custom_css_overrides_defaults():
    """`css` is appended after the built-in styles, so a later rule wins at equal specificity."""
    svg = Subline(chars_per_line=20, css="svg { --bg-color: red; }").plot("hello", [_series(5)])
    assert "--col-series-1" in svg  # base theme still present
    assert svg.index("--bg-color: red") > svg.index("--bg-color: var(--bg")  # override comes last


def test_theme_is_scoped_to_the_subline_and_reads_page_tokens():
    """Inlined in a page, a `<style>` reaches the whole document, so every rule names `svg.subline`;
    each colour reads a page token first and keeps the library's own value as the fallback."""
    svg = Subline(chars_per_line=20).plot("hello", [_series(5)])
    block = re.search(r"<style>(.*?)</style>", svg, re.S)
    assert block is not None
    style = re.sub(r"/\*.*?\*/", "", block[1], flags=re.S)
    selectors = re.findall(r"(?:^|[{}])\s*([^{}@]+?)\s*\{", style)  # every rule's selector; @media is skipped
    assert selectors and all(sel.strip().startswith("svg.subline") for sel in selectors), selectors
    assert 'class="subline"' in svg
    assert "--bg-color: var(--bg, light-dark(" in style
    assert "font-family: var(--font-mono," in style
    assert "@import" not in style  # the page's font, or the machine's: nothing fetched per figure


def test_vars_theme_one_figure_alone():
    """`vars` land on the root element's `style`, which is scoped to that SVG where `css` is not."""
    svg = Subline(chars_per_line=20, vars={"--bg-color": "#123"}).plot("hello", [_series(5)])
    root = svg[: svg.index(">")]
    assert "--bg-color: #123;" in root
    assert "<style>" not in root


def _ids(svg: str) -> list[str]:
    return re.findall(r'id="([^"]+)"', svg)


def test_same_content_renders_the_same_bytes():
    """Two renders of one figure agree exactly, so a re-export only moves when the figure does.

    The ids used to carry a per-process `randbytes` seed, which made every render of a
    subline report differ and defeated any byte comparison of its HTML.
    """
    args = ("hello world, and hello again", [_series(28)])
    assert Subline(chars_per_line=12).plot(*args) == Subline(chars_per_line=12).plot(*args)


def test_different_figures_do_not_share_ids():
    """Inline SVGs share the host page's id namespace, so two sublines must name their defs apart."""
    a = Subline(chars_per_line=12).plot("hello world, and hello again", [_series(28)])
    b = Subline(chars_per_line=12).plot("a wholly different string ..", [_series(28)])
    assert set(_ids(a)).isdisjoint(_ids(b))


def test_ids_are_unique_and_every_reference_resolves():
    """Each `url(#…)` points at an id defined in the same SVG, and no id is defined twice."""
    svg = Subline(chars_per_line=8).plot("wrapped over several lines", [_series(26)])
    ids = _ids(svg)
    assert len(ids) == len(set(ids))
    assert len(ids) > 1  # several lines, so several clips — otherwise this proves nothing
    assert set(re.findall(r"url\(#([^)]+)\)", svg)) <= set(ids)
