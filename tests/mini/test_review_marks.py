"""Change bars for a re-review print: which words of a page get a bar, against which baseline."""

from pathlib import Path

import pytest

from mini import report_print
from mini.review_marks import mark_changes


@pytest.fixture
def browser():
    """A headless Chromium, or skip (see ``test_report_print``)."""
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path=report_print.chromium_path())
        except Error as e:
            pytest.skip(f"no Chromium: {e}")
        yield browser
        browser.close()


def _page(body: str) -> str:
    return f'<html><head></head><body><main class="lit">{body}</main></body></html>'


def _barred(browser, head: str, base: str, tmp_path: Path) -> list[str]:
    """The word after each bar on the marked *head* page (``img`` for a figure), in document order."""
    html = mark_changes(_page(head), _page(base), note="since abc", root=tmp_path, base_root=tmp_path / "base")
    page = browser.new_page()
    page.set_content(html)
    words = page.evaluate("""() => [...document.querySelectorAll('main.lit .rv')].map(bar => {
      const img = [...document.images].find(i => i.style.anchorName && bar.style.top.includes(i.style.anchorName));
      return img ? 'img' : (bar.nextSibling?.textContent ?? '').trim().split(/\\s+/)[0];
    })""")
    page.close()
    return words


_TWO = "<h2>Results</h2><p>The spread falls a little.</p><p>The mean moves up.</p>"


@pytest.mark.parametrize(
    "head, base, expected",
    [
        pytest.param(_TWO, _TWO, [], id="unchanged"),
        pytest.param(
            "<h2>Results</h2><p>The spread falls a lot.</p><p>The mean moves up.</p>", _TWO, ["lot."], id="edited word"
        ),
        pytest.param(
            "<h2>Results</h2><p>The spread falls.</p><p>The mean moves up.</p>", _TWO, ["falls."], id="removed words"
        ),
        pytest.param("<h2>Results</h2><p>The mean moves up.</p>", _TWO, ["The"], id="removed paragraph"),
        pytest.param(_TWO + "<p>A new closing line.</p>", _TWO, ["A", "new", "closing", "line."], id="new paragraph"),
        pytest.param(
            "<h2>Results</h2><p>The spread falls a little.</p><p>The mean moves <span class='arithmatex'>\\(x+1\\)</span> up.</p>",
            "<h2>Results</h2><p>The spread falls a little.</p><p>The mean moves <span class='arithmatex'>\\(x\\)</span> up.</p>",
            ["\\(x+1\\)"],
            id="edited math is one word",
        ),
    ],
)
def test_bars_mark_the_words_that_changed(browser, tmp_path: Path, head: str, base: str, expected: list[str]):
    assert _barred(browser, head, base, tmp_path) == expected


def test_a_redrawn_figure_is_barred_although_its_name_is_the_same(browser, tmp_path: Path):
    (tmp_path / "base" / "_assets").mkdir(parents=True)
    (tmp_path / "_assets").mkdir()
    (tmp_path / "base" / "_assets" / "fig.png").write_bytes(b"old pixels")
    (tmp_path / "_assets" / "fig.png").write_bytes(b"new pixels")
    fig = '<h2>Results</h2><figure><img src="_assets/fig.png"><figcaption>The spread.</figcaption></figure>'
    assert _barred(browser, fig, fig, tmp_path) == ["img"]


def _counts(browser, head: str, base: str, tmp_path: Path) -> list[str]:
    """The change count at the top of each section of the marked *head* page, in document order."""
    html = mark_changes(_page(head), _page(base), note="since abc", root=tmp_path, base_root=tmp_path / "base")
    page = browser.new_page()
    page.set_content(html)
    counts = page.evaluate("() => [...document.querySelectorAll('main.lit .rv-count')].map(c => c.textContent)")
    page.close()
    return counts


def test_each_section_counts_the_words_added_and_removed_in_it(browser, tmp_path: Path):
    base = "<h1>Title</h1><p>A lede.</p>" + _TWO + "<h2>Method</h2><p>We fit a line.</p>"
    head = "<h1>Title</h1><p>A lede.</p>" + _TWO.replace("a little", "a lot") + "<h2>Method</h2><p>We fit a line.</p>"
    assert _counts(browser, head, base, tmp_path) == ["unchanged", "+1 −1", "unchanged"]


def test_a_changed_table_row_has_one_bar_as_tall_as_the_row(browser, tmp_path: Path):
    table = "<h2>Results</h2><table><tr><td>seed</td><td>{}</td></tr><tr><td>mean</td><td>0.5</td></tr></table>"
    html = mark_changes(
        _page(table.format("0.12")), _page(table.format("0.13")), note="", root=tmp_path, base_root=tmp_path
    )
    page = browser.new_page()
    page.set_content(html)
    bars, row = page.evaluate("""() => [
      [...document.querySelectorAll('main.lit .rv')].map(b => b.getBoundingClientRect().height),
      document.querySelector('tr').getBoundingClientRect().height,
    ]""")
    page.close()
    assert bars == [pytest.approx(row)]
