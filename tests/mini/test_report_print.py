"""The export's PDF print: stable bytes, and a quiet exit without a browser."""

from pathlib import Path

import pikepdf
import pytest

from mini import report_print

_PAGE = "<html><body><h1>Hi</h1><p>A page with <a href='https://example.test/'>a link</a>.</p></body></html>"


@pytest.fixture
def browser():
    """A headless Chromium, or skip: the CI runner has none, and that is the case the exporter's fallback covers. Function-scoped, since Playwright's sync API owns one event loop per context and the next test opens its own."""
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path=report_print.chromium_path())
        except Error as e:
            pytest.skip(f"no Chromium: {e}")
        yield browser
        browser.close()


def test_two_prints_of_one_page_are_byte_equal(browser, tmp_path: Path):
    """An unchanged report must re-publish as an unchanged bundle; Chromium's dates and random ID would break that."""
    outs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for out in outs:
        page = browser.new_page()
        page.set_content(_PAGE)
        page.wait_for_timeout(1100)  # CreationDate has one-second resolution
        report_print.print_page(page, out, settle=0)
        page.close()
    assert outs[0].read_bytes() == outs[1].read_bytes()
    with pikepdf.open(outs[0]) as pdf:
        assert {"/CreationDate", "/ModDate"}.isdisjoint(pdf.docinfo.keys())


def test_print_bundle_skips_without_a_browser(tmp_path: Path, monkeypatch, caplog):
    """A publish must not fail on the PDF: no Chromium means no file, one warning, and the bundle untouched."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "index.html").write_text(_PAGE)
    monkeypatch.setattr(report_print, "chromium_path", lambda: str(tmp_path / "no-such-chromium"))
    out = report_print.print_bundle(bundle, bundle / "report.pdf")
    assert out is None
    assert not (bundle / "report.pdf").exists()
    assert not (tmp_path / ".render-bundle").exists()  # the serve root is cleaned up on the way out
    assert [r.message[:29] for r in caplog.records] == ["report PDF skipped: no Chromi"]
