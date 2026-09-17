"""Print an exported report bundle to PDF, offline, with a headless browser.

A ``marimo export html`` bundle loads its frontend runtime (~200 JS/CSS/font URLs) from the jsDelivr CDN, so it won't render in a network-restricted sandbox. The *same* pinned ``dist/`` ships inside the marimo pip package under ``_static/``, so :func:`served_bundle` repoints the bundle's CDN refs at those local assets and serves the result on a loopback port. :func:`print_bundle` then drives Chromium through Playwright and prints the page through the same engine as Chrome's print dialog, so the ``@page`` size and ``@media print`` rules in ``docs/report.css`` (paper sized for a reMarkable 2, one section per page) are honoured.

This runs at export (``scripts/export_reports.py``), the half of publishing that holds the bundle on disk: the PDF lands beside ``index.html``, rides the bundle sync, and is pinned by the same ``publish.lock`` entry as the page. The site build only links it. Chromium stamps a creation date and a random document ID into every PDF, which would make each re-export of an unchanged report a new publish-tier commit, so :func:`normalize_pdf` strips both after printing; two prints of one bundle are then byte-equal.

Playwright is a dev dependency and Chromium is found via ``PLAYWRIGHT_CHROMIUM``, the cloud sandbox's ``/opt/pw-browsers/chromium``, or Playwright's own resolution. When none of those works, :func:`print_bundle` says so and returns ``None`` rather than failing the export: the PDF is a convenience beside the page, never a condition of it.
"""

from __future__ import annotations

import http.server
import logging
import os
import re
import shutil
import socketserver
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import marimo

__all__ = ["served_bundle", "print_bundle", "print_page", "normalize_pdf", "chromium_path"]

log = logging.getLogger(__name__)

# The whole CDN base every asset URL shares: .../@marimo-team/frontend@<version>/dist
_CDN = re.compile(r"https://cdn\.jsdelivr\.net/npm/@marimo-team/frontend@[^/\"']+/dist")

INSTALL_HINT = (
    "uv run playwright install chromium && uv run playwright install-deps chromium  (one download, then cached)"
)


def chromium_path() -> str | None:
    """A Chromium binary to launch, or ``None`` to let Playwright resolve its own."""
    exe = os.environ.get("PLAYWRIGHT_CHROMIUM", "/opt/pw-browsers/chromium")
    return exe if Path(exe).exists() else None


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """The runtime is ~200 requests a page; an export log has no use for them."""

    def log_message(self, format: str, *args: Any) -> None:
        pass


def _build_serve_root(bundle: Path, root: Path, *, html: str | None = None) -> None:
    """Assemble a serve root: marimo's ``_static`` assets + the bundle's CDN-rewritten HTML (or *html* in its place)."""
    # Absolute, so the copies below resolve regardless of the bundle's cwd.
    index = (bundle / "index.html" if bundle.is_dir() else bundle).resolve()
    assets = index.parent / "_assets"
    static = Path(marimo.__file__).parent / "_static"

    # marimo runtime lives under assets/ (+ favicon etc.); copy it all in at root, so a
    # rewritten "/assets/index-*.js" resolves here. The report's figures live under _assets/
    # (note the leading underscore) — a different dir, so no collision. Copies, not
    # symlinks: a write into the serve root (like index.html below) must never reach
    # through a link into the marimo package or the bundle — that once corrupted marimo's
    # export template in site-packages (and, via uv's hardlinks, the uv cache), poisoning
    # every later `marimo export`. The runtime is a few tens of MB, copied into a
    # throwaway dir; self-contained beats cheap here.
    shutil.copytree(static, root, dirs_exist_ok=True)
    if assets.is_dir():
        shutil.copytree(assets, root / "_assets")

    html = _CDN.sub("", html if html is not None else index.read_text("utf-8"))  # ".../dist/x.js" -> "/x.js"
    (root / "index.html").write_text(html, "utf-8")


@contextmanager
def served_bundle(bundle: Path, *, html: str | None = None) -> Iterator[str]:
    """Serve *bundle* (a dir with ``index.html`` + ``_assets/``, or the ``index.html``) offline; yields its URL.

    The serve root is a throwaway copy beside the bundle (``.render-<name>/``, gitignored under ``.mini``), removed on exit. *html* replaces the bundle's page for the duration, for a caller that has rewritten it (resolved links, say) without wanting that in the bundle itself.
    """
    root = bundle.parent / (".render-" + (bundle.name or "root"))
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir()
    handler = lambda *a, **k: _QuietHandler(*a, directory=str(root), **k)  # noqa: E731
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    try:
        _build_serve_root(bundle, root, html=html)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{httpd.server_address[1]}/index.html"
    finally:
        httpd.shutdown()
        httpd.server_close()
        shutil.rmtree(root, ignore_errors=True)


def print_page(page: Any, out: Path, *, settle: float = 3.0) -> Path:
    """Print an already-loaded Playwright *page* to *out* and normalize the file.

    Same engine as Chrome's print dialog: the size and margins come from the stylesheet's ``@page`` rule, and there is no header or footer. Light scheme, since the reader is paper or e-ink. Figures the site build deferred have no pixels until scrolled to (the site's ``beforeprint`` handler does the same, but headless printing may not fire it), so they are made eager and given *settle* seconds to arrive; an export bundle carries no such deferral and pays only the wait.
    """
    page.emulate_media(color_scheme="light")
    page.evaluate("document.querySelectorAll('img[loading=lazy]').forEach(i => i.loading = 'eager')")
    page.wait_for_timeout(settle * 1000)
    page.pdf(path=str(out), prefer_css_page_size=True, print_background=True)
    normalize_pdf(out)
    return out


def normalize_pdf(path: Path) -> None:
    """Strip what varies between two prints of one page, so the bytes are a function of the content.

    Chromium stamps ``CreationDate``/``ModDate`` (now) and a document ID (random) into every PDF. A re-export of an unchanged report would then upload a different file and mint a publish-tier commit for nothing, where today an identical bundle mints none. Dates go; the ID is re-derived from the content (qpdf's deterministic ID).
    """
    import pikepdf

    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        for key in ("/CreationDate", "/ModDate"):
            if key in pdf.docinfo:
                del pdf.docinfo[key]
        pdf.save(path, deterministic_id=True)


def print_bundle(
    bundle: Path, out: Path, *, html: str | None = None, timeout: float = 8.0, settle: float = 3.0
) -> Path | None:
    """Print an export bundle to *out*; ``None`` (with a log line) when no browser is available.

    *html*, if given, is printed in place of the bundle's page (see :func:`served_bundle`). Waits up to *timeout* seconds for marimo to hydrate the first cell output, then *settle* seconds for figures and fonts. A missing Playwright or Chromium is reported with the install commands and never raises: a publish must not fail on the PDF.
    """
    try:
        from playwright.sync_api import Error as PlaywrightError, sync_playwright
    except ImportError:
        log.warning("report PDF skipped: playwright is not installed (uv sync --all-groups)")
        return None
    with served_bundle(bundle, html=html) as url, sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path=chromium_path())
        except PlaywrightError as e:
            log.warning(
                "report PDF skipped: no Chromium to print with (%s). Install once: %s",
                str(e).splitlines()[0],
                INSTALL_HINT,
            )
            return None
        try:
            # marimo's frontend validates navigator.language on boot and hard-errors
            # ("Incorrect locale information provided") if the browser reports none —
            # which a bare headless Chromium in a locale-less container does. Pin one.
            page = browser.new_page(viewport={"width": 1100, "height": 1400}, locale="en-US")
            page.goto(url)
            page.locator(".output").first.wait_for(timeout=timeout * 1000)
            return print_page(page, out, settle=settle)
        finally:
            browser.close()
