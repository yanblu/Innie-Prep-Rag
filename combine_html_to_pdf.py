#!/usr/bin/env python3
"""Merge every .html file in a folder into one PDF, ordered by filename."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_NUM_PREFIX = re.compile(r"^(\d+)\.")


def _collect_html_files(folder: Path) -> list[Path]:
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".html"
    ]

    def sort_key(p: Path) -> tuple:
        m = _NUM_PREFIX.match(p.name)
        if m:
            return (0, int(m.group(1)), p.name.lower())
        return (1, p.name.lower())

    return sorted(files, key=sort_key)


def _find_chrome_binary(explicit: str | None) -> str | None:
    if explicit:
        p = Path(expanded := os.path.expanduser(explicit))
        return str(p) if p.is_file() and os.access(p, os.X_OK) else None
    for candidate in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ):
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    for name in (
        "google-chrome-stable",
        "google-chrome",
        "chromium",
        "chromium-browser",
        "microsoft-edge",
    ):
        found = shutil.which(name)
        if found:
            return found
    return None


def _html_files_to_pdfs_playwright(html_files: list[Path], tmp_dir: Path) -> list[Path]:
    from playwright.sync_api import sync_playwright

    part_paths: list[Path] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                java_script_enabled=False,
                ignore_https_errors=True,
            )
            for i, html_path in enumerate(html_files):
                part = tmp_dir / f"part_{i:04d}.pdf"
                page = context.new_page()
                try:
                    page.goto(
                        html_path.resolve().as_uri(),
                        wait_until="domcontentloaded",
                        timeout=120_000,
                    )
                    page.pdf(
                        path=str(part),
                        print_background=True,
                        display_header_footer=False,
                    )
                finally:
                    page.close()
                part_paths.append(part)
            context.close()
        finally:
            browser.close()
    return part_paths


def _html_to_pdf_chrome(chrome: str, html_path: Path, pdf_path: Path) -> None:
    uri = html_path.resolve().as_uri()
    # Fresh profile per run avoids the default profile singleton lock when
    # several headless Chrome processes run back-to-back.
    with tempfile.TemporaryDirectory(prefix="chrome_ud_") as user_data:
        cmd = [
            chrome,
            f"--user-data-dir={user_data}",
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path.resolve()}",
            uri,
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
    if proc.returncode != 0 or not pdf_path.is_file():
        raise RuntimeError(
            f"Chrome failed for {html_path.name} (exit {proc.returncode}); "
            "check that the browser runs headless on this machine."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine HTML files in a directory into a single PDF. "
        "Prefers Playwright/Chromium (no JS, fast for saved pages); use --chrome for system Chrome.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Directory containing .html files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <folder>/combined.pdf)",
    )
    parser.add_argument(
        "--chrome-binary",
        default=None,
        metavar="PATH",
        help="Use this Chrome/Chromium/Edge executable instead of Playwright (implies --chrome)",
    )
    parser.add_argument(
        "--chrome",
        action="store_true",
        help="Render with system Chrome instead of Playwright (slower; can hang on heavy saved pages)",
    )
    args = parser.parse_args()
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 1

    html_files = _collect_html_files(folder)
    if not html_files:
        print(f"No .html files found in {folder}", file=sys.stderr)
        return 1

    use_chrome = args.chrome or args.chrome_binary is not None
    chrome: str | None = None
    if use_chrome:
        chrome = _find_chrome_binary(args.chrome_binary)
        if not chrome:
            print(
                "Could not find Chrome, Chromium, or Edge. Install one or pass --chrome-binary.",
                file=sys.stderr,
            )
            return 1
    else:
        try:
            import playwright.sync_api  # noqa: F401
        except ImportError:
            print(
                "Install Playwright: pip install playwright && playwright install chromium",
                file=sys.stderr,
            )
            return 1

    from pypdf import PdfWriter

    out_path = (
        args.output.expanduser().resolve()
        if args.output
        else folder / "combined.pdf"
    )

    with tempfile.TemporaryDirectory(prefix="html2pdf_") as tmp:
        tmp_dir = Path(tmp)
        part_paths: list[Path] = []
        if use_chrome:
            assert chrome is not None
            for i, html_path in enumerate(html_files):
                part = tmp_dir / f"part_{i:04d}.pdf"
                _html_to_pdf_chrome(chrome, html_path, part)
                part_paths.append(part)
        else:
            part_paths = _html_files_to_pdfs_playwright(html_files, tmp_dir)

        writer = PdfWriter()
        for part in part_paths:
            writer.append(str(part))
        writer.write(str(out_path))

    print(f"Wrote {out_path} ({len(html_files)} HTML file(s))")
    for p in html_files:
        print(f"  - {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
