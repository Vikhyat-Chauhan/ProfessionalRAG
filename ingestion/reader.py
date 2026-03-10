"""Document ingestion — reads PDFs and extracts page-level text."""

import hashlib
import logging
from pathlib import Path

from pypdf import PdfReader

log = logging.getLogger(__name__)


def read_pdf(file_path: str) -> list[dict]:
    """Extract text from each page of a PDF, returning page dicts."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "page": i + 1})
        else:
            log.warning("Page %d is empty or unreadable", i + 1)
    return pages


def file_fingerprint(file_path: str) -> str:
    """SHA-256 hash of file contents — used to detect changes for re-ingestion."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()
