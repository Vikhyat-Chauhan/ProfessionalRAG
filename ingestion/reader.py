"""Document ingestion — reads PDFs and GitHub repos into page/file dicts."""

import hashlib
import logging
import re
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader

log = logging.getLogger(__name__)

# File extensions to include when reading a repo
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt", ".scala",
    ".sh", ".bash", ".zsh", ".yaml", ".yml", ".toml", ".json",
    ".html", ".css", ".scss", ".sql", ".r", ".R", ".md", ".txt",
    ".cfg", ".ini", ".env.example", ".dockerfile", ".tf",
}

# Directories to always skip
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".eggs", ".tox", ".mypy_cache",
}

GITHUB_URL_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)


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


def is_github_url(source: str) -> bool:
    """Check if a source string is a GitHub repo URL."""
    return bool(GITHUB_URL_RE.match(source))


def read_repo(source: str) -> list[dict]:
    """Read source files from a GitHub URL or local directory path.

    Returns a list of dicts with keys: text, page (file path), source.
    """
    if is_github_url(source):
        repo_dir = _clone_repo(source)
        cleanup = True
    else:
        repo_dir = Path(source)
        if not repo_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {source}")
        cleanup = False

    try:
        return _read_directory(repo_dir, source)
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(repo_dir, ignore_errors=True)


def _clone_repo(url: str) -> Path:
    """Shallow-clone a GitHub repo into a temp directory."""
    tmp = Path(tempfile.mkdtemp(prefix="rag_repo_"))
    log.info("Cloning %s → %s", url, tmp)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(tmp)],
        check=True,
        capture_output=True,
        text=True,
    )
    return tmp


def _read_directory(root: Path, source: str) -> list[dict]:
    """Walk a directory and read code files into page dicts."""
    docs = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # Skip hidden/unwanted dirs
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        if path.suffix not in CODE_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            log.warning("Could not read %s", path)
            continue
        if not text:
            continue

        rel_path = str(path.relative_to(root))
        docs.append({
            "text": f"# File: {rel_path}\n\n{text}",
            "page": rel_path,
            "source": source,
        })
    log.info("Read %d files from %s", len(docs), source)
    return docs


def file_fingerprint(file_path: str) -> str:
    """SHA-256 hash of file contents — used to detect changes for re-ingestion."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def dir_fingerprint(source: str) -> str:
    """SHA-256 hash based on the source identifier (URL or path)."""
    return hashlib.sha256(source.encode()).hexdigest()
