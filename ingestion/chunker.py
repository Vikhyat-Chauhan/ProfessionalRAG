"""Semantic-aware text splitter with metadata enrichment."""

import re
from dataclasses import dataclass, field


@dataclass
class ChunkConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 200
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )


# ── Section-header detection ──────────────────────────────────────────

# Markdown headers, ALL-CAPS lines (≥3 chars), or lines ending with ":"
_HEADER_RE = re.compile(
    r"^(?:"
    r"#{1,6}\s+.+"           # Markdown: ## Education
    r"|[A-Z][A-Z &/,()-]{2,}"  # ALL CAPS: WORK EXPERIENCE
    r"|.{3,80}:\s*$"         # Trailing colon: Experience:
    r")$",
    re.MULTILINE,
)

# Date patterns for metadata enrichment
_DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}[-/]\d{2}(?:[-/]\d{2})?"             # 2023-01, 2023/01/15
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"  # Jan 2023
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"  # Jan 15, 2023
    r"|\d{1,2}/\d{1,2}/\d{2,4}"                 # 01/15/2023
    r")\b",
    re.IGNORECASE,
)

# Document-type heuristics (checked against full text)
_DOC_TYPE_SIGNALS = [
    ("resume",      re.compile(r"\b(?:experience|education|skills|objective|summary)\b", re.I)),
    ("invoice",     re.compile(r"\b(?:invoice|bill\s+to|amount\s+due|subtotal)\b", re.I)),
    ("contract",    re.compile(r"\b(?:agreement|whereas|party|herein|obligations)\b", re.I)),
    ("research",    re.compile(r"\b(?:abstract|methodology|findings|references|hypothesis)\b", re.I)),
    ("meeting_notes", re.compile(r"\b(?:attendees|agenda|action\s+items|minutes)\b", re.I)),
]


def _detect_doc_type(text: str) -> str:
    """Return a best-guess document type label, or 'general'."""
    for label, pattern in _DOC_TYPE_SIGNALS:
        if len(pattern.findall(text)) >= 2:
            return label
    return "general"


def _extract_dates(text: str) -> list[str]:
    """Pull date strings from text (deduplicated, max 10)."""
    matches = _DATE_RE.findall(text)
    seen: set[str] = set()
    unique: list[str] = []
    for m in matches:
        normed = m.strip().rstrip(".,")
        if normed not in seen:
            seen.add(normed)
            unique.append(normed)
        if len(unique) >= 10:
            break
    return unique


def _extract_section_title(text: str) -> str:
    """Return the first header-like line from a chunk, or empty string."""
    for line in text.splitlines()[:5]:
        line = line.strip()
        if _HEADER_RE.match(line):
            # Clean markdown hashes
            return re.sub(r"^#+\s*", "", line).strip().rstrip(":")
    return ""


# ── Semantic section splitting ────────────────────────────────────────


_MIN_SECTION_LEN = 50  # merge tiny sections into the next one


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split text on section headers, returning (title, body) pairs.

    If no headers are found, returns the whole text as one untitled section.
    Tiny sections (header-only) are merged into the following section.
    """
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [("", text)]

    raw: list[tuple[str, str]] = []

    # Text before the first header
    pre = text[: matches[0].start()].strip()
    if pre:
        raw.append(("", pre))

    for i, m in enumerate(matches):
        title = re.sub(r"^#+\s*", "", m.group()).strip().rstrip(":")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # Include the header line in the body so it survives chunking
        full = m.group().strip() + "\n" + body if body else m.group().strip()
        raw.append((title, full))

    # Merge tiny sections (e.g. bare "## Experience" before sub-headers)
    merged: list[tuple[str, str]] = []
    carry_title = ""
    carry_body = ""
    for title, body in raw:
        if carry_body:
            body = carry_body + "\n" + body
            title = carry_title or title
            carry_title = ""
            carry_body = ""
        if len(body) < _MIN_SECTION_LEN:
            carry_title = title
            carry_body = body
        else:
            merged.append((title, body))
    if carry_body:
        if merged:
            prev_title, prev_body = merged[-1]
            merged[-1] = (prev_title, prev_body + "\n" + carry_body)
        else:
            merged.append((carry_title, carry_body))

    return merged


# ── Public API ────────────────────────────────────────────────────────


class ChunkHelper:
    """Splits text semantically on section boundaries, then recursively
    by character when sections exceed chunk_size."""

    @staticmethod
    def split(text: str, cfg: ChunkConfig | None = None) -> list[str]:
        cfg = cfg or ChunkConfig()
        sections = _split_into_sections(text)
        chunks: list[str] = []
        for _title, body in sections:
            if len(body) <= cfg.chunk_size:
                if body.strip():
                    chunks.append(body.strip())
            else:
                chunks.extend(ChunkHelper._recursive_split(body, cfg, cfg.separators))
        return chunks

    @staticmethod
    def split_documents(
        pages: list[dict], cfg: ChunkConfig | None = None
    ) -> tuple[list[str], list[dict]]:
        """Split page dicts into chunks + aligned metadata lists.

        Enriches metadata with: section, dates, doc_type.
        """
        cfg = cfg or ChunkConfig()
        all_chunks: list[str] = []
        all_metas: list[dict] = []

        # Detect doc type from combined text of first few pages
        combined = " ".join(p.get("text", "")[:1000] for p in pages[:5])
        doc_type = _detect_doc_type(combined)

        for page in pages:
            text = page["text"]
            sections = _split_into_sections(text)

            for sec_title, body in sections:
                if len(body) <= cfg.chunk_size:
                    sub_chunks = [body.strip()] if body.strip() else []
                else:
                    sub_chunks = ChunkHelper._recursive_split(body, cfg, cfg.separators)

                for chunk in sub_chunks:
                    section = sec_title or _extract_section_title(chunk)
                    dates = _extract_dates(chunk)

                    meta = {
                        "page": page.get("page", 0),
                        "source": page.get("source", ""),
                        "doc_type": doc_type,
                    }
                    if section:
                        meta["section"] = section
                    if dates:
                        meta["dates"] = ", ".join(dates)

                    all_chunks.append(chunk)
                    all_metas.append(meta)

        return all_chunks, all_metas

    @staticmethod
    def _recursive_split(
        text: str, cfg: ChunkConfig, separators: list[str]
    ) -> list[str]:
        if len(text) <= cfg.chunk_size:
            return [text] if text.strip() else []

        # Pick the first separator that exists in the text
        sep = separators[-1]
        for s in separators:
            if s == "" or s in text:
                sep = s
                break

        parts = text.split(sep) if sep else list(text)
        remaining_seps = separators[separators.index(sep) + 1:] if sep in separators else []

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()

            if len(candidate) <= cfg.chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())

                if len(part) > cfg.chunk_size and remaining_seps:
                    sub = ChunkHelper._recursive_split(part, cfg, remaining_seps)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""
                else:
                    # Overlap: carry tail of previous chunk
                    if cfg.chunk_overlap and chunks:
                        overlap = chunks[-1][-cfg.chunk_overlap:]
                        current = (overlap + sep + part).strip()
                    else:
                        current = part.strip()

        if current.strip():
            chunks.append(current.strip())

        return chunks
