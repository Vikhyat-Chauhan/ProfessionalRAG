"""Recursive character text splitter — zero external dependencies."""

from dataclasses import dataclass, field


@dataclass
class ChunkConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )


class ChunkHelper:
    """Splits text recursively using a priority list of separators."""

    @staticmethod
    def split(text: str, cfg: ChunkConfig | None = None) -> list[str]:
        cfg = cfg or ChunkConfig()
        return ChunkHelper._recursive_split(text, cfg, cfg.separators)

    @staticmethod
    def split_documents(
        pages: list[dict], cfg: ChunkConfig | None = None
    ) -> tuple[list[str], list[dict]]:
        """Split page dicts into chunks + aligned metadata lists."""
        cfg = cfg or ChunkConfig()
        all_chunks: list[str] = []
        all_metas: list[dict] = []

        for page in pages:
            chunks = ChunkHelper.split(page["text"], cfg)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metas.append({
                    "page": page.get("page", 0),
                    "source": page.get("source", ""),
                })
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
