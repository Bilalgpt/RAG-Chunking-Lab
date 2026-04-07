"""
Recursive character splitting respects natural text boundaries by trying
separators in priority order before falling back to finer ones.

Algorithm:
  recursive_split(text, separators, max_size):
      sep = separators[0]   # try coarsest separator first
      splits = text.split(sep)
      for each split:
          if word_count(split) <= max_size:
              keep as leaf chunk
          else:
              recursive_split(split, separators[1:], max_size)

Separator priority: ["\n\n", "\n", ". ", " "]

This preserves paragraph → sentence → word structure and only uses finer
splits when a segment genuinely exceeds max_size. After splitting, the
last `overlap` words of each predecessor chunk are prepended to the next
chunk so retrieval can bridge boundaries. The unmodified text is stored
in metadata["original_text"].
"""

from .base import BaseChunker, Chunk

_SEPARATORS = ["\n\n", "\n", ". ", " "]


class RecursiveChunker(BaseChunker):

    @property
    def name(self) -> str:
        return "recursive"

    @property
    def description(self) -> str:
        return "Split by a priority list of separators, using finer ones only when a segment exceeds max size."

    @property
    def default_params(self) -> dict:
        return {"max_size": 300, "overlap": 30}

    # ── public ────────────────────────────────────────────────────────────────

    def chunk(self, text: str, **kwargs) -> list[Chunk]:
        params = {**self.default_params, **kwargs}
        max_size: int = int(params["max_size"])
        overlap: int = int(params["overlap"])

        raw = self._recursive_split(text.strip(), _SEPARATORS, max_size, depth=0)
        if not raw:
            return []

        chunks: list[Chunk] = []
        for i, item in enumerate(raw):
            original = item["text"]
            if i > 0 and overlap > 0:
                prev_words = raw[i - 1]["text"].split()
                prefix = " ".join(prev_words[-overlap:])
                final_text = prefix + " " + original
            else:
                final_text = original

            chunks.append(Chunk(
                text=final_text,
                index=i,
                metadata={
                    "separator_used": item["separator_used"],
                    "depth": item["depth"],
                    "original_text": original,
                },
            ))

        return chunks

    # ── private ───────────────────────────────────────────────────────────────

    def _recursive_split(
        self,
        text: str,
        separators: list[str],
        max_size: int,
        depth: int,
    ) -> list[dict]:
        """Returns list of {text, separator_used, depth} dicts."""
        if not text.strip():
            return []

        sep = separators[0]
        parts = text.split(sep)

        results: list[dict] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part.split()) <= max_size:
                results.append({"text": part, "separator_used": sep, "depth": depth})
            elif len(separators) > 1:
                sub = self._recursive_split(part, separators[1:], max_size, depth + 1)
                results.extend(sub)
            else:
                # Last-resort: keep oversized segment as-is
                results.append({"text": part, "separator_used": sep, "depth": depth})

        return results
