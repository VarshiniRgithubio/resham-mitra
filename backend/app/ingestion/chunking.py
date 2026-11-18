"""Sentence-aware chunking with overlap.

Size is in characters, not tokens, because Kannada tokenises into many more
tokens per word than English; characters are a fairer unit across both.
"""
import re

SENT_END = re.compile(r"(?<=[.!?।])\s+")
BULLET = re.compile(r"\s*(?=•)")


def split_sentences(text: str) -> list[str]:
    parts = []
    for para in text.split("\n\n"):
        for sent in SENT_END.split(para.replace("\n", " ")):
            # each bullet item counts as its own sentence; keep the bullet mark
            parts.extend(b.strip() for b in BULLET.split(sent) if b.strip(" •"))
    return parts


def chunk_text(text: str, max_chars: int = 800, overlap_chars: int = 150) -> list[str]:
    chunks, current = [], ""
    for sent in split_sentences(text):
        if current and len(current) + len(sent) + 1 > max_chars:
            chunks.append(current)
            tail = current[-overlap_chars:]
            current = tail[tail.find(" ") + 1:] if " " in tail else ""
        if len(sent) > max_chars:  # very long sentence: hard split
            for i in range(0, len(sent), max_chars):
                piece = sent[i:i + max_chars]
                if current:
                    chunks.append(current)
                current = piece
            continue
        current = f"{current} {sent}".strip()
    if current:
        chunks.append(current)
    return chunks
