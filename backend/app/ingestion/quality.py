"""Data-quality checks. Each chunk gets a list of issues; the pipeline drops
chunks with blocking issues and writes a report so problems are visible."""
import hashlib
import re

from app.rag.text import LETTER

BLOCKING = {"too_short", "low_text_ratio", "duplicate"}


def check_chunk(text: str, seen: set[str]) -> list[str]:
    issues = []
    if len(text) < 50:
        issues.append("too_short")
    if text and len(LETTER.findall(text)) / len(text) < 0.5:
        issues.append("low_text_ratio")
    if "�" in text:
        issues.append("replacement_chars")
    # Legacy Kannada fonts (e.g. Nudi/Baraha ASCII encodings) extract as Latin
    # gibberish; many rare Latin-1 symbols is a hint.
    if len(re.findall(r"[À-ÿ]", text)) > 0.1 * max(len(text), 1):
        issues.append("possible_legacy_font_encoding")
    digest = hashlib.sha1(text.encode()).hexdigest()
    if digest in seen:
        issues.append("duplicate")
    seen.add(digest)
    return issues
