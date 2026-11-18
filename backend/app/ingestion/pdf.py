"""PDF text extraction. Pages with no extractable text are flagged as likely
scans that need OCR (decided after the PDF audit, see PLAN.md risks)."""
from pathlib import Path

from pypdf import PdfReader


def extract_pages(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    return [{"page": i + 1, "text": page.extract_text() or ""}
            for i, page in enumerate(reader.pages)]
