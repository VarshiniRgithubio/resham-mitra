from app.ingestion.chunking import chunk_text
from app.ingestion.pipeline import build_chunks
from app.ingestion.quality import check_chunk
from app.rag.text import detect_language


def test_chunks_respect_size_and_keep_text():
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = chunk_text(text, max_chars=200, overlap_chars=40)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    assert "Sentence number 99" in chunks[-1]


def test_language_detection():
    assert detect_language("ಹಿಪ್ಪುನೇರಳೆ ಎಲೆ") == "kn"
    assert detect_language("mulberry leaf") == "en"


def test_quality_flags_duplicates_and_short():
    seen = set()
    good = "A reasonably long sentence about trays and cleaning that passes checks."
    assert check_chunk(good, seen) == []
    assert "duplicate" in check_chunk(good, seen)
    assert "too_short" in check_chunk("tiny", seen)


def test_build_chunks_reports_scanned_pages():
    pages = {"a.pdf": [{"page": 1, "text": ""},
                       {"page": 2, "text": "Some real extracted text that is long enough to keep."}]}
    chunks, rejected, report = build_chunks(pages)
    assert report["empty_pages_likely_scanned"] == [{"source_doc": "a.pdf", "page": 1}]
    assert len(chunks) == 1 and chunks[0]["page"] == 2
