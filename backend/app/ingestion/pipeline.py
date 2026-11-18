"""PDF -> pages -> clean -> chunk -> quality check -> embed -> Qdrant."""
import json
from collections import Counter
from pathlib import Path

from app.ingestion.chunking import chunk_text
from app.ingestion.pdf import extract_pages
from app.ingestion.quality import BLOCKING, check_chunk
from app.rag.text import detect_language, normalize


def build_chunks(pages_by_doc: dict[str, list[dict]], max_chars=800, overlap=150):
    """Pure function (no I/O) so it is easy to test."""
    chunks, rejected, empty_pages, seen = [], [], [], set()
    for doc, pages in pages_by_doc.items():
        for p in pages:
            text = normalize(p["text"])
            if not text:
                empty_pages.append({"source_doc": doc, "page": p["page"]})
                continue
            for i, piece in enumerate(chunk_text(text, max_chars, overlap)):
                issues = check_chunk(piece, seen)
                rec = {"chunk_id": f"{doc}::p{p['page']}::c{i}", "source_doc": doc,
                       "page": p["page"], "language": detect_language(piece),
                       "text": piece, "issues": issues}
                (rejected if BLOCKING & set(issues) else chunks).append(rec)
    report = {
        "docs": len(pages_by_doc),
        "pages": sum(len(v) for v in pages_by_doc.values()),
        "empty_pages_likely_scanned": empty_pages,
        "chunks_kept": len(chunks),
        "chunks_rejected": len(rejected),
        "languages": dict(Counter(c["language"] for c in chunks)),
        "issue_counts": dict(Counter(i for c in chunks + rejected for i in c["issues"])),
    }
    return chunks, rejected, report


def ingest_dir(pdf_dir: Path, out_dir: Path, embedder, store, recreate=False):
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}")
    chunks, rejected, report = build_chunks({p.name: extract_pages(p) for p in pdfs})
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(out_dir / "rejected_chunks.jsonl", "w", encoding="utf-8") as f:
        for c in rejected:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out_dir / "quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    store.ensure_collection(recreate=recreate)
    if chunks:
        store.upsert(chunks, embedder.embed([c["text"] for c in chunks]))
    return report
