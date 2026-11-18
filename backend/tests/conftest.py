import sys
from pathlib import Path

import pytest

from app.core.config import Settings
from app.llm.base import LLMClient
from app.rag.embeddings import HashEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.store import VectorStore, make_client

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "eval/experiments"))

# Synthetic test passages. NOT real sericulture advice.
DOCS = [
    {"chunk_id": "t::p1::c0", "source_doc": "test.pdf", "page": 1, "language": "en",
     "text": "Alpha procedure: clean the tray every morning with plain water."},
    {"chunk_id": "t::p2::c0", "source_doc": "test.pdf", "page": 2, "language": "kn",
     "text": "ಬೀಟಾ ವಿಧಾನ: ಪ್ರತಿದಿನ ಬೆಳಿಗ್ಗೆ ತಟ್ಟೆಯನ್ನು ನೀರಿನಿಂದ ತೊಳೆಯಿರಿ."},
]


class StubLLM(LLMClient):
    """Scripted LLM: lets tests drive every pipeline branch deterministically."""
    name = "stub"

    def __init__(self, answer="Clean the tray every morning [1].", grounded=True, varied=False):
        self.answer, self.grounded, self.varied, self.calls = answer, grounded, varied, 0

    async def generate(self, prompt, system=None, temperature=None):
        self.calls += 1
        if prompt.startswith("You are checking an answer"):
            return '{"claims": [{"claim": "x", "supported": %s}]}' % str(self.grounded).lower()
        if prompt.startswith("Translate"):
            return "TRANSLATED"
        if self.varied and temperature == 0.8:
            return f"completely different reply number {self.calls} zzz"
        return self.answer


@pytest.fixture
def make_pipeline():
    def _make(llm=None, **overrides):
        s = Settings(_env_file=None, **overrides)
        emb = HashEmbedder()
        store = VectorStore(make_client(":memory:"), "test", emb.dim)
        store.ensure_collection()
        store.upsert(DOCS, emb.embed([d["text"] for d in DOCS]))
        return RagPipeline(llm or StubLLM(), emb, store, s)
    return _make
