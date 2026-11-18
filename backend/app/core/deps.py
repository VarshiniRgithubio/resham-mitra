"""Builds shared singletons once per process."""
from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import get_settings
from app.llm.factory import build_llm
from app.rag.embeddings import build_embedder
from app.rag.pipeline import RagPipeline
from app.rag.store import VectorStore, make_client


@lru_cache
def get_pipeline() -> RagPipeline:
    s = get_settings()
    emb = build_embedder(s)
    store = VectorStore(make_client(s.qdrant_url), s.qdrant_collection, emb.dim)
    store.ensure_collection()
    return RagPipeline(build_llm(s), emb, store, s)


@lru_cache
def get_db():
    s = get_settings()
    return AsyncIOMotorClient(s.mongo_uri, serverSelectionTimeoutMS=1500)[s.mongo_db]
