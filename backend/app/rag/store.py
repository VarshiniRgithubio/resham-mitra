"""Qdrant wrapper. Chunk payload keeps source_doc, page and language so we can
cite sources and filter by language (needed for experiment E2)."""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, MatchValue,
                                  PointStruct, VectorParams)


def make_client(url: str) -> QdrantClient:
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if url.startswith("path:"):
        return QdrantClient(path=url[5:])
    return QdrantClient(url=url)


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int):
        self.client, self.collection, self.dim = client, collection, dim

    def ensure_collection(self, recreate: bool = False):
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                self.collection, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE))

    def upsert(self, chunks: list[dict], vectors: list[list[float]]):
        points = [PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["chunk_id"])),
                              vector=v, payload=c) for c, v in zip(chunks, vectors)]
        for i in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[i:i + 256])

    def search(self, vector: list[float], k: int, language: str | None = None) -> list[dict]:
        flt = (Filter(must=[FieldCondition(key="language", match=MatchValue(value=language))])
               if language else None)
        hits = self.client.query_points(self.collection, query=vector, limit=k,
                                        query_filter=flt, with_payload=True).points
        return [{**h.payload, "score": float(h.score)} for h in hits]

    def count(self) -> int:
        return self.client.count(self.collection).count
