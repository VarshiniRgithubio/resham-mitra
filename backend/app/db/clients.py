"""Database clients plus cheap health checks."""
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import AsyncQdrantClient

from app.core.config import Settings


async def check_mongo(settings: Settings) -> bool:
    client = AsyncIOMotorClient(settings.mongo_uri, serverSelectionTimeoutMS=1500)
    try:
        await client.admin.command("ping")
        return True
    except Exception:
        return False
    finally:
        client.close()


async def check_qdrant(settings: Settings) -> bool:
    if not settings.qdrant_url.startswith("http"):
        return True  # embedded mode (":memory:" or "path:..."): runs in-process
    client = AsyncQdrantClient(url=settings.qdrant_url, timeout=2)
    try:
        await client.get_collections()
        return True
    except Exception:
        return False
    finally:
        await client.close()
