from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.db.clients import check_mongo, check_qdrant

router = APIRouter()


@router.get("/health")
async def health(settings: Settings = Depends(get_settings)):
    mongo_ok = await check_mongo(settings)
    qdrant_ok = await check_qdrant(settings)
    return {
        "status": "ok" if mongo_ok and qdrant_ok else "degraded",
        "mongo": mongo_ok,
        "qdrant": qdrant_ok,
        "llm_provider": settings.llm_provider,
    }
