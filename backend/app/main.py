"""Resham Mitra API entry point. Run from backend/: uvicorn app.main:app --reload"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.core.config import get_settings
from app.core.deps import get_db

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app):
    if get_settings().jwt_secret.startswith("dev-only"):
        log.warning("JWT_SECRET is the dev default. Set a real secret before deploying.")
    try:  # indexes are best-effort so the API still starts without MongoDB
        db = get_db()
        await db.users.create_index("phone", unique=True)
        await db.conversations.create_index([("user_id", 1), ("updated_at", -1)])
        await db.messages.create_index([("conversation_id", 1), ("created_at", 1)])
    except Exception as e:  # noqa: BLE001
        log.warning("Could not create Mongo indexes: %s", e)
    yield


app = FastAPI(title="Resham Mitra API", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"],
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chat_router)
