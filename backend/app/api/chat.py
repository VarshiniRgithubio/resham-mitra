"""Chat, history and feedback endpoints.

Guests: answers only, best-effort anonymous logging.
Signed-in users: every exchange is saved into a conversation (ChatGPT-style
history). A DB outage never blocks an answer.
"""
import logging
import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.auth import current_user, require_user
from app.core.config import Settings, get_settings
from app.core.deps import get_db, get_pipeline

router = APIRouter()
log = logging.getLogger(__name__)
TEL = re.compile(r"^\+?[\d\s-]{6,15}$")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    conversation_id: str | None = None
    session_id: str | None = None


class FeedbackRequest(BaseModel):
    message_id: str
    helpful: bool
    comment: str | None = Field(default=None, max_length=1000)


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


def now():
    return datetime.now(timezone.utc)


def officer_phone(s: Settings) -> str | None:
    """Only expose a dialable number; the placeholder stays hidden in the UI."""
    return s.officer_contact if TEL.match(s.officer_contact or "") else None


def message_view(m: dict) -> dict:
    return {"message_id": m["_id"], "question": m["question"], "answer": m["answer"],
            "abstained": m["abstained"], "greeting": m.get("greeting", False),
            "sources": m.get("sources", []), "helpful": m.get("helpful"),
            "created_at": m["created_at"]}


@router.post("/chat")
async def chat(req: ChatRequest, pipeline=Depends(get_pipeline), db=Depends(get_db),
               user=Depends(current_user), s: Settings = Depends(get_settings)):
    conv_id = req.conversation_id
    if user and conv_id:
        conv = await db.conversations.find_one({"_id": conv_id, "user_id": user["_id"]})
        if not conv:
            raise HTTPException(404, "conversation_not_found")

    result = await pipeline.answer(req.question)
    message_id = str(uuid.uuid4())
    try:
        if user and not conv_id:
            conv_id = str(uuid.uuid4())
            title = re.sub(r"\s+", " ", req.question).strip()[:40]
            await db.conversations.insert_one({"_id": conv_id, "user_id": user["_id"],
                                               "title": title, "created_at": now(),
                                               "updated_at": now()})
        elif user:
            await db.conversations.update_one({"_id": conv_id}, {"$set": {"updated_at": now()}})
        await db.messages.insert_one({**result, "_id": message_id,
                                      "user_id": user["_id"] if user else None,
                                      "conversation_id": conv_id if user else None,
                                      "session_id": req.session_id, "created_at": now()})
    except Exception as e:  # noqa: BLE001
        log.warning("Mongo write failed: %s", e)
        if not user:
            conv_id = None

    return {"message_id": message_id, "conversation_id": conv_id if user else None,
            "answer": result["answer"], "abstained": result["abstained"],
            "greeting": result.get("greeting", False), "sources": result["sources"],
            "confidence": result["confidence"], "officer_phone": officer_phone(s)}


@router.get("/conversations")
async def list_conversations(user=Depends(require_user), db=Depends(get_db)):
    cur = db.conversations.find({"user_id": user["_id"]}).sort("updated_at", -1).limit(100)
    return [{"id": c["_id"], "title": c["title"], "updated_at": c["updated_at"]}
            async for c in cur]


@router.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str, user=Depends(require_user), db=Depends(get_db),
                           s: Settings = Depends(get_settings)):
    conv = await db.conversations.find_one({"_id": conv_id, "user_id": user["_id"]})
    if not conv:
        raise HTTPException(404, "conversation_not_found")
    msgs = db.messages.find({"conversation_id": conv_id}).sort("created_at", 1)
    return {"id": conv_id, "title": conv["title"], "officer_phone": officer_phone(s),
            "messages": [message_view(m) async for m in msgs]}


@router.patch("/conversations/{conv_id}")
async def rename_conversation(conv_id: str, req: RenameRequest, user=Depends(require_user),
                              db=Depends(get_db)):
    r = await db.conversations.update_one({"_id": conv_id, "user_id": user["_id"]},
                                          {"$set": {"title": req.title.strip()}})
    if not r.matched_count:
        raise HTTPException(404, "conversation_not_found")
    return {"ok": True}


@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str, user=Depends(require_user), db=Depends(get_db)):
    r = await db.conversations.delete_one({"_id": conv_id, "user_id": user["_id"]})
    if not r.deleted_count:
        raise HTTPException(404, "conversation_not_found")
    await db.messages.delete_many({"conversation_id": conv_id, "user_id": user["_id"]})
    return {"ok": True}


@router.post("/feedback")
async def feedback(req: FeedbackRequest, db=Depends(get_db)):
    try:
        await db.feedback.insert_one({**req.model_dump(), "created_at": now()})
        await db.messages.update_one({"_id": req.message_id}, {"$set": {"helpful": req.helpful}})
    except Exception as e:  # noqa: BLE001
        log.warning("Mongo write failed: %s", e)
    return {"ok": True}


@router.get("/stats")
async def stats(pipeline=Depends(get_pipeline)):
    try:
        return {"indexed_chunks": pipeline.store.count()}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"vector store unavailable: {e}") from e
