"""Phone number + PIN sign-in.

Why not email/password: many farmers do not use email, and a short numeric PIN
is easy on a phone keypad. Why not SMS OTP yet: it needs a paid SMS gateway;
the endpoints are shaped so OTP can replace the PIN later.
Signing in is optional; guests can chat but get no saved history.
"""
import re
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings, get_settings
from app.core.deps import get_db
from app.core.security import hash_pin, make_token, read_token, verify_pin

router = APIRouter(prefix="/auth")
bearer = HTTPBearer(auto_error=False)

PHONE = re.compile(r"^[6-9]\d{9}$")  # Indian mobile numbers
PIN = re.compile(r"^\d{4,6}$")


def clean_phone(v: str) -> str:
    v = re.sub(r"[\s-]", "", v)
    v = v[3:] if v.startswith("+91") else v
    if not PHONE.match(v):
        raise ValueError("invalid_phone")
    return v


class SignupRequest(BaseModel):
    phone: str
    pin: str
    name: str | None = Field(default=None, max_length=60)

    _phone = field_validator("phone")(lambda cls, v: clean_phone(v))

    @field_validator("pin")
    @classmethod
    def _pin(cls, v):
        if not PIN.match(v):
            raise ValueError("invalid_pin")
        return v


class LoginRequest(BaseModel):
    phone: str
    pin: str

    _phone = field_validator("phone")(lambda cls, v: clean_phone(v))


def public_user(u: dict) -> dict:
    return {"id": u["_id"], "phone": u["phone"], "name": u.get("name")}


async def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer),
                       db=Depends(get_db), s: Settings = Depends(get_settings)) -> dict | None:
    """Returns the user, or None for guests. Bad/expired tokens are an error,
    so the app can send the person back to the login screen."""
    if creds is None:
        return None
    uid = read_token(creds.credentials, s.jwt_secret)
    user = await db.users.find_one({"_id": uid}) if uid else None
    if not user:
        raise HTTPException(401, "invalid_token")
    return user


async def require_user(user=Depends(current_user)) -> dict:
    if user is None:
        raise HTTPException(401, "login_required")
    return user


@router.post("/signup")
async def signup(req: SignupRequest, db=Depends(get_db), s: Settings = Depends(get_settings)):
    if await db.users.find_one({"phone": req.phone}):
        raise HTTPException(409, "phone_exists")
    user = {"_id": str(uuid.uuid4()), "phone": req.phone, "name": req.name,
            "pin_hash": hash_pin(req.pin), "failed_attempts": 0, "locked_until": None,
            "created_at": datetime.now(timezone.utc)}
    await db.users.insert_one(user)
    return {"token": make_token(user["_id"], s.jwt_secret, s.token_days), "user": public_user(user)}


@router.post("/login")
async def login(req: LoginRequest, db=Depends(get_db), s: Settings = Depends(get_settings)):
    user = await db.users.find_one({"phone": req.phone})
    now = datetime.now(timezone.utc)
    if user and user.get("locked_until"):
        locked = user["locked_until"].replace(tzinfo=timezone.utc)
        if locked > now:
            raise HTTPException(429, "locked")
    if not user or not verify_pin(req.pin, user["pin_hash"]):
        if user:  # count failures; lock after too many (PINs are guessable)
            fails = user.get("failed_attempts", 0) + 1
            upd = {"failed_attempts": fails}
            if fails >= s.max_login_attempts:
                upd = {"failed_attempts": 0,
                       "locked_until": now + timedelta(minutes=s.lockout_minutes)}
            await db.users.update_one({"_id": user["_id"]}, {"$set": upd})
        raise HTTPException(401, "wrong_credentials")  # same error: don't reveal which part
    await db.users.update_one({"_id": user["_id"]},
                              {"$set": {"failed_attempts": 0, "locked_until": None}})
    return {"token": make_token(user["_id"], s.jwt_secret, s.token_days), "user": public_user(user)}


@router.get("/me")
async def me(user=Depends(require_user)):
    return public_user(user)
