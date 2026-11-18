"""PIN hashing and session tokens.

PINs are short, so they are hashed with scrypt (slow on purpose, salted) and
login is rate-limited (see api/auth.py). Plain PINs are never stored or logged.
"""
import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone

import jwt

_N, _R, _P = 2**14, 8, 1


def hash_pin(pin: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(pin.encode(), salt=salt, n=_N, r=_R, p=_P)
    return base64.b64encode(salt + digest).decode()


def verify_pin(pin: str, stored: str) -> bool:
    raw = base64.b64decode(stored)
    salt, digest = raw[:16], raw[16:]
    return hmac.compare_digest(hashlib.scrypt(pin.encode(), salt=salt, n=_N, r=_R, p=_P), digest)


def make_token(user_id: str, secret: str, days: int) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=days)
    return jwt.encode({"sub": user_id, "exp": exp}, secret, algorithm="HS256")


def read_token(token: str, secret: str) -> str | None:
    try:
        return jwt.decode(token, secret, algorithms=["HS256"])["sub"]
    except jwt.PyJWTError:
        return None
