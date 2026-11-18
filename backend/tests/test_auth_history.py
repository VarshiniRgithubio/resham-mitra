"""Integration tests against a real local MongoDB (test database, dropped
after). Skipped automatically if MongoDB is not running."""
import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

from app.core.deps import get_db, get_pipeline
from app.core.security import hash_pin, verify_pin
from app.main import app
from tests.conftest import StubLLM

DB_NAME = f"resham_mitra_test_{uuid.uuid4().hex[:8]}"


def mongo_up():
    try:
        MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=500).admin.command("ping")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not mongo_up(), reason="MongoDB not running")


@pytest.fixture
def client(make_pipeline):
    app.dependency_overrides[get_pipeline] = lambda: make_pipeline(StubLLM())
    app.dependency_overrides[get_db] = lambda: AsyncIOMotorClient()[DB_NAME]
    yield TestClient(app)
    app.dependency_overrides.clear()
    MongoClient().drop_database(DB_NAME)


def signup(client, phone="9876543210", pin="1234"):
    r = client.post("/auth/signup", json={"phone": phone, "pin": pin, "name": "Test"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_pin_hash_roundtrip():
    h = hash_pin("1234")
    assert verify_pin("1234", h) and not verify_pin("4321", h) and "1234" not in h


def test_signup_validation(client):
    assert client.post("/auth/signup", json={"phone": "12345", "pin": "1234"}).status_code == 422
    assert client.post("/auth/signup", json={"phone": "9876543210", "pin": "12"}).status_code == 422
    signup(client)
    assert client.post("/auth/signup", json={"phone": "+91 98765 43210", "pin": "1234"}).status_code == 409


def test_login_and_lockout(client):
    signup(client)
    assert client.post("/auth/login", json={"phone": "9876543210", "pin": "1234"}).status_code == 200
    for _ in range(5):
        assert client.post("/auth/login", json={"phone": "9876543210", "pin": "0000"}).status_code == 401
    # now locked, even with the right PIN
    assert client.post("/auth/login", json={"phone": "9876543210", "pin": "1234"}).status_code == 429


def test_history_flow(client):
    h = signup(client)
    r1 = client.post("/chat", json={"question": "clean the tray every morning"}, headers=h).json()
    cid = r1["conversation_id"]
    assert cid
    client.post("/chat", json={"question": "hlo", "conversation_id": cid}, headers=h)
    convs = client.get("/conversations", headers=h).json()
    assert len(convs) == 1 and convs[0]["title"].startswith("clean the tray")
    conv = client.get(f"/conversations/{cid}", headers=h).json()
    assert [m["question"] for m in conv["messages"]] == ["clean the tray every morning", "hlo"]
    assert client.patch(f"/conversations/{cid}", json={"title": "Tray"}, headers=h).status_code == 200
    assert client.delete(f"/conversations/{cid}", headers=h).status_code == 200
    assert client.get("/conversations", headers=h).json() == []


def test_users_cannot_see_each_others_history(client):
    a, b = signup(client, "9876543210"), signup(client, "9123456789")
    cid = client.post("/chat", json={"question": "clean tray"}, headers=a).json()["conversation_id"]
    assert client.get(f"/conversations/{cid}", headers=b).status_code == 404
    assert client.post("/chat", json={"question": "x y", "conversation_id": cid},
                       headers=b).status_code == 404


def test_guest_chat_has_no_history(client):
    r = client.post("/chat", json={"question": "clean tray"}).json()
    assert r["conversation_id"] is None
    assert client.get("/conversations").status_code == 401


def test_bad_token_rejected(client):
    assert client.get("/auth/me", headers={"Authorization": "Bearer junk"}).status_code == 401
