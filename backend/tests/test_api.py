from tests.conftest import StubLLM
from fastapi.testclient import TestClient

from app.core.deps import get_db, get_pipeline
from app.main import app


class BrokenDB:
    def __getitem__(self, _):
        raise RuntimeError("mongo down")


def test_chat_works_even_if_mongo_is_down(make_pipeline):
    app.dependency_overrides[get_pipeline] = lambda: make_pipeline(StubLLM())
    app.dependency_overrides[get_db] = lambda: BrokenDB()
    try:
        r = TestClient(app).post("/chat", json={"question": "clean the tray every morning"})
        assert r.status_code == 200 and "message_id" in r.json()
    finally:
        app.dependency_overrides.clear()
