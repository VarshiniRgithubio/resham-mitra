from fastapi.testclient import TestClient

from app.api import health as health_module
from app.main import app


def test_health_reports_db_status(monkeypatch):
    async def ok(_):
        return True

    async def down(_):
        return False

    monkeypatch.setattr(health_module, "check_mongo", ok)
    monkeypatch.setattr(health_module, "check_qdrant", down)
    body = TestClient(app).get("/health").json()
    assert body["mongo"] is True and body["qdrant"] is False
    assert body["status"] == "degraded"
