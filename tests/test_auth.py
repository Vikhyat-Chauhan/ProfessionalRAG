"""Auth middleware behavior."""

import pytest


def test_health_is_open(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_requires_auth(client):
    assert client.get("/metrics").status_code == 401


def test_metrics_with_auth(client, auth):
    assert client.get("/metrics", headers=auth).status_code == 200


def test_bad_bearer_rejected(client):
    r = client.get("/metrics", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_missing_scheme_rejected(client):
    r = client.get("/metrics", headers={"Authorization": "test-key"})
    assert r.status_code == 401


def test_fail_closed_when_key_unset(monkeypatch):
    """Lifespan must refuse to start with no API key configured."""
    from config import settings
    from fastapi.testclient import TestClient
    import api.server as srv

    monkeypatch.setattr(settings, "api_key", "")
    with pytest.raises(RuntimeError, match="ProfessionalRAG_KEY"):
        with TestClient(srv.app):
            pass
