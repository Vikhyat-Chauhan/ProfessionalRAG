"""Request-model validation + endpoint contracts."""

import json


def test_ingest_requires_source(client, auth):
    r = client.post("/ingest", headers=auth, json={})
    assert r.status_code == 422


def test_ingest_accepts_single_source(client, auth):
    r = client.post("/ingest", headers=auth, json={"source": "/tmp/fake.pdf"})
    assert r.status_code == 200
    assert r.json()["chunks"] == 1


def test_ingest_accepts_list(client, auth):
    r = client.post("/ingest", headers=auth, json={"source": ["/a", "/b", "/c"]})
    assert r.status_code == 200
    assert r.json()["chunks"] == 3


def test_query_returns_expected_shape(client, auth):
    r = client.post("/query", headers=auth, json={"question": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and "sources" in body and "metrics" in body


def test_chat_streams_sse(client, auth):
    with client.stream("POST", "/chat", headers=auth, json={"message": "hi"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [line for line in r.iter_lines() if line.startswith("data: ")]
    types = [json.loads(e[6:])["type"] for e in events]
    assert types[0] == "sources"
    assert "token" in types
    assert types[-1] == "done"


def test_chat_rejects_empty_store(client, auth, monkeypatch):
    import api.server as srv
    monkeypatch.setattr(srv.pipeline.store, "_count", 0)
    r = client.post("/chat", headers=auth, json={"message": "hi"})
    assert r.status_code == 400
