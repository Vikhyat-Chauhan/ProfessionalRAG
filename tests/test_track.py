"""POST /track and GET /visits."""


def test_pageview_recorded(client, auth, visits_store):
    r = client.post("/track", headers=auth, json={
        "event": "pageview", "page": "/", "utm_source": "twitter",
    })
    assert r.status_code == 204
    assert len(visits_store) == 1
    assert visits_store[0]["pk"] == "pageview"
    assert visits_store[0]["source"] == "twitter"


def test_pageview_source_falls_back_to_direct(client, auth, visits_store):
    client.post("/track", headers=auth, json={"event": "pageview", "page": "/"})
    assert visits_store[0]["source"] == "direct"


def test_visits_aggregates_all_event_types(client, auth):
    posts = [
        {"event": "pageview", "page": "/", "utm_source": "google", "device_type": "mobile"},
        {"event": "pageview", "page": "/about", "utm_source": "google", "device_type": "desktop"},
        {"event": "tab_click", "tab": "projects"},
        {"event": "outbound_click", "hostname": "github.com", "url": "https://github.com"},
        {"event": "resume_download"},
        {"event": "chat_message", "question": "hello"},
        {"event": "time_on_site", "seconds": 30},
        {"event": "time_on_site", "seconds": 90},
    ]
    for p in posts:
        assert client.post("/track", headers=auth, json=p).status_code == 204

    body = client.get("/visits", headers=auth).json()
    assert body["pageviews"] == 2
    assert body["by_source"]["google"] == 2
    assert body["by_page"] == {"/": 1, "/about": 1}
    assert body["by_device"] == {"mobile": 1, "desktop": 1}
    assert body["by_tab"]["projects"] == 1
    assert body["outbound_clicks"]["github.com"] == 1
    assert body["resume_downloads"] == 1
    assert body["chat_messages"] == 1
    assert body["avg_time_on_site_seconds"] == 60


def test_visits_source_filter(client, auth):
    client.post("/track", headers=auth, json={"event": "pageview", "utm_source": "twitter"})
    client.post("/track", headers=auth, json={"event": "pageview", "utm_source": "google"})
    body = client.get("/visits?source=twitter", headers=auth).json()
    assert body["by_source"] == {"twitter": 1}


def test_visits_days_bounds(client, auth):
    assert client.get("/visits?days=0", headers=auth).status_code == 422
    assert client.get("/visits?days=400", headers=auth).status_code == 422
    assert client.get("/visits?days=7", headers=auth).status_code == 200


def test_track_requires_auth(client):
    r = client.post("/track", json={"event": "pageview"})
    assert r.status_code == 401
