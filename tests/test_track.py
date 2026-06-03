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


def test_track_stores_visitor_id(client, auth, visits_store):
    client.post("/track", headers=auth, json={"event": "pageview"})
    vid = visits_store[0]["visitor_id"]
    assert isinstance(vid, str) and len(vid) == 16
    # No raw client IP leaks into the visitor id.
    assert "." not in vid and ":" not in vid


def test_visits_unique_visitors_dedup(client, auth):
    # Same client (same IP + UA) within a day → one unique visitor.
    for _ in range(3):
        client.post("/track", headers=auth, json={"event": "pageview"})
    # A different user-agent is a different visitor.
    client.post("/track", headers={**auth, "User-Agent": "Other/1.0"}, json={"event": "pageview"})

    body = client.get("/visits", headers=auth).json()
    assert body["pageviews"] == 4
    assert body["unique_visitors"] == 2


def test_visits_exposes_new_analytics_keys(client, auth):
    client.post("/track", headers=auth, json={
        "event": "pageview", "page": "/", "referrer": "https://news.ycombinator.com",
        "utm_medium": "social", "language": "en-US", "theme": "dark",
    })
    body = client.get("/visits", headers=auth).json()
    assert "by_day" in body and "bounce_rate" in body
    assert body["by_referrer"]["https://news.ycombinator.com"] == 1
    assert body["by_utm_medium"]["social"] == 1
    assert body["by_language"]["en-US"] == 1
    assert body["by_theme"]["dark"] == 1
    assert set(body["time_on_site"]) == {"avg", "p50", "p90", "p95", "count"}


def test_visits_start_end_validation(client, auth):
    # Only one bound provided → 400.
    assert client.get("/visits?start=2026-06-01", headers=auth).status_code == 400
    # start after end → 400.
    assert client.get("/visits?start=2026-06-10&end=2026-06-01", headers=auth).status_code == 400
    # Valid range → 200 and echoes the window.
    r = client.get("/visits?start=2026-06-01&end=2026-06-03", headers=auth)
    assert r.status_code == 200
    assert r.json()["start"] == "2026-06-01" and r.json()["end"] == "2026-06-03"


def test_visits_date_range_filters_events(client, auth):
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date()
    client.post("/track", headers=auth, json={"event": "pageview"})
    # Events land on "today"; a past-only range excludes them.
    past = client.get("/visits?start=2000-01-01&end=2000-01-02", headers=auth).json()
    assert past["pageviews"] == 0
    # A range covering today includes them.
    now = client.get(f"/visits?start={today}&end={today}", headers=auth).json()
    assert now["pageviews"] == 1


def test_dashboard_served_without_auth(client):
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "chart.js" in r.text.lower()
    assert "/visits?" in r.text
