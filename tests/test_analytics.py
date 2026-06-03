"""Unit tests for the pure aggregation layer (no FastAPI, no AWS)."""

from monitoring.analytics import aggregate, _percentile


def _pv(day, visitor, **extra):
    """Build a pageview event item as stored in DynamoDB."""
    return {"pk": "pageview", "timestamp": f"{day}T12:00:00+00:00",
            "visitor_id": visitor, "page": "/", "source": "direct", **extra}


def test_percentile_nearest_rank():
    samples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert _percentile(samples, 50) == 50
    assert _percentile(samples, 90) == 90
    assert _percentile(samples, 95) == 100
    assert _percentile([], 95) == 0


def test_by_day_time_series():
    events = [
        _pv("2026-06-01", "a"), _pv("2026-06-01", "b"),
        _pv("2026-06-02", "a"),
    ]
    out = aggregate(events, {"days": 30})
    assert out["by_day"] == {
        "2026-06-01": {"pageviews": 2, "unique_visitors": 2},
        "2026-06-02": {"pageviews": 1, "unique_visitors": 1},
    }
    # Days are chronologically ordered for charting.
    assert list(out["by_day"]) == ["2026-06-01", "2026-06-02"]


def test_unique_visitors_across_events():
    events = [_pv("2026-06-01", "a"), _pv("2026-06-01", "a"), _pv("2026-06-01", "b")]
    assert aggregate(events, {"days": 30})["unique_visitors"] == 2


def test_bounce_rate():
    # Visitor "a" has 2 pageviews (engaged); "b" has 1 (bounced) → 1 of 2 sessions.
    events = [_pv("2026-06-01", "a"), _pv("2026-06-01", "a"), _pv("2026-06-01", "b")]
    assert aggregate(events, {"days": 30})["bounce_rate"] == 50.0


def test_time_on_site_percentiles():
    events = [{"pk": "time_on_site", "timestamp": "2026-06-01T00:00:00+00:00",
               "seconds": s} for s in (10, 20, 30, 40, 100)]
    tos = aggregate(events, {"days": 30})["time_on_site"]
    assert tos["count"] == 5
    assert tos["avg"] == 40
    assert tos["p50"] == 30
    assert tos["p95"] == 100


def test_window_echoed_back():
    out = aggregate([], {"start": "2026-06-01", "end": "2026-06-03"})
    assert out["start"] == "2026-06-01" and out["end"] == "2026-06-03"
    assert out["pageviews"] == 0 and out["unique_visitors"] == 0
