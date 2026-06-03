# monitoring/analytics.py — pure aggregation over raw visit events.
#
# Kept free of I/O so it's trivially unit-testable and reusable by both the
# /visits API and the /dashboard view. Input is the list of DynamoDB items
# returned by monitoring.visits.read_events().

from collections import defaultdict


def _sorted(d: dict) -> dict:
    """Descending-by-count ordering — matches the original /visits behavior."""
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))


def _percentile(samples: list[int], pct: float) -> int:
    """Nearest-rank percentile. Avoids a numpy/statistics dependency and is
    exact for the small sample sizes this endpoint sees."""
    if not samples:
        return 0
    ordered = sorted(samples)
    # nearest-rank: rank = ceil(pct/100 * N), clamped to [1, N]
    rank = max(1, min(len(ordered), -(-int(pct) * len(ordered) // 100)))
    return int(ordered[rank - 1])


def _event_type(d: dict) -> str:
    """DynamoDB stores the event type in `pk`; fall back to the legacy field."""
    return d.get("pk") or d.get("event", "pageview")


def _day(d: dict) -> str:
    """YYYY-MM-DD from the stored ISO timestamp (or sort key prefix)."""
    ts = d.get("timestamp") or d.get("sk", "")
    return ts[:10]


def aggregate(events: list[dict], window: dict) -> dict:
    """Roll raw visit events up into the analytics payload.

    `window` is echoed back into the response (e.g. {"days": 30} or
    {"start": "...", "end": "..."}) so callers can label the result.
    """
    by_event: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    by_page: dict[str, int] = defaultdict(int)
    by_device: dict[str, int] = defaultdict(int)
    by_tab: dict[str, int] = defaultdict(int)
    by_referrer: dict[str, int] = defaultdict(int)
    by_utm_medium: dict[str, int] = defaultdict(int)
    by_utm_campaign: dict[str, int] = defaultdict(int)
    by_language: dict[str, int] = defaultdict(int)
    by_theme: dict[str, int] = defaultdict(int)
    outbound_clicks: dict[str, int] = defaultdict(int)

    resume_downloads = 0
    chat_messages = 0
    time_samples: list[int] = []

    visitors: set[str] = set()
    # day -> {"pageviews": int, "visitors": set}; pageviews-per-(visitor,day)
    # powers the bounce-rate calculation below.
    daily: dict[str, dict] = defaultdict(lambda: {"pageviews": 0, "visitors": set()})
    session_pageviews: dict[tuple, int] = defaultdict(int)

    for d in events:
        event = _event_type(d)
        by_event[event] += 1
        day = _day(d)
        visitor = d.get("visitor_id")
        if visitor:
            visitors.add(visitor)

        if event == "pageview":
            src = d.get("source", "direct")
            by_source[src] += 1
            by_page[d.get("page", "/")] += 1
            if d.get("device_type"):
                by_device[d["device_type"]] += 1
            if d.get("referrer") and d["referrer"] != "direct":
                by_referrer[d["referrer"]] += 1
            if d.get("utm_medium"):
                by_utm_medium[d["utm_medium"]] += 1
            if d.get("utm_campaign"):
                by_utm_campaign[d["utm_campaign"]] += 1
            if d.get("language"):
                by_language[d["language"]] += 1
            if d.get("theme"):
                by_theme[d["theme"]] += 1

            daily[day]["pageviews"] += 1
            if visitor:
                daily[day]["visitors"].add(visitor)
                session_pageviews[(visitor, day)] += 1
        elif event == "tab_click":
            by_tab[d.get("tab", "unknown")] += 1
        elif event == "outbound_click":
            outbound_clicks[d.get("hostname", "unknown")] += 1
        elif event == "resume_download":
            resume_downloads += 1
        elif event == "chat_message":
            chat_messages += 1
        elif event == "time_on_site":
            secs = d.get("seconds")
            if secs:
                # DynamoDB returns numbers as Decimal — cast to int
                time_samples.append(int(secs))

    # Bounce rate: share of sessions (one visitor on one day) with a single
    # pageview. Only meaningful when we have identified sessions.
    bounce_rate = 0.0
    if session_pageviews:
        bounced = sum(1 for n in session_pageviews.values() if n == 1)
        bounce_rate = round(100 * bounced / len(session_pageviews), 1)

    by_day = {
        day: {"pageviews": v["pageviews"], "unique_visitors": len(v["visitors"])}
        for day, v in sorted(daily.items())
    }

    avg_time = round(sum(time_samples) / len(time_samples)) if time_samples else 0

    return {
        **window,
        "pageviews": by_event.get("pageview", 0),
        "unique_visitors": len(visitors),
        "bounce_rate": bounce_rate,
        "events": _sorted(dict(by_event)),
        "by_day": by_day,
        "by_source": _sorted(dict(by_source)),
        "by_page": _sorted(dict(by_page)),
        "by_device": _sorted(dict(by_device)),
        "by_tab": _sorted(dict(by_tab)),
        "by_referrer": _sorted(dict(by_referrer)),
        "by_utm_medium": _sorted(dict(by_utm_medium)),
        "by_utm_campaign": _sorted(dict(by_utm_campaign)),
        "by_language": _sorted(dict(by_language)),
        "by_theme": _sorted(dict(by_theme)),
        "outbound_clicks": _sorted(dict(outbound_clicks)),
        "resume_downloads": resume_downloads,
        "chat_messages": chat_messages,
        "time_on_site": {
            "avg": avg_time,
            "p50": _percentile(time_samples, 50),
            "p90": _percentile(time_samples, 90),
            "p95": _percentile(time_samples, 95),
            "count": len(time_samples),
        },
        # Back-compat with the original flat field.
        "avg_time_on_site_seconds": avg_time,
    }
