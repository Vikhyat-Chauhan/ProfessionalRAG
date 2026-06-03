# monitoring/visits.py — drop-in DynamoDB replacement for Firestore visit tracking

import boto3
import hashlib
import uuid
from datetime import datetime, timezone, timedelta
from boto3.dynamodb.conditions import Attr

from config import settings

TABLE_NAME = "professionalrag-visits"

# Highest valid UTF-8 code point — appended to an end date so a BETWEEN query on
# the "{ISO timestamp}#{uuid}" sort key is inclusive of every event on that day.
_SK_MAX = "￿"


def visitor_hash(ip: str | None, user_agent: str | None) -> str:
    """Privacy-friendly visitor id: sha256(salt + UTC date + ip + ua), 16 hex chars.

    The salt is combined with the current UTC date, so a given visitor's hash
    rotates every day. That makes per-day unique counts exact while keeping the
    identifier non-reversible and uncorrelatable across days (no cookies, no
    raw-IP storage). Same technique Plausible uses.
    """
    salt = settings.visit_salt or settings.api_key
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw = f"{salt}|{today}|{ip or ''}|{user_agent or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _table():
    """Lazy DynamoDB table reference."""
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    return ddb.Table(TABLE_NAME)

def create_table_if_needed():
    """Run once at startup — idempotent."""
    ddb = boto3.client("dynamodb", region_name="us-east-1")
    if TABLE_NAME in ddb.list_tables().get("TableNames", []):
        return
    ddb.create_table(
        TableName=TABLE_NAME,
        KeySchema=[
            {"AttributeName": "pk", "KeyType": "HASH"},  # event type
            {"AttributeName": "sk", "KeyType": "RANGE"},  # ISO timestamp#uuid
        ],
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",  # stays in free tier for low traffic
    )

def write_event(doc: dict):
    """Write a visit event. doc must include 'event' and 'timestamp' (datetime)."""
    ts = doc["timestamp"].isoformat()
    item = {k: v for k, v in doc.items() if v is not None}
    item["pk"] = item.pop("event", "pageview")
    item["sk"] = f"{ts}#{uuid.uuid4().hex[:8]}"
    item["timestamp"] = ts  # DynamoDB can't store datetime objects
    _table().put_item(Item=item)

def read_events(
    days: int,
    source: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> list[dict]:
    """Scan visit events within a time window.

    When `start`/`end` (YYYY-MM-DD strings) are given, returns events whose sort
    key falls in that inclusive date range. Otherwise falls back to "newer than
    `days` days ago". An optional `source` further filters pageview attribution.
    """
    if start and end:
        condition = Attr("sk").between(start, f"{end}{_SK_MAX}")
    else:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        condition = Attr("sk").gte(cutoff)
    # DynamoDB scan is fine here — visit counts are low, table is small
    if source:
        condition &= Attr("source").eq(source)
    response = _table().scan(FilterExpression=condition)
    return response.get("Items", [])