# monitoring/visits.py — drop-in DynamoDB replacement for Firestore visit tracking

import boto3
import uuid
from datetime import datetime, timezone, timedelta
from boto3.dynamodb.conditions import Attr

TABLE_NAME = "professionalrag-visits"

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

def read_events(days: int, source: str | None = None) -> list[dict]:
    """Scan events newer than `days` days ago."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    # DynamoDB scan is fine here — visit counts are low, table is small
    kwargs = {"FilterExpression": Attr("sk").gte(cutoff)}
    if source:
        kwargs["FilterExpression"] &= Attr("source").eq(source)
    response = _table().scan(**kwargs)
    return response.get("Items", [])