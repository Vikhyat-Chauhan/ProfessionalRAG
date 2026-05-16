"""S3 ObjectCreated trigger → POST /ingest on the EC2 service.

Environment variables:
  INGEST_URL        full URL to /ingest, e.g. http://3.20.124.147:8080/ingest
  API_KEY_PARAM     SSM Parameter Store name holding the Bearer token,
                    e.g. /professionalrag/api-key  (SecureString)

Deploy:
  Runtime:  Python 3.12
  Handler:  ingest_trigger.handler
  Timeout:  30 s   (we don't wait for ingestion to finish — just fire and forget)
  Memory:   128 MB
  Role:     AWSLambdaBasicExecutionRole + ssm:GetParameter on API_KEY_PARAM

The Lambda intentionally does NOT wait for embedding to finish. It POSTs
/ingest and returns as soon as the server accepts the request. Ingestion
can take a minute on large PDFs; Lambda's job is just to dispatch.
"""

import json
import logging
import os
import urllib.parse
import urllib.request

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

INGEST_URL = os.environ["INGEST_URL"]
API_KEY_PARAM = os.environ["API_KEY_PARAM"]

# Cached across warm invocations
_api_key: str | None = None


def _get_api_key() -> str:
    global _api_key
    if _api_key is None:
        ssm = boto3.client("ssm")
        resp = ssm.get_parameter(Name=API_KEY_PARAM, WithDecryption=True)
        _api_key = resp["Parameter"]["Value"]
    return _api_key


def _post_ingest(s3_uri: str) -> int:
    """POST /ingest with the S3 URI as source. Returns HTTP status code."""
    body = json.dumps({"source": s3_uri}).encode("utf-8")
    req = urllib.request.Request(
        INGEST_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {_get_api_key()}",
            "Content-Type": "application/json",
        },
    )
    # Short timeout — we just want the server to accept the request.
    # Ingestion runs synchronously server-side, so we set a generous read
    # timeout but don't actually depend on the response.
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.status


def handler(event, context):
    """Lambda entry point. S3 may batch multiple records into one event."""
    statuses = []
    for record in event.get("Records", []):
        if not record.get("eventName", "").startswith("ObjectCreated:"):
            continue

        bucket = record["s3"]["bucket"]["name"]
        # S3 keys are URL-encoded in the event payload
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        s3_uri = f"s3://{bucket}/{key}"

        try:
            status = _post_ingest(s3_uri)
            log.info("Ingest dispatched: %s → HTTP %d", s3_uri, status)
            statuses.append({"source": s3_uri, "status": status})
        except Exception as e:
            # Log and continue with other records — don't fail the whole batch
            log.exception("Ingest failed for %s: %s", s3_uri, e)
            statuses.append({"source": s3_uri, "error": str(e)})

    return {"processed": statuses}
