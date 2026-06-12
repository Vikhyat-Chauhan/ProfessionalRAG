"""S3 ObjectCreated trigger → ingest the object directly into the vector store.

Self-contained: the Lambda runs the full parse → chunk → embed → upsert pipeline
in-process. Embedding uses Voyage AI's hosted API and the vector store is Pinecone,
so there is no EC2 instance and no cleartext HTTP hop.

Environment variables (injected from Secrets Manager / Terraform):
  VOYAGE_API_KEY     hosted embeddings
  PINECONE_API_KEY   vector store
  ANTHROPIC_API_KEY  unused during ingestion but read by config at import

Deploy:
  Packaging:  container image (Dockerfile.ingest) — bundles the doc parsers
              (pypdf/python-docx/python-pptx/Pillow/pytesseract) + tesseract.
  Handler:    lambda.ingest_trigger.handler
  Timeout:    300 s   (embedding a large PDF can take a minute)
  Memory:     1024 MB
  Role:       AWSLambdaBasicExecutionRole + s3:GetObject/HeadObject on the bucket
"""

import logging
import urllib.parse

log = logging.getLogger()
log.setLevel(logging.INFO)

# Built once per warm container and reused across invocations. Imported lazily
# inside the accessor so a cold start that fails fast on config still logs.
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pipeline import RAGPipeline

        _pipeline = RAGPipeline()
    return _pipeline


def handler(event, context):
    """Lambda entry point. S3 may batch multiple records into one event."""
    results = []
    for record in event.get("Records", []):
        if not record.get("eventName", "").startswith("ObjectCreated:"):
            continue

        bucket = record["s3"]["bucket"]["name"]
        # S3 keys are URL-encoded in the event payload
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        s3_uri = f"s3://{bucket}/{key}"

        try:
            chunks = _get_pipeline().ingest(s3_uri)
            log.info("Ingested %s → %d chunks", s3_uri, chunks)
            results.append({"source": s3_uri, "chunks": chunks})
        except Exception as e:
            # Log and continue with other records — don't fail the whole batch
            log.exception("Ingest failed for %s: %s", s3_uri, e)
            results.append({"source": s3_uri, "error": str(e)})

    return {"processed": results}
