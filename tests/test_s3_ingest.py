"""S3 source dispatching — boto3 stubbed, no AWS calls."""

import sys
import types
from pathlib import Path

import pytest


def _install_fake_boto3(monkeypatch, *, etag="abc123", body=b"hello world"):
    """Inject a minimal boto3 stub before reader imports it."""
    calls = {"head": [], "download": []}

    class FakeS3:
        def head_object(self, Bucket, Key):
            calls["head"].append((Bucket, Key))
            return {"ETag": f'"{etag}"'}

        def download_file(self, Bucket, Key, Filename):
            calls["download"].append((Bucket, Key, Filename))
            Path(Filename).write_bytes(body)

    fake = types.SimpleNamespace(client=lambda name: FakeS3())
    monkeypatch.setitem(sys.modules, "boto3", fake)
    return calls


def test_parse_s3_uri():
    from ingestion.reader import parse_s3_uri, is_s3_uri

    assert is_s3_uri("s3://my-bucket/path/to/file.pdf")
    assert not is_s3_uri("https://example.com/file.pdf")
    assert not is_s3_uri("/local/path.pdf")

    bucket, key = parse_s3_uri("s3://my-bucket/nested/dir/doc.pdf")
    assert bucket == "my-bucket"
    assert key == "nested/dir/doc.pdf"


def test_parse_s3_uri_rejects_garbage():
    from ingestion.reader import parse_s3_uri
    with pytest.raises(ValueError):
        parse_s3_uri("not-an-s3-uri")


def test_s3_fingerprint_uses_etag(monkeypatch):
    calls = _install_fake_boto3(monkeypatch, etag="deadbeef")
    from ingestion.reader import s3_fingerprint

    fp = s3_fingerprint("s3://my-bucket/doc.pdf")
    assert fp == "s3://my-bucket/doc.pdf@deadbeef"
    assert calls["head"] == [("my-bucket", "doc.pdf")]


def test_read_s3_downloads_and_dispatches_to_text_reader(monkeypatch):
    _install_fake_boto3(monkeypatch, body=b"this is the document content")
    from ingestion.reader import read_s3

    pages = read_s3("s3://my-bucket/notes.txt")

    assert len(pages) == 1
    assert pages[0]["text"] == "this is the document content"
    # The S3 URI must be preserved as 'source', not the local temp path
    assert pages[0]["source"] == "s3://my-bucket/notes.txt"


def test_read_s3_cleans_up_temp_file(monkeypatch):
    captured = {"tmp": None}

    real_dispatch_targets = []
    _install_fake_boto3(monkeypatch, body=b"x")

    from ingestion import reader

    original_read_source = reader.read_source

    def tracking_read_source(src):
        # Only intercept the recursive call on the local temp file,
        # not the outer call on the s3:// URI
        if src.startswith("s3://"):
            return original_read_source(src)
        captured["tmp"] = src
        real_dispatch_targets.append(src)
        return [{"text": "x", "page": 1}]

    monkeypatch.setattr(reader, "read_source", tracking_read_source)

    reader.read_s3("s3://my-bucket/file.txt")

    assert captured["tmp"] is not None, "temp path should have been used"
    assert not Path(captured["tmp"]).exists(), "temp file must be cleaned up"


def test_pipeline_uses_s3_fingerprint(monkeypatch):
    """pipeline.ingest() should route s3:// to s3_fingerprint, not file_fingerprint."""
    _install_fake_boto3(monkeypatch, etag="version-1")

    from pipeline import RAGPipeline

    # Stub everything the pipeline touches so we don't need real models
    class FakeStore:
        def __init__(self): self.seen_fp = None
        def needs_ingestion(self, fp):
            self.seen_fp = fp
            return False  # short-circuit before embedding

    pipe = RAGPipeline.__new__(RAGPipeline)  # bypass __init__ (loads real models)
    pipe.store = FakeStore()

    result = pipe.ingest("s3://my-bucket/doc.pdf")

    assert result == 0
    assert pipe.store.seen_fp == "s3://my-bucket/doc.pdf@version-1"
