import json
import os

from pydantic_settings import BaseSettings
from pydantic import Field


def _hydrate_secrets_from_aws() -> None:
    """In Lambda, secrets arrive as a single Secrets Manager entry referenced by
    SECRETS_ARN. Fetch it once and populate os.environ so Pydantic Settings reads
    the values normally. No-op locally (no SECRETS_ARN -> .env is used instead).
    Existing env vars win, so it never clobbers an explicit override.
    """
    arn = os.environ.get("SECRETS_ARN")
    if not arn:
        return
    try:
        import boto3

        region = os.environ.get("AWS_REGION_APP") or os.environ.get("AWS_REGION")
        client = boto3.client("secretsmanager", region_name=region)
        payload = client.get_secret_value(SecretId=arn)["SecretString"]
        for key, value in json.loads(payload).items():
            os.environ.setdefault(key, str(value))
    except Exception as exc:  # fail open to .env / existing env; surfaced at startup
        import logging

        logging.getLogger(__name__).warning("Secret hydration failed: %s", exc)


_hydrate_secrets_from_aws()


class Settings(BaseSettings):
    # API
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    api_key: str = Field(default="", alias="ProfessionalRAG_KEY")

    # Analytics — salt for the daily-rotating visitor hash. Falls back to api_key
    # when empty, so it's optional. Set a dedicated value to decouple the two.
    visit_salt: str = Field(default="", alias="VISIT_SALT")

    # Models
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 2048
    # Embedding + reranking run on Voyage AI's hosted API (no local weights).
    # `embedding_dim` must match the hosted embedding model's output.
    embedding_model: str = "voyage-3"
    reranker_model: str = "rerank-2"

    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Retrieval
    candidate_count: int = 50
    top_k: int = 5

    # Hosted inference (Voyage AI — embeddings + reranking)
    voyage_api_key: str = Field(default="", alias="VOYAGE_API_KEY")

    # Storage (Pinecone)
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index: str = "professional-rag"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embedding_dim: int = 1024  # voyage-3 default output dimension

    # Cost tracking (USD per million tokens)
    cost_per_m_input_tokens: float = 3.0
    cost_per_m_output_tokens: float = 15.0

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
