from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # API
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    api_key: str = Field(default="", alias="ProfessionalRAG_KEY")

    # Models
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 1024
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    candidate_count: int = 50
    top_k: int = 5

    # Storage (Pinecone)
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index: str = "professional-rag"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embedding_dim: int = 768

    # Cost tracking (USD per million tokens)
    cost_per_m_input_tokens: float = 3.0
    cost_per_m_output_tokens: float = 15.0

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
