FROM python:3.12-slim

WORKDIR /app

# System deps for sentence-transformers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch — avoids ~3GB of CUDA wheels that sentence-transformers would otherwise pull
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download models at build time — NOT at runtime
# This bakes the weights into the image so cold start is instant
# bge-base-en-v1.5 (~440MB) + MiniLM-L-6-v2 (~66MB)
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-base-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8080/health', timeout=3)" || exit 1

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
