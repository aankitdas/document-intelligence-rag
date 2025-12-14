FROM python:3.12-slim

WORKDIR /app

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-railway.txt ./

# Install CPU-only torch first to avoid downloading CUDA deps
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies with aggressive caching cleanup
RUN pip install --no-cache-dir -r requirements-railway.txt && \
    pip cache purge && \
    find /usr/local -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local -type f -name '*.pyc' -delete && \
    rm -rf /tmp/* /var/tmp/* /root/.cache

# Copy application code
COPY src ./src
COPY frontend ./frontend

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV TORCH_HOME=/tmp/torch
ENV EMBEDDING_BACKEND=sentence-transformers

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]