FROM python:3.13-slim-bookworm AS builder

WORKDIR /app

# Install build toolchain for native extensions (numpy, faiss-cpu, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Install dependencies into a virtual env so we can copy it cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt requirements-bench.txt ./
# Install PyTorch CPU-only first (saves ~2GB vs full torch with CUDA stubs)
RUN uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --no-cache -r requirements.txt -r requirements-bench.txt

# --- Final stage: slim image without build tools ---
FROM python:3.13-slim-bookworm

WORKDIR /app

# Copy the virtual env from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project source
COPY . /app
RUN pip install --no-cache-dir -e .

# Run tests then benchmarks in one go
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

CMD ["/app/docker-entrypoint.sh"]
