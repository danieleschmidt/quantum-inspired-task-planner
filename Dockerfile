# Multi-stage Dockerfile for Quantum-Inspired Task Planner
# Optimized for production deployment with security scanning

# ===== Base Python Image =====
FROM python:3.13-slim-bullseye as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.6.1

# Install system dependencies and security updates
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Create non-root user for security
RUN groupadd -r quantum && useradd -r -g quantum quantum

# ===== Development Stage =====
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    vim \
    htop \
    tree \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry for development
RUN poetry config virtualenvs.create false \
    && poetry install --with dev,test,docs --no-root

# Copy source code
COPY . .

# Install the package in development mode
RUN poetry install --with dev,test,docs

# Change ownership to quantum user
RUN chown -R quantum:quantum /app

USER quantum

# Default command for development
CMD ["bash"]

# ===== Testing Stage =====
FROM development as testing

USER root

# Install additional test dependencies
RUN poetry install --with test --no-root

# Run tests and generate coverage
RUN poetry run pytest tests/ --cov=src/quantum_planner --cov-report=xml --cov-report=html

# Security scanning
RUN poetry run bandit -r src/ -f json -o bandit-report.json || true
RUN poetry run safety check --json --output safety-report.json || true

USER quantum

# ===== Production Build Stage =====
FROM base as builder

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install only production dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-root

# Copy source code
COPY src/ ./src/
COPY README.md LICENSE ./

# Build the package
RUN poetry build

# ===== Production Runtime Stage =====
FROM python:3.13-slim-bullseye as production

# Security: Install only essential packages and security updates
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r quantum && useradd -r -g quantum quantum

# Set working directory
WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /app/dist/*.whl /tmp/

# Install the package
RUN pip install --no-cache-dir /tmp/*.whl \
    && rm -rf /tmp/*.whl

# Create directories for application data
RUN mkdir -p /app/data /app/logs /app/config \
    && chown -R quantum:quantum /app

# Copy configuration files
COPY docker/config/ ./config/
COPY docker/entrypoint.sh ./

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Switch to non-root user
USER quantum

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port (if running as service)
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["quantum-planner", "--help"]

# ===== Minimal Production Stage =====
FROM python:3.13-alpine as minimal

# Install minimal dependencies
RUN apk add --no-cache \
    curl \
    && addgroup -g 1000 quantum \
    && adduser -D -u 1000 -G quantum quantum

WORKDIR /app

# Copy built wheel
COPY --from=builder /app/dist/*.whl /tmp/

# Install package with minimal dependencies
RUN pip install --no-cache-dir /tmp/*.whl \
    && rm -rf /tmp/*.whl \
    && rm -rf /root/.cache

USER quantum

CMD ["quantum-planner", "--version"]

# ===== GPU-Enabled Stage =====
FROM nvidia/cuda:12.9.1-runtime-ubuntu20.04 as gpu

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r quantum && useradd -r -g quantum quantum

WORKDIR /app

# Copy built wheel
COPY --from=builder /app/dist/*.whl /tmp/

# Install package with GPU acceleration
RUN pip3 install --no-cache-dir /tmp/*.whl[gpu] \
    && rm -rf /tmp/*.whl

USER quantum

CMD ["quantum-planner", "--backend", "gpu_accelerated"]

# ===== Labels for metadata =====
LABEL maintainer="Daniel Schmidt <daniel@terragon.ai>" \
      version="1.0.0" \
      description="Quantum-Inspired Task Planner" \
      org.opencontainers.image.title="quantum-inspired-task-planner" \
      org.opencontainers.image.description="QUBO-based task scheduler for agent pools" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.created="2025-01-01T00:00:00Z" \
      org.opencontainers.image.source="https://github.com/your-org/quantum-inspired-task-planner" \
      org.opencontainers.image.documentation="https://docs.your-org.com/quantum-planner" \
      org.opencontainers.image.licenses="MIT"