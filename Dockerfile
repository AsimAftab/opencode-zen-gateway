# OpenCode Zen Gateway - Docker Image
# Optimized single-stage build

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd -r opencode && useradd -r -g opencode opencode

# Set working directory and give ownership to opencode user
WORKDIR /app
RUN chown opencode:opencode /app

# Install dependencies first (better layer caching)
COPY pyproject.toml uv.lock ./
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv sync --frozen --no-dev

# Copy application code
COPY --chown=opencode:opencode . .

# Create directory for debug logs with proper permissions
RUN mkdir -p debug_logs && chown -R opencode:opencode debug_logs

# Switch to non-root user
USER opencode

# Expose port
EXPOSE 8000

# Health check
# Using httpx (our main HTTP library) instead of requests
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD uv run python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5)"

# Run the application
CMD ["uv", "run", "python", "main.py"]
