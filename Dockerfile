# ============================================================
# Credit Risk Classifier — Production Dockerfile
# Base: python:3.11-slim (minimal attack surface)
# ============================================================

FROM python:3.11-slim AS base

# --- Security: run as non-root user ---
RUN groupadd --gid 1001 appuser \
    && useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

# --- System dependencies ---
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Copy dependency manifest first (cache layer) ---
COPY requirements.txt .

# --- Install Python dependencies ---
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Copy source code ---
COPY src/ ./src/
COPY app.py .

# --- Copy pre-trained model artifacts ---
COPY models/ ./models/

# --- Set ownership ---
RUN chown -R appuser:appuser /app

# --- Switch to non-root user ---
USER appuser

# --- Environment variables ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_DEBUG=false \
    PORT=5000

# --- Expose port ---
EXPOSE 5000

# --- Health check (Docker native) ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# --- Start application with Gunicorn (production WSGI server) ---
CMD gunicorn \
    --bind 0.0.0.0:${PORT} \
    --workers 2 \
    --threads 2 \
    --timeout 60 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    app:app
