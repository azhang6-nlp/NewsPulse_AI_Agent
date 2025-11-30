FROM python:3.12-slim

# Avoid Python temp files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Install system dependencies needed by your Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency files first to leverage Docker cache
COPY pyproject.toml pyproject.toml
COPY readme.md readme.md
COPY requirements.txt requirements.txt
COPY AI_Newsletter/ AI_Newsletter/
# Upgrade pip + install build tools (needed for numpy/cython builds)
RUN pip install --upgrade pip setuptools wheel build cython numpy

# Install your package + google-adk CLI from pyproject.toml
RUN pip install --no-cache-dir .


# Expose ADK UI port
EXPOSE 8080

# Healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:${PORT}/ || exit 1

# Run ADK Web UI
CMD ["adk", "web", "--host", "0.0.0.0", "--port", "8080"]
