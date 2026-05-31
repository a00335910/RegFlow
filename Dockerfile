FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libxml2-dev \
        libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[dev]"

FROM base AS runtime

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src ./src
COPY config ./config
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "regflow.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
