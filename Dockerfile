# Stage 1: Build
FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

COPY src/ src/
RUN pip install --no-cache-dir .

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=builder /app/src/ src/
COPY migrations/ migrations/
COPY alembic.ini .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "tabletop_oracle.main:app", "--host", "0.0.0.0", "--port", "8000"]
