# Tabletop Oracle API

Knowledge graph API for tabletop games. Built with FastAPI, SQLAlchemy 2.0 (async), and PostgreSQL 16.

## Prerequisites

- Python 3.12+
- Docker and Docker Compose (for PostgreSQL)

## Quick Start

1. Start PostgreSQL:

```bash
docker compose up -d
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

3. Copy environment config:

```bash
cp .env.example .env
```

4. Run database migrations:

```bash
alembic upgrade head
```

5. Start the development server:

```bash
uvicorn tabletop_oracle.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/api/v1/docs`.

## Development

### Linting and Formatting

```bash
ruff check src/ tests/
ruff format src/ tests/
```

### Type Checking

```bash
mypy src/
```

### Running Tests

```bash
pytest tests/ --cov=src/tabletop_oracle
```

### Pre-commit Hooks

```bash
pre-commit install
```

## Project Structure

```
src/tabletop_oracle/
  api/          Route handlers and dependencies
  models/       SQLAlchemy ORM models
  schemas/      Pydantic request/response schemas
  services/     Business logic
  repositories/ Data access layer
  workers/      Background task processing
  middleware/   HTTP middleware (logging, correlation ID)
  errors/       Exception hierarchy
  storage/      Blob storage abstraction
```
