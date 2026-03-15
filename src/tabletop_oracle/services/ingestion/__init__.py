"""Document ingestion pipeline.

Multi-stage processing pipeline that transforms uploaded documents into
structured, traceable content chunks. Runs as a Celery background task
with SSE event emission at each stage transition.
"""
