"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings sourced from environment variables."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Database
    database_url: str = "postgresql://tabletop:tabletop_dev@localhost:5432/tabletop_oracle"
    database_url_async: str = (
        "postgresql+asyncpg://tabletop:tabletop_dev@localhost:5432/tabletop_oracle"
    )

    # Auth
    oauth_google_client_id: str = ""
    oauth_google_client_secret: str = ""
    oauth_microsoft_client_id: str = ""
    oauth_microsoft_client_secret: str = ""
    initial_curator_emails: str = ""
    frontend_origin: str = "http://localhost:4200"
    oauth_redirect_base_url: str = "http://localhost:8000"
    auth_session_timeout_days: int = 30
    session_cookie_secure: bool = False
    secret_key: str = "change-me-in-production"
    bypass_auth: bool = False

    # Storage
    blob_storage_backend: str = "local"
    blob_storage_local_path: str = "./storage"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Embedding
    embedding_dim: int = 1536

    # Application
    log_level: str = "INFO"


settings = Settings()
