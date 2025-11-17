"""Application configuration using Pydantic Settings."""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from decimal import Decimal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    app_name: str = "distributed-llm-platform"
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # Database
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/distributed_llm"
    database_pool_size: int = 20
    database_max_overflow: int = 10

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 50

    # Security
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Router Configuration
    router_strategy: str = "least_latency"
    health_check_interval_sec: int = 30
    provider_timeout_sec: int = 300
    retry_attempts: int = 3

    # Rate Limiting
    rate_limit_enabled: bool = True
    free_tier_rate_limit: int = 60
    starter_tier_rate_limit: int = 300
    pro_tier_rate_limit: int = 1000

    # Billing
    llm_api_commission: Decimal = Decimal("0.05")
    memory_storage_margin: Decimal = Decimal("0.20")

    # Workflow Execution
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    workflow_default_timeout: int = 300
    workflow_max_retries: int = 3

    # Monitoring
    prometheus_enabled: bool = True
    prometheus_port: int = 9090

    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL (for Alembic)."""
        return self.database_url.replace("+asyncpg", "")


# Global settings instance
settings = Settings()
