from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


class PostgresSettings(BaseSettings):
    host: str = "localhost"
    port: int = 5433     # host 5433 -> docker container 5432 (5432 taken by Safekube)
    user: str = "regflow"
    password: SecretStr = SecretStr("regflow_dev")
    database: str = "regflow"

    @property
    def dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class Neo4jSettings(BaseSettings):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("regflow_dev")
    database: str = "neo4j"


class WeaviateSettings(BaseSettings):
    host: str = "localhost"
    http_port: int = 8080
    grpc_port: int = 50051
    corpus_collection: str = "RegulatoryCorpus"
    override_collection: str = "OverrideStore"


class MinioSettings(BaseSettings):
    endpoint: str = "localhost:9000"
    access_key: str = "regflow"
    secret_key: SecretStr = SecretStr("regflow_dev_minio")
    secure: bool = False
    raw_docs_bucket: str = "regflow-raw-docs"


class LLMSettings(BaseSettings):
    """LiteLLM-compatible config. Local Ollama by default; switch base_url to vLLM on rented GPU."""

    provider: Literal["ollama", "vllm", "openai_compatible"] = "ollama"
    base_url: str = "http://localhost:11434"
    extraction_model: str = "ollama/llama3.1:8b"      # Agent 2 (obligation extraction)
    classifier_model: str = "ollama/llama3.1:8b"     # Agent 1 (severity classification)
    reasoning_model: str = "ollama/llama3.1:8b"      # Agents 3-6
    temperature: float = 0.0
    request_timeout_s: int = 120
    max_retries: int = 3


class EmbeddingSettings(BaseSettings):
    """Local sentence-transformers. BGE-M3 is multilingual; required for EU/UK/US regulatory text."""

    model_name: str = "BAAI/bge-m3"
    dimension: int = 1024
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    batch_size: int = 32
    normalize: bool = True


class OrchestratorSettings(BaseSettings):
    """Confidence thresholds drive AUTO / NOTIFY / BLOCK routing (architecture line 40)."""

    auto_confidence_threshold: float = 0.85
    notify_confidence_threshold: float = 0.60
    extraction_min_confidence: float = 0.55       # below -> mandatory human review (line 57)
    conflict_high_severity_threshold: float = 0.75  # above -> legal signoff (line 86)
    gap_high_risk_threshold: float = 0.75           # above -> compliance approval (line 99)
    max_retries_per_agent: int = 2


class IngestionSettings(BaseSettings):
    poll_interval_s: int = 3600
    # Browser UA — Cloudflare-fronted sites (EUR-Lex) challenge "research bot" UAs.
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    )
    request_timeout_s: int = 30


class LangfuseSettings(BaseSettings):
    """LLM observability via Langfuse. Disabled by default; enable by setting
    `enabled: true` AND providing both keys (via env or YAML).

    Get keys from http://localhost:3000 after `docker compose up -d` —
    Settings -> API Keys -> Create new keys for a new project."""

    enabled: bool = False
    host: str = "http://localhost:3000"
    public_key: SecretStr | None = None
    secret_key: SecretStr | None = None
    flush_at: int = 1            # flush after N events (1 = synchronous, easier for demo)
    flush_interval: float = 1.0  # flush every N seconds


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REGFLOW_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    weaviate: WeaviateSettings = Field(default_factory=WeaviateSettings)
    minio: MinioSettings = Field(default_factory=MinioSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> Settings:
        path = path or (CONFIG_DIR / "settings.yaml")
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_yaml()
