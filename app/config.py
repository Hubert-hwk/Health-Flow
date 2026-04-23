"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # MySQL配置
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "password"
    MYSQL_DATABASE: str = "healthflow"

    # Milvus配置
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # Neo4j配置
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # vLLM配置
    VLLM_HOST: str = "localhost"
    VLLM_PORT: int = 8000
    VLLM_MODEL: str = "qwen-vl-plus"

    # MiniMax配置（用于SFT数据生成）
    MINIMAX_API_KEY: str = ""
    MINIMAX_MODEL: str = "MiniMax-M2.7"

    # Embedding配置
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"

    # API配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080

    @property
    def mysql_url(self) -> str:
        """Get MySQL connection URL."""
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
