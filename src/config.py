"""Central configuration loaded from environment / .env file."""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # vLLM
    vllm_base_url: str = Field("mock://local", env="VLLM_BASE_URL")
    vllm_model: str = Field("meta-llama/Llama-3.1-8B-Instruct", env="VLLM_MODEL")
    vllm_temperature: float = Field(0.1, env="VLLM_TEMPERATURE")
    vllm_max_tokens: int = Field(1024, env="VLLM_MAX_TOKENS")

    # Database
    database_url: str = Field(
        "postgresql://raguser:ragpassword@localhost:5432/ragdb",
        env="DATABASE_URL",
    )

    # Embeddings
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(384, env="EMBEDDING_DIMENSION")

    # Retrieval
    default_top_k: int = Field(5, env="DEFAULT_TOP_K")
    default_chunk_strategy: str = Field("sentence", env="DEFAULT_CHUNK_STRATEGY")

    # MLflow
    mlflow_tracking_uri: str = Field("http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(
        "rag-chunking-comparison", env="MLFLOW_EXPERIMENT_NAME"
    )

    # Prometheus
    prometheus_port: int = Field(8001, env="PROMETHEUS_PORT")

    # API
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    log_level: str = Field("info", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
