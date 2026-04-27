from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    app_name: str = "Argus Insurance Intelligence Platform"
    app_version: str = "1.0.0"
    app_env: str = "development"
    log_level: str = "INFO"

    anthropic_api_key: str = ""

    model_path: str = "models/argus_model.joblib"
    faiss_index_path: str = "models/faiss_index"
    policies_path: str = "data/policies"

    cors_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def model_full_path(self) -> Path:
        return self.base_dir / self.model_path

    @property
    def faiss_full_path(self) -> Path:
        return self.base_dir / self.faiss_index_path

    @property
    def policies_full_path(self) -> Path:
        return self.base_dir / self.policies_path


@lru_cache()
def get_settings() -> Settings:
    return Settings()
