from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    database_url: str = "postgresql+psycopg://postgres:password@localhost:5432/guardrag"

    llm_api_key: str = Field(
        default="", validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY")
    )
    llm_model: str = Field(
        default="gpt-4o-mini", validation_alias=AliasChoices("LLM_MODEL", "OPENAI_MODEL")
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL"),
    )
    llm_timeout: float = 60.0

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    top_k: int = 4
    cache_similarity_threshold: float = 0.90

    chunk_size: int = 1000
    chunk_overlap: int = 150
    max_upload_mb: int = 20

    # Lightweight OCR for image-only (scanned) pages via the tesseract CLI.
    ocr_enabled: bool = True
    ocr_languages: str = "eng+ben"
    ocr_dpi: int = 150
    # 0 = no limit. Caps OCR work per upload (a 1000-page scan is slow).
    ocr_page_limit: int = 0


@lru_cache
def get_settings() -> Settings:
    return Settings()


def env_file_path() -> Path:
    return Path(".env")


def persist_env(updates: dict[str, str]) -> None:
    """Write setting values back to the .env file (upper-case env keys)."""
    env = {key.upper(): value for key, value in updates.items()}
    path = env_file_path()
    lines = path.read_text().splitlines() if path.exists() else []
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key and key in env:
            output.append(f"{key}={env[key]}")
            seen.add(key)
        else:
            output.append(line)
    for key, value in env.items():
        if key not in seen:
            output.append(f"{key}={value}")
    path.write_text("\n".join(output) + "\n")
