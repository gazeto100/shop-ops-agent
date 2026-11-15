"""Single source of truth for configuration. Read env vars here and nowhere else."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

type LLMProvider = Literal["openai", "anthropic", "ollama"]

# Used when LLM_MODEL is empty, so switching LLM_PROVIDER alone never sends one
# provider's model name to another. An explicit LLM_MODEL always wins.
DEFAULT_LLM_MODELS: dict[LLMProvider, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
    "ollama": "llama3.2",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: LLMProvider = "openai"
    llm_model: str = ""  # empty -> DEFAULT_LLM_MODELS[llm_provider]
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    embedding_model: str = "intfloat/multilingual-e5-small"
    chroma_path: str = "./data/chroma"
    # Live price/stock source for the tools (#6); re-read whenever the file changes.
    catalog_path: str = "./data/products.sample.csv"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    run_live_evals: bool = False
    # Faithfulness judge for the live eval (#9): a different, stronger model than the
    # agent's Haiku, so the agent isn't graded by its own blind spots. Same keys as above.
    judge_provider: LLMProvider = "anthropic"
    judge_model: str = "claude-sonnet-5-5"

    @property
    def resolved_llm_model(self) -> str:
        return self.llm_model or DEFAULT_LLM_MODELS[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
