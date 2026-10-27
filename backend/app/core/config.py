# Configuration de l'application en utilisant Pydantic pour gérer les paramètres de configuration
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The Settings class is used to manage the application's configuration using Pydantic. It loads settings from a .env file located at the root of the project and provides default values for certain parameters.
BASE_DIR = Path(__file__).resolve().parents[2]  # = backend/

# Configuration de l'application en utilisant Pydantic pour gérer les paramètres de configuration
# Les paramètres sont chargés à partir d'un fichier .env situé à la racine du projet
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        extra="ignore",
    )

    database_url: str
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    groq_api_key: str | None = None

# Instanciation de la classe Settings pour accéder aux paramètres de configuration dans l'application
settings = Settings()