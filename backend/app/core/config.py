# Configuration settings for the application, including database URL, LLM model, and embedding model.
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define a Settings class that inherits from BaseSettings to manage application configuration.
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    database_url: str
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

# Create an instance of the Settings class to access configuration values throughout the application.
settings = Settings()