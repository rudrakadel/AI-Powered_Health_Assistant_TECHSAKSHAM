"""
Configuration management using Pydantic Settings.
Loads from environment variables with .env file support.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application configuration with environment variable loading."""
    
    # Dataset
    dataset_path: str = r"D:\archive\train_data_chatbot.csv"
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    
    # OpenAI (optional, can be overridden via GUI)
    openai_api_key: Optional[str] = None
    
    # Gemini (optional, can be overridden via GUI)
    gemini_api_key: Optional[str] = None
    
    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    
    # Application
    log_level: str = "INFO"
    max_retries: int = 3
    retry_delay: int = 2
    top_k_results: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


# Global settings instance
settings = Settings()
