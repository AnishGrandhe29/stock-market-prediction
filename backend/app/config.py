"""
Configuration management using Pydantic Settings.
Loads environment variables and provides typed config access.
"""
from functools import lru_cache
from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the directory where this config.py file is located
ENV_FILE_PATH = Path(__file__).parent / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Database
    database_url: str = Field(default="postgresql+asyncpg://postgres:password@localhost:5432/nifty50_predict")
    redis_url: str = Field(default="redis://localhost:6379")
    
    # Security
    secret_key: str = Field(...)  # Requires environment variable setup. No fallback strings.
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7
    
    # Google OAuth
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/api/v1/auth/google/callback"
    
    # Reddit API
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "NIFTY50Predictor/1.0"
    
    # Stock Data APIs (backup for Yahoo Finance)
    alpha_vantage_api_key: str = ""
    
    # Frontend
    frontend_url: str = "http://localhost:3000"
    cors_origins: List[str] = ["http://localhost:3000"]
    
    # Model
    model_path: str = "../models/nifty50_model.pt"
    price_scaler_path: str = "../models/price_scaler.pkl"
    tech_scaler_path: str = "../models/tech_scaler.pkl"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
