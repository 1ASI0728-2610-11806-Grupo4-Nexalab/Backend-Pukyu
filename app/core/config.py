from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Configuración centralizada del backend Parky AI.
    Lee automáticamente las variables de entorno o del archivo .env.
    """
    # Supabase
    SUPABASE_URL: str
    SUPABASE_KEY: str
    
    # Groq (Chatbot LLM)
    GROQ_API_KEY: str
    
    # YOLO Model
    YOLO_MODEL: str = "yolov8n"
    
    # App Config
    APP_NAME: str = "Parky AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Devuelve la instancia singleton de Settings (cacheada)."""
    return Settings()
