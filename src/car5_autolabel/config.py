from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_host: str = "127.0.0.1"
    app_port: int = 8090
    model_device: str = "cuda:0"
    model_path: str = "weights/detector.pt"
    label_studio_url: str = "http://localhost:8080"
    label_studio_api_token: str = ""


settings = Settings()
