import os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


class Settings:
    PROJECT_NAME: str = "Wan2.2 API"
    VERSION: str = "1.0.0"

    # Environment: "local" | "production"
    ENV: str = os.getenv("ENV", "local")

    # Database — SQLite for local, PostgreSQL for production
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./wan22.db" if os.getenv("ENV", "local") == "local"
        else "postgresql://postgres:postgres@localhost:5432/wan22"
    )

    # Redis / Celery (optional in local mode)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = REDIS_URL
    CELERY_RESULT_BACKEND: str = REDIS_URL

    # S3 / DO Spaces
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT", "https://nyc3.digitaloceanspaces.com")
    S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY", "")
    S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY", "")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "wan22-videos")

    # Wan2.2 paths
    WAN2_2_PATH: str = os.getenv("WAN2_2_PATH", str(Path(__file__).parent.parent.parent / "wan2.2"))
    WAN2_2_CKPT_DIR: str = os.getenv("WAN2_2_CKPT_DIR", "cache/TI2V-5B")
    WAN2_2_DEFAULT_TASK: str = "ti2v-5B"
    WAN2_2_DEFAULT_SIZE: str = "704*1280"

    # Mock mode — simulate generation without GPU
    MOCK_MODE: bool = os.getenv("MOCK_MODE", "true" if os.getenv("ENV", "local") == "local" else "false").lower() == "true"

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Output directory (local)
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", str(Path(__file__).parent.parent / "outputs"))

    @property
    def is_local(self) -> bool:
        return self.ENV == "local"

    @property
    def is_mock(self) -> bool:
        return self.MOCK_MODE or self.is_local


settings = Settings()

# Ensure output dir exists
Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
