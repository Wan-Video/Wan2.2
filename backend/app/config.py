import os
from dotenv import load_dotenv
load_dotenv()


class Settings:
    PROJECT_NAME: str = "Wan2.2 API"
    VERSION: str = "1.0.0"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/wan22")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = REDIS_URL
    CELERY_RESULT_BACKEND: str = REDIS_URL
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT", "https://nyc3.digitaloceanspaces.com")
    S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY", "")
    S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY", "")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "wan22-videos")
    WAN2_2_PATH: str = os.getenv("WAN2_2_PATH", "/workspace/Wan2.2")
    WAN2_2_CKPT_DIR: str = os.getenv("WAN2_2_CKPT_DIR", "cache/TI2V-5B")
    WAN2_2_DEFAULT_TASK: str = "ti2v-5B"
    WAN2_2_DEFAULT_SIZE: str = "704*1280"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()
