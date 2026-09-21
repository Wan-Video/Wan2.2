from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    model: str = "ti2v-5b"
    prompt: str
    image: Optional[str] = None
    steps: int = 30
    guidance_scale: float = 6.0
    seed: Optional[int] = None
    size: str = "704*1280"


class JobResponse(BaseModel):
    job_id: str
    status: str
    estimated_seconds: int = 540


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    output_url: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
