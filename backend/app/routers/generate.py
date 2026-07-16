import uuid
from fastapi import APIRouter, HTTPException
from app.schemas.job import GenerateRequest, JobResponse, JobStatus

router = APIRouter()
jobs_store = {}

@router.post("/generate", response_model=JobResponse)
async def create_job(req: GenerateRequest):
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {"status": "queued", "progress": 0.0,
                          "request": req.model_dump(), "output_url": None, "error": None}
    return JobResponse(job_id=job_id, status="queued", estimated_seconds=540)

@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    job = jobs_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobStatus(job_id=job_id, status=job["status"], progress=job["progress"],
                     output_url=job["output_url"], error=job["error"])

@router.get("/download/{job_id}")
async def download_video(job_id: str):
    job = jobs_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "completed":
        raise HTTPException(400, "Not completed")
    if not job["output_url"]:
        raise HTTPException(404, "No output")
    return {"url": job["output_url"]}
