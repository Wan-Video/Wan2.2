import uuid, threading, os
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.job import Job
from app.schemas.job import GenerateRequest, JobResponse, JobStatus
from app.config import settings
from app.services.wan_worker import Wan2_2Worker
from app.services.mock_worker import MockWorker
from pathlib import Path

router = APIRouter()

# Initialize worker based on environment
if settings.is_mock or not Path(settings.WAN2_2_PATH).exists():
    worker = MockWorker()
else:
    worker = Wan2_2Worker(wan_path=settings.WAN2_2_PATH, ckpt_dir=settings.WAN2_2_CKPT_DIR)


def run_in_background(job_id: str, params: dict, db_session_factory):
    """Run generation in a background thread"""
    db = db_session_factory()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return

        job.status = "processing"
        db.commit()

        def update_progress(progress: float):
            job.progress = progress
            db.commit()

        image_path = None
        if params.get("image"):
            import base64
            image_dir = Path(settings.OUTPUT_DIR) / "inputs"
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = str(image_dir / f"{job_id}_input.png")
            try:
                img_data = base64.b64decode(params["image"])
                with open(image_path, "wb") as f:
                    f.write(img_data)
            except Exception:
                image_path = params["image"]

        output_filename = f"{job_id}.mp4"
        output_path = str(Path(settings.OUTPUT_DIR) / output_filename)

        out = worker.generate(
            prompt=params["prompt"],
            image_path=image_path,
            task=params.get("model", "ti2v-5B"),
            size=params.get("size", "704*1280"),
            steps=params.get("steps", 30),
            guidance=params.get("guidance_scale", 6.0),
            seed=params.get("seed"),
            output_path=output_path,
            progress_callback=update_progress if hasattr(worker, "generate") and isinstance(worker, MockWorker) else None,
        )

        job.status = "completed"
        job.progress = 1.0
        job.output_path = out
        from datetime import datetime
        job.completed_at = datetime.utcnow()
        db.commit()

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        db.commit()
    finally:
        db.close()


@router.post("/generate", response_model=JobResponse)
async def create_job(req: GenerateRequest, db: Session = Depends(get_db)):
    job = Job(
        model=req.model,
        prompt=req.prompt,
        image_path=req.image,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        size=req.size,
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Run in background thread
    from app.database import SessionLocal
    thread = threading.Thread(
        target=run_in_background,
        args=(job.id, req.model_dump(), SessionLocal),
        daemon=True,
    )
    thread.start()

    return JobResponse(job_id=job.id, status="queued", estimated_seconds=540)


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    return JobStatus(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        output_url=f"/api/download/{job.id}" if job.status == "completed" else None,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.get("/gallery")
async def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(Job).order_by(Job.created_at.desc()).limit(50).all()
    return [
        {
            "job_id": j.id,
            "status": j.status,
            "prompt": j.prompt[:100] if j.prompt else "",
            "output_url": f"/api/download/{j.id}" if j.status == "completed" else None,
            "created_at": j.created_at.isoformat() if j.created_at else None,
        }
        for j in jobs
    ]


@router.get("/download/{job_id}")
async def download_video(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "completed":
        raise HTTPException(400, "Job not completed yet")
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(404, "Output file not found")
    return FileResponse(job.output_path, media_type="video/mp4", filename=f"{job_id}.mp4")
