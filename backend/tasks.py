"""Celery tasks for GPU workers (used when not in mock mode)
Run with: celery -A backend.tasks worker --loglevel=info
"""
import os, sys

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from celery import Task
from celery_app import celery_app
from app.config import settings
from app.services.wan_worker import Wan2_2Worker


class WanTask(Task):
    _worker = None

    @property
    def worker(self):
        if self._worker is None:
            self._worker = Wan2_2Worker(
                wan_path=settings.WAN2_2_PATH,
                ckpt_dir=settings.WAN2_2_CKPT_DIR,
            )
        return self._worker


@celery_app.task(base=WanTask, bind=True, name="generate_video")
def generate_video(self, job_id: str, params: dict):
    """Run Wan2.2 inference in background"""
    from app.database import SessionLocal
    from app.models.job import Job
    from pathlib import Path

    self.update_state(state="PROCESSING", meta={"progress": 0.0})

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.status = "processing"
        db.commit()

        outputs_dir = Path(settings.OUTPUT_DIR)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(outputs_dir / f"{job_id}.mp4")

        self.worker.generate(
            prompt=params["prompt"],
            image_path=params.get("image"),
            task=params.get("model", "ti2v-5B"),
            size=params.get("size", "704*1280"),
            steps=params.get("steps", 30),
            guidance=params.get("guidance_scale", 6.0),
            seed=params.get("seed"),
            output_path=output_path,
        )

        job.status = "completed"
        job.progress = 1.0
        job.output_path = output_path
        from datetime import datetime
        job.completed_at = datetime.utcnow()
        db.commit()

        return {"output_url": f"/api/download/{job_id}", "job_id": job_id}

    except Exception as e:
        if job:
            job.status = "failed"
            job.error = str(e)
            db.commit()
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
    finally:
        db.close()
