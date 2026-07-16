import os, tempfile, shutil
from pathlib import Path
from celery import Task
from celery_app import celery_app
from app.config import settings
from app.services.wan_worker import Wan2_2Worker


class WanTask(Task):
    _worker = None
    @property
    def worker(self):
        if self._worker is None:
            self._worker = Wan2_2Worker(wan_path=settings.WAN2_2_PATH,
                                        ckpt_dir=settings.WAN2_2_CKPT_DIR)
        return self._worker


@celery_app.task(base=WanTask, bind=True, name="generate_video")
def generate_video(self, job_id: str, params: dict):
    self.update_state(state="PROCESSING", meta={"progress": 0.0})
    wan_path = Path(settings.WAN2_2_PATH)
    outputs_dir = wan_path / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    output_path = outputs_dir / f"{job_id}.mp4"

    image_path = None
    if params.get("image"):
        import base64
        image_dir = wan_path / "keyframes"
        image_dir.mkdir(exist_ok=True)
        image_path = str(image_dir / f"{job_id}_input.png")
        try:
            img_data = base64.b64decode(params["image"])
            with open(image_path, "wb") as f: f.write(img_data)
        except Exception:
            image_path = params["image"]

    try:
        self.update_state(state="PROCESSING", meta={"progress": 0.1})
        out = self.worker.generate(
            prompt=params["prompt"], image_path=image_path,
            task=params.get("model", "ti2v-5B"), size=params.get("size", "704*1280"),
            steps=params.get("steps", 30), guidance=params.get("guidance_scale", 6.0),
            seed=params.get("seed"), output_path=str(output_path))
        self.update_state(state="SUCCESS", meta={"progress": 1.0, "output_url": f"/api/download/{job_id}"})
        return {"output_url": f"/api/download/{job_id}", "job_id": job_id}
    except Exception as e:
        self.update_state(state="FAILURE", meta={"progress": 0.0, "error": str(e)})
        raise
