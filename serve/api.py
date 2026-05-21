from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .auth import require_bearer
from .config import Settings
from .job_build import request_to_job
from .schemas import (
    HealthResponse,
    ModelEnum,
    OutputTaskId,
    TaskStatusBody,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from .store import TaskStore

_settings: Optional[Settings] = None
_store: Optional[TaskStore] = None


def get_settings() -> Settings:
    assert _settings is not None
    return _settings


def get_store() -> TaskStore:
    assert _store is not None
    return _store


def _validate_model_input(body: VideoGenerationRequest) -> None:
    """Per-model field validation — return clear 400 errors for missing required inputs."""
    model = body.model.value
    inp = body.input
    params = body.parameters

    if not inp.prompt:
        raise HTTPException(status_code=400, detail=f"model '{model}' requires input.prompt")

    # i2v-A14B and ti2v-5B require image
    if model in (ModelEnum.i2v_a14b.value, ModelEnum.ti2v_5b.value) and not inp.image:
        raise HTTPException(
            status_code=400,
            detail=f"model '{model}' requires input.image",
        )

    # animate-14B requires video + pose (src_root_path)
    if model == ModelEnum.animate_14b.value and not inp.video:
        raise HTTPException(
            status_code=400,
            detail=f"model '{model}' requires input.video (reference video path)",
        )

    # s2v-14B requires image + audio (or enable_tts)
    if model == ModelEnum.s2v_14b.value:
        if not inp.image:
            raise HTTPException(
                status_code=400,
                detail=f"model '{model}' requires input.image",
            )
        if not inp.audio and not params.enable_tts:
            raise HTTPException(
                status_code=400,
                detail=f"model '{model}' requires input.audio or parameters.enable_tts=true",
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _settings, _store
    _settings = Settings.from_env()
    Path(_settings.job_dir).mkdir(parents=True, exist_ok=True)
    Path(_settings.output_dir).mkdir(parents=True, exist_ok=True)
    _store = TaskStore(_settings)
    yield
    _store = None
    _settings = None


app = FastAPI(
    title="Wan2.2 Video Generation API",
    version="1.0.0",
    lifespan=lifespan,
)


async def _auth_dep(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
):
    await require_bearer(authorization, settings=settings)


@app.get("/healthz", response_model=HealthResponse)
def healthz():
    return HealthResponse()


@app.post(
    "/api/v1/video/generation",
    response_model=VideoGenerationResponse,
    dependencies=[Depends(_auth_dep)],
)
def create_video_job(
    body: VideoGenerationRequest,
    settings: Settings = Depends(get_settings),
    store: TaskStore = Depends(get_store),
):
    if not settings.ckpt_dir and not (body.parameters.ckpt_dir):
        raise HTTPException(
            status_code=400,
            detail="Set WAN_CKPT_DIR or parameters.ckpt_dir",
        )

    _validate_model_input(body)

    task_id = f"wan-{uuid.uuid4().hex}"
    request_id = str(uuid.uuid4())
    job = request_to_job(body, task_id=task_id, settings=settings)
    store.create_task(task_id, request_id, job)
    return VideoGenerationResponse(
        request_id=request_id,
        output=OutputTaskId(task_id=task_id),
    )


@app.get(
    "/api/v1/tasks/{task_id}",
    dependencies=[Depends(_auth_dep)],
)
def get_task(task_id: str, store: TaskStore = Depends(get_store)):
    doc = store.get_public(task_id)
    if not doc:
        raise HTTPException(status_code=404, detail="task not found")
    return TaskStatusBody(
        task_id=doc["task_id"],
        task_status=doc["task_status"],
        message=doc.get("message", ""),
        output=doc.get("output") or {},
        request_id=doc.get("request_id"),
        model=doc.get("model"),
    )


@app.get(
    "/api/v1/files/by-task/{task_id}",
    dependencies=[Depends(_auth_dep)],
)
def download_task_video(task_id: str, store: TaskStore = Depends(get_store)):
    doc = store.get_internal(task_id)
    if not doc or doc.get("status") != "SUCCEEDED":
        raise HTTPException(status_code=404, detail="video not ready")
    path = doc.get("output_path")
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="file missing on server")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{task_id}.mp4",
    )


def create_app() -> FastAPI:
    return app


# File upload endpoint for WebUI
@app.post(
    "/api/v1/files/upload",
    dependencies=[Depends(_auth_dep)],
)
async def upload_file(
    file: UploadFile = File(...),
    category: str = Form("general"),
    settings: Settings = Depends(get_settings),
):
    """Upload a file (image, audio, video) to the server for use in generation."""
    import shutil
    upload_dir = Path(settings.job_dir) / "uploads" / category
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Use only ASCII hex name to avoid encoding issues across containers
    ext = Path(file.filename).suffix.lower()
    # Convert webp/heic to jpg for PIL compatibility
    if ext in (".webp", ".heic", ".heif"):
        ext = ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest = upload_dir / unique_name

    if category == "image" and ext == ".jpg" and Path(file.filename).suffix.lower() in (".webp", ".heic", ".heif"):
        # Convert webp/heic to jpg via PIL
        from PIL import Image as PILImage
        img = PILImage.open(file.file).convert("RGB")
        img.save(dest, "JPEG", quality=95)
    else:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

    return {"path": str(dest), "filename": unique_name}


# WebUI — serve root page
_static_dir = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def webui():
    index = _static_dir / "index.html"
    if index.is_file():
        return index.read_text(encoding="utf-8")
    return HTMLResponse("<h1>WebUI not found</h1>", status_code=404)


if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir), html=True), name="static")