from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import health, generate
from app.config import settings

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(health.router, tags=["health"])
app.include_router(generate.router, prefix="/api", tags=["generate"])
