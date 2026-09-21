from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import health, generate
from app.config import settings
from app.database import init_db

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(generate.router, prefix="/api", tags=["generate"])


@app.on_event("startup")
async def startup():
    # Initialize database tables
    init_db()
    print(f"  Mode: {'MOCK' if settings.is_mock else 'LIVE'}")
    print(f"  DB: {settings.DATABASE_URL}")
    print(f"  Worker: {'MockWorker' if settings.is_mock else 'Wan2_2Worker'}")
    print(f"  Output: {settings.OUTPUT_DIR}")
    print(f"  API: http://localhost:{settings.PORT}")
