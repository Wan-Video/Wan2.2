from __future__ import annotations

from fastapi import Header, HTTPException

from .config import Settings


async def require_bearer(
    authorization: str | None = Header(default=None),
    *,
    settings: Settings,
) -> None:
    if not settings.api_keys:
        raise HTTPException(
            status_code=503,
            detail="Server misconfigured: set WAN_SERVE_API_KEYS",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token not in settings.api_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
