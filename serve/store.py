from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import redis

from .config import Settings


def _task_key(settings: Settings, task_id: str) -> str:
    return f"{settings.task_key_prefix}{task_id}"


class TaskStore:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._r = redis.Redis.from_url(settings.redis_url, decode_responses=True)

    def create_task(self, task_id: str, request_id: str, job: Dict[str, Any]) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        doc = {
            "task_id": task_id,
            "request_id": request_id,
            "status": "PENDING",
            "message": "",
            "output_path": None,
            "created_at": now,
            "updated_at": now,
            "job": job,
        }
        self._r.set(_task_key(self._settings, task_id), json.dumps(doc))
        self._r.lpush(self._settings.queue_name, task_id)

    def get_public(self, task_id: str) -> Optional[Dict[str, Any]]:
        raw = self._r.get(_task_key(self._settings, task_id))
        if not raw:
            return None
        doc = json.loads(raw)
        job = doc.get("job") or {}
        out: Dict[str, Any] = {
            "task_id": doc["task_id"],
            "task_status": doc["status"],
            "message": doc.get("message") or "",
            "output": {},
        }
        if doc.get("output_path"):
            out["output"]["video_url"] = (
                f"/api/v1/files/by-task/{doc['task_id']}"
            )
            out["output"]["path"] = doc["output_path"]
        out["request_id"] = doc.get("request_id")
        out["model"] = job.get("model") or job.get("task")
        return out

    def get_internal(self, task_id: str) -> Optional[Dict[str, Any]]:
        raw = self._r.get(_task_key(self._settings, task_id))
        if not raw:
            return None
        return json.loads(raw)

    def update(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        message: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> None:
        raw = self._r.get(_task_key(self._settings, task_id))
        if not raw:
            return
        doc = json.loads(raw)
        if status is not None:
            doc["status"] = status
        if message is not None:
            doc["message"] = message
        if output_path is not None:
            doc["output_path"] = output_path
        doc["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._r.set(_task_key(self._settings, task_id), json.dumps(doc))

    def acquire_cluster_lock(self) -> bool:
        ok = self._r.set(
            self._settings.lock_key,
            "1",
            nx=True,
            ex=self._settings.cluster_lock_ttl_sec,
        )
        return bool(ok)

    def release_cluster_lock(self) -> None:
        self._r.delete(self._settings.lock_key)

    def brpop_task_id(self, timeout: int = 5) -> Optional[str]:
        item = self._r.brpop(self._settings.queue_name, timeout=timeout)
        if not item:
            return None
        return item[1]

    def requeue(self, task_id: str) -> None:
        self._r.rpush(self._settings.queue_name, task_id)

    def publish_signal(self, payload: str) -> None:
        self._r.lpush(self._settings.signal_key, payload)

    def brpop_signal(self, timeout: int = 10) -> Optional[str]:
        item = self._r.brpop(self._settings.signal_key, timeout=timeout)
        if not item:
            return None
        return item[1]
