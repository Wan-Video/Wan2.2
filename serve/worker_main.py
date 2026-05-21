from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from .config import Settings
from .launcher import launch_generate_job
from .store import TaskStore

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)


def main_master(settings: Settings, store: TaskStore):
    """Master node: pull tasks from Redis queue, signal worker1 via Redis, run torchrun."""
    logging.info("Master worker started; queue=%s, nnodes=%d", settings.queue_name, settings.nnodes)

    while True:
        task_id = store.brpop_task_id(timeout=10)
        if not task_id:
            continue
        doc = store.get_internal(task_id)
        if not doc:
            logging.warning("Missing task doc for %s", task_id)
            continue

        if not store.acquire_cluster_lock():
            logging.info("Cluster busy; requeue %s", task_id)
            store.requeue(task_id)
            time.sleep(3)
            continue

        try:
            store.update(task_id, status="RUNNING")
            job = doc["job"]
            job_path = Path(settings.job_dir) / f"{task_id}.json"
            job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
            rdzv_id = f"{settings.rdzv_id_prefix}-{task_id}"

            # Signal worker1 to join this torchrun job
            if settings.nnodes > 1:
                signal = json.dumps({
                    "task_id": task_id,
                    "rdzv_id": rdzv_id,
                    "job": job,
                })
                store.publish_signal(signal)
                logging.info("Published signal for worker1: %s / rdzv_id=%s", task_id, rdzv_id)

            rc = launch_generate_job(settings, job_path, rdzv_id)
            out_path = job.get("save_file")
            if rc != 0:
                store.update(
                    task_id,
                    status="FAILED",
                    message=f"torchrun exited with code {rc}",
                )
            elif out_path and Path(out_path).is_file():
                store.update(
                    task_id,
                    status="SUCCEEDED",
                    output_path=out_path,
                )
            else:
                store.update(
                    task_id,
                    status="FAILED",
                    message="save_file missing after run",
                )
        except Exception as e:
            logging.exception("task %s failed", task_id)
            store.update(task_id, status="FAILED", message=str(e))
        finally:
            store.release_cluster_lock()


def main_worker(settings: Settings, store: TaskStore):
    """Worker node (secondary): listen for signal from master, write local job JSON, run torchrun."""
    logging.info("Worker node started; waiting for signals on %s", settings.signal_key)

    while True:
        raw = store.brpop_signal(timeout=10)
        if not raw:
            continue

        try:
            signal = json.loads(raw)
            task_id = signal["task_id"]
            rdzv_id = signal["rdzv_id"]
            job = signal["job"]
            logging.info("Received signal: task=%s, rdzv_id=%s", task_id, rdzv_id)

            # Write job JSON locally so generate_job.py can read it
            job_path = Path(settings.job_dir) / f"{task_id}.json"
            job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")

            rc = launch_generate_job(settings, job_path, rdzv_id)
            logging.info("Worker torchrun for %s finished with rc=%d", task_id, rc)
        except Exception as e:
            logging.exception("worker failed processing signal: %s", raw)


def main():
    settings = Settings.from_env()
    Path(settings.job_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    store = TaskStore(settings)

    # Clear stale cluster lock on startup
    store.release_cluster_lock()

    if settings.node_role == "master":
        main_master(settings, store)
    else:
        main_worker(settings, store)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)